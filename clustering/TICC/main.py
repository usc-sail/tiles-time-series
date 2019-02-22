#!/usr/bin/env python3

import os
import sys
from configparser import ConfigParser
import argparse
import pandas as pd
import numpy as np
from TICC_solver import TICC


###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'plot')))

import load_data_path, load_sensor_data, load_data_basic
import config


def main(tiles_data_path, config_path, experiement):
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiement)
    
    # Load preprocess folder
    load_data_path.load_preprocess_path(data_config, process_data_path, data_name='preprocess_data')
    
    # Load segmentation folder
    load_data_path.load_segmentation_path(data_config, process_data_path, data_name='segmentation')
    
    # Load clustering folder
    load_data_path.load_clustering_path(data_config, process_data_path, data_name='clustering')
    
    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')
    
    ###########################################################
    # Read ground truth data
    ###########################################################
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    if os.path.exists(os.path.join(process_data_path, experiement, 'participant_id.csv.gz')) is True:
        top_participant_id_df = pd.read_csv(os.path.join(process_data_path, experiement, 'participant_id.csv.gz'), index_col=0, compression='gzip')
    else:
        participant_id_list = load_data_basic.return_participant(tiles_data_path)
        participant_id_list.sort()
        top_participant_id_df = load_data_basic.return_top_k_participant(participant_id_list, k=150, data_config=data_config)
        top_participant_id_df.to_csv(os.path.join(process_data_path, experiement, 'participant_id.csv.gz'), compression='gzip')

    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    top_participant_id_list = top_participant_id_list[0:]

    ###########################################################
    # 2. Learn TICC
    ###########################################################
    signal_type, imputation_folder_postfix = 'om_signal', '_imputation_all'

    ticc = TICC(signal_type=signal_type, main_folder=ticc_folder,
                process_hyper=default_process_hype,
                signal_hyper=default_om_signal,
                window_size=5, number_of_clusters=10,
                lambda_parameter=1e-2, beta=100, maxIters=300,
                threshold=2e-5, write_out_file=False,
                prefix_string="output_folder/", num_proc=1)
    ticc.load_model_parameters()

    ###########################################################
    # 3.1 All method
    ###########################################################
    method_array = ['brits', 'trmf', 'mean', 'naive']
    # method_array = ['mean']

    ###########################################################
    # 3.2 Missing rate
    ###########################################################
    mp_array = [0.05, 0.1, 0.25, 0.5]

    ###########################################################
    # 4. Iterate over settings
    ###########################################################
    for mp in mp_array:
        ###########################################################
        # 4.1 Init imputed data loader
        ###########################################################
        signal_type, imputation_folder_postfix = 'om_signal', '_imputation_all'
    
        data_loader = ImputedDataLoader(signal_type=signal_type,
                                        main_folder=imputed_folder,
                                        participant_id_array=participant_id_array,
                                        mp=mp, postfix=imputation_folder_postfix, method_hyper=None,
                                        original_main_folder_postfix='_set', method='mean',
                                        process_hyper=default_process_hype,
                                        signal_hyper=default_om_signal)
    
        ###########################################################
        # 4.2 Load ground truth data and mask
        ###########################################################
        data_loader.load_ground_truth_data()

        ###########################################################
        # 4.3 Load global statistics of data
        ###########################################################
        data_loader.load_data_stats(method='mean')
        data_loader.load_data_stats(method='std')

        for method in method_array:
            
            ###########################################################
            # 4.4 Load imputed values
            ###########################################################
            if method == 'trmf' or method == 'brits' or method == 'brits_multitask':
                method_dict_array = init_process_hyperparameter(method=method)
                for method_dict in method_dict_array:
                    ###########################################################
                    # Update method
                    ###########################################################
                    data_loader.update_method(method=method, method_hyper=method_dict, mp=mp)
                    print('mp: %s, method %s, param: %s' % (str(mp), method, data_loader.method_str))

                    ticc.update_imputation_method(method=method, method_hyper=method_dict, mp=mp)
                    
                    data_loader.load_imputed_data()
                    if batch_norm == True:
                        ticc.predict_data(data_loader.imputed_data_array)
                    else:
                        ticc.predict_data(data_loader.imputed_data_array, global_stats=data_loader.global_stat_dict)
            else:
                ###########################################################
                # Update method
                ###########################################################
                data_loader.update_method(method=method, mp=mp)
                print('mp: %s, method %s' % (str(mp), method))
                
                ticc.update_imputation_method(method=method, mp=mp)
                
                data_loader.load_imputed_data()
                if batch_norm == True:
                    ticc.predict_data(data_loader.imputed_data_array)
                else:
                    ticc.predict_data(data_loader.imputed_data_array, global_stats=data_loader.global_stat_dict)

        del data_loader
     
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiement", required=False, help="Experiement name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiement = 'ticc' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiement)
