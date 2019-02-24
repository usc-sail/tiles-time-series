#!/usr/bin/env python3

import os
import sys
from configparser import ConfigParser
import argparse
import pandas as pd
import numpy as np

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'plot')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'ticc')))

from TICC_solver import TICC
import load_data_path, load_sensor_data, load_data_basic
import config


def main(tiles_data_path, config_path, experiment):
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
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
    if os.path.exists(os.path.join(process_data_path, experiment, 'participant_id.csv.gz')) is True:
        top_participant_id_df = pd.read_csv(os.path.join(process_data_path, experiment, 'participant_id.csv.gz'), index_col=0, compression='gzip')
    else:
        participant_id_list = load_data_basic.return_participant(tiles_data_path)
        participant_id_list.sort()
        top_participant_id_df = load_data_basic.return_top_k_participant(participant_id_list, k=150, data_config=data_config)
        top_participant_id_df.to_csv(os.path.join(process_data_path, experiment, 'participant_id.csv.gz'), compression='gzip')

    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    top_participant_id_list = top_participant_id_list[100:]

    ###########################################################
    # 3. Learn ticc
    ###########################################################
    for idx, participant_id in enumerate(top_participant_id_list):
    
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        ###########################################################
        # 3. Create segmentation class
        ###########################################################
        ticc = TICC(data_config=data_config, maxIters=300, threshold=2e-5, num_proc=1,
                    lambda_parameter=data_config.fitbit_sensor_dict['ticc_sparsity'],
                    beta=data_config.fitbit_sensor_dict['ticc_switch_penalty'],
                    window_size=data_config.fitbit_sensor_dict['ticc_window'],
                    number_of_clusters=data_config.fitbit_sensor_dict['num_cluster'], participant_id=participant_id)
        
        if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['clustering_path'], participant_id + '.csv.gz')) is True:
            continue
    
        ###########################################################
        # 4. Read segmentation data
        ###########################################################
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, participant_id)
    
        if fitbit_df is None:
            continue
            
        ticc.fit(fitbit_df, fitbit_mean, fitbit_std)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiment)
