"""
Filter the data
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'preprocess')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import segmentation
import load_sensor_data, load_data_path, load_data_basic
import pandas as pd
import argparse


def main(tiles_data_path, config_path, experiment):
    ###########################################################
    # 1. Create Config
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess_data', segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data', clustering_data_identifier='clustering')
    
    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, experiment, 'participant_id.csv.gz'), tiles_data_path, k=150, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        ###########################################################
        # 3. Create filter class
        ###########################################################
        ggs_segmentation = segmentation.Segmentation(data_config=data_config, participant_id=participant_id)
        
        ###########################################################
        # 4. Read segmentation data
        ###########################################################
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, participant_id)
        
        if fitbit_df is None:
            continue
        
        ###########################################################
        # 5. Segmentation
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']
        
        ggs_segmentation.segment_data_by_sleep(fitbit_mean, fitbit_std, fitbit_df, fitbit_summary_df=fitbit_summary_df)
        
        del ggs_segmentation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiement = 'baseline' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiement)