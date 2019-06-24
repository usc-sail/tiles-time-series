#!/usr/bin/env python3

import os
import sys
import pandas as pd
import argparse

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))


import load_data_path, load_data_basic
import config
from preprocess import Preprocess


def main(tiles_data_path, config_path, experiment):
    ###########################################################
    # 0. Read configs
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))

    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)

    # Load preprocess folder
    load_data_path.load_preprocess_path(data_config, process_data_path, data_name='preprocess')

    ###########################################################
    # 1. Read all participant
    ###########################################################
    participant_id_list = load_data_basic.return_participant(tiles_data_path)

    ###########################################################
    # 2. Iterate all participant
    ###########################################################
    for i, participant_id in enumerate(participant_id_list[:]):
    
        print('Complete process for %s: %.2f' % (participant_id, 100 * i / len(participant_id_list)))
        
        ###########################################################
        # Read ppg and step count file path
        ###########################################################
        ppg_file = participant_id + '_heartRate.csv.gz'
        step_file = participant_id + '_stepCount.csv.gz'

        ppg_file_abs_path = os.path.join(tiles_data_path, '3_preprocessed_data/fitbit/', ppg_file)
        step_file_abs_path = os.path.join(tiles_data_path, '3_preprocessed_data/fitbit/', step_file)

        ###########################################################
        # Read ppg and step count data
        ###########################################################
        ppg_df = pd.read_csv(ppg_file_abs_path, index_col=0)
        ppg_df = ppg_df.sort_index()
        
        step_df = pd.read_csv(step_file_abs_path, index_col=0)
        step_df = step_df.sort_index()

        ###########################################################
        # 2.0 Iterate all fitbit files
        ###########################################################
        if len(ppg_df) > 0:
            
            ###########################################################
            # 2.1 Init fitbit preprocess
            ###########################################################
            fitbit_preprocess = Preprocess(data_config=data_config, participant_id=participant_id)
        
            ###########################################################
            # 2.2 Preprocess fitbit raw data array (No slice)
            ###########################################################
            fitbit_preprocess.process_fitbit(ppg_df, step_df)
            
            ###########################################################
            # 2.3 Delete current fitbit preprocess object
            ###########################################################
            del fitbit_preprocess
      
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiment)

