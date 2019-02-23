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

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

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
    load_data_path.load_preprocess_path(data_config, process_data_path, data_name='preprocess_data')
    
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
        # Read audio file path
        ###########################################################
        
        ###########################################################
        # Read audio data
        ###########################################################
        # should be pd.read_csv
        audio_data_df = pd.DataFrame()
        
        ###########################################################
        # 2.0 Iterate all fitbit files
        ###########################################################
        if len(audio_data_df) > 0:
            ###########################################################
            # 2.1 Init fitbit preprocess
            ###########################################################
            audio_preprocess = Preprocess(data_config=data_config, participant_id=participant_id)
            
            ###########################################################
            # 2.2 Preprocess audio data array
            ###########################################################
            # add your code inside this function
            audio_preprocess.preprocess_audio(audio_data_df)
            
            del audio_preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'baseline' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiment)

