#!/usr/bin/env python3

import os
import sys
import argparse

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))


import load_data_path, load_data_basic, load_sensor_data
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
    # 2. Iterate all realizd files
    ###########################################################
    for participant_id in participant_id_list[150:]:
        
        ###########################################################
        # 2.0 Init realizd preprocess
        ###########################################################
        realizd_preprocess = Preprocess(data_config=data_config, participant_id=participant_id)

        ###########################################################
        # 2.1 Read realizd data
        ###########################################################
        realizd_df = load_sensor_data.read_realizd(os.path.join(tiles_data_path, '2_raw_csv_data/realizd/'), participant_id)

        ###########################################################
        # 2.2 Preprocess data
        ###########################################################
        if len(realizd_df) > 0:
            realizd_preprocess.preprocess_realizd(realizd_df)

        ###########################################################
        # 2.3 Delete current realizd_preprocess object
        ###########################################################
        del realizd_preprocess
      
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiment)
