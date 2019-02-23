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

import load_data_basic, load_data_path, load_sensor_data
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
    # 1. Read all om_signal folder
    ###########################################################
    omsignal_folder = os.path.join(main_folder, '3_preprocessed_data/omsignal/')
    omsignal_file_list = os.listdir(omsignal_folder)
    
    for omsignal_file in omsignal_file_list:
        if 'DS' in omsignal_file or '.zip' in omsignal_file:
            omsignal_file_list.remove(omsignal_file)

    omsignal_file_list.sort()

    ###########################################################
    # 2. Iterate all omsignal files
    ###########################################################
    for participant_id in participant_id_list[150:]:
        
        # Read data
        omsignal_df = load_sensor_data.read_omsignal(os.path.join(tiles_data_path, '3_preprocessed_data/omsignal/'), participant_id)

        ###########################################################
        # 2.0 Iterate all omsignal files
        ###########################################################
        
        if len(omsignal_df) > 0:
            
            ###########################################################
            # 2.1 Init om_signal preprocess
            ###########################################################
            omsignal_preprocess = Preprocess(data_config=data_config, participant_id=participant_id)
        
            ###########################################################
            # 2.2 Slice the raw data array
            ###########################################################
            omsignal_preprocess.slice_raw_data(method='block')

            ###########################################################
            # 2.3 Preprocess data
            ###########################################################
            omsignal_preprocess.preprocess_slice_raw_data_full_feature(check_saved=True)
            
            ###########################################################
            # 2.5 Delete current omsignal_preprocess object
            ###########################################################
            del omsignal_preprocess
      
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath( os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'baseline' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiment)

