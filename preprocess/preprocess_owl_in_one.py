#!/usr/bin/env python3

import os
import sys
import argparse

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path

import load_data_basic
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
    participant_id_list.sort()
    
    ###########################################################
    # 2. Iterate over participant
    ###########################################################
    for idx, participant_id in enumerate(participant_id_list[:]):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(participant_id_list)))

        ###########################################################
        # 3. Initialize preprocess
        ###########################################################
        owl_in_one_preprocess = Preprocess(data_config, participant_id)

        ###########################################################
        # 4. Read owl_in_one data
        ###########################################################
        owl_in_one_data_df = load_sensor_data.read_owl_in_one(os.path.join(tiles_data_path, '3_preprocessed_data/owl_in_one/'), participant_id)

        ###########################################################
        # 5. Process owl_in_one data
        ###########################################################
        owl_in_one_preprocess.preprocess_owl_in_one(owl_in_one_data_df)
        
        del owl_in_one_preprocess


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
