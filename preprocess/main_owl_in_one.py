#!/usr/bin/env python3

import os
import sys

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

from load_data_basic import *
from preprocess import Preprocess

process_hyper_param = {'method': 'ma', 'offset': 60, 'overlap': 0, 'preprocess_cols': None}


def return_participant(main_folder):
    ###########################################################
    # Read all owl_in_one file
    ###########################################################
    owl_in_one_folder = os.path.join(main_folder, '3_preprocessed_data/owl_in_one/')
    owl_in_one_file_list = os.listdir(owl_in_one_folder)
    owl_in_one_file_dict_list = {}
    
    for owl_in_one_file in owl_in_one_file_list:
        
        if '.DS' in owl_in_one_file:
            continue
        
        participant_id = owl_in_one_file.split('_')[0]
        if participant_id not in list(owl_in_one_file_dict_list.keys()):
            owl_in_one_file_dict_list[participant_id] = {}
    return list(owl_in_one_file_dict_list.keys())


def main(main_folder):
    ###########################################################
    # 1. Create Config
    ###########################################################
    owl_in_one_read_config = config.Config(data_type='3_preprocessed_data', sensor='owl_in_one',
                                           read_folder=main_folder, return_full_feature=False,
                                           process_hyper=process_hyper_param)
    
    owl_in_one_save_config = config.Config(data_type='preprocess_data', sensor='owl_in_one',
                                           read_folder=os.path.abspath(os.path.join(os.pardir, '..')),
                                           return_full_feature=False,
                                           process_hyper=process_hyper_param)

    participant_id_list = return_participant(main_folder)
    participant_id_list.sort()

    ###########################################################
    # 2. Iterate over participant
    ###########################################################
    for idx, participant_id in enumerate(participant_id_list[200:]):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(participant_id_list)))

        ###########################################################
        # 3. Initialize preprocess
        ###########################################################
        owl_in_one_preprocess = Preprocess(participant_id=participant_id, process_hyper=process_hyper_param,
                                           read_config=owl_in_one_read_config, save_config=owl_in_one_save_config,
                                           save_main_folder=os.path.abspath(os.path.join(os.pardir, '../preprocess_data')))

        ###########################################################
        # 4. Read owl_in_one data
        ###########################################################
        owl_in_one_data_df = load_sensor_data.read_owl_in_one(owl_in_one_read_config, participant_id)

        ###########################################################
        # 5. Process owl_in_one data
        ###########################################################
        owl_in_one_preprocess.process_owl_in_one(owl_in_one_data_df)
        
        del owl_in_one_preprocess


if __name__ == "__main__":
    # Main Data folder
    main_folder = '../../../data/keck_wave_all/'
    
    main(main_folder)
