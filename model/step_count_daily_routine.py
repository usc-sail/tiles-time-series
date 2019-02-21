#!/usr/bin/env python3

import os
import sys
import pdb
import argparse
import pandas as pd
import numpy as np
from configparser import ConfigParser
from dtw import dtw

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'plot')))

import config
import segmentation
import load_sensor_data
import load_data_basic
import summarize_sequence
import clustering

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

default_signal = {'MinPeakDistance': 100, 'MinPeakHeight': 0.04,
                  'raw_cols': ['Cadence', 'HeartRate', 'Intensity', 'Steps', 'BreathingDepth', 'BreathingRate']}

segmentation_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                     'segmentation': 'ggs', 'segmentation_lamb': 10e0, 'sub_segmentation_lamb': None,
                     'preprocess_cols': ['HeartRatePPG', 'StepCount'], 'imputation': 'iterative'}

preprocess_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                   'preprocess_cols': ['HeartRatePPG', 'StepCount'], 'imputation': 'iterative'}

owl_in_one_hype = {'method': 'ma', 'offset': 60, 'overlap': 0, 'imputation': None}


def return_participant(tiles_data_path):
    ###########################################################
    # 2. Read all fitbit file
    ###########################################################
    fitbit_folder = os.path.join(tiles_data_path, '3_preprocessed_data/fitbit/')
    fitbit_file_list = os.listdir(fitbit_folder)
    fitbit_file_dict_list = {}
    
    for fitbit_file in fitbit_file_list:
        
        if '.DS' in fitbit_file:
            continue
        
        participant_id = fitbit_file.split('_')[0]
        if participant_id not in list(fitbit_file_dict_list.keys()):
            fitbit_file_dict_list[participant_id] = {}
    return list(fitbit_file_dict_list.keys())


def compute_daily_difference(tiles_data_path):
    ###########################################################
    # 1. Create Config
    ###########################################################
    fitbit_config = config.Config(data_type='preprocess_data', sensor='fitbit',
                                  read_folder=os.path.abspath(os.path.join(os.pardir, 'data')),
                                  return_full_feature=False, process_hyper=preprocess_hype, signal_hyper=default_signal)
    
    owl_in_one_config = config.Config(data_type='preprocess_data', sensor='owl_in_one',
                                      read_folder=os.path.abspath(os.path.join(os.pardir, 'data')),
                                      return_full_feature=False, process_hyper=owl_in_one_hype,
                                      signal_hyper=default_signal)
    
    ggs_config = config.Config(data_type='segmentation', sensor='fitbit',
                               read_folder=os.path.abspath(os.path.join(os.pardir, 'data')), return_full_feature=False,
                               process_hyper=segmentation_hype, signal_hyper=default_signal)
    
    fitbit_summary_config = config.Config(data_type='3_preprocessed_data', sensor='fitbit', read_folder=tiles_data_path,
                                          return_full_feature=False, process_hyper=preprocess_hype,
                                          signal_hyper=default_signal)
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    participant_id_list = return_participant(tiles_data_path)
    participant_id_list.sort()
    
    top_participant_id_df = pd.read_csv(os.path.join(ggs_config.process_folder, 'participant_id.csv.gz'), index_col=0,
                                        compression='gzip')
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    for idx, participant_id in enumerate(top_participant_id_list):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(participant_id_list)))
        
        
        ###########################################################
        # 4.1 Read basic data for each participant
        ###########################################################
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        current_job_position = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
        
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
        
        ###########################################################
        # 4.2 Read owl_in_one data
        ###########################################################
        owl_in_one_df = load_sensor_data.read_processed_owl_in_one(owl_in_one_config, participant_id)
        
        ###########################################################
        # 4.3 Read fitbit summary data
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_config, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']
        
        ###########################################################
        # 4.4 Read fitbit data
        ###########################################################
        fitbit_df = load_sensor_data.read_processed_fitbit_with_pad(fitbit_config, participant_id)

        ###########################################################
        # 5. Read step count data only
        ###########################################################
        step_count_folder = os.path.abspath(os.path.join(os.pardir, 'data', 'routine'))
        if os.path.exists(step_count_folder) is False:
            os.mkdir(step_count_folder)
        
        if os.path.exists(os.path.join(step_count_folder, participant_id + '_step.csv.gz')) is False:

            step_data = np.array(fitbit_df.StepCount)
            num_of_days = (pd.to_datetime(fitbit_df.index[-1]) - pd.to_datetime(fitbit_df.index[0])).days
            
            step_data_per_day_list = []
            for i in range(num_of_days):
                day_step = step_data[i*1440:(i+1)*1440]
                day_valid_step = np.where(day_step >= 0)[0]
                
                if len(day_valid_step) > 1440 * 0.9:
                    # step_mean = np.mean(day_step[day_valid_step])
                
                    for j in np.where(day_step < 0)[0]:
                        day_step[j] = np.nan
                    step_data_per_day_list.append(day_step)
    
            step_data_per_day_array = np.array(step_data_per_day_list)
            step_data_per_day_df = pd.DataFrame(step_data_per_day_array)
            step_data_per_day_df = step_data_per_day_df.fillna(step_data_per_day_df.mean())
            
            step_data_per_day_df.to_csv(os.path.join(step_count_folder, participant_id + '_step.csv.gz'), compression='gzip')
        else:
            step_data_per_day_df = pd.read_csv(os.path.join(step_count_folder, participant_id + '_step.csv.gz'), index_col=0)
        
        if len(step_data_per_day_df) > 20:
            step_data_per_day_array = np.array(step_data_per_day_df)
            euclidean_norm = lambda x, y: np.abs(x - y)
            d, cost_matrix, acc_cost_matrix, path = dtw(step_data_per_day_array[0, :], step_data_per_day_array[1, :], dist=euclidean_norm)
            print()
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False,
                        help="Path to a config file specifying how to perform the clustering")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = 'configs/baseline.cfg' if args.config is None else args.config
    
    compute_daily_difference(tiles_data_path)
