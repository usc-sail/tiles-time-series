"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import numpy as np
import pandas as pd
from datetime import timedelta
import itertools
import operator
import math
from sympy import *

from test_on_simulated_data import *
import cyclic_HMM

np.random.seed(42)
random.seed(42)
confirm_results_do_not_change = True

missing_data_val = -9999999


def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    mgt_df = load_data_basic.read_MGT(tiles_data_path)
    igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    data_dict = {}

    n_states = 3
    max_duration = 20
    duration_distribution_name = 'geometric'

    # 'geometric', 'poisson'
    process_cond_list = ['night_nurse', 'day_nurse', 'all_nurse']
    # process_cond_list = ['all_nurse', 'all']
    # process_cond_list = ['all_nurse']
    
    for process_cond in process_cond_list:
        participant_type = process_cond
    
        param_file_path = participant_type + '_state_' + str(n_states) + '_max_duration_' + str(max_duration) + '_' + duration_distribution_name + '_param.csv.gz'
        cycle_file_path = participant_type + '_state_' + str(n_states) + '_max_duration_' + str(max_duration) + '_' + duration_distribution_name + '_cycle.pkl'
    
        save_param_path = os.path.join(data_config.fitbit_sensor_dict['preprocess_path'], param_file_path)
        save_cycle_path = os.path.join(data_config.fitbit_sensor_dict['preprocess_path'], cycle_file_path)
        
        pkl_file = open(save_cycle_path, 'rb')
        data_dict = pickle.load(pkl_file)
        
        param_df = pd.read_csv(save_param_path, index_col=0)
        
        mean_list, std_list = [], []
        all_df = pd.DataFrame()
        
        for participant_id in list(data_dict.keys()):
            data = data_dict[participant_id]
            
            mean_list.append(data['cycle_mean'])
            std_list.append(data['cycle_std'])

            uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
            participant_igtb_df = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][igtb_cols]
            participant_igtb_df['cycle_mean'] = data['cycle_mean']
            participant_igtb_df['cycle_mean_diff'] = np.abs(data['cycle_mean'] - 24)
            participant_igtb_df['cycle_std'] = data['cycle_std']

            nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
            primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]

            nurse_cond = nurse == 1 and 'Dialysis' not in primary_unit
            day_shift_cond = shift == 'Day shift'
            night_shift_cond = shift == 'Night shift'
            lab_cond = 'Lab' not in primary_unit

            if nurse_cond and day_shift_cond:
                participant_igtb_df['cond'] = 1
            elif nurse_cond and night_shift_cond:
                participant_igtb_df['cond'] = 2
            elif nurse_cond is False and lab_cond:
                participant_igtb_df['cond'] = 3

            all_df = all_df.append(participant_igtb_df)
            
        print('type: %s' % process_cond)
        print('mean: %.2f +/- %.2f' % (np.mean(mean_list), np.std(mean_list)))
        print('std: %.2f +/- %.2f' % (np.mean(std_list), np.std(std_list)))

        print('type: day nurse')
        print('mean: %.2f +/- %.2f' % (np.mean(all_df.loc[all_df['cond'] == 1]['cycle_mean']),
                                       np.std(all_df.loc[all_df['cond'] == 1]['cycle_mean'])))
        print('std: %.2f +/- %.2f' % (np.mean(all_df.loc[all_df['cond'] == 1]['cycle_std']),
                                      np.std(all_df.loc[all_df['cond'] == 1]['cycle_std'])))

        print('type: night nurse')
        print('mean: %.2f +/- %.2f' % (np.mean(all_df.loc[all_df['cond'] == 2]['cycle_mean']),
                                       np.std(all_df.loc[all_df['cond'] == 2]['cycle_mean'])))
        print('std: %.2f +/- %.2f' % (np.mean(all_df.loc[all_df['cond'] == 2]['cycle_std']),
                                      np.std(all_df.loc[all_df['cond'] == 2]['cycle_std'])))
        
        print()


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)