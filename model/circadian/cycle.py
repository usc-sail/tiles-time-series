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
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    data_dict = {}
    
    if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['preprocess_path'], 'data.pkl')) is False:
        for idx, participant_id in enumerate(top_participant_id_list):
            
            print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
            
            uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
            participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
            
            nurse_cond = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
            primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            
            fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad_and_norm(data_config, participant_id, pad=missing_data_val)
            missing_rate = len(fitbit_df.loc[fitbit_df['HeartRatePPG'] < -10000]) / len(fitbit_df)
            
            if missing_rate < 0.25:
                data_dict[participant_id] = {}
                data_dict[participant_id]['data'] = np.array(fitbit_df)
                data_dict[participant_id]['nurse'] = nurse_cond
                data_dict[participant_id]['primary_unit'] = primary_unit
                data_dict[participant_id]['shift'] = shift
                
            print('success: %s' % participant_id)
            
        # Save
        output = open(os.path.join(data_config.fitbit_sensor_dict['preprocess_path'], 'data.pkl'), 'wb')
        pickle.dump(data_dict, output)

    else:
        pkl_file = open(os.path.join(data_config.fitbit_sensor_dict['preprocess_path'], 'data.pkl'), 'rb')
        data_dict = pickle.load(pkl_file)

        day_nurse_data_list, night_nurse_data_list,lab_data_list, all_nurse_data_list, all_data_list = [], [], [], [], []
        day_nurse_id_list, night_nurse_id_list, lab_id_list, all_nurse_id_list, all_id_list = [], [], [], [], []

        for participant_id in list(data_dict.keys()):
            data = data_dict[participant_id]['data']
            
            nurse_cond = data_dict[participant_id]['nurse'] == 1 and 'Dialysis' not in data_dict[participant_id]['primary_unit']
            day_shift_cond = data_dict[participant_id]['shift'] == 'Day shift'
            night_shift_cond = data_dict[participant_id]['shift'] == 'Night shift'
            lab_cond = 'Lab' not in data_dict[participant_id]['primary_unit']
            
            if nurse_cond and day_shift_cond:
                day_nurse_data_list.append(data)
                day_nurse_id_list.append(participant_id)
            elif nurse_cond and night_shift_cond:
                night_nurse_data_list.append(data)
                night_nurse_id_list.append(participant_id)
            elif nurse_cond is False and lab_cond:
                lab_data_list.append(data)
                lab_id_list.append(participant_id)
                
            if nurse_cond:
                all_nurse_data_list.append(data)
                all_nurse_id_list.append(participant_id)
                
            all_data_list.append(data)
            all_id_list.append(participant_id)
                
                
        # process_cond_list = ['night_nurse', 'day_nurse', 'lab']
        # process_cond_list = ['night_nurse', 'day_nurse']
        process_cond_list = ['night_nurse', 'day_nurse', 'all_nurse']
        # process_cond_list = ['all_nurse', 'all']
        for process_cond in process_cond_list:
            if process_cond == 'night_nurse':
                samples = night_nurse_data_list
                participant_id_list = night_nurse_id_list
                participant_type = process_cond
            elif process_cond == 'day_nurse':
                samples = day_nurse_data_list
                participant_id_list = day_nurse_id_list
                participant_type = process_cond
            elif process_cond == 'all_nurse':
                samples = all_nurse_data_list
                participant_id_list = all_nurse_id_list
                participant_type = process_cond
            elif process_cond == 'all':
                samples = all_data_list
                participant_id_list = all_id_list
                participant_type = process_cond
            else:
                samples = lab_data_list
                participant_id_list = lab_id_list
                participant_type = process_cond
                
            # 'geometric', 'poisson'
            model = cyclic_HMM.fit_cyhmm_model(n_states=2, samples=samples, data_config=data_config,
                                               symptom_names=['HeartRatePPG', 'StepCount'], max_iterations=100,
                                               duration_distribution_name='geometric',
                                               emission_distribution_name='normal_with_missing_data',
                                               hypothesized_duration=24, max_duration=16,
                                               participant_id_list=participant_id_list, participant_type=participant_type,
                                               verbose=True, n_processes=1, min_iterations=10)
    
    print()
    
    
if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)