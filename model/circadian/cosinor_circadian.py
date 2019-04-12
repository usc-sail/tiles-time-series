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
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # Read all data
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']
        
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
        
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, participant_id)
        
        data_start, data_end = fitbit_df.index[0], fitbit_df.index[-1]
        days = (pd.to_datetime(data_end) - pd.to_datetime(data_start)).total_seconds() / (3600 * 24)
        valid_days = np.zeros([int(days)])

        # Read days with valid data
        for day_index in range(int(days)):
            
            day_start = (pd.to_datetime(data_start) + timedelta(days=day_index)).strftime(load_data_basic.date_time_format)
            day_end = (pd.to_datetime(data_start) + timedelta(days=day_index+1)).strftime(load_data_basic.date_time_format)
            day_data_df = fitbit_df[day_start:day_end]
            
            if len(np.where(np.array(day_data_df.StepCount) >= 0)[0]) / 1440 > 0.5:
                valid_days[day_index] = 1
                
        if len(np.where(valid_days == 1)[0]) < 10:
            continue

        # Read longest days with valid data
        valid_region_list = [[i for i,value in it] for key,it in itertools.groupby(enumerate(valid_days), key=operator.itemgetter(1)) if key != 0]
        valid_region_list = sorted(valid_region_list, key=len)

        longest_valid_region = valid_region_list[-1]
        if len(longest_valid_region) < 10:
            continue

        # Read longest days of valid data
        longest_valid_start = (pd.to_datetime(data_start) + timedelta(days=longest_valid_region[0])).strftime(load_data_basic.date_time_format)
        longest_valid_end = (pd.to_datetime(data_start) + timedelta(days=longest_valid_region[-1])).strftime(load_data_basic.date_time_format)

        longest_valid_data = fitbit_df[longest_valid_start:longest_valid_end]
        longest_valid_data = np.array(longest_valid_data.StepCount)
        
        valid_data_index = np.where(longest_valid_data >= 0)[0]
        valid_data = longest_valid_data[np.where(longest_valid_data >= 0)[0]]

        cosinor(valid_data_index / 1440, valid_data, 2 * math.pi, 0.05)

def cosinor(t, y, w, alpha):
    # I.Parameter Estimation
    n = len(t)

    # Substituition
    x = np.cos(w * t)
    z = np.sin(w * t)

    # Set up and solve the normal equations simultaneously
    NE = Matrix([[n, np.sum(x), np.sum(z), np.sum(y)],
                 [np.sum(x), np.sum(np.power(x, 2)), np.sum(x * z), np.sum(x * y)],
                 [np.sum(z), np.sum(x * z), np.sum(np.power(z, 2)), np.sum(z * y)]])

    RNE = NE.rref()
    M = RNE[0][0, 3]
    beta = RNE[0][1, 3]
    gamma = RNE[0][2, 3]

    # Calculate amplitude and acrophase from beta and gamma
    Amp = np.sqrt(np.array(np.power(beta, 2) + np.power(gamma, 2), dtype=np.float64))
    theta = np.arctan(np.array(abs(gamma / beta), dtype=np.float64))

    a = np.sign(beta)
    b = np.sign(gamma)
    if (a == 1 or a == 0) and b == 1:
        phi = -theta
    elif a == -1 and (b == 1 or b == 0):
        phi = -np.pi + theta
    elif (a == -1 or a == 0) and b == -1:
        phi = -np.pi - theta
    elif a == 1 and (b == -1 or b == 0):
        phi = -2 * np.pi + theta

    f = M + Amp * np.cos(w * t + phi)
    plt.plot(t, f)
    plt.show()

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)