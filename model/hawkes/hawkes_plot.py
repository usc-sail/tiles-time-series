"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
from random import shuffle

from tick.hawkes import (HawkesSumGaussians)
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.special import erf

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'ticc')))

from TICC_solver import TICC
import config
import load_sensor_data, load_data_path, load_data_basic
import parser
import matplotlib.pyplot as plt

# igtb values
igtb_label_list = ['neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb',
                   'pos_af_igtb', 'neg_af_igtb', 'stai_igtb', 'audit_igtb',
                   'shipley_abs_igtb', 'shipley_voc_igtb', 'Inflexbility', 'Flexbility',
                   'LifeSatisfaction', 'General_Health', 'Emotional_Wellbeing', 'Engage', 'Perceivedstress',
                   'itp_igtb', 'irb_igtb', 'iod_id_igtb', 'iod_od_igtb', 'ocb_igtb']

# fitbit columns
# 'Peak_caloriesOut_mean', 'Peak_caloriesOut_std', 'Peak_caloriesOut_min', 'Peak_caloriesOut_max', 'Peak_caloriesOut_median',
# 'Fat_Burn_caloriesOut_mean', 'Fat_Burn_caloriesOut_std', 'Fat_Burn_caloriesOut_min', 'Fat_Burn_caloriesOut_max', 'Fat_Burn_caloriesOut_median',
# 'Cardio_caloriesOut_mean', 'Cardio_caloriesOut_std', 'Cardio_caloriesOut_min', 'Cardio_caloriesOut_max', 'Cardio_caloriesOut_median',

fitbit_cols = ['Cardio_minutes_mean', 'Cardio_minutes_std', 'Cardio_minutes_min', 'Cardio_minutes_max',
               'Cardio_minutes_median',
               'Peak_minutes_mean', 'Peak_minutes_std', 'Peak_minutes_min', 'Peak_minutes_max', 'Peak_minutes_median',
               'Fat_Burn_minutes_mean', 'Fat_Burn_minutes_std', 'Fat_Burn_minutes_min', 'Fat_Burn_minutes_max',
               'Fat_Burn_minutes_median',
               'Out_of_range_minutes_mean', 'Out_of_range_minutes_std', 'Out_of_range_minutes_min',
               'Out_of_range_minutes_max', 'Out_of_range_minutes_median',
               'NumberSteps_mean', 'NumberSteps_std', 'NumberSteps_min', 'NumberSteps_max', 'NumberSteps_median',
               'RestingHeartRate_mean', 'RestingHeartRate_std', 'RestingHeartRate_min', 'RestingHeartRate_max',
               'RestingHeartRate_median',
               'SleepMinutesInBed_mean', 'SleepMinutesInBed_std', 'SleepMinutesInBed_min', 'SleepMinutesInBed_max',
               'SleepMinutesInBed_median',
               'SleepEfficiency_mean', 'SleepEfficiency_std', 'SleepEfficiency_min', 'SleepEfficiency_max',
               'SleepEfficiency_median']

group_label_list = ['shift', 'position', 'fatigue', 'Inflexbility', 'Flexbility', 'Sex',
                    'LifeSatisfaction', 'General_Health', 'Emotional_Wellbeing', 'Engage', 'Perceivedstress',
                    'pos_af_igtb', 'neg_af_igtb', 'stai_igtb',
                    'neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb']


def Kernel_Integration(dT_array, landmark, w, kernel='gauss'):
    dT_array_repmat = np.tile(dT_array, [1, len(landmark)])
    landmark_repmat = np.tile(np.array(landmark), [len(dT_array), 1])
    
    distance = dT_array_repmat - landmark_repmat
    
    if kernel == 'gauss':
        erf_distance = erf(np.divide(distance, np.sqrt(2) * w))
        erf_landmark = erf(np.divide(landmark_repmat, np.sqrt(2) * w))
        
        G = 0.5 * (erf_distance + erf_landmark)
    else:
        G = 1 - np.exp(np.multiply(-w, distance))
        G[np.where(G < 0)] = 0
    
    return G


def get_infective_matrix(D_max, adjencent_A, landmark, w, dt=1):
    A = np.zeros([D_max, D_max])
    T_max = 1680  # 60 * 28
    
    for u in range(D_max):
        for v in range(D_max):
            dT_array = np.reshape(T_max, [1, 1])
            basis_int = Kernel_Integration(dT_array, landmark, w)
            
            # A_tmp = np.array(adjencent_A[v, u, :]).reshape([1, len(adjencent_A[v, u, :])])
            A_tmp = np.array(adjencent_A[v, u, :]).reshape([1, len(adjencent_A[v, u, :])])
            basis_int = np.array(basis_int)
            basis_int = basis_int.reshape([basis_int.shape[1], 1])
            
            A[u, v] = np.matmul(A_tmp, basis_int)
    
    return A


def init_param(num_of_days, D_max, dist_list, point_dict):
    est = {(i, j): [] for i in range(0, D_max) for j in range(0, D_max)}
    
    sigma = np.zeros([D_max, D_max])
    Tmax_array = np.zeros([D_max, D_max])
    
    try:
        # Append time difference array
        for n in range(num_of_days):
            for i in range(1, len(dist_list[n])):
                ti = dist_list[n][i][2]
                di = point_dict[(dist_list[n][i][0], dist_list[n][i][1])]
                
                for j in range(i - 1):
                    tj = dist_list[n][j][2]
                    dj = point_dict[(dist_list[n][j][0], dist_list[n][j][1])]
                    est[di, dj].append(ti - tj)
    except:
        print('Error')
    
    # Compute sigma and Tmax
    for di in range(D_max):
        for dj in range(D_max):
            est_std = 4 * np.power(np.nanstd(est[di, dj]), 5)
            sigma[di][dj] = np.power(est_std / (3 * len(est[di, dj])), 0.2)
            Tmax_array[di, dj] = np.nanmean(est[di, dj])
    
    Tmax = np.nanmin(Tmax_array[:]) / 2
    sigma = np.array(sigma[:])
    sigma = sigma[sigma != 0]
    
    w = np.nanmin(sigma) / 2
    # landmark = int(np.ceil(Tmax / w))
    landmark = [w * i for i in range(int(np.ceil(Tmax / w)))]
    # if landmark < 1:
    #     landmark = 4
    
    return landmark, w


def get_point_list_kernel(data_config, participant_id, num_of_days, remove_col_index=2, num_of_gaussian=10):
    print('hawkes: participant: %s' % (participant_id))
    row_df = pd.DataFrame(index=[participant_id])
    
    # Read per participant clustering
    clustering_data_list = load_sensor_data.load_filter_clustering(data_config.fitbit_sensor_dict['clustering_path'],
                                                                   participant_id)
    
    # Read per participant data
    participant_data_dict = load_sensor_data.load_filter_data(data_config.fitbit_sensor_dict['filter_path'],
                                                              participant_id, filter_logic=None, valid_data_rate=0.9,
                                                              threshold_dict={'min': 20, 'max': 28})
    
    if clustering_data_list is None or participant_data_dict is None:
        return None, None, None, None, None, None
    
    # Read filtered data
    filter_data_list = participant_data_dict['filter_data_list']
    workday_point_list, offday_point_list = [], []
    workday_point_dist_list, offday_point_dist_list = [], []
    workday_cluster_sum_list, offday_cluster_sum_list = [], []
    
    shuffle(clustering_data_list)
    
    index = 0
    point_dict = {}
    activity_tran_dict = {}
    dict_activity = {0: 'Rest', 2: 'LA', 3: 'MA'}
    for i in range(data_config.fitbit_sensor_dict['num_cluster']):
        for j in range(data_config.fitbit_sensor_dict['num_cluster']):
            if i != j and i != remove_col_index and j != remove_col_index:
                point_dict[(i, j)] = index
                activity_tran_dict[dict_activity[i] + '->' + dict_activity[i]] = index
                index += 1
    
    # Iterate clustering data
    for clustering_data_dict in clustering_data_list:
        start = clustering_data_dict['start']
        
        for filter_data_index, filter_data_dict in enumerate(filter_data_list):
            if np.abs((pd.to_datetime(start) - pd.to_datetime(filter_data_dict['start'])).total_seconds()) > 300:
                continue
            
            cluster_data = clustering_data_dict['data']
            
            cluster_array = np.array(cluster_data)
            change_point = cluster_array[1:] - cluster_array[:-1]
            change_point_index = np.where(change_point != 0)

            cluster_sum_array = np.zeros([1, data_config.fitbit_sensor_dict['num_cluster']])
            for i in range(data_config.fitbit_sensor_dict['num_cluster']):
                cluster_sum_array[0, i] = len(np.where(cluster_array == i)[0]) / len(cluster_array)

            if int(cluster_array[0][0]) != remove_col_index and int(
                    cluster_array[change_point_index[0][0] + 1][0]) != remove_col_index:
                change_list = [(int(cluster_array[0][0]), int(cluster_array[change_point_index[0][0] + 1][0]),
                                change_point_index[0][0] + 1)]
            else:
                change_list = []
            
            for i in range(len(change_point_index[0]) - 1):
                start = change_point_index[0][i] + 1
                end = change_point_index[0][i + 1] + 1
                if cluster_array[start][0] != remove_col_index and int(cluster_array[end][0]) != remove_col_index:
                    change_list.append((int(cluster_array[start][0]), int(cluster_array[end][0]),
                                        change_point_index[0][i + 1] + 1))
            
            # Initiate list for counter
            day_point_list = []
            for i in range(data_config.fitbit_sensor_dict['num_cluster']):
                for j in range(data_config.fitbit_sensor_dict['num_cluster']):
                    if i != j and i != remove_col_index and j != remove_col_index:
                        day_point_list.append(np.zeros(1))
            
            for change_tuple in change_list:
                day_point_list[point_dict[(change_tuple[0], change_tuple[1])]] = np.append(
                        day_point_list[point_dict[(change_tuple[0], change_tuple[1])]], change_tuple[2])
            
            for i, day_point_array in enumerate(day_point_list):
                day_point_list[i] = np.array(len(cluster_array)) if len(day_point_list[i]) == 0 else np.sort(day_point_list[i][1:])
            
            # If we have no point data
            if len(day_point_list) == 0:
                continue
            
            if filter_data_dict['work'] == 1:
                # from collections import Counter
                # data = Counter(elem[0] for elem in change_list)
                if len(workday_point_list) < num_of_days:
                    workday_point_list.append([point for point in day_point_list])
                    workday_point_dist_list.append(change_list)
                    workday_cluster_sum_list.append(cluster_sum_array)
            else:
                if len(offday_point_list) < num_of_days:
                    offday_point_list.append([point for point in day_point_list])
                    offday_point_dist_list.append(change_list)
                    offday_cluster_sum_list.append(cluster_sum_array)
    
    return workday_point_list, workday_point_dist_list, workday_cluster_sum_list, \
           offday_point_list, offday_point_dist_list, offday_cluster_sum_list


def learn_hawkes(num_of_days, length, point_dist_list, point_list, col_list, point_dict):
    landmark, w = init_param(num_of_days, length, point_dist_list, point_dict)
    
    # Learn causality
    hawkes_learner = HawkesSumGaussians(1000, n_gaussians=len(landmark), max_iter=100)
    hawkes_learner.fit(point_list)
    ineffective_array = get_infective_matrix(len(col_list), hawkes_learner.amplitudes, landmark, w, dt=1)
    
    return ineffective_array


def predict(data_config, groundtruth_df, top_participant_id_list, num_of_days=5, num_of_gaussian=10, remove_col_index=2):
    # Initiate data df
    groundtruth_df[igtb_label_list] = groundtruth_df[igtb_label_list].fillna(groundtruth_df[igtb_label_list].mean())
    
    median_dict = { 'pos_af_igtb': np.nanmedian(groundtruth_df['pos_af_igtb']),
                    'neg_af_igtb': np.nanmedian(groundtruth_df['neg_af_igtb']),
                    'stai_igtb': np.nanmedian(groundtruth_df['stai_igtb']),
                    'psqi_igtb': np.nanmedian(groundtruth_df['psqi_igtb']),
                    'neu_igtb': np.nanmedian(groundtruth_df['neu_igtb']),
                    'con_igtb': np.nanmedian(groundtruth_df['con_igtb']),
                    'ext_igtb': np.nanmedian(groundtruth_df['ext_igtb']),
                    'agr_igtb': np.nanmedian(groundtruth_df['agr_igtb']),
                    'ope_igtb': np.nanmedian(groundtruth_df['ope_igtb']),
                    'Inflexbility': np.nanmedian(groundtruth_df['Inflexbility']),
                    'Flexbility': np.nanmedian(groundtruth_df['Flexbility']),
                    'LifeSatisfaction': np.nanmedian(groundtruth_df['LifeSatisfaction']),
                    'General_Health': np.nanmedian(groundtruth_df['General_Health']),
                    'Emotional_Wellbeing': np.nanmedian(groundtruth_df['Emotional_Wellbeing']),
                    'Engage': np.nanmedian(groundtruth_df['Engage']),
                    'Perceivedstress': np.nanmedian(groundtruth_df['Perceivedstress'])}
    
    nurse_workday_point_list, nurse_workday_point_dist_list = [], []
    nurse_offday_point_list, nurse_offday_point_dist_list = [], []
    non_nurse_workday_point_list, non_nurse_workday_point_dist_list = [], []
    non_nurse_offday_point_list, non_nurse_offday_point_dist_list = [], []
    nurse_workday_cluster_sum_list, nurse_offday_cluster_sum_list = [], []
    non_nurse_workday_cluster_sum_list, non_nurse_offday_cluster_sum_list = [], []
    
    high_con_workday_point_list, high_con_workday_point_dist_list = [], []
    high_con_offday_point_list, high_con_offday_point_dist_list = [], []
    low_con_workday_point_list, low_con_workday_point_dist_list = [], []
    low_con_offday_point_list, low_con_offday_point_dist_list = [], []
    high_con_workday_cluster_sum_list, high_con_offday_cluster_sum_list = [], []
    low_con_workday_cluster_sum_list, low_con_offday_cluster_sum_list = [], []

    high_pos_workday_point_list, high_pos_workday_point_dist_list = [], []
    high_pos_offday_point_list, high_pos_offday_point_dist_list = [], []
    low_pos_workday_point_list, low_pos_workday_point_dist_list = [], []
    low_pos_offday_point_list, low_pos_offday_point_dist_list = [], []
    high_pos_workday_cluster_sum_list, high_pos_offday_cluster_sum_list = [], []
    low_pos_workday_cluster_sum_list, low_pos_offday_cluster_sum_list = [], []

    for idx, participant_id in enumerate(top_participant_id_list[:]):
        
        ###########################################################
        # Print out
        ###########################################################
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        ###########################################################
        # Read hawkes feature, if None, continue
        ###########################################################
        workday_point_list, workday_point_dist_list, workday_cluster_sum_list, \
        offday_point_list, offday_point_dist_list, offday_cluster_sum_list = get_point_list_kernel(data_config, participant_id, num_of_days, remove_col_index=remove_col_index, num_of_gaussian=num_of_gaussian)
        
        if workday_point_list is None:
            continue
        
        ###########################################################
        # Conditions for label
        ###########################################################
        cond1 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['currentposition'].values[0] == 1
        cond2 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['currentposition'].values[0] == 2
        cond3 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['gender'].values[0] > 0
        cond4 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['Shift'].values[0] == 'Day shift'
        cond5 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['Sex'].values[0] == 'Female'
        if not cond3:
            continue
        
        ###########################################################
        # Feature data
        ###########################################################
        
        for group_label in group_label_list:
            if group_label == 'position':
                if cond1 or cond2:
                    [nurse_workday_point_list.append(point_list) for point_list in workday_point_list]
                    [nurse_offday_point_list.append(point_list) for point_list in offday_point_list]
                    
                    [nurse_workday_point_dist_list.append(point_list) for point_list in workday_point_dist_list]
                    [nurse_offday_point_dist_list.append(point_list) for point_list in offday_point_dist_list]

                    [nurse_workday_cluster_sum_list.append(point_list) for point_list in workday_cluster_sum_list]
                    [nurse_offday_cluster_sum_list.append(point_list) for point_list in offday_cluster_sum_list]

                else:
                    [non_nurse_workday_point_list.append(point_list) for point_list in workday_point_list]
                    [non_nurse_offday_point_list.append(point_list) for point_list in offday_point_list]
                    
                    [non_nurse_workday_point_dist_list.append(point_list) for point_list in workday_point_dist_list]
                    [non_nurse_offday_point_dist_list.append(point_list) for point_list in offday_point_dist_list]

                    [non_nurse_workday_cluster_sum_list.append(point_list) for point_list in workday_cluster_sum_list]
                    [non_nurse_offday_cluster_sum_list.append(point_list) for point_list in offday_cluster_sum_list]
                    
            elif 'con_igtb' in group_label:
                score = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][group_label].values[0]
                cond_igtb = score >= median_dict[group_label]
                
                if cond_igtb:
                    [high_con_workday_point_list.append(point_list) for point_list in workday_point_list]
                    [high_con_offday_point_list.append(point_list) for point_list in offday_point_list]
                    
                    [high_con_workday_point_dist_list.append(point_list) for point_list in workday_point_dist_list]
                    [high_con_offday_point_dist_list.append(point_list) for point_list in offday_point_dist_list]

                    [high_con_workday_cluster_sum_list.append(point_list) for point_list in workday_cluster_sum_list]
                    [high_con_offday_cluster_sum_list.append(point_list) for point_list in offday_cluster_sum_list]
                else:
                    [low_con_workday_point_list.append(point_list) for point_list in workday_point_list]
                    [low_con_offday_point_list.append(point_list) for point_list in offday_point_list]
                    
                    [low_con_workday_point_dist_list.append(point_list) for point_list in workday_point_dist_list]
                    [low_con_offday_point_dist_list.append(point_list) for point_list in offday_point_dist_list]

                    [low_con_workday_cluster_sum_list.append(point_list) for point_list in workday_cluster_sum_list]
                    [low_con_offday_cluster_sum_list.append(point_list) for point_list in offday_cluster_sum_list]
                    
            elif 'pos_af_igtb' in group_label:
                score = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][group_label].values[0]
                pos_af_igtb = score >= median_dict[group_label]

                if pos_af_igtb:
                    [high_pos_workday_point_list.append(point_list) for point_list in workday_point_list]
                    [high_pos_offday_point_list.append(point_list) for point_list in offday_point_list]
    
                    [high_pos_workday_point_dist_list.append(point_list) for point_list in workday_point_dist_list]
                    [high_pos_offday_point_dist_list.append(point_list) for point_list in offday_point_dist_list]

                    [high_pos_workday_cluster_sum_list.append(point_list) for point_list in workday_cluster_sum_list]
                    [high_pos_offday_cluster_sum_list.append(point_list) for point_list in offday_cluster_sum_list]
                else:
                    [low_pos_workday_point_list.append(point_list) for point_list in workday_point_list]
                    [low_pos_offday_point_list.append(point_list) for point_list in offday_point_list]
    
                    [low_pos_workday_point_dist_list.append(point_list) for point_list in workday_point_dist_list]
                    [low_pos_offday_point_dist_list.append(point_list) for point_list in offday_point_dist_list]

                    [low_pos_workday_cluster_sum_list.append(point_list) for point_list in workday_cluster_sum_list]
                    [low_pos_offday_cluster_sum_list.append(point_list) for point_list in offday_cluster_sum_list]

    ###########################################################
    # Learn Hawkes data
    ###########################################################
    index = 0
    point_dict, dict_activity = {}, {0: 'Rest', 2: 'LA', 3: 'MA'}
    col_list = []
    for i in range(data_config.fitbit_sensor_dict['num_cluster']):
        for j in range(data_config.fitbit_sensor_dict['num_cluster']):
            if i != j and i != remove_col_index and j != remove_col_index:
                point_dict[(i, j)] = index
                col_list.append(dict_activity[i] + '->' + dict_activity[j])
                index += 1
                
    workday_col_list, offday_col_list = [], []

    for i in range(data_config.fitbit_sensor_dict['num_cluster']):
        for j in range(data_config.fitbit_sensor_dict['num_cluster']):
            if i != j and i != remove_col_index and j != remove_col_index:
                workday_col_list.append('workday:' + str(i) + ',' + str(j))
                offday_col_list.append('offday:' + str(i) + ',' + str(j))

    high_con_workday = learn_hawkes(num_of_days, len(workday_col_list), high_con_workday_point_dist_list,
                                    high_con_workday_point_list, workday_col_list, point_dict)
    high_con_workday = pd.DataFrame(high_con_workday, index=col_list, columns=col_list)
    high_con_workday.to_csv(os.path.join(os.curdir, 'result', 'high_con_workday.csv'))

    high_con_offday = learn_hawkes(num_of_days, len(offday_col_list), high_con_offday_point_dist_list,
                                   high_con_offday_point_list, offday_col_list, point_dict)
    high_con_offday = pd.DataFrame(high_con_offday, index=col_list, columns=col_list)
    high_con_offday.to_csv(os.path.join(os.curdir, 'result', 'high_con_offday.csv'))

    low_con_workday = learn_hawkes(num_of_days, len(workday_col_list), low_con_workday_point_dist_list,
                                   low_con_workday_point_list, workday_col_list, point_dict)
    low_con_workday = pd.DataFrame(low_con_workday, index=col_list, columns=col_list)
    low_con_workday.to_csv(os.path.join(os.curdir, 'result', 'low_con_workday.csv'))

    low_con_offday = learn_hawkes(num_of_days, len(offday_col_list), low_con_offday_point_dist_list,
                                  low_con_offday_point_list, offday_col_list, point_dict)
    low_con_offday = pd.DataFrame(low_con_offday, index=col_list, columns=col_list)
    low_con_offday.to_csv(os.path.join(os.curdir, 'result', 'low_con_offday.csv'))
    
    # Affect
    high_pos_workday = learn_hawkes(num_of_days, len(workday_col_list), high_pos_workday_point_dist_list,
                                    high_pos_workday_point_list, workday_col_list, point_dict)
    high_pos_workday = pd.DataFrame(high_pos_workday, index=col_list, columns=col_list)
    high_pos_workday.to_csv(os.path.join(os.curdir, 'result', 'high_pos_workday.csv'))

    high_pos_offday = learn_hawkes(num_of_days, len(offday_col_list), high_pos_offday_point_dist_list,
                                   high_pos_offday_point_list, offday_col_list, point_dict)
    high_pos_offday = pd.DataFrame(high_pos_offday, index=col_list, columns=col_list)
    high_pos_offday.to_csv(os.path.join(os.curdir, 'result', 'high_pos_offday.csv'))

    low_pos_workday = learn_hawkes(num_of_days, len(workday_col_list), low_pos_workday_point_dist_list,
                                   low_pos_workday_point_list, workday_col_list, point_dict)
    low_pos_workday = pd.DataFrame(low_pos_workday, index=col_list, columns=col_list)
    low_pos_workday.to_csv(os.path.join(os.curdir, 'result', 'low_pos_workday.csv'))

    low_pos_offday = learn_hawkes(num_of_days, len(offday_col_list), low_pos_offday_point_dist_list,
                                  low_pos_offday_point_list, offday_col_list, point_dict)
    low_pos_offday = pd.DataFrame(low_pos_offday, index=col_list, columns=col_list)
    low_pos_offday.to_csv(os.path.join(os.curdir, 'result', 'low_pos_offday.csv'))

    print('high_con_workday_cluster_sum_list')
    print(np.mean(np.mean(np.array(high_con_workday_cluster_sum_list), axis=1), axis=0))
    print('high_con_offday_cluster_sum_list')
    print(np.mean(np.mean(np.array(high_pos_offday_cluster_sum_list), axis=1), axis=0))

    print('low_con_workday_cluster_sum_list')
    print(np.mean(np.mean(np.array(low_con_workday_cluster_sum_list), axis=1), axis=0))
    print('low_con_offday_cluster_sum_list')
    print(np.mean(np.mean(np.array(low_con_offday_cluster_sum_list), axis=1), axis=0))

    print('\nhigh_pos_workday_cluster_sum_list')
    print(np.mean(np.mean(np.array(high_pos_workday_cluster_sum_list), axis=1), axis=0))
    print('high_pos_offday_cluster_sum_list')
    print(np.mean(np.mean(np.array(high_pos_offday_cluster_sum_list), axis=1), axis=0))

    print('low_pos_workday_cluster_sum_list')
    print(np.mean(np.mean(np.array(low_pos_workday_cluster_sum_list), axis=1), axis=0))
    print('low_pos_offday_cluster_sum_list')
    print(np.mean(np.mean(np.array(low_pos_offday_cluster_sum_list), axis=1), axis=0))
    
    print()

def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path, filter_data=True,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    # Get participant id list, k=10, read 10 participants with most data in fitbit
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, k=150, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    ###########################################################
    # Read ground truth data
    ###########################################################
    groundtruth_df = load_data_basic.read_AllBasic(tiles_data_path)
    groundtruth_df = groundtruth_df.drop_duplicates(keep='first')
    
    if os.path.exists(os.path.join(os.curdir, 'result')) is False:
        os.mkdir(os.path.join(os.curdir, 'result'))
    
    save_model_path = os.path.join(os.curdir, 'result', data_config.fitbit_sensor_dict['clustering_path'].split('/')[-1])
    if os.path.exists(save_model_path) is False:
        os.mkdir(save_model_path)
    
    for index, row_series in groundtruth_df.iterrows():
        # Extra process for feature
        groundtruth_df.loc[index, 'nurseyears'] = groundtruth_df.loc[index, 'nurseyears'] if groundtruth_df.loc[index, 'nurseyears'] != ' ' else np.nan
        groundtruth_df.loc[index, 'housing'] = float(groundtruth_df.loc[index, 'housing']) if groundtruth_df.loc[index, 'housing'] != ' ' else np.nan
        groundtruth_df.loc[index, 'overtime'] = float(groundtruth_df.loc[index, 'overtime']) if groundtruth_df.loc[index, 'overtime'] != ' ' else np.nan
        
        # Extra process for label
        groundtruth_df.loc[index, 'Flexbility'] = float(groundtruth_df.loc[index, 'Flexbility']) if groundtruth_df.loc[index, 'Flexbility'] != ' ' else np.nan
        groundtruth_df.loc[index, 'Inflexbility'] = float(groundtruth_df.loc[index, 'Inflexbility']) if groundtruth_df.loc[index, 'Inflexbility'] != ' ' else np.nan
        
        groundtruth_df.loc[index, 'LifeSatisfaction'] = float(groundtruth_df.loc[index, 'LifeSatisfaction']) if groundtruth_df.loc[index, 'LifeSatisfaction'] != ' ' else np.nan
        groundtruth_df.loc[index, 'General_Health'] = float(groundtruth_df.loc[index, 'General_Health']) if groundtruth_df.loc[index, 'General_Health'] != ' ' else np.nan
        groundtruth_df.loc[index, 'Emotional_Wellbeing'] = float(groundtruth_df.loc[index, 'Emotional_Wellbeing']) if groundtruth_df.loc[index, 'Emotional_Wellbeing'] != ' ' else np.nan
        groundtruth_df.loc[index, 'Engage'] = float(groundtruth_df.loc[index, 'Engage']) if groundtruth_df.loc[index, 'Engage'] != ' ' else np.nan
        groundtruth_df.loc[index, 'Perceivedstress'] = float(groundtruth_df.loc[index, 'Perceivedstress']) if groundtruth_df.loc[index, 'Perceivedstress'] != ' ' else np.nan
    
    
    high_con_workday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'high_con_workday.csv'), index_col=0)
    high_con_offday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'high_con_offday.csv'), index_col=0)
    low_con_workday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'low_con_workday.csv'), index_col=0)
    low_con_offday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'low_con_offday.csv'), index_col=0)
    
    '''
    high_con_workday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'high_pos_workday.csv'), index_col=0)
    high_con_offday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'high_pos_offday.csv'), index_col=0)
    low_con_workday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'low_pos_workday.csv'), index_col=0)
    low_con_offday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'low_pos_offday.csv'), index_col=0)
    '''
    
    
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    cbar_ax = fig.add_axes([.93, .325, .02, .55])

    sns.heatmap(high_con_workday_df, cmap="jet", linewidths=.5, ax=ax[0], cbar_ax=cbar_ax, cbar_kws={'format': '%.2f'},
                vmin=-0.0, vmax=1, xticklabels=list(high_con_workday_df.index), yticklabels=list(high_con_workday_df.index))
    sns.heatmap(high_con_offday_df, cmap="jet", linewidths=.5, ax=ax[2], cbar=False, cbar_ax=None,
                vmin=-0.0, vmax=1, xticklabels=list(high_con_offday_df.index), yticklabels=False)
    
    sns.heatmap(low_con_workday_df, cmap="jet", linewidths=.5, ax=ax[1], cbar=False, cbar_ax=None,
                vmin=-0.0, vmax=1, xticklabels=list(high_con_offday_df.index), yticklabels=False)
    sns.heatmap(low_con_offday_df, cmap="jet", linewidths=.5, ax=ax[3], cbar=False, cbar_ax=None,
                vmin=-0.0, vmax=1, xticklabels=list(high_con_offday_df.index), yticklabels=False)

    ax[0].tick_params(axis=u'both', which=u'both', length=0)
    ax[1].tick_params(axis=u'both', which=u'both', length=0)
    ax[2].tick_params(axis=u'both', which=u'both', length=0)
    ax[3].tick_params(axis=u'both', which=u'both', length=0)
    
    ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0, fontsize=14, fontweight="bold")
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90, fontsize=14, fontweight="bold")
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90, fontsize=14, fontweight="bold")
    ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=90, fontsize=14, fontweight="bold")
    ax[3].set_xticklabels(ax[3].get_xticklabels(), rotation=90, fontsize=14, fontweight="bold")

    # ax[0].set_title('High Conscientiousness', fontweight="bold", fontsize=13, y=1.2)
    # ax[1].set_title('Low Conscientiousness', fontweight="bold", fontsize=13, y=1.2)
    
    '''
    ax[0].set_title('High Conscientiousness \n Workdays', fontweight="bold", fontsize=16, y=1.2)
    ax[1].set_title('Low Conscientiousness: \n Workdays', fontweight="bold", fontsize=16, y=1.2)
    ax[2].set_title('High Conscientiousness: \n Off-days', fontweight="bold", fontsize=16, y=1.2)
    ax[3].set_title('Low Conscientiousness: \n Off-days', fontweight="bold", fontsize=16, y=1.2)
    '''
    '''
    ax[0].set_title('High Positive Affect \n Workdays', fontweight="bold", fontsize=16, y=1.2)
    ax[1].set_title('Low Positive Affect: \n Workdays', fontweight="bold", fontsize=16, y=1.2)
    ax[2].set_title('High Positive Affect: \n Off-days', fontweight="bold", fontsize=16, y=1.2)
    ax[3].set_title('Low Positive Affect: \n Off-days', fontweight="bold", fontsize=16, y=1.2)
    '''

    ax[0].set_title('High Conscientiousness \n Workdays', fontweight="bold", fontsize=14, y=1.2)
    ax[1].set_title('Low Conscientiousness \n Workdays', fontweight="bold", fontsize=14, y=1.2)
    ax[2].set_title('High Conscientiousness \n Off-days', fontweight="bold", fontsize=14, y=1.2)
    ax[3].set_title('Low Conscientiousness \n Off-days', fontweight="bold", fontsize=14, y=1.2)
    
    '''
    ax[0].set_title('High Level of Neuroticism \n Workdays', fontweight="bold", fontsize=15, y=1.2)
    ax[1].set_title('Low Level of Neuroticism \n Workdays', fontweight="bold", fontsize=15, y=1.2)
    ax[2].set_title('High Level of Neuroticism \n Off-days', fontweight="bold", fontsize=15, y=1.2)
    ax[3].set_title('Low Level of Neuroticism \n Off-days', fontweight="bold", fontsize=15, y=1.2)
    '''
    
    ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    
    # fig.text(0.2, 1.2, 'Workday', dict(size=16))
    # ax[2].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    # ax[3].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    
    plt.gcf().subplots_adjust(bottom=0.3)
    # plt.yticks(rotation=180)
    # plt.tight_layout()
    plt.show()
    # fig.savefig(os.path.join(os.curdir, 'result', 'workday_job.png'), dpi=300)
    fig.savefig(os.path.join(os.curdir, 'result', 'con_job.png'), dpi=300)
    # fig.savefig(os.path.join(os.curdir, 'result', 'affect_job.png'), dpi=300)
    
    
    '''
    ticc_num_cluster_6_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 2
    arima_15_ticc_num_cluster_3_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 1
    arima_15_ticc_num_cluster_4_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 1
    arima_15_ticc_num_cluster_5_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 1
    ticc_num_cluster_6_window_10_penalty_10.0_sparsity_0.1_cluster_days_7: 5
    ticc_num_cluster_4_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 3
    ticc_num_cluster_4_window_10_penalty_10.0_sparsity_0.1_cluster_days_7: 3
    ticc_num_cluster_5_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 3
    ticc_num_cluster_5_window_10_penalty_10.0_sparsity_0.1_cluster_days_7: 3
    '''
    
    # for i in range(3, 8, 2):
    for i in range(5, 6):
        final_result_per_day_setting_df, final_fitbit_result_per_day_setting_df = pd.DataFrame(), pd.DataFrame()
        final_feat_importance_result, final_fitbit_feat_importance_result = pd.DataFrame(), pd.DataFrame()
    
        predict(data_config, groundtruth_df, top_participant_id_list,
                num_of_days=i, remove_col_index=1)
            
            
if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir,
                                               'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)
