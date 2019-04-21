"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
from random import shuffle


from tick.hawkes import (HawkesSumGaussians, SimuHawkesSumExpKernels)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


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
from scipy.special import erf

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

fitbit_cols = ['Cardio_minutes_mean', 'Cardio_minutes_std', 'Cardio_minutes_min', 'Cardio_minutes_max', 'Cardio_minutes_median',
               'Peak_minutes_mean', 'Peak_minutes_std', 'Peak_minutes_min', 'Peak_minutes_max', 'Peak_minutes_median',
               'Fat_Burn_minutes_mean', 'Fat_Burn_minutes_std', 'Fat_Burn_minutes_min', 'Fat_Burn_minutes_max', 'Fat_Burn_minutes_median',
               'Out_of_range_minutes_mean', 'Out_of_range_minutes_std', 'Out_of_range_minutes_min', 'Out_of_range_minutes_max', 'Out_of_range_minutes_median',
               'NumberSteps_mean', 'NumberSteps_std', 'NumberSteps_min', 'NumberSteps_max', 'NumberSteps_median',
               'RestingHeartRate_mean', 'RestingHeartRate_std', 'RestingHeartRate_min', 'RestingHeartRate_max', 'RestingHeartRate_median',
               'SleepMinutesInBed_mean', 'SleepMinutesInBed_std', 'SleepMinutesInBed_min', 'SleepMinutesInBed_max', 'SleepMinutesInBed_median',
               'SleepEfficiency_mean', 'SleepEfficiency_std', 'SleepEfficiency_min', 'SleepEfficiency_max', 'SleepEfficiency_median']

'''
group_label_list = ['shift', 'position', 'fatigue', 'Inflexbility', 'Flexbility', 'Sex',
                    'LifeSatisfaction', 'General_Health', 'Emotional_Wellbeing', 'Engage', 'Perceivedstress',
                    'pos_af_igtb', 'neg_af_igtb', 'stai_igtb',
                    'neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb']
'''
group_label_list = ['shift', 'position', 'fatigue', 'Sex',
                    'pos_af_igtb', 'neg_af_igtb', 'stai_igtb',
                    'neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb']

'''
def Kernel_Integration(dt, para):

    # % dt = t_current - t_hist(:);
    distance = repmat(dt(:), [1, length(para.landmark(:))]) - ...
    repmat(para.landmark(:)', [length(dt), 1]);
    landmark = repmat(para.landmark(:)', [length(dt), 1]);
    
    switch
    para.kernel
    case
    'exp'
    G = 1 - exp(-para.w * (distance - landmark));
    G(G < 0) = 0;
    
    case
    'gauss'
    G = 0.5 * (erf(distance. / (sqrt(2) * para.w))...
               + erf(landmark. / (sqrt(2) * para.w)));
'''


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
    M = 300
    T_max = 1680 # 60 * 28

    for u in range(D_max):
        for v in range(D_max):
            dT_array = np.reshape(T_max, [1, 1])
            basis_int = Kernel_Integration(dT_array, landmark, w)
            
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
    

def get_hawkes_kernel(data_config, participant_id, num_of_days, remove_col_index=2, num_of_gaussian=10):
    
    print('hawkes: participant: %s' % (participant_id))
    row_df = pd.DataFrame(index=[participant_id])
    
    # Read per participant clustering
    clustering_data_list = load_sensor_data.load_filter_clustering(data_config.fitbit_sensor_dict['clustering_path'], participant_id)
    
    # Read per participant data
    participant_data_dict = load_sensor_data.load_filter_data(data_config.fitbit_sensor_dict['filter_path'], participant_id, filter_logic=None, valid_data_rate=0.9, threshold_dict={'min': 20, 'max': 28})
    
    if clustering_data_list is None or participant_data_dict is None:
        return None
    
    # Read filtered data
    filter_data_list = participant_data_dict['filter_data_list']
    workday_point_list, offday_point_list = [], []
    workday_point_dist_list, offday_point_dist_list = [], []
    shuffle(clustering_data_list)

    index = 0
    point_dict = {}
    for i in range(data_config.fitbit_sensor_dict['num_cluster']):
        for j in range(data_config.fitbit_sensor_dict['num_cluster']):
            if i != j and i != remove_col_index and j != remove_col_index:
                point_dict[(i, j)] = index
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
            
            if int(cluster_array[0][0]) != remove_col_index and int(cluster_array[change_point_index[0][0]+1][0]) != remove_col_index:
                change_list = [(int(cluster_array[0][0]), int(cluster_array[change_point_index[0][0]+1][0]), change_point_index[0][0]+1)]
            else:
                change_list = []
            
            for i in range(len(change_point_index[0])-1):
                start = change_point_index[0][i] + 1
                end = change_point_index[0][i+1] + 1
                if cluster_array[start][0] != remove_col_index and int(cluster_array[end][0]) != remove_col_index:
                    change_list.append((int(cluster_array[start][0]), int(cluster_array[end][0]), change_point_index[0][i+1] + 1))
            
            # Initiate list for counter
            day_point_list = []
            for i in range(data_config.fitbit_sensor_dict['num_cluster']):
                for j in range(data_config.fitbit_sensor_dict['num_cluster']):
                    if i != j and i != remove_col_index and j != remove_col_index:
                        day_point_list.append(np.zeros(1))
                        
            for change_tuple in change_list:
                day_point_list[point_dict[(change_tuple[0], change_tuple[1])]] = np.append(day_point_list[point_dict[(change_tuple[0], change_tuple[1])]], change_tuple[2])
            
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
            else:
                if len(offday_point_list) < num_of_days:
                    offday_point_list.append([point for point in day_point_list])
                    offday_point_dist_list.append(change_list)

    workday_col_list, offday_col_list = [], []

    for i in range(data_config.fitbit_sensor_dict['num_cluster']):
        for j in range(data_config.fitbit_sensor_dict['num_cluster']):
            if i != j and i != remove_col_index and j != remove_col_index:
                workday_col_list.append('workday:' + str(i) + ',' + str(j))
                offday_col_list.append('offday:' + str(i) + ',' + str(j))

    landmark, w = init_param(num_of_days, len(offday_col_list), workday_point_dist_list, point_dict)
    
    # Learn causality
    workday_learner = HawkesSumGaussians(1000, n_gaussians=len(landmark), max_iter=100)
    workday_learner.fit(workday_point_list)
    # ineffective_array = np.array(workday_learner.get_kernel_norms())

    ineffective_array = get_infective_matrix(len(workday_col_list), workday_learner.amplitudes, landmark, w, dt=1)

    for i in range(ineffective_array.shape[0]):
        for j in range(ineffective_array.shape[1]):
            index = i * ineffective_array.shape[0] + j
            row_df[workday_col_list[i] + '->' + workday_col_list[j]] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][index]

    landmark, w = init_param(num_of_days, len(offday_col_list), offday_point_dist_list, point_dict)

    offday_learner = HawkesSumGaussians(1000, n_gaussians=len(landmark), max_iter=100)
    offday_learner.fit(offday_point_list)
    # ineffective_array = np.array(offday_learner.get_kernel_norms())
    
    ineffective_array = get_infective_matrix(len(offday_col_list), offday_learner.amplitudes, landmark, w, dt=1)

    for i in range(ineffective_array.shape[0]):
        for j in range(ineffective_array.shape[1]):
            index = i * ineffective_array.shape[0] + j
            row_df[offday_col_list[i] + '->' + offday_col_list[j]] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][index]

    return row_df
    

def predict(data_config, groundtruth_df, top_participant_id_list, index, fitbit=False, fusion=False,
            num_of_days=5, num_of_gaussian=10, remove_col_index=2):
    
    # Initiate data df
    hawkes_kernel_df, data_df = pd.DataFrame(), pd.DataFrame()
    
    groundtruth_df[igtb_label_list] = groundtruth_df[igtb_label_list].fillna(groundtruth_df[igtb_label_list].mean())

    mean_dict = {'pos_af_igtb': np.nanmedian(groundtruth_df['pos_af_igtb']), 'neg_af_igtb': np.nanmedian(groundtruth_df['neg_af_igtb']),
                 'stai_igtb': np.nanmedian(groundtruth_df['stai_igtb']), 'psqi_igtb': np.nanmedian(groundtruth_df['psqi_igtb']),
                 'neu_igtb': np.nanmedian(groundtruth_df['neu_igtb']), 'con_igtb': np.nanmedian(groundtruth_df['con_igtb']),
                 'ext_igtb': np.nanmedian(groundtruth_df['ext_igtb']), 'agr_igtb': np.nanmedian(groundtruth_df['agr_igtb']), 'ope_igtb': np.nanmedian(groundtruth_df['ope_igtb']),
                 'Inflexbility': np.nanmedian(groundtruth_df['Inflexbility']), 'Flexbility': np.nanmedian(groundtruth_df['Flexbility']),
                 'LifeSatisfaction': np.nanmedian(groundtruth_df['LifeSatisfaction']), 'General_Health': np.nanmedian(groundtruth_df['General_Health']),
                 'Emotional_Wellbeing': np.nanmedian(groundtruth_df['Emotional_Wellbeing']), 'Engage': np.nanmedian(groundtruth_df['Engage']),
                 'Perceivedstress': np.nanmedian(groundtruth_df['Perceivedstress'])}

    for idx, participant_id in enumerate(top_participant_id_list):
    
        ###########################################################
        # Print out
        ###########################################################
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
    
        ###########################################################
        # Read hawkes feature, if None, continue
        ###########################################################
        row_df = get_hawkes_kernel(data_config, participant_id, num_of_days, remove_col_index=remove_col_index, num_of_gaussian=num_of_gaussian)
        
        if row_df is None:
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
        # Read Fitbit summary
        ###########################################################
        fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']

        ###########################################################
        # Feature data
        ###########################################################
        hawkes_kernel_df = hawkes_kernel_df.append(row_df)
        
        if fitbit or fusion:
            # Summary features
            row_df['Cardio_caloriesOut_mean'] = np.nanmean(fitbit_summary_df.Cardio_caloriesOut)
            row_df['Cardio_caloriesOut_std'] = np.nanstd(fitbit_summary_df.Cardio_caloriesOut)
            row_df['Cardio_caloriesOut_min'] = np.nanmin(fitbit_summary_df.Cardio_caloriesOut)
            row_df['Cardio_caloriesOut_max'] = np.nanmax(fitbit_summary_df.Cardio_caloriesOut)
            row_df['Cardio_caloriesOut_median'] = np.nanmedian(fitbit_summary_df.Cardio_caloriesOut)
            
            row_df['Cardio_minutes_mean'] = np.nanmean(fitbit_summary_df.Cardio_minutes)
            row_df['Cardio_minutes_std'] = np.nanstd(fitbit_summary_df.Cardio_minutes)
            row_df['Cardio_minutes_min'] = np.nanmin(fitbit_summary_df.Cardio_minutes)
            row_df['Cardio_minutes_max'] = np.nanmax(fitbit_summary_df.Cardio_minutes)
            row_df['Cardio_minutes_median'] = np.nanmedian(fitbit_summary_df.Cardio_minutes)
            
            row_df['Peak_caloriesOut_mean'] = np.nanmean(fitbit_summary_df.Peak_caloriesOut)
            row_df['Peak_caloriesOut_std'] = np.nanstd(fitbit_summary_df.Peak_caloriesOut)
            row_df['Peak_caloriesOut_min'] = np.nanmin(fitbit_summary_df.Peak_caloriesOut)
            row_df['Peak_caloriesOut_max'] = np.nanmax(fitbit_summary_df.Peak_caloriesOut)
            row_df['Peak_caloriesOut_median'] = np.nanmedian(fitbit_summary_df.Peak_caloriesOut)
            
            row_df['Peak_minutes_mean'] = np.nanmean(fitbit_summary_df.Peak_minutes)
            row_df['Peak_minutes_std'] = np.nanstd(fitbit_summary_df.Peak_minutes)
            row_df['Peak_minutes_min'] = np.nanmin(fitbit_summary_df.Peak_minutes)
            row_df['Peak_minutes_max'] = np.nanmax(fitbit_summary_df.Peak_minutes)
            row_df['Peak_minutes_median'] = np.nanmedian(fitbit_summary_df.Peak_minutes)
            
            row_df['Fat_Burn_caloriesOut_mean'] = np.nanmean(fitbit_summary_df['Fat Burn_caloriesOut'])
            row_df['Fat_Burn_caloriesOut_std'] = np.nanstd(fitbit_summary_df['Fat Burn_caloriesOut'])
            row_df['Fat_Burn_caloriesOut_min'] = np.nanmin(fitbit_summary_df['Fat Burn_caloriesOut'])
            row_df['Fat_Burn_caloriesOut_max'] = np.nanmax(fitbit_summary_df['Fat Burn_caloriesOut'])
            row_df['Fat_Burn_caloriesOut_median'] = np.nanmedian(fitbit_summary_df['Fat Burn_caloriesOut'])
            
            row_df['Fat_Burn_minutes_mean'] = np.nanmean(fitbit_summary_df['Fat Burn_minutes'])
            row_df['Fat_Burn_minutes_std'] = np.nanstd(fitbit_summary_df['Fat Burn_minutes'])
            row_df['Fat_Burn_minutes_min'] = np.nanmin(fitbit_summary_df['Fat Burn_minutes'])
            row_df['Fat_Burn_minutes_max'] = np.nanmax(fitbit_summary_df['Fat Burn_minutes'])
            row_df['Fat_Burn_minutes_median'] = np.nanmedian(fitbit_summary_df['Fat Burn_minutes'])

            row_df['Out_of_range_minutes_mean'] = np.nanmean(fitbit_summary_df['Out of Range_minutes'])
            row_df['Out_of_range_minutes_std'] = np.nanstd(fitbit_summary_df['Out of Range_minutes'])
            row_df['Out_of_range_minutes_min'] = np.nanmin(fitbit_summary_df['Out of Range_minutes'])
            row_df['Out_of_range_minutes_max'] = np.nanmax(fitbit_summary_df['Out of Range_minutes'])
            row_df['Out_of_range_minutes_median'] = np.nanmedian(fitbit_summary_df['Out of Range_minutes'])
            
            row_df['NumberSteps_mean'] = np.nanmean(fitbit_summary_df.NumberSteps)
            row_df['NumberSteps_std'] = np.nanstd(fitbit_summary_df.NumberSteps)
            row_df['NumberSteps_min'] = np.nanmin(fitbit_summary_df.NumberSteps)
            row_df['NumberSteps_max'] = np.nanmax(fitbit_summary_df.NumberSteps)
            row_df['NumberSteps_median'] = np.median(fitbit_summary_df.NumberSteps)
            
            row_df['RestingHeartRate_mean'] = np.nanmean(fitbit_summary_df.RestingHeartRate)
            row_df['RestingHeartRate_std'] = np.nanstd(fitbit_summary_df.RestingHeartRate)
            row_df['RestingHeartRate_min'] = np.nanmin(fitbit_summary_df.RestingHeartRate)
            row_df['RestingHeartRate_max'] = np.nanmax(fitbit_summary_df.RestingHeartRate)
            row_df['RestingHeartRate_median'] = np.nanmedian(fitbit_summary_df.RestingHeartRate)
            
            row_df['SleepMinutesInBed_mean'] = np.nanmean(fitbit_summary_df.SleepMinutesInBed)
            row_df['SleepMinutesInBed_std'] = np.nanstd(fitbit_summary_df.SleepMinutesInBed)
            row_df['SleepMinutesInBed_min'] = np.nanmax(fitbit_summary_df.SleepMinutesInBed)
            row_df['SleepMinutesInBed_max'] = np.nanmin(fitbit_summary_df.SleepMinutesInBed)
            row_df['SleepMinutesInBed_median'] = np.nanmedian(fitbit_summary_df.SleepMinutesInBed)
            
            row_df['SleepEfficiency_mean'] = np.nanmean(np.array(list(fitbit_summary_df.Sleep1Efficiency) + list(fitbit_summary_df.Sleep2Efficiency)))
            row_df['SleepEfficiency_std'] = np.nanstd(np.array(list(fitbit_summary_df.Sleep1Efficiency) + list(fitbit_summary_df.Sleep2Efficiency)))
            row_df['SleepEfficiency_min'] = np.nanmin(np.array(list(fitbit_summary_df.Sleep1Efficiency) + list(fitbit_summary_df.Sleep2Efficiency)))
            row_df['SleepEfficiency_max'] = np.nanmax(np.array(list(fitbit_summary_df.Sleep1Efficiency) + list(fitbit_summary_df.Sleep2Efficiency)))
            row_df['SleepEfficiency_median'] = np.nanmedian(np.array(list(fitbit_summary_df.Sleep1Efficiency) + list(fitbit_summary_df.Sleep2Efficiency)))
        
        for group_label in group_label_list:
            if group_label == 'position':
                row_df[group_label] = 1 if cond1 or cond2 else 2
            elif group_label == 'shift':
                row_df[group_label] = 1 if cond4 else 2
            elif group_label == 'Sex':
                row_df[group_label] = 1 if cond5 else 2
            elif 'igtb' in group_label or 'Flexbility' in group_label or 'Inflexbility' in group_label \
                    or 'LifeSatisfaction' in group_label or 'General_Health' in group_label \
                    or 'Emotional_Wellbeing' in group_label or 'Engage' in group_label or 'Perceivedstress' in group_label:
                score = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][group_label].values[0]
                cond_igtb = score >= mean_dict[group_label]
                row_df[group_label] = 1 if cond_igtb else 2
            elif group_label == 'fatigue':
                score = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][group_label].values[0]
                if score == ' ' or score == 'nan' or score == np.nan:
                    row_df[group_label] = 1
                else:
                    row_df[group_label] = 1 if float(score) > 60 else 2
            else:
                row_df[group_label] = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][group_label].values[0]
    
        data_df = data_df.append(row_df)

    for group_label in group_label_list:
        print('label: %s' % group_label)
        print('class balance: %d, %d, %.2f' % (len(data_df.loc[data_df[group_label] == 1]), len(data_df.loc[data_df[group_label] == 2]),
                                               len(data_df.loc[data_df[group_label] == 1]) / len(data_df)))

    param_grid = {"max_depth": [4, 5, 6], "min_samples_split": [2, 3, 5], "bootstrap": [True, False], "n_estimators": [10, 20, 30], "criterion": ["gini", "entropy"]}

    # for group_label in group_label_list:
    result = pd.DataFrame(columns=group_label_list, index=[index])
    feat_importance_result = pd.DataFrame(columns=hawkes_kernel_df.columns, index=group_label_list)
    
    if not fitbit and not fusion:
        for group_label in group_label_list:
            print('--------------------------------------------------')
            print('predict %s' % group_label)
            
            # ML data
            X = np.array(hawkes_kernel_df)
            y = np.array(data_df[group_label])
            
            clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_macro')
            clf.fit(X, y)
            feat_importance_result.loc[group_label, :] = clf.best_estimator_.feature_importances_
        
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print(clf.best_score_)
            print()
            print("Grid scores on development set:")
            print()
            print('--------------------------------------------------')
            result[group_label] = clf.best_score_

    fitbit_result = pd.DataFrame(columns=group_label_list, index=[index])
    fitbit_feat_importance_result = pd.DataFrame(columns=fitbit_cols, index=group_label_list)
    if fitbit:
        for group_label in group_label_list:
            print('--------------------------------------------------')
            print('predict %s' % group_label)

            # ML data
            X = np.array(data_df[fitbit_cols])
            y = np.array(data_df[group_label])
            clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_macro')
            clf.fit(X, y)

            fitbit_feat_importance_result.loc[group_label, :] = clf.best_estimator_.feature_importances_

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print(clf.best_score_)
            print()
            print("Grid scores on development set:")
            print()
            print('--------------------------------------------------')
            fitbit_result[group_label] = clf.best_score_

    fusion_result = pd.DataFrame(columns=group_label_list, index=[index])
    fusion_feat_importance_result = pd.DataFrame(columns=fitbit_cols+list(hawkes_kernel_df.columns), index=group_label_list)
    if fusion:
        for group_label in group_label_list:
            print('--------------------------------------------------')
            print('predict %s' % group_label)

            # ML data
            X = np.array(data_df[fitbit_cols+list(hawkes_kernel_df.columns)])
            y = np.array(data_df[group_label])
            clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_macro')
            clf.fit(X, y)

            fusion_feat_importance_result.loc[group_label, :] = clf.best_estimator_.feature_importances_

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print(clf.best_score_)
            print()
            print("Grid scores on development set:")
            print()
            print('--------------------------------------------------')
            fusion_result[group_label] = clf.best_score_
    
    return result, fitbit_result, fusion_result, feat_importance_result, fitbit_feat_importance_result, fusion_feat_importance_result


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

    # Save data and path
    final_result_df, fitbit_final_result_df, fusion_final_result_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    num_of_gaussian, fitbit_enable, fusion_enabled = 100, False, False
    prefix = data_config.fitbit_sensor_dict['clustering_path'].split('_impute_')[0]
    prefix = prefix.split('clustering/fitbit/')[1]
    save_path = prefix + '_num_of_gaussian_' + str(num_of_gaussian) + '_transitional.csv'
    save_path_feat = prefix + '_num_of_gaussian_' + str(num_of_gaussian) + '_transitional_feat_imp.csv'
    
    '''
    ticc_num_cluster_6_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 2
    auto_arima_15_ticc_num_cluster_3_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 1
    auto_arima_15_ticc_num_cluster_4_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 1
    auto_arima_15_ticc_num_cluster_5_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 1
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
        final_result_per_day_setting_df, final_fitbit_result_per_day_setting_df, final_fusion_result_per_day_setting_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        final_feat_importance_result, final_fitbit_feat_importance_result, final_fusion_feat_importance_result = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        for j in range(5):
            # save_hawkes_kernel(data_config, top_participant_id_list, save_model_path, num_of_days=i)
            # result_df = predict_demographic(groundtruth_df, save_model_path, top_participant_id_list, j)
            result_df, fitbit_result_df, fusion_result_df, \
            feat_importance_result, \
            fitbit_feat_importance_result, \
            fusion_feat_importance_result = predict(data_config, groundtruth_df, top_participant_id_list, j,
                                                    fusion=fusion_enabled, fitbit=fitbit_enable,
                                                    num_of_days=i, num_of_gaussian=num_of_gaussian, remove_col_index=1)
            final_result_per_day_setting_df = final_result_per_day_setting_df.append(result_df)
            final_fitbit_result_per_day_setting_df = final_fitbit_result_per_day_setting_df.append(fitbit_result_df)
            final_fusion_result_per_day_setting_df = final_fusion_result_per_day_setting_df.append(fusion_result_df)
            
            if i == 5:
                final_feat_importance_result = feat_importance_result if len(final_feat_importance_result) == 0 else final_feat_importance_result + feat_importance_result
                final_fusion_feat_importance_result = fusion_feat_importance_result if len(final_fusion_feat_importance_result) == 0 else final_fusion_feat_importance_result + fusion_feat_importance_result
                final_fitbit_feat_importance_result = fitbit_feat_importance_result if len(final_fitbit_feat_importance_result) == 0 else final_fitbit_feat_importance_result + fitbit_feat_importance_result
        
        if not fitbit_enable and not fusion_enabled:
            tmp_df = pd.DataFrame(np.mean(np.array(final_result_per_day_setting_df), axis=0).reshape([1, -1]), index=[i], columns=final_result_per_day_setting_df.columns)
            final_result_df = final_result_df.append(tmp_df)
            final_result_df.to_csv(os.path.join(os.curdir, 'result', save_path))
            final_feat_importance_result = final_feat_importance_result / 5
            final_feat_importance_result.to_csv(os.path.join(os.curdir, 'result', save_path_feat))
        
        if fitbit_enable is True:
            tmp_df = pd.DataFrame(np.mean(np.array(final_fitbit_result_per_day_setting_df), axis=0).reshape([1, -1]), index=[i], columns=final_fitbit_result_per_day_setting_df.columns)
            fitbit_final_result_df = fitbit_final_result_df.append(tmp_df)
            fitbit_final_result_df.to_csv(os.path.join(os.curdir, 'result', 'fitbit_result.csv'))
            final_fitbit_feat_importance_result = final_fitbit_feat_importance_result / 5
            final_fitbit_feat_importance_result.to_csv(os.path.join(os.curdir, 'result', 'fitbit_feat_result.csv'))
            
        if fusion_enabled is True:
            tmp_df = pd.DataFrame(np.mean(np.array(final_fusion_result_per_day_setting_df), axis=0).reshape([1, -1]), index=[i], columns=final_fusion_result_per_day_setting_df.columns)
            fusion_final_result_df = fusion_final_result_df.append(tmp_df)
            fusion_final_result_df.to_csv(os.path.join(os.curdir, 'result', 'fusion_' + save_path_feat))
            final_fusion_feat_importance_result = final_fusion_feat_importance_result / 5
            final_fusion_feat_importance_result.to_csv(os.path.join(os.curdir, 'result', 'fusion_feat_'+save_path_feat))
            
    print(final_result_df)


if __name__ == '__main__':
    
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)
