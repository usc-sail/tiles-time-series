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


def get_hawkes_kernel(data_config, participant_id, num_of_days, remove_col_index=2, num_of_gaussian=10):
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
        return None, None
    
    # Read filtered data
    filter_data_list = participant_data_dict['filter_data_list']
    workday_point_list, offday_point_list = [], []
    shuffle(clustering_data_list)
    
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
            
            if int(cluster_array[0][0]) != remove_col_index and int(
                    cluster_array[change_point_index[0][0] + 1][0]) != remove_col_index:
                change_list = [(int(cluster_array[0][0]), int(cluster_array[change_point_index[0][0] + 1][0]), change_point_index[0][0] + 1)]
            else:
                change_list = []
            
            for i in range(len(change_point_index[0]) - 1):
                start = change_point_index[0][i] + 1
                end = change_point_index[0][i + 1] + 1
                if cluster_array[start][0] != remove_col_index and int(cluster_array[end][0]) != remove_col_index:
                    change_list.append((int(cluster_array[start][0]), int(cluster_array[end][0]), change_point_index[0][i + 1] + 1))
            
            # Initiate list for counter
            day_point_list = []
            point_dict = {}
            
            index = 0
            for i in range(data_config.fitbit_sensor_dict['num_cluster']):
                for j in range(data_config.fitbit_sensor_dict['num_cluster']):
                    if i != j and i != remove_col_index and j != remove_col_index:
                        day_point_list.append(np.zeros(1))
                        point_dict[(i, j)] = index
                        index += 1
            
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
                    workday_point_list.append([point / 60 for point in day_point_list])
            else:
                if len(offday_point_list) < num_of_days:
                    offday_point_list.append([point / 60 for point in day_point_list])
    
    workday_col_list, offday_col_list = [], []
    
    dict_activity = {0: 'Rest', 2: 'LA', 3: 'MA'}
    
    for i in range(data_config.fitbit_sensor_dict['num_cluster']):
        for j in range(data_config.fitbit_sensor_dict['num_cluster']):
            if i != j and i != remove_col_index and j != remove_col_index:
                workday_col_list.append('workday:' + str(i) + ',' + str(j))
                offday_col_list.append(dict_activity[i] + '->' + dict_activity[j])
    
    # Learn causality
    workday_learner = HawkesSumGaussians(num_of_gaussian, max_iter=10)
    workday_learner.fit(workday_point_list)
    ineffective_array = np.array(workday_learner.get_kernel_norms())
    
    for i in range(ineffective_array.shape[0]):
        for j in range(ineffective_array.shape[1]):
            index = i * ineffective_array.shape[0] + j
            row_df[workday_col_list[i] + '->' + workday_col_list[j]] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][index]

    workdata_df = pd.DataFrame(columns=offday_col_list, index=offday_col_list)
    for i in range(ineffective_array.shape[0]):
        for j in range(ineffective_array.shape[1]):
            index = i * ineffective_array.shape[0] + j
            workdata_df.loc[offday_col_list[i], offday_col_list[j]] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][index]

    offday_learner = HawkesSumGaussians(num_of_gaussian, max_iter=10)
    offday_learner.fit(offday_point_list)
    ineffective_array = np.array(offday_learner.get_kernel_norms())
    
    for i in range(ineffective_array.shape[0]):
        for j in range(ineffective_array.shape[1]):
            index = i * ineffective_array.shape[0] + j
            row_df[offday_col_list[i] + '->' + offday_col_list[j]] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][index]
    
    offdata_df = pd.DataFrame(columns=offday_col_list, index=offday_col_list)
    for i in range(ineffective_array.shape[0]):
        for j in range(ineffective_array.shape[1]):
            index = i * ineffective_array.shape[0] + j
            offdata_df.loc[offday_col_list[i], offday_col_list[j]] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][index]

    return workdata_df, offdata_df


def predict(data_config, groundtruth_df, top_participant_id_list, num_of_days=5, num_of_gaussian=10, remove_col_index=2):
    # Initiate data df
    groundtruth_df[igtb_label_list] = groundtruth_df[igtb_label_list].fillna(groundtruth_df[igtb_label_list].mean())
    
    mean_dict = {'pos_af_igtb': np.nanmedian(groundtruth_df['pos_af_igtb']),
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
    
    group_1_workday_df, group_1_offday_df = pd.DataFrame(), pd.DataFrame()
    group_2_workday_df, group_2_offday_df = pd.DataFrame(), pd.DataFrame()
    group_3_workday_df, group_3_offday_df = pd.DataFrame(), pd.DataFrame()
    group_4_workday_df, group_4_offday_df = pd.DataFrame(), pd.DataFrame()
    
    group_1_cnt, group_2_cnt, group_3_cnt, group_4_cnt = 0, 0, 0, 0

    for idx, participant_id in enumerate(top_participant_id_list):
        
        ###########################################################
        # Print out
        ###########################################################
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        ###########################################################
        # Read hawkes feature, if None, continue
        ###########################################################
        workdata_df, offdata_df = get_hawkes_kernel(data_config, participant_id, num_of_days, remove_col_index=remove_col_index, num_of_gaussian=num_of_gaussian)
        
        if workdata_df is None:
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
                    group_1_workday_df = workdata_df if len(group_1_workday_df) == 0 else group_1_workday_df + workdata_df
                    group_1_offday_df = offdata_df if len(group_1_offday_df) == 0 else group_1_offday_df + offdata_df
                    group_1_cnt += 1
                else:
                    group_2_workday_df = workdata_df if len(group_2_workday_df) == 0 else group_2_workday_df + workdata_df
                    group_2_offday_df = offdata_df if len(group_2_offday_df) == 0 else group_2_offday_df + offdata_df
                    group_2_cnt += 1
            elif 'agr_igtb' in group_label:
                score = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][group_label].values[0]
                cond_igtb = score >= mean_dict[group_label]
                
                if cond_igtb:
                    group_3_workday_df = workdata_df if len(group_3_workday_df) == 0 else group_3_workday_df + workdata_df
                    group_3_offday_df = offdata_df if len(group_3_offday_df) == 0 else group_3_offday_df + offdata_df
                    group_3_cnt += 1
                else:
                    group_4_workday_df = workdata_df if len(group_4_workday_df) == 0 else group_4_workday_df + workdata_df
                    group_4_offday_df = offdata_df if len(group_4_offday_df) == 0 else group_4_offday_df + offdata_df
                    group_4_cnt += 1

    group_1_workday_df = group_1_workday_df / group_1_cnt
    group_2_workday_df = group_2_workday_df / group_2_cnt
    group_3_workday_df = group_3_workday_df / group_3_cnt
    group_4_workday_df = group_4_workday_df / group_4_cnt
    
    group_1_offday_df = group_1_offday_df / group_1_cnt
    group_2_offday_df = group_2_offday_df / group_2_cnt
    group_3_offday_df = group_3_offday_df / group_3_cnt
    group_4_offday_df = group_4_offday_df / group_4_cnt
    
    group_1_workday_df.to_csv(os.path.join(os.curdir, 'result', 'group1_worday.csv'))
    group_2_workday_df.to_csv(os.path.join(os.curdir, 'result', 'group2_worday.csv'))
    group_3_workday_df.to_csv(os.path.join(os.curdir, 'result', 'group3_worday.csv'))
    group_4_workday_df.to_csv(os.path.join(os.curdir, 'result', 'group4_worday.csv'))

    group_1_offday_df.to_csv(os.path.join(os.curdir, 'result', 'group1_offday.csv'))
    group_2_offday_df.to_csv(os.path.join(os.curdir, 'result', 'group2_offday.csv'))
    group_3_offday_df.to_csv(os.path.join(os.curdir, 'result', 'group3_offday.csv'))
    group_4_offday_df.to_csv(os.path.join(os.curdir, 'result', 'group4_offday.csv'))
    
    
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
    
    save_model_path = os.path.join(os.curdir, 'result',
                                   data_config.fitbit_sensor_dict['clustering_path'].split('/')[-1])
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
    
    num_of_gaussian, fitbit_enable = 4, False
    prefix = data_config.fitbit_sensor_dict['clustering_path'].split('_impute_')[0]
    prefix = prefix.split('clustering/fitbit/')[1]
    
    group_1_workday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'group1_worday.csv'), index_col=0)
    group_2_workday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'group2_worday.csv'), index_col=0)
    group_3_workday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'group3_worday.csv'), index_col=0)
    group_4_workday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'group4_worday.csv'), index_col=0)

    group_1_offday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'group1_offday.csv'), index_col=0)
    group_2_offday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'group2_offday.csv'), index_col=0)
    group_3_offday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'group3_offday.csv'), index_col=0)
    group_4_offday_df = pd.read_csv(os.path.join(os.curdir, 'result', 'group4_offday.csv'), index_col=0)
    
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    cbar_ax = fig.add_axes([.93, .325, .02, .55])

    sns.heatmap(group_1_workday_df, cmap="jet", linewidths=.5, ax=ax[0], cbar_ax=cbar_ax, cbar_kws={'format': '%.2f'},
                vmin=-0.0, vmax=1.5, xticklabels=list(group_2_offday_df.index), yticklabels=list(group_2_offday_df.index))
    sns.heatmap(group_2_workday_df, cmap="jet", linewidths=.5, ax=ax[1], cbar=False, cbar_ax=None,
                vmin=-0.0, vmax=1.5, xticklabels=list(group_2_offday_df.index), yticklabels=False)
    
    sns.heatmap(group_1_offday_df, cmap="jet", linewidths=.5, ax=ax[2], cbar=False, cbar_ax=None,
                vmin=-0.0, vmax=1.5, xticklabels=list(group_2_offday_df.index), yticklabels=False)
    sns.heatmap(group_2_offday_df, cmap="jet", linewidths=.5, ax=ax[3], cbar=False, cbar_ax=None,
                vmin=-0.0, vmax=1.5, xticklabels=list(group_2_offday_df.index), yticklabels=False)

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

    ax[0].set_title('Nurses: Workdays', fontweight="bold", fontsize=16, y=1.2)
    ax[1].set_title('Non-Nurses: Workdays', fontweight="bold", fontsize=16, y=1.2)
    ax[2].set_title('Nurses: Off-days', fontweight="bold", fontsize=16, y=1.2)
    ax[3].set_title('Non-Nurses: Off-days', fontweight="bold", fontsize=16, y=1.2)
    
    ax[0].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    
    # fig.text(0.2, 1.2, 'Workday', dict(size=16))
    # ax[2].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    # ax[3].set_yticklabels(ax[0].get_yticklabels(), rotation=0)
    
    plt.gcf().subplots_adjust(bottom=0.3)
    # plt.yticks(rotation=180)
    # plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(os.curdir, 'result', 'workday_job.png'), dpi=300)
    
    
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
                num_of_days=i, num_of_gaussian=num_of_gaussian, remove_col_index=1)
            
            
if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir,
                                               'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)
