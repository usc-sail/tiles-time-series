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

# igtb values
igtb_label_list = ['neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb',
                   'pos_af_igtb', 'neg_af_igtb', 'stai_igtb', 'audit_igtb',
                   'shipley_abs_igtb', 'shipley_voc_igtb', 'Inflexbility', 'Flexbility',
                   'LifeSatisfaction', 'General_Health', 'Emotional_Wellbeing', 'Engage', 'Perceivedstress',
                   'itp_igtb', 'irb_igtb', 'iod_id_igtb', 'iod_od_igtb', 'ocb_igtb']

# fitbit columns
fitbit_cols = ['Cardio_caloriesOut_mean', 'Cardio_caloriesOut_std', 'Cardio_minutes_mean', 'Cardio_minutes_std',
               'Peak_caloriesOut_mean', 'Peak_caloriesOut_std', 'Peak_minutes_mean', 'Peak_minutes_std',
               'Fat_Burn_caloriesOut_mean', 'Fat_Burn_caloriesOut_std', 'NumberSteps_mean', 'NumberSteps_std',
               'RestingHeartRate_std', 'SleepMinutesInBed_mean', 'SleepMinutesInBed_std', 'SleepEfficiency_mean', 'SleepEfficiency_std']

group_label_list = ['shift', 'position', 'fatigue', 'Inflexbility', 'Flexbility', 'Sex',
                    'LifeSatisfaction', 'General_Health', 'Emotional_Wellbeing', 'Engage', 'Perceivedstress',
                    'pos_af_igtb', 'neg_af_igtb', 'stai_igtb',
                    'neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb']


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
                    workday_point_list.append(day_point_list)
            else:
                if len(offday_point_list) < num_of_days:
                    offday_point_list.append(day_point_list)
    
    # Learn causality
    workday_learner = HawkesSumGaussians(num_of_gaussian, max_iter=50)
    workday_learner.fit(workday_point_list)
    ineffective_array = np.array(workday_learner.get_kernel_norms())
    for i in range(ineffective_array.shape[0] * ineffective_array.shape[1]):
        row_df['feat' + str(i)] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][i]

    offday_learner = HawkesSumGaussians(num_of_gaussian, max_iter=50)
    offday_learner.fit(offday_point_list)
    ineffective_array = np.array(offday_learner.get_kernel_norms())
    
    for i in range(ineffective_array.shape[0] * ineffective_array.shape[1]):
        row_df['feat' + str(ineffective_array.shape[0] * ineffective_array.shape[1] + i)] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][i]
    
    return row_df
    

def predict(data_config, groundtruth_df, top_participant_id_list, index, fitbit=False,
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
        
        if fitbit:
            # Summary features
            row_df['Cardio_caloriesOut_mean'] = np.nanmean(fitbit_summary_df.Cardio_caloriesOut)
            row_df['Cardio_caloriesOut_std'] = np.nanstd(fitbit_summary_df.Cardio_caloriesOut)
            row_df['Cardio_minutes_mean'] = np.nanmean(fitbit_summary_df.Cardio_minutes)
            row_df['Cardio_minutes_std'] = np.nanstd(fitbit_summary_df.Cardio_minutes)
            
            row_df['Peak_caloriesOut_mean'] = np.nanmean(fitbit_summary_df.Peak_caloriesOut)
            row_df['Peak_caloriesOut_std'] = np.nanstd(fitbit_summary_df.Peak_caloriesOut)
            row_df['Peak_minutes_mean'] = np.nanmean(fitbit_summary_df.Peak_minutes)
            row_df['Peak_minutes_std'] = np.nanstd(fitbit_summary_df.Peak_minutes)
            
            row_df['Fat_Burn_caloriesOut_mean'] = np.nanmean(fitbit_summary_df['Fat Burn_caloriesOut'])
            row_df['Fat_Burn_caloriesOut_std'] = np.nanstd(fitbit_summary_df['Fat Burn_caloriesOut'])
            row_df['NumberSteps_mean'] = np.nanmean(fitbit_summary_df.NumberSteps)
            row_df['NumberSteps_std'] = np.nanstd(fitbit_summary_df.NumberSteps)
            row_df['RestingHeartRate_std'] = np.nanstd(fitbit_summary_df.RestingHeartRate)
            
            row_df['SleepMinutesInBed_mean'] = np.nanmean(fitbit_summary_df.SleepMinutesInBed)
            row_df['SleepMinutesInBed_std'] = np.nanstd(fitbit_summary_df.SleepMinutesInBed)
            row_df['SleepEfficiency_mean'] = np.nanmean(np.array(list(fitbit_summary_df.Sleep1Efficiency) + list(fitbit_summary_df.Sleep2Efficiency)))
            row_df['SleepEfficiency_std'] = np.nanstd(np.array(list(fitbit_summary_df.Sleep1Efficiency) + list(fitbit_summary_df.Sleep2Efficiency)))
        
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
    for group_label in group_label_list:
        print('--------------------------------------------------')
        print('predict %s' % group_label)
        
        # ML data
        X = np.array(hawkes_kernel_df)
        y = np.array(data_df[group_label])
        
        clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_micro')
        clf.fit(X, y)
    
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
    if fitbit:
        for group_label in group_label_list:
            print('--------------------------------------------------')
            print('predict %s' % group_label)

            # ML data
            X = np.array(data_df[fitbit_cols])
            y = np.array(data_df[group_label])
            clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='f1_micro')
            clf.fit(X, y)
        
            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print(clf.best_score_)
            print()
            print("Grid scores on development set:")
            print()
            print('--------------------------------------------------')
            fitbit_result[group_label] = clf.best_score_
    
    return result, fitbit_result


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
    final_result_df, fitbit_final_result_df = pd.DataFrame(), pd.DataFrame()

    num_of_gaussian, fitbit_enable = 5, False
    prefix = data_config.fitbit_sensor_dict['clustering_path'].split('_impute_')[0]
    prefix = prefix.split('clustering/fitbit/')[1]
    save_path = prefix + '_num_of_gaussian_' + str(num_of_gaussian) + '_transitional.csv'
    
    '''
    ticc_num_cluster_6_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 2
    ticc_num_cluster_6_window_10_penalty_10.0_sparsity_0.1_cluster_days_7: 5
    ticc_num_cluster_4_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 3
    ticc_num_cluster_4_window_10_penalty_10.0_sparsity_0.1_cluster_days_7: 3
    ticc_num_cluster_5_window_10_penalty_10.0_sparsity_0.1_cluster_days_5: 3
    ticc_num_cluster_5_window_10_penalty_10.0_sparsity_0.1_cluster_days_7: 3
    '''

    # for i in range(3, 8, 2):
    for i in range(3, 6):
        final_result_per_day_setting_df, final_fitbit_result_per_day_setting_df = pd.DataFrame(), pd.DataFrame()
        
        for j in range(5):
            # save_hawkes_kernel(data_config, top_participant_id_list, save_model_path, num_of_days=i)
            # result_df = predict_demographic(groundtruth_df, save_model_path, top_participant_id_list, j)
            result_df, fitbit_result_df = predict(data_config, groundtruth_df, top_participant_id_list, j,
                                                  fitbit=fitbit_enable, num_of_days=i, num_of_gaussian=num_of_gaussian, remove_col_index=3)
            final_result_per_day_setting_df = final_result_per_day_setting_df.append(result_df)
            final_fitbit_result_per_day_setting_df = final_fitbit_result_per_day_setting_df.append(fitbit_result_df)

        tmp_df = pd.DataFrame(np.mean(np.array(final_result_per_day_setting_df), axis=0).reshape([1, -1]), index=[i], columns=final_result_per_day_setting_df.columns)
        final_result_df = final_result_df.append(tmp_df)
        final_result_df.to_csv(os.path.join(os.curdir, 'result', save_path))
        
        if fitbit_enable is True:
            tmp_df = pd.DataFrame(np.mean(np.array(final_fitbit_result_per_day_setting_df), axis=0).reshape([1, -1]), index=[i], columns=final_fitbit_result_per_day_setting_df.columns)
            fitbit_final_result_df = fitbit_final_result_df.append(tmp_df)
            fitbit_final_result_df.to_csv(os.path.join(os.curdir, 'result', 'fitbit_result.csv'))
            
    print(final_result_df)


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)