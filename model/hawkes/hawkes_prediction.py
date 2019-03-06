"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np
from random import shuffle

from tick.dataset import fetch_hawkes_bund_data
from tick.hawkes import HawkesConditionalLaw
from tick.plot import plot_hawkes_kernel_norms

from tick.hawkes import (SimuHawkes, SimuHawkesMulti, HawkesKernelExp,
                         HawkesKernelTimeFunc, HawkesKernelPowerLaw,
                         HawkesKernel0, HawkesSumGaussians)

from tick.hawkes import (HawkesCumulantMatching, SimuHawkesExpKernels, SimuHawkesMulti)

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


def save_hawkes_kernel(data_config, top_participant_id_list, save_model_path, num_of_days):
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # Read per participant clustering
        clustering_data_list = load_sensor_data.load_filter_clustering(data_config.fitbit_sensor_dict['clustering_path'], participant_id)
        
        # Read per participant data
        participant_data_dict = load_sensor_data.load_filter_data(data_config.fitbit_sensor_dict['filter_path'], participant_id, filter_logic=None, valid_data_rate=0.9, threshold_dict={'min': 20, 'max': 28})
        
        if clustering_data_list is None or participant_data_dict is None:
            continue
        
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
                
                change_list = [(cluster_array[0][0], 0)]
                
                for i in change_point_index[0]:
                    change_list.append((cluster_array[i + 1][0], i + 1))
                
                # Initiate list for counter
                day_point_list = []
                for i in range(data_config.fitbit_sensor_dict['num_cluster']):
                    day_point_list.append(np.zeros(1))
                
                for change_tuple in change_list:
                    day_point_list[int(change_tuple[0])] = np.append(day_point_list[int(change_tuple[0])],
                                                                     change_tuple[1])
                
                for i, day_point_array in enumerate(day_point_list):
                    if len(day_point_list[i]) == 0:
                        day_point_list[i] = np.array(len(cluster_array))
                    else:
                        day_point_list[i] = np.sort(day_point_list[i][1:])
                
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
        
        if os.path.exists(os.path.join(save_model_path, participant_id)) is False:
            os.mkdir(os.path.join(save_model_path, participant_id))
        
        # Learn causality
        workday_learner = HawkesSumGaussians(5, max_iter=20)
        workday_learner.fit(workday_point_list)
        ineffective_df = pd.DataFrame(workday_learner.get_kernel_norms())
        ineffective_df.to_csv(os.path.join(save_model_path, participant_id, 'workday.csv.gz'), compression='gzip')
        
        offday_learner = HawkesSumGaussians(5, max_iter=20)
        offday_learner.fit(offday_point_list)
        ineffective_df = pd.DataFrame(offday_learner.get_kernel_norms())
        ineffective_df.to_csv(os.path.join(save_model_path, participant_id, 'offday.csv.gz'), compression='gzip')
    print('Successfully cluster all participant filter data')
    

def predict(groundtruth_df, save_model_path, top_participant_id_list, index):
    data_dict_list = []
    data_cluster_df, data_df = pd.DataFrame(), pd.DataFrame()
    
    predict_label_list = ['neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb',
                          'pos_af_igtb', 'neg_af_igtb', 'stai_igtb', 'audit_igtb',
                          'shipley_abs_igtb', 'shipley_voc_igtb',
                          'itp_igtb', 'irb_igtb', 'iod_id_igtb', 'iod_od_igtb', 'ocb_igtb']
    
    # group_label_list = ['gender', 'position']
    group_label_list = ['shift', 'position']
    
    groundtruth_df[predict_label_list] = groundtruth_df[predict_label_list].fillna(groundtruth_df[predict_label_list].mean())

    for idx, participant_id in enumerate(top_participant_id_list):
    
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
    
        if os.path.exists(os.path.join(save_model_path, participant_id, 'offday.csv.gz')) is False:
            continue
    
        cond1 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['currentposition'].values[0] == 1
        cond2 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['currentposition'].values[0] == 2
        cond3 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['gender'].values[0] > 0
        cond4 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['Shift'].values[0] == 'Day shift'
        if not cond3:
            continue

        # if not cond1 and not cond2:
        #    continue
    
        ineffective_df = pd.read_csv(os.path.join(save_model_path, participant_id, 'workday.csv.gz'), index_col=0)
        ineffective_array = np.array(ineffective_df)
        ineffective_array = np.delete(ineffective_array, 2, axis=0)
        ineffective_array = np.delete(ineffective_array, 2, axis=1)

        participant_dict = {}
        participant_dict['participant_id'] = participant_id
        participant_dict['groundtruth_df'] = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]
        participant_dict['data'] = ineffective_array
    
        data_dict_list.append(participant_dict)
    
        row_df = pd.DataFrame(index=[participant_id])
        for i in range(ineffective_array.shape[0] * ineffective_array.shape[1]):
            row_df['feat' + str(i)] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][i]
    
        ineffective_df = pd.read_csv(os.path.join(save_model_path, participant_id, 'offday.csv.gz'), index_col=0)
        ineffective_array = np.array(ineffective_df)
        ineffective_array = np.delete(ineffective_array, 2, axis=0)
        ineffective_array = np.delete(ineffective_array, 2, axis=1)
        
        for i in range(ineffective_array.shape[0] * ineffective_array.shape[1]):
            row_df['feat' + str(ineffective_array.shape[0] * ineffective_array.shape[1] + i)] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][i]
    
        data_cluster_df = data_cluster_df.append(row_df)
    
        for predict_label in predict_label_list:
            row_df[predict_label] = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][predict_label].values[0]
    
        for group_label in group_label_list:
            if group_label == 'position':
                if cond1 or cond2:
                    row_df[group_label] = 1
                else:
                    row_df[group_label] = 2
            elif group_label == 'shift':
                if cond4:
                    row_df[group_label] = 1
                else:
                    row_df[group_label] = 2
            else:
                row_df[group_label] = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][group_label].values[0]
    
        data_df = data_df.append(row_df)

    from sklearn.svm import SVR, SVC
    from sklearn.metrics import r2_score

    from sklearn.decomposition import PCA
    pca = PCA(n_components=20)
    X = pca.fit_transform(np.array(data_cluster_df))
    X = np.array(data_cluster_df)

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    param_grid = {"max_depth": [3, 5, 6],
                  "max_features": [5, 10, 15],
                  "min_samples_split": [2, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # for group_label in group_label_list:
    result = pd.DataFrame(columns=group_label_list, index=[index])
    for group_label in group_label_list:
        print('--------------------------------------------------')
        print('predict %s' % group_label)
        y = np.array(data_df[group_label])
        svc = SVC()
        clf = GridSearchCV(RandomForestClassifier(n_estimators=20), param_grid, cv=5, scoring='f1')
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
    
    return result


def predict_demographic(groundtruth_df, save_model_path, top_participant_id_list, index):
    data_dict_list = []
    data_cluster_df, data_df = pd.DataFrame(), pd.DataFrame()
    
    predict_label_list = ['nurseyears', 'age']
    
    # group_label_list = ['gender', 'position']
    group_label_list = ['shift', 'position']
    
    # groundtruth_df[predict_label_list] = groundtruth_df[predict_label_list].fillna(groundtruth_df[predict_label_list].mean())

    feat_len = 0
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        if os.path.exists(os.path.join(save_model_path, participant_id, 'offday.csv.gz')) is False:
            continue
        
        cond1 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['currentposition'].values[0] == 1
        cond2 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['currentposition'].values[0] == 2
        cond3 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['gender'].values[0] > 0
        cond4 = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]['Shift'].values[0] == 'Day shift'
        if not cond3:
            continue
        
        # if not cond1 and not cond2:
        #    continue
        
        ineffective_df = pd.read_csv(os.path.join(save_model_path, participant_id, 'workday.csv.gz'), index_col=0)
        ineffective_array = np.array(ineffective_df)
        ineffective_array = np.delete(ineffective_array, 2, axis=0)
        ineffective_array = np.delete(ineffective_array, 2, axis=1)
        
        participant_dict = {}
        participant_dict['participant_id'] = participant_id
        participant_dict['groundtruth_df'] = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]
        participant_dict['data'] = ineffective_array
        
        data_dict_list.append(participant_dict)
        
        row_df = pd.DataFrame(index=[participant_id])
        for i in range(ineffective_array.shape[0] * ineffective_array.shape[1]):
            row_df['feat' + str(i)] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][i]
        
        ineffective_df = pd.read_csv(os.path.join(save_model_path, participant_id, 'offday.csv.gz'), index_col=0)
        ineffective_array = np.array(ineffective_df)
        ineffective_array = np.delete(ineffective_array, 2, axis=0)
        ineffective_array = np.delete(ineffective_array, 2, axis=1)
        
        for i in range(ineffective_array.shape[0] * ineffective_array.shape[1]):
            row_df['feat' + str(ineffective_array.shape[0] * ineffective_array.shape[1] + i)] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][i]
        
        feat_len = len(row_df)
        
        invalid_data = False
        for predict_label in predict_label_list:
            cond_null = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][predict_label].values[0] != ' '
            if cond_null:
                row_df[predict_label] = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][predict_label].values[0]
            else:
                invalid_data = True
        if invalid_data:
            continue
            
        data_df = data_df.append(row_df)

    data_df = data_df.dropna()
    feat_cols = ['feat' + str(i) for i in range(feat_len)]
    data_cluster_df = data_df[feat_cols]

    X = np.array(data_cluster_df)

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    param_grid = {"max_depth": [3, 4, 5],
                  "min_samples_split": [2, 3, 5],
                  "bootstrap": [True, False],
                  'max_features': ['auto', 'sqrt'],
                  'n_estimators': [5, 10, 20, 30]}
    
    # for group_label in group_label_list:
    result = pd.DataFrame(columns=predict_label_list, index=[index])
    for group_label in predict_label_list:
        print('--------------------------------------------------')
        print('predict %s' % group_label)
        y = np.array(data_df[group_label])
        clf = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='r2')
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
    
    return result


def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path, filter_data=True,
                                           preprocess_data_identifier='preprocess_data',
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

    final_result_df = pd.DataFrame()
    for i in range(3, 6):
        final_result_per_day_setting_df = pd.DataFrame()
        for j in range(5):
            save_hawkes_kernel(data_config, top_participant_id_list, save_model_path, num_of_days=i)
            result_df = predict_demographic(groundtruth_df, save_model_path, top_participant_id_list, j)
            final_result_per_day_setting_df = final_result_per_day_setting_df.append(result_df)

        tmp_df = pd.DataFrame(np.mean(np.array(final_result_per_day_setting_df), axis=0).reshape([1, -1]), index=[i], columns=final_result_per_day_setting_df.columns)
        final_result_df = final_result_df.append(tmp_df)
    
    print(final_result_df)


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)