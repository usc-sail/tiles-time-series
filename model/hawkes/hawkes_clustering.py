"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

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
    
    save_model_path = os.path.join(os.curdir, 'result', data_config.fitbit_sensor_dict['clustering_path'].split('/')[-1])
    
    data_dict_list = []
    data_cluster_df, data_df = pd.DataFrame(), pd.DataFrame()
    
    predict_label_list = ['neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb',
                          'pos_af_igtb', 'neg_af_igtb', 'stai_igtb', 'audit_igtb',
                          'shipley_abs_igtb', 'shipley_voc_igtb',
                          'itp_igtb', 'irb_igtb', 'iod_id_igtb', 'iod_od_igtb', 'ocb_igtb']

    groundtruth_df[predict_label_list] = groundtruth_df[predict_label_list].fillna(groundtruth_df[predict_label_list].mean())
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        if os.path.exists(os.path.join(save_model_path, participant_id, 'workday.csv.gz')) is False:
            continue

        ineffective_df = pd.read_csv(os.path.join(save_model_path, participant_id, 'workday.csv.gz'), index_col=0)
        ineffective_array = np.array(ineffective_df)
        ineffective_array = np.delete(ineffective_array, 5, axis=0)
        ineffective_array = np.delete(ineffective_array, 5, axis=1)

        # ineffective_array = np.delete(ineffective_array, 3, axis=0)
        # ineffective_array = np.delete(ineffective_array, 3, axis=1)
        
        participant_dict = {}
        participant_dict['participant_id'] = participant_id
        participant_dict['groundtruth_df'] = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id]
        participant_dict['data'] = ineffective_array

        data_dict_list.append(participant_dict)
        
        row_df = pd.DataFrame(index=[participant_id])
        for i in range(ineffective_array.shape[0] * ineffective_array.shape[1]):
            row_df['feat'+ str(i)] = np.reshape(ineffective_array, [1, ineffective_array.shape[0] * ineffective_array.shape[1]])[0][i]
        data_cluster_df = data_cluster_df.append(row_df)
        
        # 'neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb'
        # 'pos_af_igtb', 'neg_af_igtb', 'stai_igtb'
        for predict_label in predict_label_list:
            row_df[predict_label] = groundtruth_df.loc[groundtruth_df['ParticipantID'] == participant_id][predict_label].values[0]
        data_df = data_df.append(row_df)

    from sklearn.model_selection import KFold
    from sklearn.manifold import TSNE
    from sklearn.svm import SVR
    from sklearn.metrics import r2_score

    X = np.array(data_cluster_df)
    
    from sklearn import svm, datasets
    from sklearn.model_selection import GridSearchCV
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    
    for predict_label in predict_label_list:
        
        print('--------------------------------------------------')
        print('predict %s' % predict_label)
        y = np.array(data_df[predict_label])
        svr_rbf = SVR()
        clf = GridSearchCV(svr_rbf, tuned_parameters, cv=5, scoring='r2')
        clf.fit(X, y)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print(clf.best_score_)
        print()
        print("Grid scores on development set:")
        print()
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print('--------------------------------------------------')

    '''
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        svr_rbf = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.1)
        svr_rbf.fit(X_train, y_train)
        y_pred = svr_rbf.predict(X_test)
        print(r2_score(y_test, y_pred))
        
    
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(np.array(data_cluster_df))

    data_df['x-tsne'] = tsne_results[:, 0]
    data_df['y-tsne'] = tsne_results[:, 1]

    plt.scatter(data_df['x-tsne'], data_df['y-tsne'], c=data_df['neu_igtb'])
    plt.show()
    '''
    
    print('Successfully cluster all participant filter data')


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)
    