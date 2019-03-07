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


def predict(feature_array, label_array, label_name, feature_name):
    
    X = feature_array
    y = label_array

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    
    param_grid = {"max_depth": [3, 4, 5],
                  "min_samples_split": [2, 3, 5],
                  "bootstrap": [True, False],
                  'max_features': ['auto', 'sqrt'],
                  'n_estimators': [5, 10, 20, 30]}
    
    print('--------------------------------------------------')
    print('predict %s' % label_name)
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
    feature_importance = np.array(clf.best_estimator_.feature_importances_).reshape([1, -1])
    result_df = pd.DataFrame(feature_importance, index=[label_name], columns=feature_name)
    result_df['best_r2'] = clf.best_score_
    
    return result_df


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

    ###########################################################
    # Get igtb cols
    ###########################################################
    igtb_cols = [col for col in groundtruth_df.columns if 'igtb' in col]
    igtb_cols.append('Flexbility')
    igtb_cols.append('Inflexbility')
    
    ''' 'housing', 'nurseyears', 'hours', 'overtime' 'extrajob', 'extrahours' '''
    demographic_cols = ['gender', 'age', 'bornUS', 'language', 'education',
                        'supervise', 'supervise_size', 'employer_duration', 'income',
                        'commute_type', 'commute_time', 'nurseyears', 'housing', 'overtime']

    demographic_cols = ['gender', 'age', 'language', 'income',
                        'nurseyears', 'housing', 'overtime']

    ml_df = groundtruth_df[demographic_cols+igtb_cols]
    ml_df[~(ml_df >= 0)] = np.nan
    ml_df = ml_df.dropna()
    
    # Handling shift case with str
    for index, row_series in ml_df.iterrows():
        con1 = groundtruth_df.loc[index, 'currentposition'] == 1
        con2 = groundtruth_df.loc[index, 'currentposition'] == 2
        
        # Extra process for feature
        ml_df.loc[index, 'nurseyears'] = ml_df.loc[index, 'nurseyears'] if ml_df.loc[index, 'nurseyears'] != ' ' else np.nan
        ml_df.loc[index, 'shift'] = 1 if groundtruth_df.loc[index, 'Shift'] == 'Day shift' else 2
        ml_df.loc[index, 'housing'] = float(ml_df.loc[index, 'housing']) if ml_df.loc[index, 'housing'] != ' ' else np.nan
        ml_df.loc[index, 'currentposition'] = 1 if con1 or con2 else 2
        ml_df.loc[index, 'overtime'] = float(ml_df.loc[index, 'overtime']) if ml_df.loc[index, 'overtime'] != ' ' else np.nan

        # Extra process for label
        ml_df.loc[index, 'Flexbility'] = float(ml_df.loc[index, 'Flexbility']) if ml_df.loc[index, 'Flexbility'] != ' ' else np.nan
        ml_df.loc[index, 'Inflexbility'] = float(ml_df.loc[index, 'Inflexbility']) if ml_df.loc[index, 'Inflexbility'] != ' ' else np.nan

    ml_df = ml_df.dropna()

    for index, row_series in ml_df.iterrows():
        ml_df.loc[index, 'age'] = 1 if ml_df.loc[index, 'age'] <= 40 else 2
        ml_df.loc[index, 'nurseyears'] = 1 if int(ml_df.loc[index, 'nurseyears']) <= 15 else 2
        ml_df.loc[index, 'overtime'] = int(int(ml_df.loc[index, 'nurseyears']) / 10)
        
    # Read feature and label
    feature_df, label_df = ml_df[demographic_cols], ml_df[igtb_cols]
    
    # Initialize result df
    final_result_df = pd.DataFrame()

    # Iterate the igtb cols
    for igtb_col in igtb_cols:
        # Training
        label_array = np.array(label_df[igtb_col])
        result_df = predict(np.array(feature_df), label_array, igtb_col, feature_df.columns)
        
        # Append the score
        final_result_df = final_result_df.append(result_df)
        # Save the score
        final_result_df.to_csv(os.path.join(os.curdir, 'result', 'result_condense.csv'))
    

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)