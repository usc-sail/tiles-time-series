"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np

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

    # Read path
    save_data_path = os.path.join(process_data_path, data_config.experiement, 'filter_data_' + data_config.filter_method)

    # Read fitbit norm data and dict
    fitbit_norm_data_df = pd.read_csv(os.path.join(save_data_path, 'norm_data.csv.gz'), index_col=3)
    fitbit_dict_df = pd.read_csv(os.path.join(save_data_path, 'dict.csv.gz'), index_col=0)

    # fitbit_dict_df = fitbit_dict_df.loc[list(fitbit_dict_df.index)[, :]
    # fitbit_norm_data_df = fitbit_norm_data_df.iloc[:fitbit_dict_df.iloc[-1].end, :]
    fitbit_norm_data_df = fitbit_norm_data_df.loc[:, list(fitbit_norm_data_df.columns)[1:]]
    

    ###########################################################
    # 3. Create segmentation class
    ###########################################################
    ticc = TICC(data_config=data_config, maxIters=300, threshold=2e-5, num_proc=1,
                lambda_parameter=data_config.fitbit_sensor_dict['ticc_sparsity'],
                beta=data_config.fitbit_sensor_dict['ticc_switch_penalty'],
                window_size=data_config.fitbit_sensor_dict['ticc_window'],
                number_of_clusters=data_config.fitbit_sensor_dict['num_cluster'])


    ticc.fit_multiple_sequences(data_df=fitbit_norm_data_df, dict_df=fitbit_dict_df)
    
    print('Successfully cluster all participant filter data')


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)