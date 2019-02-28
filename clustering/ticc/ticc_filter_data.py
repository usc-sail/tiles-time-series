"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import pandas as pd

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
    
    ticc_data_df = {'norm_data': pd.DataFrame(), 'dict': pd.DataFrame(), 'data': pd.DataFrame()}
    ticc_data_index = 0

    save_data_path = os.path.join(process_data_path, data_config.experiement, 'filter_data_' + data_config.filter_method)
    
    '''
    for idx, participant_id in enumerate(top_participant_id_list):

        print('merge data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        # Read per participant data
        participant_data_dict = load_sensor_data.load_filter_data(data_config.fitbit_sensor_dict['filter_path'],
                                                                  participant_id, filter_logic=None, valid_data_rate=0.8,
                                                                  threshold_dict={'min': 20, 'max': 28})

        if participant_data_dict is not None:
            
            if len(participant_data_dict['filter_data_list']) < 20:
                print('Not enough data for %s' % participant_id)
                continue
            
            for data_dict in participant_data_dict['filter_data_list']:
                norm_data_df = data_dict['data'].copy()
                norm_data_df.loc[:, 'HeartRatePPG'] = (data_dict['data'].loc[:, 'HeartRatePPG'] - participant_data_dict['HeartRatePPG_mean']) / participant_data_dict['HeartRatePPG_std']
                norm_data_df.loc[:, 'StepCount'] = (data_dict['data'].loc[:, 'StepCount'] - participant_data_dict['StepCount_mean']) / participant_data_dict['StepCount_std']

                dict_df = pd.DataFrame(index=[ticc_data_index])
                dict_df['start'] = ticc_data_index
                ticc_data_index = ticc_data_index + len(norm_data_df)
                dict_df['end'] = ticc_data_index
                dict_df['participant_id'] = participant_id

                ticc_data_df['norm_data'] = ticc_data_df['norm_data'].append(norm_data_df)
                ticc_data_df['data'] = ticc_data_df['data'].append(data_dict['data'].copy())
                ticc_data_df['dict'] = ticc_data_df['dict'].append(dict_df)

        ticc_data_df['norm_data'].to_csv(os.path.join(save_data_path, 'norm_data.csv.gz'), compression='gzip')
        ticc_data_df['data'].to_csv(os.path.join(save_data_path, 'data.csv.gz'), compression='gzip')
        ticc_data_df['dict'].to_csv(os.path.join(save_data_path, 'dict.csv.gz'), compression='gzip')
    '''
    # Read fitbit norm data and dict
    fitbit_norm_data_df = pd.read_csv(os.path.join(save_data_path, 'norm_data.csv.gz'))
    fitbit_dict_df = pd.read_csv(os.path.join(save_data_path, 'dict.csv.gz'))
    
    ###########################################################
    # 3. Create segmentation class
    ###########################################################
    
    ticc = TICC(data_config=data_config, maxIters=300, threshold=2e-5, num_proc=2,
                lambda_parameter=data_config.fitbit_sensor_dict['ticc_sparsity'],
                beta=data_config.fitbit_sensor_dict['ticc_switch_penalty'],
                window_size=data_config.fitbit_sensor_dict['ticc_window'],
                number_of_clusters=data_config.fitbit_sensor_dict['num_cluster'])
    
    
    ticc.fit_multiple_sequences(data_df=fitbit_norm_data_df, dict_df=fitbit_dict_df)

    print('Successfully load all participant filter data')
    

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)