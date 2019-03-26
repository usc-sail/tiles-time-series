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
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    # Get participant id list, k=10, read 10 participants with most data in fitbit
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, k=150, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    ticc_data_dict = {'norm_data_df': pd.DataFrame(), 'dict_df': pd.DataFrame(), 'data_df': pd.DataFrame()}
    save_data_path = os.path.join(process_data_path, data_config.experiement, 'filter_data_' + data_config.filter_method)
    
    ticc_data_index = 0
    for idx, participant_id in enumerate(top_participant_id_list):

        print('merge data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        # Read per participant data
        participant_data_dict = load_sensor_data.load_filter_data(data_config.fitbit_sensor_dict['filter_path'], participant_id, filter_logic=None, valid_data_rate=0.9, threshold_dict={'min': 20, 'max': 28})

        if participant_data_dict is not None:
    
            work_days, off_days = 0, 0
            for data_dict in participant_data_dict['filter_data_list']:
                if data_dict['work'] == 1:
                    work_days = work_days + 1
                else:
                    off_days = off_days + 1
    
            if work_days < data_config.fitbit_sensor_dict['ticc_cluster_days'] or off_days < data_config.fitbit_sensor_dict['ticc_cluster_days']:
                print('Not enough data for %s' % participant_id)
                continue
            
            # Randomly select data
            work_day_index, off_day_index = [], []

            for i, data_dict in enumerate(participant_data_dict['filter_data_list']):
                if data_dict['work'] == 1:
                    work_day_index.append(i)
                else:
                    off_day_index.append(i)
            
            index_array = np.random.choice(range(len(work_day_index)), data_config.fitbit_sensor_dict['ticc_cluster_days'], replace=False)
            final_work_day_index = [work_day_index[i] for i in index_array]

            index_array = np.random.choice(range(len(off_day_index)), data_config.fitbit_sensor_dict['ticc_cluster_days'], replace=False)
            final_off_day_index = [off_day_index[i] for i in index_array]

            final_data_index = final_off_day_index + final_work_day_index
            
            for i in final_data_index:
                data_dict = participant_data_dict['filter_data_list'][i]
                
                # Data
                norm_data_df = data_dict['data'].copy()
                data_df = data_dict['data'].copy()
                
                norm_heart_rate = (data_dict['data'].loc[:, 'HeartRatePPG'] - participant_data_dict['HeartRatePPG_mean']) / participant_data_dict['HeartRatePPG_std']
                norm_step_count = (data_dict['data'].loc[:, 'StepCount'] - participant_data_dict['StepCount_mean']) / participant_data_dict['StepCount_std']

                norm_data_df.loc[:, 'HeartRatePPG'] = np.array(norm_heart_rate)
                norm_data_df.loc[:, 'StepCount'] = np.array(norm_step_count)
                norm_data_df.loc[:, 'time'] = list(norm_data_df.index)

                data_df.loc[:, 'time'] = list(norm_data_df.index)

                ticc_data_dict['norm_data_df'] = ticc_data_dict['norm_data_df'].append(norm_data_df)
                ticc_data_dict['data_df'] = ticc_data_dict['data_df'].append(data_dict['data'].copy())
                
                # Dict
                dict_df = pd.DataFrame(index=[ticc_data_index])
                dict_df['start'] = ticc_data_index
                dict_df['end'] = ticc_data_index + len(data_dict['data'])
                dict_df['participant_id'] = participant_id
                dict_df['work'] = data_dict['work']
                dict_df['duration'] = len(data_dict['data'])

                ticc_data_dict['dict_df'] = ticc_data_dict['dict_df'].append(dict_df)
                
                ticc_data_index = ticc_data_index + len(data_dict['data'])

    if os.path.exists(os.path.join(save_data_path, data_config.fitbit_sensor_dict['filter_path'].split('/')[-1])) is False:
        os.mkdir(os.path.join(save_data_path, data_config.fitbit_sensor_dict['filter_path'].split('/')[-1]))
    
    ticc_data_dict['norm_data_df'].to_csv(os.path.join(save_data_path, data_config.fitbit_sensor_dict['filter_path'].split('/')[-1], 'norm_data_ticc_cluster_days_' + str(data_config.fitbit_sensor_dict['ticc_cluster_days']) + '.csv.gz'), compression='gzip')
    ticc_data_dict['dict_df'].to_csv(os.path.join(save_data_path, data_config.fitbit_sensor_dict['filter_path'].split('/')[-1], 'dict_norm_data_cluster_days_' + str(data_config.fitbit_sensor_dict['ticc_cluster_days']) + '.csv.gz'), compression='gzip')
    # ticc_data_dict['data_df'].to_csv(os.path.join(save_data_path, 'data_ticc_cluster_days_' + str(data_config.fitbit_sensor_dict['ticc_cluster_days']) + '.csv.gz'), compression='gzip')


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)