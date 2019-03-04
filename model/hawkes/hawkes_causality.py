"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import pandas as pd
import numpy as np

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

    if os.path.exists(os.path.join(os.curdir, 'result')) is False:
        os.mkdir(os.path.join(os.curdir, 'result'))

    save_model_path = os.path.join(os.curdir, 'result', data_config.fitbit_sensor_dict['clustering_path'].split('/')[-1])
    if os.path.exists(save_model_path) is False:
        os.mkdir(save_model_path)
    
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

        # Iterate clustering data
        for clustering_data_dict in clustering_data_list:
            start = clustering_data_dict['start']

            for filter_data_dict in filter_data_list:
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

                # If we have point data
                if len(day_point_list) == 0:
                    continue
                    
                if filter_data_dict['work'] == 1:
                    # from collections import Counter
                    # data = Counter(elem[0] for elem in change_list)
                    workday_point_list.append(day_point_list)
                else:
                    offday_point_list.append(day_point_list)
        
        if os.path.exists(os.path.join(save_model_path, participant_id)) is False:
            os.mkdir(os.path.join(save_model_path, participant_id))

        # Learn causality
        workday_learner = HawkesSumGaussians(10, max_iter=20)
        workday_learner.fit(workday_point_list)
        ineffective_df = pd.DataFrame(workday_learner.get_kernel_norms())
        ineffective_df.to_csv(os.path.join(save_model_path, participant_id, 'workday.csv.gz'), compression='gzip')

        offday_learner = HawkesSumGaussians(10, max_iter=20)
        offday_learner.fit(offday_point_list)
        ineffective_df = pd.DataFrame(offday_learner.get_kernel_norms())
        ineffective_df.to_csv(os.path.join(save_model_path, participant_id, 'offday.csv.gz'), compression='gzip')
    print('Successfully cluster all participant filter data')


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment

    main(tiles_data_path, config_path, experiment)