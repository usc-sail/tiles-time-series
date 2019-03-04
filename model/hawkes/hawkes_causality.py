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
                    
        # cond1 = len(offday_point_list) != int(data_config.fitbit_sensor_dict['ticc_cluster_days'])
        # cond2 = len(workday_point_list) != int(data_config.fitbit_sensor_dict['ticc_cluster_days'])
        #if cond1 or cond2:
        #    continue
        
        # Learn causality
        workday_learner = HawkesSumGaussians(3, max_iter=100)
        workday_learner.fit(workday_point_list)
        
        if os.path.exists(os.path.join(save_model_path, participant_id)) is False:
            os.mkdir(os.path.join(save_model_path, participant_id))
        
        for i, causality_array in enumerate(workday_learner.amplitudes):
    
            # for i in range(data_config.fitbit_sensor_dict['num_cluster']):
            #    for j in range(data_config.fitbit_sensor_dict['num_cluster']):
            ineffective_df = workday_learner.get_kernel_norms()
            
            causality_return_array = np.zeros([1, causality_array.shape[0] * causality_array.shape[1]])
            causality_return_col = []
            for row_index, causality_row_array in enumerate(causality_array):
                for col_index, element in enumerate(causality_row_array):
                    causality_return_array[0][row_index*causality_array.shape[0]+col_index] = element
                    causality_return_col.append(str(row_index) + '->' + str(col_index))
                    
            causality_df = pd.DataFrame(causality_return_array, index=['cluster_' + str(i)], columns=causality_return_col)
            causality_df.to_csv(os.path.join(save_model_path, participant_id, 'workday.csv.gz'), compression='gzip')

        offday_learner = HawkesSumGaussians(3, max_iter=100)
        offday_learner.fit(offday_point_list)

        for i, causality in enumerate(offday_learner.amplitudes):
            causality_return_array = np.zeros([1, causality_array.shape[0] * causality_array.shape[1]])
            causality_return_col = []
            for row_index, causality_row_array in enumerate(causality_array):
                for col_index, element in enumerate(causality_row_array):
                    causality_return_array[0][row_index * causality_array.shape[0] + col_index] = element
                    causality_return_col.append(str(row_index) + '->' + str(col_index))
    
            causality_df = pd.DataFrame(causality_return_array, index=['cluster_' + str(i)],
                                        columns=causality_return_col)
            causality_df.to_csv(os.path.join(save_model_path, participant_id, 'offday.csv.gz'), compression='gzip')

    print('Successfully cluster all participant filter data')


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment

    end_time = 1000
    n_nodes = 2
    n_realizations = 10
    n_gaussians = 10
    
    kernel_timefunction = HawkesKernelTimeFunc(t_values=np.array([0., .7, 2.5, 3., 4.]),
                                               y_values=np.array([.3, .03, .03, .2, 0.]))
    kernels = [[HawkesKernelExp(.2, 2.), HawkesKernelPowerLaw(.2, .5, 1.3)], [HawkesKernel0(), kernel_timefunction]]

    hawkes = SimuHawkes(baseline=[.5, .2], kernels=kernels, end_time=end_time, verbose=False, seed=1039)

    multi = SimuHawkesMulti(hawkes, n_simulations=n_realizations)

    multi.simulate()

    learner = HawkesSumGaussians(n_gaussians, max_iter=10)
    time = multi.timestamps
    learner.fit(time)

    from tick.plot import plot_hawkes_kernels
    # plot_hawkes_kernels(learner, hawkes=hawkes, support=4)

    main(tiles_data_path, config_path, experiment)
    
    '''
    from tick.dataset import fetch_hawkes_bund_data
    from tick.hawkes import HawkesConditionalLaw
    from tick.plot import plot_hawkes_kernel_norms

    timestamps_list = fetch_hawkes_bund_data()

    kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))
    hawkes_learner = HawkesConditionalLaw(
            claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
            quad_method="log", n_quad=10, min_support=1e-4, max_support=1, n_threads=4)

    hawkes_learner.fit(timestamps_list)
    
    
    np.random.seed(7168)

    n_nodes = 3
    baselines = 0.3 * np.ones(n_nodes)
    decays = 0.5 + np.random.rand(n_nodes, n_nodes)
    adjacency = np.array([
        [1, 1, -0.5],
        [0, 1, 0],
        [0, 0, 2],], dtype=float)

    adjacency /= 4

    end_time = 1e5
    integration_support = 5
    n_realizations = 5

    simu_hawkes = SimuHawkesExpKernels(baseline=baselines, adjacency=adjacency, decays=decays, end_time=end_time, verbose=False, seed=7168)
    simu_hawkes.threshold_negative_intensity(True)

    multi = SimuHawkesMulti(simu_hawkes, n_simulations=n_realizations, n_threads=-1)
    multi.simulate()

    nphc = HawkesCumulantMatching(integration_support, cs_ratio=.15, tol=1e-10, step=0.3)
    nphc.fit(multi.timestamps)
    
    '''

'''
if __name__ == '__main__':
    timestamps_list = fetch_hawkes_bund_data()
    
    kernel_discretization = np.hstack((0, np.logspace(-5, 0, 50)))
    hawkes_learner = HawkesConditionalLaw(claw_method="log", delta_lag=0.1, min_lag=5e-4, max_lag=500,
                                          quad_method="log", n_quad=10, min_support=1e-4, max_support=1, n_threads=4)
    
    hawkes_learner.fit(timestamps_list)
'''
'''
end_time = 1000
n_nodes = 2
n_realizations = 10
n_gaussians = 5

timestamps_list = []

if __name__ == '__main__':
    

    kernel_timefunction = HawkesKernelTimeFunc(t_values=np.array([0., .7, 2.5, 3., 4.]), y_values=np.array([.3, .03, .03, .2, 0.]))
    kernels = [[HawkesKernelExp(.2, 2.), HawkesKernelPowerLaw(.2, .5, 1.3)], [HawkesKernel0(), kernel_timefunction]]
    
    hawkes = SimuHawkes(baseline=[.5, .2], kernels=kernels, end_time=end_time, verbose=False, seed=1039)
    
    multi = SimuHawkesMulti(hawkes, n_simulations=n_realizations)
    
    multi.simulate()
    
    learner = HawkesSumGaussians(n_gaussians, max_iter=10)
    learner.fit(multi.timestamps)
    
    plot_hawkes_kernels(learner, hawkes=hawkes, support=4)
'''