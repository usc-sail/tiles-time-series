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

    # Get participant id list, k=10, read 10 participants with most data in fitbit
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, k=150, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    # Read path
    save_data_path = os.path.join(process_data_path, data_config.experiement, 'filter_data_' + data_config.filter_method)
    
    # Read fitbit norm data and dict
    # fitbit_norm_data_df = pd.read_csv(os.path.join(save_data_path, 'norm_data_ticc_cluster_days_' + str(data_config.fitbit_sensor_dict['ticc_cluster_days']) + '.csv.gz'), index_col=3)
    # fitbit_dict_df = pd.read_csv(os.path.join(save_data_path, 'dict_norm_data_cluster_days_' + str(data_config.fitbit_sensor_dict['ticc_cluster_days']) + '.csv.gz'), index_col=0)
    # fitbit_norm_data_df = fitbit_norm_data_df.loc[:, list(fitbit_norm_data_df.columns)[1:]]

    for idx, participant_id in enumerate(top_participant_id_list):
    
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        clustering_data_list = load_sensor_data.load_filter_clustering(data_config.fitbit_sensor_dict['clustering_path'], participant_id)

        # Read per participant data
        participant_data_dict = load_sensor_data.load_filter_data(data_config.fitbit_sensor_dict['filter_path'], participant_id, filter_logic=None, valid_data_rate=0.9, threshold_dict={'min': 20, 'max': 28})
        
        if clustering_data_list is None or participant_data_dict is None:
            continue
        
        filter_data_list = participant_data_dict['filter_data_list']
        
        print()

    print('Successfully cluster all participant filter data')


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)

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