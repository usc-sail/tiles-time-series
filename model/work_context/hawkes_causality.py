# !/usr/bin/env python3

import numpy as np

from tick.plot import plot_hawkes_kernels
from tick.hawkes import (SimuHawkes, SimuHawkesMulti, HawkesKernelExp, HawkesKernelTimeFunc, HawkesKernelPowerLaw, HawkesKernel0, HawkesSumGaussians)
from tick.dataset import fetch_hawkes_bund_data
from tick.hawkes import HawkesConditionalLaw

import os
import sys
import argparse
import pandas as pd

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'plot')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'ticc')))

import load_data_path, load_sensor_data, load_data_basic
import config


def main(tiles_data_path, config_path, experiment):
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load preprocess folder
    load_data_path.load_preprocess_path(data_config, process_data_path, data_name='preprocess_data')
    
    # Load segmentation folder
    load_data_path.load_segmentation_path(data_config, process_data_path, data_name='segmentation')
    
    # Load clustering folder
    load_data_path.load_clustering_path(data_config, process_data_path, data_name='clustering')
    
    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')
    
    ###########################################################
    # Read ground truth data
    ###########################################################
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    if os.path.exists(os.path.join(process_data_path, experiment, 'participant_id.csv.gz')) is True:
        top_participant_id_df = pd.read_csv(os.path.join(process_data_path, experiment, 'participant_id.csv.gz'), index_col=0, compression='gzip')
    else:
        participant_id_list = load_data_basic.return_participant(tiles_data_path)
        participant_id_list.sort()
        top_participant_id_df = load_data_basic.return_top_k_participant(participant_id_list, k=150, data_config=data_config)
        top_participant_id_df.to_csv(os.path.join(process_data_path, experiment, 'participant_id.csv.gz'), compression='gzip')
    
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    top_participant_id_list = top_participant_id_list[:]
    
    ###########################################################
    # 3. Learn ticc
    ###########################################################
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['clustering_path'], participant_id + '.csv.gz')) is False:
            continue
        
        clustering_df = load_sensor_data.load_clustering_data(data_config.fitbit_sensor_dict['clustering_path'], participant_id)
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.config is None else args.config
    
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