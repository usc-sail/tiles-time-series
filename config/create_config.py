#!/usr/bin/env python3
import os
from config import Config
import argparse

'''
fitbit_param = {'data_type': 'fitbit', 'imputation': 'iterative', 'feature': 'original', 'offset': 60, 'overlap': 0,
                'preprocess_cols': ['HeartRatePPG', 'StepCount'],
                'cluster_method': 'kmeans', 'num_cluster': 5,
                'segmentation_method': 'gaussian', 'segmentation_lamb': 10e0}

'''
fitbit_param = {'data_type': 'fitbit', 'imputation': 'iterative', 'feature': 'original', 'offset': 60, 'overlap': 0,
                'preprocess_cols': ['HeartRatePPG', 'StepCount'],
                'cluster_method': 'ticc', 'num_cluster': 6, 'ticc_window': 10,
                'ticc_switch_penalty': 10, 'ticc_sparsity': 1e-1, 'ticc_cluster_days': 7,
                'segmentation_method': None, 'segmentation_lamb': 10e0}


om_param = {'data_type': 'om_signal', 'imputation': None, 'feature': 'original', 'offset': 60, 'overlap': 0,
            'preprocess_cols': ['BreathingDepth', 'BreathingRate', 'Cadence', 'HeartRate', 'Intensity', 'Steps']}

realizd_param = {'data_type': 'realizd', 'imputation': None, 'feature': 'original', 'offset': 60, 'overlap': 0}

owl_in_one_param = {'data_type': 'owl_in_one', 'imputation': None, 'feature': 'original', 'offset': 60, 'overlap': 0}

audio_param = {'data_type': 'audio', 'feature': 'original', 'offset': 60}

feature_engineering_param = {'features': ['duration', 'count']}

# segmentation_param = {'method': 'gaussian', 'segmentation_lamb': 10e0}
segmentation_param = None

cluster_param = {'method': 'kmeans', 'num_cluster': 5}

global_param = {'enable_plot': False, 'filter_method': 'awake_period'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=False, help="Name of the experiment, which will become the name of the output configuration file")
    args = parser.parse_args()

    args.experiment = 'baseline' if args.experiment is None else args.experiment
    config = Config()
    config.saveConfig(om_param, fitbit_param, owl_in_one_param, realizd_param, audio_param, segmentation_param, cluster_param, feature_engineering_param, global_param, args.experiment)

    # Create config files
    config.createConfigFile(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file'), args.experiment)

    # Read config files
    config.readConfigFile(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file'), args.experiment)
