from config import Config
import os

'''
fitbit_param = {'data_type': 'fitbit', 'imputation': 'iterative', 'feature': 'original', 'offset': 60, 'overlap': 0,
                'preprocess_cols': ['HeartRatePPG', 'StepCount'],
                'cluster_method': 'kmeans', 'num_cluster': 5,
                'segmentation_method': 'gaussian', 'segmentation_lamb': 10e0}
'''

fitbit_param = {'data_type': 'fitbit', 'imputation': 'iterative', 'feature': 'original', 'offset': 60, 'overlap': 0,
                'preprocess_cols': ['HeartRatePPG', 'StepCount'],
                'cluster_method': 'ticc', 'num_cluster': 5, 'ticc_window': 10,
                'ticc_switch_penalty': 1e-2, 'ticc_sparsity': 100,
                'segmentation_method': None, 'segmentation_lamb': 10e0}

om_param = {'data_type': 'om_signal', 'imputation': None, 'feature': 'original', 'offset': 60, 'overlap': 0,
            'preprocess_cols': ['BreathingDepth', 'BreathingRate', 'Cadence', 'HeartRate', 'Intensity', 'Steps']}

realizd_param = {'data_type': 'realizd', 'imputation': None, 'feature': 'original', 'offset': 60, 'overlap': 0}

owl_in_one_param = {'data_type': 'owl_in_one', 'imputation': None, 'feature': 'original', 'offset': 60, 'overlap': 0}

# segmentation_param = {'method': 'gaussian', 'segmentation_lamb': 10e0}
segmentation_param = None

cluster_param = {'method': 'kmeans', 'num_cluster': 5}

global_param = {'enable_plot': False}

if __name__ == '__main__':
    
    experiement = 'ticc'
    config = Config()
    
    # Save parameters
    config.saveConfig(om_param, fitbit_param, owl_in_one_param, realizd_param, segmentation_param, cluster_param, global_param, experiement)

    # Create config files
    config.createConfigFile(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file'), experiement)

    # Read config files
    config.readConfigFile(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file'), experiement)
    
    