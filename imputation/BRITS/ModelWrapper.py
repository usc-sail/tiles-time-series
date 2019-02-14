import numpy as np
import pandas as pd
import os
import copy

from preprocess.autoencoder.data_funcs import DataLoader
from preprocess.imputation.metrics import ND, NRMSE, RMSE


class ModelWrapper():
    
    """
    A class that inherits from the generic wrapper, enabling the testing and evaluation
    of different hyperparameters settings for use with a Multimodal Autoencoder (MMAE).
    Performs a grid search over every combination of settings.
    """
    _process_hype = {'method': 'ma', 'offset': 30, 'overlap': 0}

    _default_om_signal = {'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence',
                                       'HeartRate', 'Intensity', 'Steps'],
                          'MinPeakDistance': 100, 'MinPeakHeight': 0.04}

    def __init__(self, signal_type='om_signal', main_folder=None, participant_id_array=None,
                 process_hyper=_process_hype, signal_hyper=_default_om_signal, model_hyper=None,
                 cont=False, model_name='BRITS', temp_model_path='temp_saved_models', length_seq=10,
                 check_test=False, normalize_and_fill=False, normalization='min_max',
                 optimize_for='val_score', save_results_every_nth=1, cross_validation=True):
        
        self.temp_model_path = temp_model_path + '_' + str(length_seq)
    
        self.participant_array = participant_id_array
        
        self.model_name = model_name
    
        ###########################################################
        # 1. om_signal type
        ###########################################################
        self.signal_hypers = copy.deepcopy(self._default_om_signal)
        self.process_hyper = copy.deepcopy(self._process_hype)
    
        if signal_type == 'om_signal':
            ###########################################################
            # Update hyper paramters for signal and preprocess method
            ###########################################################
            self.signal_hypers.update(signal_hyper)
            self.process_hyper.update(process_hyper)
    
        self.signal_type = signal_type
        self.save_imputation_folder = main_folder

        process_col_array = list(self.process_hyper['preprocess_cols'])
        process_col_array.sort()
        self.process_col_str = '-'.join(process_col_array)

        self.process_basic_str = 'method_' + self.process_hyper['method'] + \
                                 '_offset_' + str(self.process_hyper['offset']) + \
                                 '_overlap_' + str(self.process_hyper['overlap'])

        signal_type_folder = os.path.join(self.save_imputation_folder, self.signal_type)
        self.process_basic_folder = os.path.join(signal_type_folder, self.process_basic_str)
        self.process_col_folder = os.path.join(self.process_basic_folder, self.process_col_str)

        ###########################################################
        # Hyperparameters to test
        ###########################################################
        if model_hyper is not None:
            self.layer_sizes = model_hyper['layer_sizes'] if 'layer_sizes' in model_hyper.keys() else None
            self.dropout_probs = model_hyper['dropout_probs'] if 'dropout_probs' in model_hyper.keys() else None
            self.weight_penalties = model_hyper['weight_penalties'] if 'weight_penalties' in model_hyper.keys() else None
            self.weight_initializers = model_hyper['weight_initializers'] if 'weight_initializers' in model_hyper.keys() else None
            self.activation_funcs = model_hyper['activation_funcs'] if 'activation_funcs' in model_hyper.keys() else None
            self.test_variational = model_hyper['test_variational'] if 'test_variational' in model_hyper.keys() else None
        
        self.length_seq = length_seq

        self.define_params()

    def load_train_data(self):
        ###########################################################
        # Load training data
        ###########################################################
        self.data_loader.load_train_data_brits(length_seq=self.length_seq)
        
    def init_data_loader(self):
        """Loads data from csv files using the DataLoader class."""
        self.data_loader = DataLoader(signal_type='om_signal', participant_id_array=self.participant_array,
                                      main_folder=os.path.abspath(os.path.join(os.pardir, '../imputation_test')),
                                      process_hyper=self.process_hyper, signal_hyper=self.signal_hypers)
    
    def load_test_data(self):
        ###########################################################
        # Load test data
        ###########################################################
        self.data_loader.load_test_dict_data()

    def define_params(self):
        """Defines the list of hyperparameters that will be tested."""
        self.params = {}
