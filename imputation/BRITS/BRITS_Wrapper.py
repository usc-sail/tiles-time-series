# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:48:54 2018
@author: Tiantian
"""

import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import copy
import numpy as np
import pandas as pd
import time
import os
from preprocess.main.BRITS.models.data_funcs import DataLoader
from preprocess.main.BRITS.ModelWrapper import ModelWrapper
from preprocess.main.BRITS.models.brits import BRITSModel
from preprocess.main.BRITS.utils import to_var

_process_hype = {'method': 'ma', 'offset': 30, 'overlap': 0}

_default_om_signal = {'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence',
                                   'HeartRate', 'Intensity', 'Steps'],
                      'MinPeakDistance': 100, 'MinPeakHeight': 0.04}


class BRITSWrapper(ModelWrapper):
    
    def __init__(self, signal_type='om_signal', main_folder=None, participant_id_array=None,
                 mp=0.05, postfix='_set', method='brits',
                 process_hyper=_process_hype, signal_hyper=_default_om_signal, model_dict=None,
                 cont=False, temp_model_path='temp_saved_models', length_seq=10, batch_size=256,
                 norm='z_norm',  lossFunc=torch.nn.MSELoss, learning_rate=0.0001, optimizer=torch.optim.RMSprop):
        
        self.postfix = postfix
        self.temp_model_path = temp_model_path + '_' + str(length_seq)
        self.participant_array = participant_id_array
        
        self.mp = mp
        self.method = method
        
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
        self.save_imputation_folder = os.path.join(main_folder, signal_type + '_imputation_all')
        
        ###########################################################
        # 2. Data folder, data loader
        ###########################################################
        self.main_folder = main_folder

        if 'length_seq' in list(self.process_hyper.keys()):
            self.process_basic_str = 'method_' + self.process_hyper['method'] + \
                                     '_offset_' + str(self.process_hyper['offset']) + \
                                     '_overlap_' + str(self.process_hyper['overlap']) + \
                                     '_length_seq_' + str(self.process_hyper['length_seq']) + \
                                     '_training_percent_' + str(self.process_hyper['training_percent'])
            self.length_seq = self.process_hyper['length_seq']
            self.training_percent = self.process_hyper['training_percent']
        else:
            self.process_basic_str = 'method_' + self.process_hyper['method'] + \
                                     '_offset_' + str(self.process_hyper['offset']) + \
                                     '_overlap_' + str(self.process_hyper['overlap'])
        
        process_col_array = list(self.process_hyper['preprocess_cols'])
        process_col_array.sort()
        self.process_col_array = process_col_array
        self.process_col_str = '-'.join(process_col_array)
        
        self.num_features = len(process_col_array)
        self.length_seq = length_seq
        
        self.init_data_loader()
        
        ###########################################################
        # 2. Define the models
        ###########################################################
        self.model_dict = model_dict
        self.model = BRITSModel(rnn_hid_size=self.model_dict['rnn_hid_size'],
                                impute_weight=self.model_dict['impute_weight'],
                                num_of_units=len(process_col_array), drop_out=self.model_dict['drop_out'],
                                include_lable=self.model_dict['include_lable'], seq_len=self.model_dict['seq_len'])
        
        self.model_str = 'rnn_hid_size_' + str(self.model_dict['rnn_hid_size']) + \
                         '_drop_out_' + str(self.model_dict['drop_out']) + \
                         '_seq_len_' + str(self.model_dict['seq_len'])
        
        self.loss_func = lossFunc
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate, alpha=0.99)
        
        self.batch_size = batch_size
        self.norm = norm

        # Initializes the parent class.
        ModelWrapper.__init__(self, signal_type=self.signal_type, main_folder=self.main_folder,
                              participant_id_array=self.participant_array,
                              process_hyper=self.process_hyper, signal_hyper=self.signal_hypers, model_hyper=None,
                              cont=False, model_name='BRITS', temp_model_path='temp_saved_models', length_seq=self.length_seq,
                              check_test=False, normalize_and_fill=False, normalization='min_max',
                              optimize_for='val_score', save_results_every_nth=1, cross_validation=True)

        ###########################################################
        # 3. Create Folder
        ###########################################################
        self.signal_folder = os.path.join(self.main_folder, signal_type + '_imputation_all')
        self.process_basic_folder = os.path.join(self.signal_folder, self.process_basic_str)
        self.process_col_folder = os.path.join(self.process_basic_folder, self.process_col_str)
        self.process_col_folder = os.path.join(self.process_col_folder, 'data')
        self.process_col_folder = os.path.join(self.process_col_folder, 'mp_' + str(self.mp))
        self.model_folder = os.path.join(self.process_col_folder, self.method)
        self.create_folder(self.model_folder)
        self.model_folder = os.path.join(self.model_folder, self.model_str)
        self.create_folder(self.model_folder)

    def load_train_data_brits(self, mp=0.1):
        ###########################################################
        # Load training data
        ###########################################################
        self.data_loader.load_train_data_brits(length_seq=self.length_seq, norm=self.norm, mp=mp)
        
    def load_data_brits(self):
        ###########################################################
        # Load training data
        ###########################################################
        self.data_loader.load_data_brits(length_seq=self.length_seq, norm=self.norm, mp=self.mp)
    
    def load_batch_brits(self, index=0):
        ###########################################################
        # Load training data
        ###########################################################
        return self.data_loader.loadBatch(index=index, batch_size=self.batch_size)
    
    def append_test_data_to_train_data(self, mp=0.1):
        ###########################################################
        # Append test data to training data
        ###########################################################
        self.data_loader.append_test_data_to_train_data_brits(length_seq=self.length_seq, norm=self.norm, mp=mp)
    
    def init_data_loader(self):
        """Loads data from csv files using the DataLoader class."""
        self.data_loader = DataLoader(signal_type='om_signal', participant_id_array=self.participant_array,
                                      main_folder=self.main_folder, mp=self.mp, postfix=self.postfix,
                                      process_hyper=self.process_hyper, signal_hyper=self.signal_hypers)
    
    def get_data_stat(self, stat='std'):
        ###########################################################
        # Get mean train data
        ###########################################################
        self.data_loader.get_global_stat_from_data(stat)
        
        return self.data_loader.data_stat_df
    
    def train(self, optimizer=None):
        ###########################################################
        # Train the model
        ###########################################################
        batch_length = int(len(self.data_loader.train_X['original']) / self.batch_size)
        loss_df = pd.DataFrame()

        for epoch in range(100):
    
            self.model.train()
            run_loss = 0.0
            for idx in range(batch_length):
                
                data = self.load_batch_brits(index=idx)
                data = to_var(data)
                ret = self.model.run_on_batch(data, optimizer, epoch)
        
                if ret is not None:
                    run_loss += ret['loss'].item()
                    batch_loss = ret['loss'].item()
                    print('Epoch: %d, Batch %d, loss: %.2f' % (epoch, idx + 1, batch_loss))
    
            print("Run loss: %.2f" % run_loss)
            run_loss_df = pd.DataFrame(np.array(run_loss).reshape([1, 1]), index=[epoch], columns=['loss'])
            loss_df = loss_df.append(run_loss_df)

        ###########################################################
        # Evaluate the model
        ###########################################################
        self.evaluate()
        loss_df.to_csv(os.path.join(self.model_folder, 'loss.csv'))

    def evaluate(self):
        ###########################################################
        # Evaluate the performance
        ###########################################################
        self.model.eval()

        batch_length = int(len(self.data_loader.train_X['original']) / self.batch_size)

        save_impute = []
        
        for idx in range(batch_length):
            data = self.load_batch_brits(index=idx)

            ###########################################################
            # Get mean and std
            ###########################################################
            self.model.eval()
            mean_array, std_array = data['mean'], data['std']

            ###########################################################
            # Get participant name, file name, index
            ###########################################################
            participant_array, file_name_array, index_array = data['id'], data['file_name'], data['index']

            ###########################################################
            # Run and return
            ###########################################################
            data = to_var(data)
            ret = self.model.run_on_batch(data, None)
    
            if ret is not None:
                # save the imputation results which is used to test the improvement of traditional methods with imputed values
                save_impute.append(ret['imputations'].data.cpu().numpy())
                eval_masks = ret['masks'].data.cpu().numpy()
                imputation = ret['imputations'].data.cpu().numpy()

                for i in range(eval_masks.shape[0]):
                    mask_index = np.where(eval_masks[i] == 0)
                    
                    participant = participant_array[i]
                    file_name = file_name_array[i]
                    index = index_array[i]

                    row_index_array = self.data_loader.data_array[participant][file_name].index
                    col_index_array = self.data_loader.data_array[participant][file_name].columns

                    if len(mask_index) > 0:
                        
                        tmp_imputation_values = np.multiply(imputation[i], std_array) + mean_array
                        
                        for j in range(len(mask_index[0])):
                            
                            row_idx = row_index_array[index + mask_index[0][j]]
                            col_idx = col_index_array[mask_index[1][j]]

                            row_num = mask_index[0][j]
                            col_num = mask_index[1][j]
                            
                            if tmp_imputation_values[row_num, col_num] < 0:
                                self.data_loader.data_array[participant][file_name].loc[row_idx, col_idx] = 0
                            else:
                                self.data_loader.data_array[participant][file_name].loc[row_idx, col_idx] = tmp_imputation_values[row_num, col_num]
        
        for participant in self.data_loader.data_array.keys():
            
            save_path = os.path.join(self.model_folder, participant)
            self.create_folder(save_path)
            
            for file_name in self.data_loader.data_array[participant].keys():
                self.data_loader.data_array[participant][file_name].to_csv(os.path.join(save_path, file_name))
    
    def create_folder(self, folder):
        if os.path.exists(folder) is False:
            os.mkdir(folder)