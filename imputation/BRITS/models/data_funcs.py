""" This file provides functions for loading data from a saved
    .csv file.

    To use this code with classification, the file should contain
    at least one column with 'label' in the name. This column
    should be an outcome we are trying to classify; e.g. if we
    are trying to predict if someone is happy or not, we might
    have a column 'happy_label' that has values that are either
    0 or 1.

    We also assume the file contains a column named 'dataset'
    containing the dataset the example belongs to, which can be
    either 'Train', 'Val', or 'Test'.

    Other typical columns include 'user_id', 'timestamp', 'ppt_id',
    or columns with 'logistics_' as a prefix. A common logistics
    column is 'logistics_noisy', which describes whether the data
    has missing modalities or not.
"""

import pandas as pd
import numpy as np
import operator
import copy
import matplotlib.pyplot as plt
import os
import torch

NUM_CROSS_VAL_FOLDS = 5


class DataLoader:
    _process_hype = {'method': 'ma', 'offset': 30, 'overlap': 0}
    
    _default_om_signal = {'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence',
                                       'HeartRate', 'Intensity', 'Steps'],
                          'MinPeakDistance': 100, 'MinPeakHeight': 0.04}
    
    def __init__(self, signal_type=None, main_folder=None, participant_id_array=None,
                 mp=0.05, postfix='_set', process_hyper=_process_hype, signal_hyper=_default_om_signal,
                 supervised=True, suppress_output=False,
                 normalize_and_fill=True, normalization='min_max', fill_missing_with=0,
                 fill_gaps_with=None, separate_noisy_data=True):
        """
        Class that handles extracting numpy data matrices for train, validation,
        and test sets from a .csv file. Also normalizes and fills the data.
        """
        if not suppress_output:
            print("-----Loading data-----")

        ###########################################################
        # Assert if these parameters are not parsed
        ###########################################################
        assert signal_type is not None and main_folder is not None and participant_id_array is not None
        self.signal_type = signal_type
        
        # memorize arguments
        self.supervised = supervised
        self.normalize_and_fill = normalize_and_fill
        self.normalization = normalization
        self.suppress_output = suppress_output
        self.fill_missing_with = fill_missing_with
        self.fill_gaps_with = fill_gaps_with
        self.separate_noisy_data = separate_noisy_data

        ###########################################################
        # 1. om_signal type
        ###########################################################
        if signal_type == 'om_signal':
    
            ###########################################################
            # Update hyper paramters for signal and preprocess method
            ###########################################################
            self.signal_hypers = copy.deepcopy(self._default_om_signal)
            self.signal_hypers.update(signal_hyper)
    
            self.process_hyper = copy.deepcopy(self._process_hype)
            self.process_hyper.update(process_hyper)
            
        ###########################################################
        # 2. Data folder
        ###########################################################
        self.main_folder = main_folder
        self.process_basic_str = 'method_' + self.process_hyper['method'] + \
                                 '_offset_' + str(self.process_hyper['offset']) + \
                                 '_overlap_' + str(self.process_hyper['overlap'])

        process_col_array = list(self.process_hyper['preprocess_cols'])
        process_col_array.sort()
        self.process_col_str = '-'.join(process_col_array)
        
        self.process_col_array = process_col_array
        self.num_modalities = len(process_col_array)
        
        self.main_folder = os.path.join(self.main_folder, self.signal_type + postfix)
        self.main_folder = os.path.join(self.main_folder, self.process_col_str)
        self.main_folder = os.path.join(self.main_folder, self.process_basic_str)
        self.main_folder = os.path.join(self.main_folder, 'data')
        self.main_folder = os.path.join(self.main_folder, 'mp_' + str(mp))
        
        self.participant_id_array = participant_id_array
        
        # stats
        self.data_stat_df = {}

    def get_global_stat_from_data(self, funcName=None):
        
        if funcName == 'mean':
            func = np.nanmean
        elif funcName == 'std':
            func = np.nanstd
        elif funcName == 'min':
            func = np.nanmin
        elif funcName == 'max':
            func = np.nanmax
        
        if os.path.exists(os.path.join(self.main_folder, funcName + '.csv')) is False:
            data_full_df = pd.DataFrame()
        
            ###########################################################
            # Iterate participant array to get stat
            ###########################################################
            for participant_id in self.participant_id_array:
            
                ###########################################################
                # Read train data
                ###########################################################
                if os.path.exists(os.path.join(self.main_folder, participant_id)) is False:
                    continue
            
                file_array = os.listdir(os.path.join(self.main_folder, participant_id))
            
                for file_name in file_array:
                    data_df = pd.read_csv(os.path.join(self.main_folder, participant_id, file_name), index_col=0)
                    data_full_df = data_full_df.append(data_df)
                    
            ###########################################################
            # Assign stat for data
            ###########################################################
            data_stat_df = pd.DataFrame(np.array(func(np.array(data_full_df), axis=0)).reshape([1, data_full_df.shape[1]]), columns=data_full_df.columns)
        
            self.data_stat_df[funcName] = data_stat_df
            self.data_stat_df[funcName].to_csv(os.path.join(self.main_folder, funcName + '.csv'))
        else:
            self.data_stat_df[funcName] = pd.read_csv(os.path.join(self.main_folder, funcName + '.csv'), index_col=0)

    def load_test_dict_data(self, window=10):
    
        self.test_X = []
    
        ###########################################################
        # Iterate participant array to get min and max
        ###########################################################
        for participant_id in self.participant_id_array:
        
            if os.path.exists(os.path.join(self.test_folder, participant_id)) is False:
                continue
        
            file_array = os.listdir(os.path.join(self.test_folder, participant_id))
        
            for file_name in file_array:
                data_df = pd.read_csv(os.path.join(self.test_folder, participant_id, file_name), index_col=0)
                data_dict = {}
                data_dict['participant_id'] = participant_id
                data_dict['file_name'] = file_name
                data_dict['data'] = data_df
                self.test_X.append(data_dict)
        self.num_feats = len(self.process_col_array) * window

    def get_unsupervised_train_batch(self, batch_size):
        """Get a random batch of data from the X training matrix

        Args:
            batch_size: Integer number of examples in the batch.
        """
        idx = np.random.choice(len(self.train_X), size=batch_size)
        return self.train_X[idx]
    
    def get_unsupervised_val_batch(self, batch_size):
        """Randomly sample a set of X data from the validation set

        Args:
            batch_size: Integer number of examples in the batch.
        """
        idx = np.random.choice(len(self.val_X), size=batch_size)
        return self.val_X[idx]

    def load_train_dict_data(self, window=10):
        self.train_X = []
    
        ###########################################################
        # Iterate participant array to get min and max
        ###########################################################
        for participant_id in self.participant_id_array:
        
            if os.path.exists(os.path.join(self.train_folder, participant_id)) is False:
                continue
        
            file_array = os.listdir(os.path.join(self.train_folder, participant_id))
        
            for file_name in file_array:
                data_df = pd.read_csv(os.path.join(self.train_folder, participant_id, file_name), index_col=0)
                data_dict = {}
                data_dict['participant_id'] = participant_id
                data_dict['file_name'] = file_name
                data_dict['data'] = data_df
                self.train_X.append(data_dict)
        self.num_feats = len(self.process_col_array) * window

    def load_train_data_brits(self, length_seq=10, norm="min_max", mp=0):
        """
        Load the data for brits training
        """
        self.train_X = {'original': [], 'backward': [], 'forward': []}

        ###########################################################
        # Iterate participant array to get min and max
        ###########################################################
        for participant_id in self.participant_id_array:
            
            if os.path.exists(os.path.join(self.train_folder, participant_id)) is False:
                continue
    
            file_array = os.listdir(os.path.join(self.train_folder, participant_id))
    
            for file_name in file_array:
                data_df = pd.read_csv(os.path.join(self.train_folder, participant_id, file_name), index_col=0)
                data_df = data_df.fillna(data_df.mean())
                mask = (np.random.random(data_df.shape) > mp).astype(int)

                if norm == "min_max":
                    data_array = np.array(data_df) - np.array(self.data_stat_df['min'])
                    data_array = np.divide(data_array, np.array(self.data_stat_df['max'] - self.data_stat_df['min']))
                else:
                    data_array = np.array(data_df) - np.array(self.data_stat_df['mean'])
                    data_array = np.divide(data_array, np.array(self.data_stat_df['std']))
                for i in range(int((len(data_array) - length_seq) / length_seq)):
                    forwardDict, backwardDict = {}, {}
    
                    start_idx = int(i * length_seq)
                    end_idx = int((i + 1) * length_seq)
                    forwardDict['masks'] = mask[start_idx:end_idx, :]
                    forwardDict['values'] = data_array[start_idx:end_idx, :].copy()
                    forwardDict['values'][forwardDict['masks'] == 0] = 0
                    forwardDict['deltas'] = np.tile(np.arange(0, length_seq), [data_array.shape[1], 1]).T

                    backwardDict['masks'] = np.flip(mask[start_idx:end_idx, :], axis=0).copy()
                    backwardDict['values'] = np.flip(forwardDict['values'].copy(), axis=0).copy()
                    backwardDict['deltas'] = np.tile(np.arange(0, length_seq), [data_array.shape[1], 1]).T * (-1)

                    self.train_X['backward'].append(backwardDict)
                    self.train_X['forward'].append(forwardDict)
                    self.train_X['original'].append(data_array[start_idx:end_idx, :].copy())
                    
    def append_test_data_to_train_data_brits(self, length_seq=10, norm="min_max", mp=0):
        """
        Load the data for brits training
        """
        ###########################################################
        # Iterate participant array to get min and max
        ###########################################################
        for participant_id in self.participant_id_array:
        
            if os.path.exists(os.path.join(self.train_folder, participant_id)) is False:
                continue
        
            file_array = os.listdir(os.path.join(self.train_folder, participant_id))
        
            for file_name in file_array:
                data_df = pd.read_csv(os.path.join(self.train_folder, participant_id, file_name), index_col=0)
                data_df = data_df.fillna(data_df.mean())
                mask = (np.random.random(data_df.shape) > mp).astype(int)
            
                if norm == "min_max":
                    data_array = np.array(data_df) - np.array(self.data_stat_df['min'])
                    data_array = np.divide(data_array, np.array(self.data_stat_df['max'] - self.data_stat_df['min']))
                else:
                    data_array = np.array(data_df) - np.array(self.data_stat_df['mean'])
                    data_array = np.divide(data_array, np.array(self.data_stat_df['std']))
                for i in range(int((len(data_array) - length_seq) / length_seq)):
                    forwardDict, backwardDict = {}, {}
                
                    start_idx = int(i * length_seq)
                    end_idx = int((i + 1) * length_seq)
                    forwardDict['masks'] = mask[start_idx:end_idx, :]
                    forwardDict['values'] = data_array[start_idx:end_idx, :].copy()
                    forwardDict['values'][forwardDict['masks'] == 0] = 0
                    forwardDict['deltas'] = np.tile(np.arange(0, length_seq), [data_array.shape[1], 1]).T
                
                    backwardDict['masks'] = np.flip(mask[start_idx:end_idx, :], axis=0).copy()
                    backwardDict['values'] = np.flip(forwardDict['values'].copy(), axis=0).copy()
                    backwardDict['deltas'] = np.tile(np.arange(0, length_seq), [data_array.shape[1], 1]).T * (-1)
                
                    self.train_X['backward'].append(backwardDict)
                    self.train_X['forward'].append(forwardDict)
                    self.train_X['original'].append(data_array[start_idx:end_idx, :].copy())

    def load_data_brits(self, length_seq=10, norm="min_max", mp=0.1):
        """
        Load the data for brits training
        """
        self.train_X = {'original': [], 'backward': [], 'forward': []}
        self.data_array = {}
    
        ###########################################################
        # Iterate participant array to get min and max
        ###########################################################
        for participant_id in self.participant_id_array:
        
            if os.path.exists(os.path.join(self.main_folder, participant_id)) is False:
                continue
        
            file_array = os.listdir(os.path.join(self.main_folder, participant_id))

            self.data_array[participant_id] = {}
        
            for file_name in file_array:
                
                if self.check_error_file(participant_id, file_name) is True:
                    continue
                
                data_df = pd.read_csv(os.path.join(self.main_folder, participant_id, file_name), index_col=0)
                self.data_array[participant_id][file_name] = data_df
                
                data_missing_df = data_df.copy()
                data_missing_df = data_missing_df.fillna(-1)
                
                mask = np.array(np.array(data_missing_df) != -1).astype(int)
                
                data_array = np.array(data_df)
                self.cols = data_df.columns
                
                for i in range(int((len(data_array)) / length_seq)):
                    forwardDict, backwardDict = {}, {}
                    
                    start_idx = i * length_seq
                    end_idx = (i + 1) * length_seq
                    forwardDict['masks'] = mask[start_idx:end_idx, :]
                    forwardDict['values'] = data_array[start_idx:end_idx, :].copy()
                    forwardDict['values'][forwardDict['masks'] == 0] = 0
                    forwardDict['deltas'] = np.tile(np.arange(0, length_seq), [data_array.shape[1], 1]).T
                    forwardDict['participant_id'] = participant_id
                    forwardDict['file_name'] = file_name
                    forwardDict['index'] = start_idx
                
                    backwardDict['masks'] = np.flip(mask[start_idx:end_idx, :], axis=0).copy()
                    backwardDict['values'] = np.flip(forwardDict['values'].copy(), axis=0).copy()
                    backwardDict['deltas'] = np.tile(np.arange(0, length_seq), [data_array.shape[1], 1]).T * (-1)
                
                    self.train_X['backward'].append(backwardDict)
                    self.train_X['forward'].append(forwardDict)
                    self.train_X['original'].append(data_array[start_idx:end_idx, :].copy())
                
                if len(data_array) % length_seq != 0:
                    forwardDict, backwardDict = {}, {}
    
                    start_idx = len(data_array) - length_seq
                    end_idx = len(data_array)
                    forwardDict['masks'] = mask[start_idx:end_idx, :]
                    forwardDict['values'] = data_array[start_idx:end_idx, :].copy()
                    forwardDict['values'][forwardDict['masks'] == 0] = 0
                    forwardDict['deltas'] = np.tile(np.arange(0, length_seq), [data_array.shape[1], 1]).T
                    forwardDict['participant_id'] = participant_id
                    forwardDict['file_name'] = file_name
                    forwardDict['index'] = start_idx
    
                    backwardDict['masks'] = np.flip(mask[start_idx:end_idx, :], axis=0).copy()
                    backwardDict['values'] = np.flip(forwardDict['values'].copy(), axis=0).copy()
                    backwardDict['deltas'] = np.tile(np.arange(0, length_seq), [data_array.shape[1], 1]).T * (-1)
    
                    self.train_X['backward'].append(backwardDict)
                    self.train_X['forward'].append(forwardDict)
                    self.train_X['original'].append(data_array[start_idx:end_idx, :].copy())

    def loadBatch(self, index=True, batch_size=256, data_type='train'):
        
        if data_type == 'train':
            data = self.train_X
        else:
            data = self.test_X
            
        start_idx = index * batch_size
        end_idx = (index + 1) * batch_size

        backDataValues, backDataMasks, backDataDeltas = [], [], []
        forwardDataValues, forwardDataMasks, forwardDataDeltas = [], [], []
        forwardParticipant, forwardFile, forwardIndex = [], [], []
        originalValues = []
        
        originalArray = data['original'][start_idx]

        for j in range(start_idx + 1, end_idx):
            originalArray = np.append(originalArray, data['original'][j], axis=0)
            
        mean_array = np.nanmean(originalArray, axis=0)
        std_array = np.nanstd(originalArray, axis=0)
            
        for j in range(start_idx, end_idx):
            
            back_array = data['backward'][j]['values'] - np.array(mean_array)
            back_array = np.divide(back_array, np.array(std_array))
        
            backDataValues.append(back_array)
            backDataMasks.append(data['backward'][j]['masks'])
            backDataDeltas.append(data['backward'][j]['deltas'])

            forward_array = data['forward'][j]['values'] - np.array(mean_array)
            forward_array = np.divide(forward_array, np.array(std_array))

            forwardDataValues.append(forward_array)
            forwardDataMasks.append(data['forward'][j]['masks'])
            forwardDataDeltas.append(data['forward'][j]['deltas'])

            forwardParticipant.append(data['forward'][j]['participant_id'])
            forwardFile.append(data['forward'][j]['file_name'])
            forwardIndex.append(data['forward'][j]['index'])

            originalValues.append(data['original'][j])

        backDataValuesTensor = torch.FloatTensor(backDataValues)
        backDataMasksTensor = torch.FloatTensor(backDataMasks)
        backDataDeltasTensor = torch.FloatTensor(backDataDeltas)
        
        backDict = {'values': backDataValuesTensor, 'masks': backDataMasksTensor, 'deltas': backDataDeltasTensor}

        forwardDataValuesTensor = torch.FloatTensor(forwardDataValues)
        forwardDataMasksTensor = torch.FloatTensor(forwardDataMasks)
        forwardDataDeltasTensor = torch.FloatTensor(forwardDataDeltas)

        originalValuesTensor = torch.FloatTensor(originalValues)

        forwardDict = {'values': forwardDataValuesTensor, 'masks': forwardDataMasksTensor,
                       'deltas': forwardDataDeltasTensor}

        ret_dict = {'forward': forwardDict, 'backward': backDict, 'original': originalValuesTensor,
                    'mean': mean_array.copy(), 'std': std_array.copy(), 'id': forwardParticipant,
                    'file_name': forwardFile, 'index': forwardIndex}
        
        return ret_dict
    
    def create_folder(self, folder):
        if os.path.exists(folder) is False:
            os.mkdir(folder)

    def check_error_file(self, participant_id, data_file):
        if participant_id == 'a1623554-43d6-4038-b28a-bd74a96b9c97' \
                and data_file == '2018-05-05T08:54:00.000.csv' and self.process_hyper['offset'] == 30:
            return True
    
        if participant_id == 'aa8d1b7a-14c6-4490-a57b-d3893ac28b03' \
                and data_file == '2018-04-24T06:56:00.000.csv' and self.process_hyper['offset'] == 30:
            return True
    
        if participant_id == 'cc25830a-254a-487f-acec-c0afb3962679' \
                and data_file == '2018-06-16T06:24:30.000.csv' and self.process_hyper['offset'] == 30:
            return True
    
        if participant_id == 'e5be76f8-6461-481b-834e-b44f8f880273' \
                and data_file == '2018-06-01T06:18:30.000.csv' and self.process_hyper['offset'] == 30:
            return True
    
        if participant_id == 'f610ffea-f6cb-4182-bbe7-19ff8fbe66ee' \
                and data_file == '2018-06-16T06:03:30.000.csv' and self.process_hyper['offset'] == 30:
            return True
    
        return False


