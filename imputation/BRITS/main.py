"""
Created on Sat May 12 16:49:49 2018
@author: Tiantian Feng
"""

from preprocess.main.BRITS.BRITS_Wrapper import BRITSWrapper
import os
import torch.optim as optim
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from preprocess.main.BRITS.utils import to_var

offset = 60
length_seq = 30
training_percent = 0.1

raw_cols = ['BreathingDepth', 'BreathingRate', 'Cadence', 'HeartRate', 'Intensity', 'Steps', 'RR0', 'RR1', 'RR2', 'RR3']

process_hype = {'method': 'ma', 'offset': offset, 'overlap': 0,
                'length_seq': length_seq, 'training_percent':training_percent,
                'preprocess_cols': ['Cadence', 'HeartRate', 'Intensity', 'Steps',
                                    'BreathingDepth', 'BreathingRate']}

default_om_signal = {'MinPeakDistance': 100, 'MinPeakHeight': 0.04,
                     'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence',
                                  'HeartRate', 'Intensity', 'Steps',
                                  'RR0', 'RR1', 'RR2', 'RR3']}


def main(main_folder, mp=0.05):
    ###########################################################
    # 1. Read all om_signal folder
    ###########################################################
    omsignal_folder = os.path.join(main_folder, '2_raw_csv_data/omsignal/')
    omsignal_file_list = os.listdir(omsignal_folder)
    omsignal_file_list.sort()
    
    for omsignal_file in omsignal_file_list:
        if 'DS' in omsignal_file:
            omsignal_file_list.remove(omsignal_file)
    
    participant_id_array = [omsignal_file.split('_omsignal')[0] for omsignal_file in omsignal_file_list]
    participant_id_array.sort()
    
    ###########################################################
    # 2. Training
    ###########################################################
    rnn_hid_size_list, drop_out_list = [10, 20, 30], [0.5]
    
    for rnn_hid_size in rnn_hid_size_list:
        for drop_out in drop_out_list:
            model_dict = {'rnn_hid_size': rnn_hid_size, 'impute_weight': 1, 'batch_size': 128,
                          'drop_out': drop_out, 'include_lable': False, 'seq_len': length_seq}

            brits_wrapper = BRITSWrapper(signal_type='om_signal', length_seq=model_dict['seq_len'],
                                         mp=mp, postfix='_set', method='brits',
                                         model_dict=model_dict, batch_size=model_dict['batch_size'],
                                         main_folder=os.path.abspath(os.path.join(os.pardir, '../../imputation_set_np')),
                                         participant_id_array=participant_id_array,
                                         process_hyper=process_hype, signal_hyper=default_om_signal)

            ###########################################################
            # 2.1 Initiate data loader
            ###########################################################
            brits_wrapper.init_data_loader()
        
            # brits_wrapper.get_data_stat(stat='min')
            # brits_wrapper.get_data_stat(stat='max')
            # brits_wrapper.get_data_stat(stat='mean')
            # brits_wrapper.get_data_stat(stat='std')

            ###########################################################
            # 2.2 Load the data
            ###########################################################
            brits_wrapper.load_data_brits()

            ###########################################################
            # 2.3 Train the model
            ###########################################################
            brits_wrapper.train(optimizer=optim.Adam(brits_wrapper.model.parameters(), lr=1e-3))
            
            del brits_wrapper
    

if __name__ == '__main__':
    # Main Data folder
    main_folder = '../../../../data/keck_wave_all/'
    
    mp_array = [0.05, 0.1, 0.25, 0.5]
    # mp_array = [0.1, 0.25, 0.5]
    
    for mp in mp_array:
        main(main_folder, mp=mp)
