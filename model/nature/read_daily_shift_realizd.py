"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import numpy as np
import pandas as pd
import pickle
import preprocess
from scipy import stats
from datetime import timedelta
import collections


def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    chi_data_config = config.Config()
    chi_data_config.readChiConfigFile(config_path)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    load_data_path.load_chi_preprocess_path(chi_data_config, process_data_path)
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
    psqi_raw_igtb = load_data_basic.read_PSQI_Raw(tiles_data_path)
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    if os.path.exists(os.path.join(os.path.dirname(__file__), 'shift_trans.pkl')) is True:
        pkl_file = open(os.path.join(os.path.dirname(__file__), 'shift_trans.pkl'), 'rb')
        shift_dict = pickle.load(pkl_file)
        
        freq_add_list = []
        freq_decrease_list = []
        for shift in ['day', 'night']:
            shift_data = shift_dict[shift][4]
            shift_array = np.zeros([len(shift_data), len(shift_data[0].index), len(shift_data[0].columns)])
            
            for i in range(len(shift_data)):
                shift_array[i, :, :] = np.array(shift_data[i])
                if shift_data[i]['frequency'][1] + shift_data[i]['frequency'][2] > shift_data[i]['frequency'][0] + shift_data[i]['frequency'][3]:
                    freq_add_list.append(shift)
                else:
                    freq_decrease_list.append(shift)
            
            print()
        print()
    else:
        participant_dict = {}
        shift_data = {}
        shift_data['day'] = {}
        shift_data['night'] = {}
    
        for i in [4, 5, 6]:
            shift_data['day'][i] = []
            shift_data['night'][i] = []
    
        for idx, participant_id in enumerate(top_participant_id_list[:]):
            print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
            
            nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
            primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            job_str = 'nurse' if nurse == 1 and 'Dialysis' not in primary_unit else 'non_nurse'
            shift_str = 'day' if shift == 'Day shift' else 'night'
            
            uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
            realizd_df = load_sensor_data.read_preprocessed_realizd(data_config.realizd_sensor_dict['preprocess_path'], participant_id)
            
            if os.path.exists(os.path.join(data_config.phone_usage_path, participant_id + '.pkl')) is False:
                continue
                
            if realizd_df is None:
                continue
    
            pkl_file = open(os.path.join(data_config.phone_usage_path, participant_id + '.pkl'), 'rb')
            participant_id_shift_dict = pickle.load(pkl_file)
            
            if participant_id_shift_dict is None:
                continue
            
            '''
            data_cols = ['above_1min', 'less_than_1min', 'frequency', 'mean_time', 'total_time',
                         'shift_total_time', 'shift_mean_time', 'shift_frequency',
                         'off_total_time', 'off_mean_time', 'off_frequency']
            '''
            data_cols = ['above_1min', 'less_than_1min', 'frequency', 'mean_time', 'total_time']
    
            participant_dict[participant_id] = {}
            for i in [4, 5, 6]:
                valid_data = 0
                for key in list(participant_id_shift_dict.keys()):
                    data = participant_id_shift_dict[key]['data']
                    if len(data) == i:
                        valid_data += 1
                
                if valid_data < 3:
                    continue
               
                data_array = np.zeros([valid_data, i, len(data_cols)])
                valid_data = 0
                for key in list(participant_id_shift_dict.keys()):
                    data = participant_id_shift_dict[key]['data']
                    if len(data) == i:
                        data_df = data[data_cols]
                        data_array[valid_data, :, :] = np.array(data_df)
                        valid_data += 1
        
                data_array = np.nanmean(data_array, axis=0)
                data_df = pd.DataFrame(data_array, columns=data_cols)
                participant_dict[participant_id][i] = data_df
                
                if shift_str == 'day':
                    shift_data['day'][i].append(data_df)
                else:
                    shift_data['night'][i].append(data_df)
    
        output = open(os.path.join(os.path.dirname(__file__), 'shift_trans.pkl'), 'wb')
        pickle.dump(shift_data, output)
    
        output = open(os.path.join(os.path.dirname(__file__), 'participant_trans.pkl'), 'wb')
        pickle.dump(participant_dict, output)
    


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)