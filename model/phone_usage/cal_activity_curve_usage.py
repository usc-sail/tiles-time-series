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

from statsmodels.tsa.stattools import grangercausalitytests

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


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
    agg, sliding = 10, 2
    
    load_data_path.load_chi_preprocess_path(chi_data_config, process_data_path)
    load_data_path.load_chi_activity_curve_path(chi_data_config, process_data_path, agg=agg, sliding=sliding)
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')

    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list[:]):
    
        nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
        job_str = 'nurse' if nurse == 1 else 'non_nurse'
        shift_str = 'day' if shift == 'Day shift' else 'night'

        icu_str = 'non_icu'
        if 'ICU' in primary_unit:
            icu_str = 'icu'

        for unit in icu_list:
            if unit in primary_unit:
                icu_str = 'icu'
        
        if job_str == 'non_nurse':
            continue
        
        # if icu_str == 'non_icu':
        #    continue
        
        # print('job shift: %s, job type: %s' % (shift_str, job_str))
        # print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        if os.path.exists(os.path.join(chi_data_config.activity_curve_path, participant_id + '_physio.pkl')):
            physio_dict = np.load(os.path.join(chi_data_config.activity_curve_path, participant_id + '_physio.pkl'))
            realizd_dict = np.load(os.path.join(chi_data_config.activity_curve_path, participant_id + '_realizd.pkl'))
            
            regular_physio_dict, shuffle_physio_dict = physio_dict['regular'], physio_dict['shuffle']
            regular_realizd_dict, shuffle_realizd_dict = realizd_dict['regular'], realizd_dict['shuffle']
            
            dist_array = np.zeros([len(regular_physio_dict.keys()), 2])
            for dates_idx, dates in enumerate(list(regular_physio_dict.keys())):
                physio_dist = np.sum(regular_physio_dict[dates]['physio_dist'])
                realizd_dist = np.sum(regular_realizd_dict[dates]['realizd_dist'])

                physio_shuffle_dist = shuffle_physio_dict[dates]
                realizd_shuffle_dist = shuffle_realizd_dict[dates]
                
                physio_change, realizd_change = 0, 0
                
                for shuffle_idx in list(physio_shuffle_dist.keys()):
                    if np.sum(physio_shuffle_dist[shuffle_idx]['physio_dist']) > physio_dist:
                        physio_change += 1
                    
                    if np.sum(realizd_shuffle_dist[shuffle_idx]['realizd_dist']) > realizd_dist:
                        realizd_change += 1

                dist_array[dates_idx, 0] = physio_change / 100
                dist_array[dates_idx, 1] = realizd_change / 100

            dist_array = dist_array[:-2, :]
            # print(dist_array)
            # print(np.corrcoef(dist_array[:, 0], dist_array[:, 1]))
            print(np.corrcoef(dist_array[:, 0], dist_array[:, 1])[0, 1])

            dist_array_inv = dist_array
            dist_array_inv[:, 1] = dist_array[:, 0]
            dist_array_inv[:, 0] = dist_array[:, 1]
            # grangercausalitytests(dist_array, maxlag=3)
            

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)