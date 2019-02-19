"""
Top level classes for the preprocess model.
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own pyspark path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'preprocess')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import segmentation
import load_sensor_data
import numpy as np
import pandas as pd

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

segmentation_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                     'segmentation': 'ggs', 'segmentation_lamb': 10e0, 'sub_segmentation_lamb': None,
                     'preprocess_cols': ['HeartRatePPG', 'StepCount'], 'imputation': 'iterative'}

preprocess_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                   'preprocess_cols': ['HeartRatePPG', 'StepCount'], 'imputation': 'iterative'}


def return_participant(main_folder):
    ###########################################################
    # 2. Read all fitbit file
    ###########################################################
    fitbit_folder = os.path.join(main_folder, '3_preprocessed_data/fitbit/')
    fitbit_file_list = os.listdir(fitbit_folder)
    fitbit_file_dict_list = {}
    
    for fitbit_file in fitbit_file_list:
        
        if '.DS' in fitbit_file:
            continue
        
        participant_id = fitbit_file.split('_')[0]
        if participant_id not in list(fitbit_file_dict_list.keys()):
            fitbit_file_dict_list[participant_id] = {}
    return list(fitbit_file_dict_list.keys())


def return_top_k_participant(participant_id_list, k=10, fitbit_config=None, ggs_config=None):
    ###########################################################
    # Get participant with most fitbit data
    ###########################################################
    fitbit_len_list = []
    for idx, participant_id in enumerate(participant_id_list):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(participant_id_list)))

        fitbit_df = load_sensor_data.read_processed_fitbit(fitbit_config, participant_id)
        
        if fitbit_df is not None:
            fitbit_len_list.append(len(fitbit_df))
        else:
            fitbit_len_list.append(0)
    
    top_participant_list = [participant_id_list[i] for i in np.argsort(fitbit_len_list)[::-1][:k]]
    fitbit_len_sort = np.sort(fitbit_len_list)[::-1][:k]
    
    fitbit_len_df = pd.DataFrame(fitbit_len_sort, index=top_participant_list)
    fitbit_len_df.to_csv(os.path.join(ggs_config.process_folder, 'participant_id.csv.gz'), compression='gzip')
    
    return fitbit_len_df


def main(main_folder):
    ###########################################################
    # 1. Create Config
    ###########################################################
    fitbit_config = config.Config(data_type='preprocess_data', sensor='fitbit',
                                  read_folder=os.path.abspath(os.path.join(os.pardir, 'data')),
                                  return_full_feature=False, process_hyper=preprocess_hype)
    
    ggs_config = config.Config(data_type='segmentation_no_comb', sensor='fitbit',
                               read_folder=os.path.abspath(os.path.join(os.pardir, '..')),
                               return_full_feature=False, process_hyper=segmentation_hype)
    
    fitbit_summary_config = config.Config(data_type='3_preprocessed_data', sensor='fitbit', read_folder=main_folder,
                                          return_full_feature=False, process_hyper=preprocess_hype)
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    if os.path.exists(os.path.join(ggs_config.process_folder, 'participant_id.csv.gz')) is True:
        top_participant_id_df = pd.read_csv(os.path.join(ggs_config.process_folder, 'participant_id.csv.gz'), index_col=0, compression='gzip')
    else:
        participant_id_list = return_participant(main_folder)
        participant_id_list.sort()
        top_participant_id_df = return_top_k_participant(participant_id_list, k=150, fitbit_config=fitbit_config, ggs_config=ggs_config)

    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    top_participant_id_list = top_participant_id_list[120:]
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        ###########################################################
        # 3. Create segmentation class
        ###########################################################
        ggs_segmentation = segmentation.Segmentation(read_config=fitbit_config, save_config=ggs_config,
                                                     participant_id=participant_id)
        
        ###########################################################
        # 4. Read segmentation data
        ###########################################################
        ggs_segmentation.read_preprocess_data_all()
        
        ###########################################################
        # 5. Segmentation
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_config, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']
        
        success = ggs_segmentation.segment_data_by_sleep(fitbit_summary_df=fitbit_summary_df, threshold=0)
        # ggs_segmentation.read_inactive_df(participant_id)
        # success = ggs_segmentation.segment_data_by_sleep(participant_id, fitbit_summary_df=fitbit_summary_df)

        del ggs_segmentation


if __name__ == '__main__':
    # Main Data folder
    main_folder = '../../../../data/keck_wave_all/'
    
    main(main_folder)
