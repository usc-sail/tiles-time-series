#!/usr/bin/env python3

import os
import sys
import pandas as pd

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

from load_data_basic import *
from preprocess import Preprocess


def main(main_folder, return_feature=False):
    
    ###########################################################
    # 1. Read all om_signal folder
    ###########################################################
    realizd_folder = os.path.join(main_folder, '2_raw_csv_data/realizd/')
    realizd_file_list = os.listdir(realizd_folder)
    for realizd_file in realizd_file_list:
        if 'DS' in realizd_file:
            realizd_file_list.remove(realizd_file)
    realizd_file_list.sort()
    process_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                    'preprocess_cols': ['HeartRatePPG', 'StepCount']}

    ###########################################################
    # 2. Iterate all realizd files
    ###########################################################
    for realizd_file in realizd_file_list[:]:
        
        # Read data and participant id first
        realizd_file_abs_path = os.path.join(realizd_folder, realizd_file)
        realizd_df = pd.read_csv(realizd_file_abs_path, index_col=0)
        realizd_df = realizd_df.sort_index()
        
        participant_id = realizd_file.split('_realizd')[0]

        ###########################################################
        # 2.0 Iterate all realizd files
        ###########################################################
        if len(realizd_df) > 0:
            ###########################################################
            # 2.1 Init realizd preprocess
            ###########################################################
            realizd_preprocess = Preprocess(data_df=realizd_df, signal_type = 'realizd', process_hyper=process_hype,
                                            participant_id=participant_id, return_full_feature=return_feature,
                                            save_main_folder=os.path.abspath(os.path.join(os.pardir, '../preprocess_data')))
        
            ###########################################################
            # 2.2 Preprocess data (No slicing)
            ###########################################################
            realizd_preprocess.process_realizd(realizd_df, offset=60)

            ###########################################################
            # 2.3 Save preprocess data
            ###########################################################
            # realizd_preprocess.save_preprocess_slice_raw_data()
            if len(realizd_preprocess.preprocess_data_all_df) > 0:
                realizd_preprocess.preprocess_data_all_df.to_csv(os.path.join(realizd_preprocess.participant_folder, realizd_preprocess.participant_id + '.csv.gz'), compression='gzip')

            ###########################################################
            # 2.4 Delete current realizd_preprocess object
            ###########################################################
            del realizd_preprocess
      
        
if __name__ == "__main__":
    
    # Main Data folder
    main_folder = '../../../data/keck_wave_all/'

    main(main_folder, return_feature=False)

