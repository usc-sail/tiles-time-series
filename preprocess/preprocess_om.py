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

sys.path.append(os.path.join(os.path.curdir, '../', 'utils'))
from load_data_basic import *
from preprocess import Preprocess

raw_cols = ['BreathingDepth', 'BreathingRate', 'Cadence', 'HeartRate', 'Intensity', 'Steps', 'RR0', 'RR1', 'RR2', 'RR3']


def main(main_folder, return_feature=False):
    
    ###########################################################
    # 1. Read all om_signal folder
    ###########################################################
    omsignal_folder = os.path.join(main_folder, '3_preprocessed_data/omsignal/')
    omsignal_file_list = os.listdir(omsignal_folder)
    
    for omsignal_file in omsignal_file_list:
        if 'DS' in omsignal_file or '.zip' in omsignal_file:
            omsignal_file_list.remove(omsignal_file)

    omsignal_file_list.sort()

    ###########################################################
    # 2. Iterate all omsignal files
    ###########################################################
    for omsignal_file in omsignal_file_list[150:]:
        
        # Read data and participant id first
        omsignal_file_abs_path = os.path.join(omsignal_folder, omsignal_file)
        omsignal_df = pd.read_csv(omsignal_file_abs_path, index_col=0)

        omsignal_df = omsignal_df.fillna(0)
        omsignal_df = omsignal_df.drop_duplicates(keep='first')
        omsignal_df = omsignal_df.sort_index()
        
        participant_id = omsignal_file.split('_omsignal')[0]

        ###########################################################
        # 2.0 Iterate all omsignal files
        ###########################################################
        process_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                        'preprocess_cols': ['Cadence', 'HeartRate', 'Intensity', 'Steps',
                                            'BreathingDepth', 'BreathingRate', 'RR']}
                
        default_om_signal = {'MinPeakDistance': 100, 'MinPeakHeight': 0.04,
                             'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence',
                                          'HeartRate', 'Intensity', 'Steps',
                                          'RR0', 'RR1', 'RR2', 'RR3']}
        
        if len(omsignal_df) > 0:
            
            ###########################################################
            # 2.1 Init om_signal preprocess
            ###########################################################
            omsignal_preprocess = Preprocess(data_df=omsignal_df, signal_type = 'om_signal', participant_id=participant_id,
                                             save_main_folder=os.path.abspath(os.path.join(os.pardir, '../preprocess_data')),
                                             process_hyper=process_hype, signal_hyper=default_om_signal,
                                             return_full_feature=return_feature)
        
            ###########################################################
            # 2.2 Slice the raw data array
            ###########################################################
            omsignal_preprocess.slice_raw_data(method='block')

            ###########################################################
            # 2.3 Preprocess data
            ###########################################################
            if return_feature == True:
                omsignal_preprocess.preprocess_slice_raw_data_full_feature(check_saved=True)
            else:
                omsignal_preprocess.preprocess_slice_raw_data(check_saved=True)

            ###########################################################
            # 2.4 Save preprocess data
            ###########################################################
            omsignal_preprocess.save_preprocess_slice_raw_data()

            ###########################################################
            # 2.5 Delete current omsignal_preprocess object
            ###########################################################
            del omsignal_preprocess
      
        
if __name__ == "__main__":
    
    # Main Data folder
    main_folder = '../../../data/keck_wave_all/'

    main(main_folder, return_feature=True)

