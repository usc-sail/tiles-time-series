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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'plot')))

from om_signal.helper import *
from fitbit.helper import *
from realizd.helper import *
import pandas as pd

from GGS.ggs import *
import numpy as np

__all__ = ['Filter']


class Filter(object):
    
    def __init__(self, data_config=None, participant_id=None):
        """
        Initialization method
        """
        ###########################################################
        # Assert if these parameters are not parsed
        ###########################################################
        assert data_config is not None and participant_id is not None
        
        self.data_config = data_config
        self.participant_id = participant_id
        
        self.processed_data_dict_array = {}
    
    def filter_data(self, data_df, fitbit_summary_df=None, mgt_df=None, omsignal_df=None, owl_in_one_df=None):
        
        participant_id = self.participant_id
        
        ###########################################################
        # If folder not exist
        ###########################################################
        save_participant_folder = os.path.join(self.data_config.fitbit_sensor_dict['filter_path'], participant_id)
        if not os.path.exists(save_participant_folder):
            os.mkdir(save_participant_folder)
        
        ###########################################################
        # Read data
        ###########################################################
        data_df = data_df.sort_index()
        
        fitbit_summary_df = fitbit_summary_df
        sleep_df = pd.DataFrame()
        
        if self.data_config.filter_method == 'awake_period':
            ###########################################################
            # Parse sleep df
            ###########################################################
            if fitbit_summary_df is not None:
                if len(fitbit_summary_df) > 0:
                    for index, row in fitbit_summary_df.iterrows():
                        '''
                        return_df = self.add_sleep_data_frame(row.Sleep1BeginTimestamp, row.Sleep1EndTimestamp)
                        if return_df is not None:
                            if (pd.to_datetime(row.Sleep1EndTimestamp) - pd.to_datetime(row.Sleep1BeginTimestamp)).total_seconds() > 60 * 60 * 3:
                                sleep_df = sleep_df.append(return_df)
                                
                        return_df = self.add_sleep_data_frame(row.Sleep2BeginTimestamp, row.Sleep2EndTimestamp)
                        if return_df is not None:
                            if (pd.to_datetime(row.Sleep2EndTimestamp) - pd.to_datetime(row.Sleep2BeginTimestamp)).total_seconds() > 60 * 60 * 3:
                                sleep_df = sleep_df.append(return_df)
                        
                        return_df = self.add_sleep_data_frame(row.Sleep3BeginTimestamp, row.Sleep3EndTimestamp)
                        if return_df is not None:
                            if (pd.to_datetime(row.Sleep3EndTimestamp) - pd.to_datetime(row.Sleep3BeginTimestamp)).total_seconds() > 60 * 60 * 3:
                                sleep_df = sleep_df.append(return_df)
                        '''
                        return_df = self.add_sleep_data_frame(row.Sleep1BeginTimestamp, row.Sleep1EndTimestamp)
                        if return_df is not None:
                            if row.Sleep1BeginTimestamp in list(sleep_df.index):
                                if pd.to_datetime(sleep_df.loc[row.Sleep1BeginTimestamp, 'end']) < pd.to_datetime(row.Sleep1EndTimestamp):
                                    sleep_df.loc[row.Sleep1BeginTimestamp, 'end'] = row.Sleep1EndTimestamp
                            else:
                                sleep_df = sleep_df.append(return_df)
                        return_df = self.add_sleep_data_frame(row.Sleep2BeginTimestamp, row.Sleep2EndTimestamp)
                        if return_df is not None:
                            if row.Sleep2BeginTimestamp in list(sleep_df.index):
                                if pd.to_datetime(sleep_df.loc[row.Sleep2BeginTimestamp, 'end']) < pd.to_datetime(row.Sleep2EndTimestamp):
                                    sleep_df.loc[row.Sleep2BeginTimestamp, 'end'] = row.Sleep2EndTimestamp
                            else:
                                sleep_df = sleep_df.append(return_df)
                                
                        return_df = self.add_sleep_data_frame(row.Sleep3BeginTimestamp, row.Sleep3EndTimestamp)
                        if return_df is not None:
                            if row.Sleep3BeginTimestamp in list(sleep_df.index):
                                if pd.to_datetime(sleep_df.loc[row.Sleep3BeginTimestamp, 'end']) < pd.to_datetime(row.Sleep3EndTimestamp):
                                    sleep_df.loc[row.Sleep3BeginTimestamp, 'end'] = row.Sleep3EndTimestamp
                            else:
                                sleep_df = sleep_df.append(return_df)
                        
            sleep_df = sleep_df.sort_index()
            sleep_df = sleep_df.drop_duplicates(keep='first')
        
            if len(sleep_df) < 5:
                return False
            
            # Enumerate sleep time
            sleep_index_list = list(sleep_df.index)
            
            last_sleep_df = sleep_df.loc[sleep_index_list[0], :]
            last_start_str, last_end_str = last_sleep_df.start, last_sleep_df.end
            
            filter_df = pd.DataFrame()
            for i, sleep_index in enumerate(sleep_index_list):
                
                # Current sleep
                current_sleep_df = sleep_df.loc[sleep_index, :]
                curr_start_str, curr_end_str = current_sleep_df.start, current_sleep_df.end

                # Duration of last sleep longer than 4 hours
                cond1 = (pd.to_datetime(last_end_str) - pd.to_datetime(last_start_str)).total_seconds() > 60 * 60 * 4
                # Duration between sleep longer than 8 hours
                cond2 = (pd.to_datetime(curr_start_str) - pd.to_datetime(last_end_str)).total_seconds() > 60 * 60 * 10
                # Duration between sleep longer than 20 hours
                cond3 = (pd.to_datetime(curr_start_str) - pd.to_datetime(last_end_str)).total_seconds() > 60 * 60 * 16
                if cond3 or (cond1 and cond2):
                    row_filter_df = pd.DataFrame(index=[last_start_str])
                    row_filter_df['start'] = last_start_str
                    row_filter_df['end'] = curr_start_str
                    row_filter_df['sleep_start'] = last_start_str
                    row_filter_df['sleep_end'] = last_end_str
                    row_filter_df['duration'] = (pd.to_datetime(curr_start_str) - pd.to_datetime(last_start_str)).total_seconds() / 3600

                    owl_in_one_cond, omsignal_cond, mgt_cond = False, False, False
                    
                    if owl_in_one_df is not None:
                        owl_in_one_cond = len(owl_in_one_df[last_start_str:curr_start_str]) > 5
                    if omsignal_df is not None:
                        omsignal_cond = len(omsignal_df[last_start_str:curr_start_str]) > 60 * 30
                    
                    day_mgt = mgt_df[last_start_str:curr_start_str]
                    if len(day_mgt) > 0:
                        location_array = np.array(day_mgt.location_mgt)
                        for location in location_array:
                            if location == 2:
                                mgt_cond = True
                    
                    row_filter_df['work'] = 1 if owl_in_one_cond or omsignal_cond or mgt_cond else 0
                    
                    filter_df = filter_df.append(row_filter_df)
                    last_start_str, last_end_str = curr_start_str, curr_end_str
                
                filter_data_df = data_df[last_start_str:curr_start_str]
                filter_data_df.to_csv(os.path.join(save_participant_folder, last_start_str + '.csv.gz'), compression='gzip')
            filter_df.to_csv(os.path.join(save_participant_folder, 'filter_dict.csv.gz'), compression='gzip')
    
    def add_sleep_data_frame(self, sleep_begin_str, sleep_end_str):
        
        if pd.to_datetime(sleep_begin_str).year > 0:
            
            index = pd.to_datetime(sleep_begin_str).strftime(date_time_format)[:-3]
            start = pd.to_datetime(sleep_begin_str).strftime(date_time_format)[:-3]
            end = pd.to_datetime(sleep_end_str).strftime(date_time_format)[:-3]
            
            return_df = pd.DataFrame(index=[index], columns=['start', 'end'])
            return_df.loc[index, 'start'] = start
            return_df.loc[index, 'end'] = end
            return return_df
        
        else:
            return None