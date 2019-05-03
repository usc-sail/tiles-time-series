"""
Top level classes for the preprocess model.
"""
from __future__ import print_function

import os
import pandas as pd
import numpy as np
from datetime import timedelta

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

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
    
    def filter_data(self, data_df=None, audio_df=None, raw_audio_df=None, fitbit_summary_df=None, mgt_df=None, omsignal_df=None, owl_in_one_df=None, realizd_df=None):
        """
        Filter data based on the config file being initiated
        """
        participant_id = self.participant_id
        
        sleep_df = pd.DataFrame()
        
        if self.data_config.filter_method == 'realizd':
            
            ###########################################################
            # If there is no realizd data or not enough data, return
            ###########################################################
            if realizd_df is None:
                return

            if len(realizd_df.loc[realizd_df['SecondsOnPhone'] > 0]) < 500:
                return

            ###########################################################
            # Save the realizd data and fitbit data for now
            ###########################################################
            realizd_df.to_csv(os.path.join(self.data_config.realizd_sensor_dict['filter_path'], participant_id + '.csv.gz'), compression='gzip')
            data_df.to_csv(os.path.join(self.data_config.fitbit_sensor_dict['filter_path'], participant_id + '.csv.gz'), compression='gzip')

        elif self.data_config.filter_method == 'audio':
    
            ###########################################################
            # If there is no realizd data or not enough data, return
            ###########################################################
            if audio_df is None:
                return
            if owl_in_one_df is None:
                return
    
            ###########################################################
            # Save the realizd data and fitbit data for now
            ###########################################################
            owl_in_one_df.to_csv(os.path.join(self.data_config.owl_in_one_sensor_dict['filter_path'], participant_id + '.csv.gz'), compression='gzip')
            data_df.to_csv(os.path.join(self.data_config.fitbit_sensor_dict['filter_path'], participant_id + '.csv.gz'), compression='gzip')
            audio_df.to_csv(os.path.join(self.data_config.audio_sensor_dict['filter_path'], participant_id + '.csv.gz'), compression='gzip')

        elif self.data_config.filter_method == 'raw_audio':

            ###########################################################
            # If there is no realizd data or not enough data, return
            ###########################################################
            if raw_audio_df is None:
                return
            if owl_in_one_df is None:
                return

            ###########################################################
            # Find valid owl-in-one data
            ###########################################################
            time_diff = list((pd.to_datetime(list(owl_in_one_df.index[1:])) - pd.to_datetime(list(owl_in_one_df.index[:-1]))).total_seconds())
            
            change_point_start_list = [0]
            change_point_end_list = list(np.where(np.array(time_diff) > 3600 * 2)[0])
            
            if len(change_point_end_list) < 4:
                return
            
            [change_point_start_list.append(change_point_end+1) for change_point_end in change_point_end_list]
            change_point_end_list.append(len(owl_in_one_df.index)-1)
            
            time_start_end_list = []
            for i, change_point_end in enumerate(change_point_end_list):
                if 240 < change_point_end - change_point_start_list[i] < 900:
                    time_start_end_list.append([list(owl_in_one_df.index)[change_point_start_list[i]], list(owl_in_one_df.index)[change_point_end]])
            
            if len(time_start_end_list) < 5:
                return
            
            ###########################################################
            # Filter raw audio data
            ###########################################################
            for time_start_end in time_start_end_list:
                start_time = (pd.to_datetime(time_start_end[0]) - timedelta(hours=1)).strftime(date_time_format)[:-3]
                end_time = (pd.to_datetime(time_start_end[1]) + timedelta(hours=1)).strftime(date_time_format)[:-3]
                tmp_raw_audio_df = raw_audio_df[start_time:end_time]
                tmp_owl_in_one_df = owl_in_one_df[start_time:end_time]
                
                if len(tmp_raw_audio_df) > 100 * 30:
                    # tmp_raw_audio_df = tmp_raw_audio_df.loc[tmp_raw_audio_df['F0final_sma'] > 0]

                    sum = 0
                    if 'other' in list(tmp_owl_in_one_df.columns):
                        sum += np.nansum(np.array(tmp_owl_in_one_df.other_floor))
                    if 'unknown' in list(tmp_owl_in_one_df.columns):
                        sum += np.nansum(np.array(tmp_owl_in_one_df.unknown))
                        
                    if sum / len(tmp_owl_in_one_df) < 0.5:
                        if os.path.exists(os.path.join(self.data_config.audio_sensor_dict['filter_path'], participant_id)) is False:
                            os.mkdir(os.path.join(self.data_config.audio_sensor_dict['filter_path'], participant_id))
                        if os.path.exists(os.path.join(self.data_config.owl_in_one_sensor_dict['filter_path'], participant_id)) is False:
                            os.mkdir(os.path.join(self.data_config.owl_in_one_sensor_dict['filter_path'], participant_id))
                        
                        tmp_raw_audio_df.to_csv(os.path.join(self.data_config.audio_sensor_dict['filter_path'],
                                                             participant_id, start_time + '.csv.gz'), compression='gzip')
                        if len(tmp_owl_in_one_df) > 0:
                            tmp_owl_in_one_df.to_csv(os.path.join(self.data_config.owl_in_one_sensor_dict['filter_path'],
                                                                  participant_id, list(tmp_owl_in_one_df.index)[0] + '.csv.gz'), compression='gzip')

        elif self.data_config.filter_method == 'awake_period':
    
            ###########################################################
            # If folder not exist
            ###########################################################
            save_participant_folder = os.path.join(self.data_config.fitbit_sensor_dict['filter_path'], participant_id)
            if not os.path.exists(save_participant_folder):
                os.mkdir(save_participant_folder)
    
            ###########################################################
            # Parse sleep df
            ###########################################################
            if fitbit_summary_df is not None:
                if len(fitbit_summary_df) > 0:
                    for index, row in fitbit_summary_df.iterrows():
                        
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
            # If we have sleep data to split
            if len(sleep_df) < 5:
                print('%s has not enough sleep data' % participant_id)
                return False
            sleep_df = sleep_df.sort_index()
            sleep_df = sleep_df.drop_duplicates(keep='first')
            
            # Enumerate sleep time
            sleep_index_list = list(sleep_df.index)
            
            last_sleep_df = sleep_df.loc[sleep_index_list[0], :]
            last_start_str, last_end_str = last_sleep_df.start, last_sleep_df.end
            
            filter_df = pd.DataFrame()
            filter_data_all_df = pd.DataFrame()

            ###########################################################
            # Sort the data
            ###########################################################
            data_df = data_df.sort_index()

            for i, sleep_index in enumerate(sleep_index_list):
                
                # Current sleep
                current_sleep_df = sleep_df.loc[sleep_index, :]
                curr_start_str, curr_end_str = current_sleep_df.start, current_sleep_df.end

                # Duration of last sleep longer than 4 hours
                cond1 = (pd.to_datetime(last_end_str) - pd.to_datetime(last_start_str)).total_seconds() > 60 * 60 * 4
                # Duration between sleep longer than 10 hours, it would be a main sleep
                cond2 = (pd.to_datetime(curr_start_str) - pd.to_datetime(last_end_str)).total_seconds() > 60 * 60 * 10
                # Duration between sleep longer than 16 hours, even it is short, it is like a main sleep
                cond3 = (pd.to_datetime(curr_start_str) - pd.to_datetime(last_end_str)).total_seconds() > 60 * 60 * 16
                
                # If the condition holds, we have a day of data
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
                            if location == 2: mgt_cond = True
                    
                    row_filter_df['work'] = 1 if owl_in_one_cond or omsignal_cond or mgt_cond else 0

                    # The real data for the day
                    filter_data_df = data_df[last_start_str:curr_start_str]
                    for col in list(data_df.columns):
                        row_filter_df[col + '_mean'], row_filter_df[col + '_std'] = np.nan, np.nan
                    
                    # If we have enough amount of data
                    if len(np.where(filter_data_df.StepCount >= 0)[0]) > 3 * 60:
                        
                        # Calculate stats on valid data
                        mean = np.nanmean(filter_data_df[filter_data_df >= 0].dropna(), axis=0)
                        std = np.std(filter_data_df[filter_data_df >= 0].dropna(), axis=0)
                        
                        row_filter_df['valid_length'] = len(np.where(filter_data_df.StepCount >= 0)[0])

                        # Save mean and std for each stream
                        for i, col in enumerate(list(data_df.columns)):
                            row_filter_df[col + '_mean'] = mean[i]
                            row_filter_df[col + '_std'] = std[i]
                        
                        # Save the data df
                        filter_data_df.to_csv(os.path.join(save_participant_folder, last_start_str + '.csv.gz'), compression='gzip')

                        # Save filter dict
                        filter_df = filter_df.append(row_filter_df)

                        filter_data_all_df = filter_data_all_df.append(filter_data_df)
                    
                    # Update curr and last recording start
                    last_start_str, last_end_str = curr_start_str, curr_end_str

            filter_data_all_df.to_csv(os.path.join(save_participant_folder, participant_id + '.csv.gz'), compression='gzip')
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