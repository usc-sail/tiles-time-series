#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import argparse
import datetime

# add util into the file path, so we can import helper functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from files import *

# current path
current_path = os.getcwd()

# main_data_folder path
main_data_directory = os.path.join(current_path, '../../data')

# output_data_folder path
output_data_folder_path = os.path.join(current_path, '../output')

# csv's
id_csv = 'mitreids.csv'

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

# save_type
save_type = 'combined'

# sleep_related_header from fitbit summary
sleep_header = ['Timestamp',
                'Sleep1BeginTimestamp', 'Sleep1Efficiency', 'Sleep1EndTimestamp', 'Sleep1MinutesAwake',
                'Sleep1MinutesStageDeep', 'Sleep1MinutesStageLight', 'Sleep1MinutesStageRem', 'Sleep1MinutesStageWake',
                # 'Sleep1Main_Sleep', 'Sleep1Time_In_Bed',
                'Sleep2BeginTimestamp', 'Sleep2Efficiency', 'Sleep2EndTimestamp', 'Sleep2MinutesAwake',
                'Sleep2MinutesStageDeep', 'Sleep2MinutesStageLight', 'Sleep2MinutesStageRem', 'Sleep2MinutesStageWake',
                # 'Sleep2Main_Sleep', 'Sleep2Time_In_Bed',
                'Sleep3BeginTimestamp', 'Sleep3Efficiency', 'Sleep3EndTimestamp', 'Sleep3MinutesAwake',
                'Sleep3MinutesStageDeep', 'Sleep3MinutesStageLight', 'Sleep3MinutesStageRem', 'Sleep3MinutesStageWake',
                # 'Sleep3Main_Sleep', 'Sleep3Time_In_Bed',
                'SleepMinutesAsleep', 'SleepMinutesInBed',
                'SleepPerDay']

# output df sleep header
sleep_df_header = ['BeginTimestamp', 'EndTimestamp',
                   'Main_Sleep', 'Time_In_Bed',
                   'MinutesAwake', 'MinutesStageDeep', 'MinutesStageLight',
                   'MinutesStageRem', 'MinutesStageWake', 'Efficiency']

# step count window
step_count_window = 10

# Inactive threshold
inactive_threshold = 15

# Sleep hour threshold
sleep_hour_threshold = 4


def construct_frame_data(user_id, om_id, sleep_data):
    """
    Construct frame data

    Parameters
    ----------
    user_id: str
    om_id: str
    sleep_data: Pandas Frame

    Returns
    -------
    frame_data : Dict
        A Dict of (user_id : str,
                   om_id: str,
                   sleep_data: DataFrame).

        'user_id' is the user id for each participant.

        'om_id' is the om id for each participant assigned by Evidation.

        'sleep_timeline` is the sleep time for each sleep, contains 'SleepBeginTimestamp', 'SleepEndTimestamp',
                    'SleepMinutesAwake', 'SleepMinutesStageDeep', 'SleepMinutesStageLight',
                    'SleepMinutesStageRem', 'SleepMinutesStageWake', 'SleepEfficiency'

    """
    frame_data = {}
    frame_data['user_id'] = user_id
    frame_data['om_id'] = om_id
    frame_data['sleep_data'] = sleep_data
    
    return frame_data


def combine_step_count_sleep_and_sleep_summary(ground_truth_folder_path,
                                               sleep_stepcount_data_array,
                                               sleep_summary_data_array,
                                               output_data_folder_path):
    """
    Read Full Sleep data for each participant from step count data and sleep summary,
    note we don't have sleep summaries with different stages if sleep is coming from step count data
    
    Parameters
    ----------
    sleep_stepcount_data_array: List
        A Dict of (user_id : str,
                       om_id: str,
                       sleep_data: DataFrame).
    sleep_summary_data_array:
        A Dict of (user_id : str,
                   om_id: str,
                   sleep_data: DataFrame).
        
    output_data_folder_path: str
        output folder.

    Returns
    -------
    sleep_timeline_data_array : List
        A Dict of (user_id : str,
                   om_id: str,
                   sleep_data: DataFrame).

        'user_id' is the user id for each participant.

        'om_id' is the om id for each participant assigned by Evidation.

        'sleep_timeline` is the sleep time for each sleep, contains 'SleepBeginTimestamp', 'SleepEndTimestamp',
                    'SleepMinutesAwake', 'SleepMinutesStageDeep', 'SleepMinutesStageLight',
                    'SleepMinutesStageRem', 'SleepMinutesStageWake', 'SleepEfficiency'

    """
    print('-----------------------------------------------------')
    print('--------------- Combine Sleep Summary ---------------')
    print('-----------------------------------------------------')
    
    combine_sleep_timeline_data_array = []
    
    combined_sleep_summary_file_array = [file_name for file_name in os.listdir(output_data_folder_path)
                                          if '_Sleep_Combine' in file_name]

    # Construct id file path
    id_data_df = pd.read_csv(os.path.join(ground_truth_folder_path, id_csv))

    # if we have extract the feature before and save as csv file, just read the data
    if len(combined_sleep_summary_file_array) > 20:
        for file_name in combined_sleep_summary_file_array:
            
            # Read parameters
            om_id = file_name.split('_Sleep_Combine')[0]
            user_id = id_data_df.loc[id_data_df['OMuser_id'] == om_id]['user_id'].values[0]
            combined_sleep_timeline = pd.read_csv(os.path.join(output_data_folder_path,
                                                               om_id + '_Sleep_Combine.csv'))
            print('Read combined data for ' + user_id)
            
            combine_sleep_timeline_data_array.append(construct_frame_data(user_id, om_id, combined_sleep_timeline))
    else:
        for individual_sleep_stepcount_data in sleep_stepcount_data_array:
            # read om_id, user_id, sleep_timeline_step_count_array, sleep_timeline_summary
            om_id = individual_sleep_stepcount_data['om_id']
            user_id = individual_sleep_stepcount_data['user_id']
            sleep_timeline_step_count_df = individual_sleep_stepcount_data['sleep_data']
            
            print('Combine data ' + user_id)
            
            sleep_timeline_summary_df = [individual_sleep_summary_data['sleep_data']
                                         for individual_sleep_summary_data in sleep_summary_data_array
                                         if individual_sleep_stepcount_data['user_id'] == user_id]
            
            if len(sleep_timeline_summary_df[0]) > 0:
                for index, sleep_timeline_step_count in sleep_timeline_step_count_df.iterrows():
                    
                    sleep_timeline_step_count = sleep_timeline_step_count.to_frame().transpose()
                    
                    # Read sleep time from step count
                    start_sleep_time_step_count = datetime.datetime.strptime(
                            sleep_timeline_step_count['SleepBeginTimestamp'].values[0], date_time_format)
                    end_sleep_time_step_count = datetime.datetime.strptime(
                            sleep_timeline_step_count['SleepEndTimestamp'].values[0], date_time_format)
                    
                    # Calculate sleep time from step count
                    mid_sleep_time_step_count = start_sleep_time_step_count + (end_sleep_time_step_count - start_sleep_time_step_count) / 2
        
                    is_sleep_timeline_step_count_exist = False
                    
                    # If mid-point of sleep time from step count inference is not in sleep summary list, then add it!
                    for idx, sleep_timeline_summary in sleep_timeline_summary_df[0].iterrows():
                        # Read sleep time from summary
                        start_sleep_time_summary = datetime.datetime.strptime(sleep_timeline_summary['SleepBeginTimestamp'], date_time_format)
                        end_sleep_time_summary = datetime.datetime.strptime(sleep_timeline_summary['SleepEndTimestamp'], date_time_format)
    
                        # So it is a sleep
                        # if start_sleep_time_summary < mid_sleep_time_step_count < end_sleep_time_summary \
                        #        or start_sleep_time_step_count < mid_sleep_time_summary < end_sleep_time_step_count:
                        if start_sleep_time_summary < start_sleep_time_step_count < end_sleep_time_summary \
                                or start_sleep_time_summary < end_sleep_time_step_count < end_sleep_time_summary:
                            is_sleep_timeline_step_count_exist = True
                            
                        if start_sleep_time_step_count < start_sleep_time_summary < end_sleep_time_step_count \
                                or start_sleep_time_step_count < end_sleep_time_summary < end_sleep_time_step_count:
                            is_sleep_timeline_step_count_exist = True
                            
                    # If sleep timeline is not happened in the sleep summary add it!
                    if is_sleep_timeline_step_count_exist is False:
                        sleep_timeline_summary_df[0] = sleep_timeline_summary_df[0].append(sleep_timeline_step_count)
    
            frame_data = {}
            frame_data['user_id'] = user_id
            frame_data['om_id'] = om_id
            frame_data['sleep_data'] = sleep_timeline_summary_df[0].sort_values('SleepBeginTimestamp')

            frame_data['sleep_data'].to_csv(os.path.join(output_data_folder_path, om_id + '_Sleep_Combine.csv'), index=False)
    
            combine_sleep_timeline_data_array.append(frame_data)
    
    return combine_sleep_timeline_data_array


def extract_sleep_from_fitbit_step_count(fitbit_data_folder_path, ground_truth_folder_path, output_data_folder_path):
    """
    Read Sleep timeline for each participant from step count

    Parameters
    ----------
    fitbit_data_folder_path: str
        Fitbit data folder.
        
    ground_truth_folder_path: str
        ground truth data folder, like id.
        
    output_data_folder_path: str
        output folder.

    Returns
    -------
    sleep_timeline_data_array : List
        A Dict of (user_id : str,
                   om_id: str,
                   sleep_data: DataFrame).

        'user_id' is the user id for each participant.
        
        'om_id' is the om id for each participant assigned by Evidation.
        
        'sleep_timeline` is the sleep time for each sleep, contains 'SleepBeginTimestamp', 'SleepEndTimestamp',
                    'SleepMinutesAwake', 'SleepMinutesStageDeep', 'SleepMinutesStageLight',
                    'SleepMinutesStageRem', 'SleepMinutesStageWake', 'SleepEfficiency'
                    
    """
    print('------------------------------------------------------------------')
    print('--------------- Read Sleep Timeline from StepCount ---------------')
    print('------------------------------------------------------------------')
    
    # Construct id file path
    id_data_df = pd.read_csv(os.path.join(ground_truth_folder_path, id_csv))
    
    # Read step_count_data
    step_count_data_array = [[file_name.split('_stepCount.csv')[0],
                              pd.read_csv(os.path.join(fitbit_data_folder_path, file_name))]
                             for file_name in os.listdir(fitbit_data_folder_path) if 'stepCount' in file_name]
    
    # Initialize sleep_timeline_data_array to return and output df col header (match all)
    sleep_timeline_data_array = []
    output_df_header = ['Sleep' + header for header in sleep_df_header]
    
    step_count_sleep_file_array = [file_name for file_name in os.listdir(output_data_folder_path) if
                                   '_Sleep_StepCount' in file_name]
    
    # if we have extract the feature before and save as csv file
    if len(step_count_sleep_file_array) > 20:
        for file_name in step_count_sleep_file_array:
            om_id = file_name.split('_Sleep_StepCount')[0]
            user_id = id_data_df.loc[id_data_df['OMuser_id'] == om_id]['user_id'].values[0]
            sleep_timeline = pd.read_csv(os.path.join(output_data_folder_path,
                                                      om_id + '_Sleep_StepCount.csv'))
            
            print('Read data for ' + user_id)
            
            frame_data = {}
            frame_data['user_id'] = user_id
            frame_data['om_id'] = om_id
            frame_data['sleep_data'] = sleep_timeline
            
            sleep_timeline_data_array.append(frame_data)
    
    else:
        # Iterate each participant's step count data
        for individual_step_count_data in step_count_data_array:
            
            # Extract user id from id_df
            user_id = id_data_df.loc[id_data_df['OMuser_id'] == individual_step_count_data[0]]['user_id'].values[0]
            
            sleep_timeline = []
            
            print('Extract data for ' + user_id)
            
            # We want to windowing the step count data, window 10 minutes of data
            timestamp_window, window_step_count_data = [], []
            sleep_start_time = datetime.datetime.strptime('2015-01-01T00:00:00.000', date_time_format)
            sleep_end_time = datetime.datetime.strptime('2015-01-01T00:00:00.000', date_time_format)
            
            # Iterate individual step count in minute level
            for idx, row in individual_step_count_data[1].iterrows():
                timestamp = row['Timestamp']
                step_count = row['StepCount']
                
                # Always delete the first element from window timestamp and step count data
                if len(timestamp_window) == step_count_window:
                    timestamp_window.pop(0)
                    window_step_count_data.pop(0)
                
                # Append the data
                timestamp_window.append(timestamp)
                window_step_count_data.append(step_count)
                
                # First detect whether we have timestamp inconsistency
                timestamp_delta = datetime.datetime.strptime(timestamp, date_time_format) - datetime.datetime.strptime(
                        timestamp_window[len(timestamp_window) - 1], date_time_format)
                
                # If delta is bigger than an hour
                if timestamp_delta.total_seconds() > 3600:
                    
                    # If inactive for 1.5 hours, we assume it is a sleep or nap
                    if 3600 * sleep_hour_threshold < (sleep_end_time - sleep_start_time).total_seconds() < 3600 * 10:
                        # Append the data
                        sleep_data_temp = pd.DataFrame([sleep_start_time.strftime(date_time_format)[:-3],
                                                        sleep_end_time.strftime(date_time_format)[:-3],
                                                        np.nan, np.nan, np.nan,
                                                        np.nan, np.nan, np.nan]).transpose()
                        sleep_data_temp.columns = output_df_header
                        sleep_data_temp['data_source'] = 0
                        if len(sleep_timeline) > 0:
                            sleep_timeline = sleep_timeline.append(sleep_data_temp)
                        else:
                            sleep_timeline = sleep_data_temp
                    
                    # Initialize the data again
                    timestamp_window, window_step_count_data = [], []
                    sleep_start_time = datetime.datetime.strptime('2015-01-01T00:00:00.000', date_time_format)
                    sleep_end_time = datetime.datetime.strptime('2015-01-01T00:00:00.000', date_time_format)
                else:
                    # We have found the inactive region
                    if len(window_step_count_data) == step_count_window and np.sum(
                            np.array(window_step_count_data)) < inactive_threshold:
                        # We have not initialize the start before
                        if sleep_start_time.year != 2018:
                            sleep_start_time = datetime.datetime.strptime(timestamp_window[0], date_time_format)
                        
                        # Just update end time
                        sleep_end_time = datetime.datetime.strptime(timestamp_window[len(timestamp_window) - 1],
                                                                    date_time_format)
                    
                    # End inactive region
                    elif len(window_step_count_data) == step_count_window and np.sum(
                            np.array(window_step_count_data)) >= inactive_threshold:
                        
                        # Delta time
                        timestamp_delta = sleep_end_time - sleep_start_time
                        
                        # If timestamp_delta is greater than 1.5 hours
                        if 3600 * sleep_hour_threshold < timestamp_delta.total_seconds() < 3600 * 10:
                            # Append the data
                            sleep_data_temp = pd.DataFrame([sleep_start_time.strftime(date_time_format)[:-3],
                                                            sleep_end_time.strftime(date_time_format)[:-3],
                                                            np.nan, np.nan, np.nan,
                                                            np.nan, np.nan, np.nan]).transpose()
                            sleep_data_temp.columns = output_df_header
                            sleep_data_temp['data_source'] = 0
                            if len(sleep_timeline) > 0:
                                sleep_timeline = sleep_timeline.append(sleep_data_temp)
                            else:
                                sleep_timeline = sleep_data_temp
                        # Initialize the data again
                        timestamp_window, window_step_count_data = [], []
                        sleep_start_time = datetime.datetime.strptime('2015-01-01T00:00:00.000', date_time_format)
                        sleep_end_time = datetime.datetime.strptime('2015-01-01T00:00:00.000', date_time_format)
            
            if len(sleep_timeline) > 0:
                frame_data = {}
                frame_data['user_id'] = user_id
                frame_data['om_id'] = individual_step_count_data[0]
                frame_data['sleep_data'] = sleep_timeline
                
                sleep_timeline_data_array.append(frame_data)
                sleep_timeline.to_csv(os.path.join(output_data_folder_path,
                                                   individual_step_count_data[0] + '_Sleep_StepCount.csv'),
                                      index=False)
    
    return sleep_timeline_data_array


def extract_sleep_from_fitbit_sleep_summary(fitbit_data_folder_path, ground_truth_folder_path, output_data_folder_path):
    """
    Read Sleep Summary for each participant

    Parameters
    ----------
    fitbit_data_folder_path: str
        Fitbit data folder.
        
    ground_truth_folder_path: str
        ground truth data folder, like id.
        
    output_data_folder_path: str
        output folder.

    Returns
    -------
    sleep_summary_data_array : List
        A Dict of (user_id : str,
                   om_id: str,
                   sleep_data: DataFrame).

        'user_id' is the user id for each participant.
        
        'om_id' is the om id for each participant assigned by Evidation.

        'sleep_data` is the sleep summary for each sleep, contains 'SleepBeginTimestamp', 'SleepEndTimestamp',
                   'SleepMinutesAwake', 'SleepMinutesStageDeep', 'SleepMinutesStageLight',
                   'SleepMinutesStageRem', 'SleepMinutesStageWake', 'SleepEfficiency'
    """
    print('--------------------------------------------------------------')
    print('--------------- Read Sleep Summary from Fitbit ---------------')
    print('--------------------------------------------------------------')
    
    # Construct id file path
    id_data_df = pd.read_csv(os.path.join(ground_truth_folder_path, id_csv))
    
    # read daily summary
    summary_data_array = [[file_name.split('_dailySummary.csv')[0],
                           pd.read_csv(os.path.join(fitbit_data_folder_path, file_name))[sleep_header]]
                          for file_name in os.listdir(fitbit_data_folder_path) if 'dailySummary' in file_name]
    
    # initialize sleep_summary_data_array to return and output df col header
    sleep_summary_data_array = []
    output_df_header = ['Sleep' + header for header in sleep_df_header]

    sleep_summary_file_array = [file_name for file_name in os.listdir(output_data_folder_path) if
                                '_Sleep_Summary' in file_name]

    # if we have extract the feature before and save as csv file
    if len(sleep_summary_file_array) > 20:
        for file_name in sleep_summary_file_array:
            participant_id = file_name.split('_Sleep_Summary')[0]
            user_id = id_data_df.loc[id_data_df['participant_id'] == participant_id]['mitre_id'].values[0]
            sleep_summary = pd.read_csv(os.path.join(output_data_folder_path, participant_id + '_Sleep_Summary.csv'))
        
            print('Read data for ' + user_id)
        
            frame_data = {}
            frame_data['user_id'] = user_id
            frame_data['participant_id'] = participant_id
            frame_data['sleep_data'] = sleep_summary

            sleep_summary_data_array.append(frame_data)
    else:
        for individual_summary_data in summary_data_array:
            user_id = id_data_df.loc[id_data_df['participant_id'] == individual_summary_data[0]]['mitre_id'].values[0]
            participant_id = individual_summary_data[0]
            sleep_data = []
    
            print('Extract data for ' + user_id)
            
            for idx, row in individual_summary_data[1].iterrows():
                # if we have more than one sleep per day
                if row['SleepPerDay'] != 0:
                    # iterate each sleep for that day
                    sleep_per_day = row['SleepPerDay'] if row['SleepPerDay'] <= 3 else 3
                    for i in range(sleep_per_day):
                        extract_header = ['Sleep' + str(i + 1) + header for header in sleep_df_header]
                        
                        # Aggregate each sleep data
                        if len(sleep_data) is 0:
                            sleep_data = row[extract_header].to_frame().transpose()
                            sleep_data.columns = output_df_header
                            sleep_data['data_source'] = 1
                        else:
                            temp = row[extract_header].to_frame().transpose()
                            temp.columns = output_df_header
                            temp['data_source'] = 1
                            sleep_data = sleep_data.append(temp)
            
            # Construct the frame level data per sleep
            frame_data = {}
            frame_data['user_id'] = user_id
            frame_data['participant_id'] = individual_summary_data[0]
            frame_data['sleep_data'] = sleep_data
            
            # if sleep data is not null, we add the frame data
            if len(sleep_data) > 0:
                sleep_summary_data_array.append(frame_data)
                sleep_data = sleep_data.sort_values('SleepBeginTimestamp')
                sleep_data.to_csv(os.path.join(output_data_folder_path, participant_id + '_Sleep_Summary.csv'), index=False)

    return sleep_summary_data_array


if __name__ == '__main__':
    
    DEBUG = 1
    
    if DEBUG == 0:
        # Define the parser
        parser = argparse.ArgumentParser(description='Parse sleep timeline.')
        
        parser.add_argument('-t', '--save_type', type=str, required=False,
                            help='Save data type.')
        parser.add_argument('-i', '--main_data_directory', type=str, required=False,
                            help='Directory with source data.')
        parser.add_argument('-o', '--sleep_directory', type=str, required=False,
                            help='File with processed data.')
        
        args = parser.parse_args()
        
        # if we have these parser information, then read them
        if args.save_type is not None: save_type = args.save_type
        if args.main_data_directory is not None: main_data_directory = args.main_data_directory
        if args.sleep_directory is not None: sleep_directory = args.sleep_directory
        
    else:
        main_data_directory = '../../data'
        save_type = 'sleep_summary'
        sleep_directory = '../output'
        

    # fitbit_data_folder path
    fitbit_data_folder_path = get_fitbit_data_folder(main_data_directory)

    # ground_truth path
    ground_truth_folder_path = get_ground_truth_folder(main_data_directory)

    # om signal start and end recording time
    om_signal_start_end_recording_path = get_om_signal_start_end_recording_folder(sleep_directory)

    # sleep timeline
    sleep_timeline_data_folder_path = get_sleep_timeline_data_folder(sleep_directory)

    # Create folder if not exist
    if os.path.exists(sleep_timeline_data_folder_path) is False: os.mkdir(sleep_timeline_data_folder_path)

    if 'sleep_summary' in save_type:
        extract_sleep_from_fitbit_sleep_summary(fitbit_data_folder_path, ground_truth_folder_path, sleep_timeline_data_folder_path)
    elif 'step_count' in save_type:
        extract_sleep_from_fitbit_step_count(fitbit_data_folder_path, ground_truth_folder_path, sleep_timeline_data_folder_path)
    else:
        # combined sleep data
        sleep_timeline_stepcount_data_array = extract_sleep_from_fitbit_step_count(fitbit_data_folder_path, ground_truth_folder_path, sleep_timeline_data_folder_path)
        sleep_summary_data_array = extract_sleep_from_fitbit_sleep_summary(fitbit_data_folder_path, ground_truth_folder_path, sleep_timeline_data_folder_path)
        
        combine_step_count_sleep_and_sleep_summary(ground_truth_folder_path, sleep_timeline_stepcount_data_array, sleep_summary_data_array, sleep_timeline_data_folder_path)
