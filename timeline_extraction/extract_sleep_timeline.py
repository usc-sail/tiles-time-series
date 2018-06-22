#!/usr/bin/env python3

import os
import sys
import glob
import numpy as np
import pandas as pd
import argparse
import datetime
import math

# current path
current_path = os.getcwd()

# fitbit_data_folder path
fitbit_data_folder_path = os.path.join(current_path, '../../data/keck_wave1/2_preprocessed_data/fitbit/fitbit')

# ground_truth path
ground_truth_folder_path = os.path.join(current_path, '../../data/keck_wave1/2_preprocessed_data/ground_truth')

# om signal start and end recording time
om_signal_start_end_recording_path = os.path.join(current_path, '../output/om_signal_timeline')

# output folder path
output_data_folder_path = os.path.join(current_path, '../output/sleep_timeline')

# csv's
id_csv = 'IDs.csv'

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

# save_type
save_type = 'combined'

# sleep_related_header from fitbit summary
sleep_header = ['Timestamp',
                'Sleep1BeginTimestamp', 'Sleep1Efficiency', 'Sleep1EndTimestamp', 'Sleep1MinutesAwake',
                'Sleep1MinutesStageDeep', 'Sleep1MinutesStageLight', 'Sleep1MinutesStageRem', 'Sleep1MinutesStageWake',
                'Sleep2BeginTimestamp', 'Sleep2Efficiency', 'Sleep2EndTimestamp', 'Sleep2MinutesAwake',
                'Sleep2MinutesStageDeep', 'Sleep2MinutesStageLight', 'Sleep2MinutesStageRem', 'Sleep2MinutesStageWake',
                'Sleep3BeginTimestamp', 'Sleep3Efficiency', 'Sleep3EndTimestamp', 'Sleep3MinutesAwake',
                'Sleep3MinutesStageDeep', 'Sleep3MinutesStageLight', 'Sleep3MinutesStageRem', 'Sleep3MinutesStageWake',
                'SleepMinutesAsleep', 'SleepMinutesInBed',
                'SleepPerDay']

# output df sleep header
sleep_df_header = ['BeginTimestamp', 'EndTimestamp',
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


def combine_step_count_sleep_and_sleep_summary(sleep_stepcount_data_array, sleep_summary_data_array):
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
    
                        mid_sleep_time_summary = start_sleep_time_summary + (end_sleep_time_summary - start_sleep_time_step_count) / 2
                        
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


def read_sleep_from_fitbit_step_count():
    """
    Read Sleep timeline for each participant from step count

    Parameters
    ----------
    None

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


def read_sleep_from_fitbit_sleep_summary():
    """
    Read Sleep Summary for each participant

    Parameters
    ----------
    None

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
            om_id = file_name.split('_Sleep_Summary')[0]
            user_id = id_data_df.loc[id_data_df['OMuser_id'] == om_id]['user_id'].values[0]
            sleep_summary = pd.read_csv(os.path.join(output_data_folder_path,
                                                      om_id + '_Sleep_Summary.csv'))
        
            print('Read data for ' + user_id)
        
            frame_data = {}
            frame_data['user_id'] = user_id
            frame_data['om_id'] = om_id
            frame_data['sleep_data'] = sleep_summary_data_array

            sleep_summary_data_array.append(frame_data)
    else:
        for individual_summary_data in summary_data_array:
            user_id = id_data_df.loc[id_data_df['OMuser_id'] == individual_summary_data[0]]['user_id'].values[0]
            om_id = individual_summary_data[0]
            sleep_data = []
    
            print('Extract data for ' + user_id)
            
            for idx, row in individual_summary_data[1].iterrows():
                # if we have more than one sleep per day
                if row['SleepPerDay'] != 0:
                    # iterate each sleep for that day
                    for i in range(row['SleepPerDay']):
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
            frame_data['om_id'] = individual_summary_data[0]
            frame_data['sleep_data'] = sleep_data
            
            # if sleep data is not null, we add the frame data
            if len(sleep_data) > 0:
                sleep_summary_data_array.append(frame_data)
                sleep_data.to_csv(os.path.join(output_data_folder_path, om_id + '_Sleep_Summary.csv'), index=False)

    return sleep_summary_data_array


if __name__ == '__main__':
    
    # Create folder if not exist
    if os.path.exists(output_data_folder_path) is False: os.mkdir(output_data_folder_path)

    if 'sleep_summary' in save_type:
        read_sleep_from_fitbit_sleep_summary()
    elif 'step_count' in save_type:
        read_sleep_from_fitbit_step_count()
    else:
        # combined sleep data
        sleep_timeline_stepcount_data_array = read_sleep_from_fitbit_step_count()
        sleep_summary_data_array = read_sleep_from_fitbit_sleep_summary()
        
        combine_step_count_sleep_and_sleep_summary(sleep_timeline_stepcount_data_array, sleep_summary_data_array)

    # Define the parser
    parser = argparse.ArgumentParser(description='Parse read folders and output folders.')

    parser.add_argument('-t', '--save_type', type=str, required=False,
                        help='Save data type.')
    parser.add_argument('-f', '--fitbit_directory', type=str, required=False,
                        help='Directory with Fitbit data.')
    parser.add_argument('-g', '--ground_truth_directory', type=str, required=False,
                        help='Directory with ground truth data.')
    parser.add_argument('-o', '--output_directory', type=str, required=False,
                        help='File with processed data.')
    parser.add_argument('-m', '--om_signal_recording_time_directory', type=str, required=False,
                        help='Directory with OM Signal recording time.')

    args = parser.parse_args()

    # if we have these parser information, then read them
    if len(args.save_type) > 0: save_type = args.save_type
    if len(args.fitbit_directory) > 0: om_signal_data_folder_path = args.om_signal_directory
    if len(args.ground_truth_directory) > 0: ground_truth_folder_path = args.ground_truth_directory
    if len(
            args.om_signal_recording_time_directory) > 0: om_signal_recording_time_directory = args.om_signal_recording_time_directory
    if len(args.output_directory) > 0: output_data_folder_path = args.output_directory
    if len(args.valid_om_signal_recording_gap_hour) > 0: valid_om_signal_recording_gap_hour = int(
        args.valid_om_signal_recording_gap_hour)
