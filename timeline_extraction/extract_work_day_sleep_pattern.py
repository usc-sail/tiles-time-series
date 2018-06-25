#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import argparse
import datetime
import math

from extract_sleep_timeline import *
from extract_work_schedule import *

# add util into the file path, so we can import helper functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from files import *

# current path
current_path = os.getcwd()

# main_data_folder path
main_data_directory = os.path.join(current_path, '../../data')

# output_data_folder path
output_data_folder_path = os.path.join(current_path, '../output')

# read_data_type
read_data_type = 'combined'

# csv's
job_shift_sleep_csv = 'workday_sleep_routine.csv'

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

# sleep_related_header
sleep_header = ['Timestamp',
                'Sleep1BeginTimestamp', 'Sleep1Efficiency', 'Sleep1EndTimestamp', 'Sleep1MinutesAwake',
                'Sleep1MinutesStageDeep', 'Sleep1MinutesStageLight', 'Sleep1MinutesStageRem', 'Sleep1MinutesStageWake',
                'Sleep2BeginTimestamp', 'Sleep2Efficiency', 'Sleep2EndTimestamp', 'Sleep2MinutesAwake',
                'Sleep2MinutesStageDeep', 'Sleep2MinutesStageLight', 'Sleep2MinutesStageRem', 'Sleep2MinutesStageWake',
                'Sleep3BeginTimestamp', 'Sleep3Efficiency', 'Sleep3EndTimestamp', 'Sleep3MinutesAwake',
                'Sleep3MinutesStageDeep', 'Sleep3MinutesStageLight', 'Sleep3MinutesStageRem', 'Sleep3MinutesStageWake',
                'SleepMinutesAsleep', 'SleepMinutesInBed',
                'SleepPerDay']

output_sleep_header = ['MinutesAwake', 'MinutesStageDeep', 'MinutesStageLight',
                       'MinutesStageRem', 'MinutesStageWake', 'Efficiency']

sleep_df_header = ['BeginTimestamp', 'EndTimestamp',
                   'MinutesAwake', 'MinutesStageDeep', 'MinutesStageLight',
                   'MinutesStageRem', 'MinutesStageWake', 'Efficiency']


def extract_work_day_sleep_pattern_from_om_signal(sleep_data_array, om_signal_start_end_recording_path, sleep_routine_work_folder_path):
    """
    Read Sleep pattern for each participant on workdays

    Parameters
    ----------
    sleep_data_array: List
        sleep data array
        
    job_shift_folder_path: str
        job shift type folder
        
    om_signal_start_end_recording_path: str
        om signal recording time line data folder
        
    sleep_routine_work_folder_path: str
        output folder

    Returns
    -------
    None

    """
    
    # read om signal recording time line
    om_recording_start_end_time_df = pd.read_csv(os.path.join(om_signal_start_end_recording_path, 'OM_Signal_Start_End.csv'))
    workday_sleep_schedule = []

    for index, row in om_recording_start_end_time_df.iterrows():
        print(row)

        user_id, work_date, work_shift = row['user_id'], row['recording_date'], row['work_shift_type']
        start_recording_time, end_recording_time = row['start_recording_time'], row['end_recording_time']

        work_date_datetime = datetime.datetime.strptime(work_date, date_only_date_time_format)

        # initialize standard work shift datetime based on shift type
        if work_shift == 1:
            start_work_datetime = work_date_datetime.replace(hour=7, minute=0, second=0)
            end_work_datetime = work_date_datetime.replace(hour=19, minute=0, second=0)
        else:
            start_work_datetime = work_date_datetime.replace(hour=19, minute=0, second=0)
            end_work_datetime = (work_date_datetime + datetime.timedelta(days=1)).replace(hour=7, minute=0, second=0)

        # iterate sleep data
        for sleep_summary_data in sleep_data_array:
            if user_id in sleep_summary_data['user_id']:
                sleep_summary_data_df = sleep_summary_data['sleep_data']

                # initilize df and header
                output_df_header = ['Sleep' + header for header in sleep_df_header]
                output_df_init_data = [np.nan for header in sleep_df_header]
                
                min_time_before_work_standard, min_time_after_work_standard = math.inf, math.inf
                min_time_before_work_om_signal, min_time_after_work_om_signal = math.inf, math.inf

                sleep_before_work_df = pd.DataFrame(output_df_init_data).transpose()
                sleep_before_work_df.columns = output_df_header
                sleep_before_work_df['data_source'] = 1

                sleep_after_work_df = pd.DataFrame(output_df_init_data).transpose()
                sleep_after_work_df.columns = output_df_header
                sleep_after_work_df['data_source'] = 1

                # iterate sleep data and find sleep data before and after work
                for idx, sleep_data in sleep_summary_data_df.iterrows():
                    sleep_data_df = sleep_data.to_frame().transpose().fillna(0)
    
                    time_begin_sleep = sleep_data_df['SleepBeginTimestamp'].values[0]
                    time_end_sleep = sleep_data_df['SleepEndTimestamp'].values[0]

                    # After work, when sleep and wake
                    if time_begin_sleep != 0:
                        time_begin_sleep = datetime.datetime.strptime(time_begin_sleep, date_time_format)
                        
                        sleep_time_after_work = time_begin_sleep - end_work_datetime
                        sleep_time_after_om_signal = time_begin_sleep - datetime.datetime.strptime(end_recording_time, date_time_format)

                        sleep_after_om_signal_in_hour = round(
                                sleep_time_after_om_signal.days * 24 + sleep_time_after_om_signal.seconds / 3600, 2)
                        
                        sleep_time_after_work_in_hour = round(
                                sleep_time_after_work.days * 24 + sleep_time_after_work.seconds / 3600, 2)
                        if sleep_time_after_work.days > -1 and sleep_time_after_work_in_hour < min_time_after_work_standard and sleep_time_after_work_in_hour < 15:
                            min_time_after_work_standard = sleep_time_after_work_in_hour
                            min_time_after_work_om_signal = sleep_after_om_signal_in_hour
                            sleep_after_work_df = sleep_data_df
    
                    # Before work, when sleep and wake
                    if time_end_sleep != 0:
                        time_end_sleep = datetime.datetime.strptime(time_end_sleep, date_time_format)
                        
                        wake_time_before_work = start_work_datetime - time_end_sleep
                        wake_time_before_om_signal = datetime.datetime.strptime(start_recording_time, date_time_format) - time_end_sleep
                        
                        wake_time_before_om_signal_in_hour = round(
                                wake_time_before_om_signal.days * 24 + wake_time_before_om_signal.seconds / 3600, 2)
                        wake_time_before_work_in_hour = round(
                                wake_time_before_work.days * 24 + wake_time_before_work.seconds / 3600, 2)
                        
                        if wake_time_before_work.days > -1 and wake_time_before_work_in_hour < min_time_before_work_standard and wake_time_before_work_in_hour < 15:
                            min_time_before_work_standard = wake_time_before_work_in_hour
                            min_time_before_work_om_signal = wake_time_before_om_signal_in_hour
                            sleep_before_work_df = sleep_data_df
                            
                # If we can't find one, then save as nan
                if min_time_before_work_standard is math.inf: min_time_before_work_standard = np.nan
                if min_time_after_work_standard is math.inf: min_time_after_work_standard = np.nan
                if min_time_before_work_om_signal is math.inf: min_time_before_work_om_signal = np.nan
                if min_time_after_work_om_signal is math.inf: min_time_after_work_om_signal = np.nan

                frame_data = [user_id, work_date, work_shift,
                              min_time_before_work_standard, min_time_after_work_standard,
                              min_time_before_work_om_signal, min_time_after_work_om_signal,
                              sleep_before_work_df['SleepBeginTimestamp'].values[0],
                              sleep_before_work_df['SleepEndTimestamp'].values[0],
                              sleep_before_work_df['data_source'].values[0],
                              start_work_datetime.strftime(date_time_format)[:-3], start_recording_time,
                              end_work_datetime.strftime(date_time_format)[:-3], end_recording_time,
                              sleep_after_work_df['SleepBeginTimestamp'].values[0],
                              sleep_after_work_df['SleepEndTimestamp'].values[0],
                              sleep_after_work_df['data_source'].values[0]]

                [frame_data.append(sleep_before_work_df['Sleep' + header].values[0]) for header in output_sleep_header]
                [frame_data.append(sleep_after_work_df['Sleep' + header].values[0]) for header in output_sleep_header]

                workday_sleep_schedule.append(frame_data)

    # save the data in the format provided
    frame_header = ['user_id', 'work_date', 'work_shift_type',
                    'wake_before_work_standard_work_time', 'sleep_after_work_standard_work_time',
                    'wake_before_work_om_signal_start_time', 'sleep_after_work_om_signal_end_time',
                    'sleep_before_work_SleepBeginTimestamp', 'sleep_before_work_SleepEndTimestamp',
                    'sleep_before_work_DataResource',
                    'start_work_time', 'start_recording_time',
                    'end_work_time', 'end_recording_time',
                    'sleep_after_work_SleepBeginTimestamp', 'sleep_after_work_SleepEndTimestamp',
                    'sleep_after_work_DataResource']

    [frame_header.append('sleep_before_work_' + header) for header in output_sleep_header]
    [frame_header.append('sleep_after_work_' + header) for header in output_sleep_header]

    workday_sleep_schedule_df = pd.DataFrame(workday_sleep_schedule, columns=frame_header)
    workday_sleep_schedule_df.to_csv(os.path.join(sleep_routine_work_folder_path, job_shift_sleep_csv), index=False)


if __name__ == '__main__':
    
    # Define the parser
    parser = argparse.ArgumentParser(description='Parse main data folders and output folders.')

    parser.add_argument('-t', '--read_type', type=str, required=False,
                        help='Read data type.')
    parser.add_argument('-i', '--input_data_directory', type=str, required=False,
                        help='Directory with source data.')
    parser.add_argument('-o', '--output_directory', type=str, required=False,
                        help='File with processed data.')
    
    args = parser.parse_args()
    
    print(args)
    
    # Read the args
    if args.input_data_directory is not None: main_data_directory = args.input_data_directory
    if args.output_directory is not None: output_data_folder_path = args.output_directory

    # fitbit_data_folder path
    fitbit_data_folder_path = get_fitbit_data_folder(main_data_directory)

    # omsignal folder path
    omsignal_data_folder_path = get_omsignal_data_folder(main_data_directory)

    # ground_truth path
    ground_truth_folder_path = get_ground_truth_folder(main_data_directory)

    # job_shift path
    job_shift_folder_path = get_job_shift_folder(main_data_directory)

    # sleep_routine_work path
    sleep_routine_work_folder_path = get_sleep_routine_work_folder(output_data_folder_path)

    # om signal start and end recording time
    om_signal_start_end_recording_path = get_om_signal_start_end_recording_folder(output_data_folder_path)

    # sleep timeline
    sleep_timeline_data_folder_path = get_sleep_timeline_data_folder(output_data_folder_path)
    
    # Read sleep timeline
    if os.path.exists(sleep_routine_work_folder_path) is False: os.mkdir(sleep_routine_work_folder_path)

    # If we used combined sleep summary input
    if 'combined' in read_data_type:
        # Read sleep data
        # Combined sleep data, read step count based sleep and sleep summary data
        sleep_timeline_stepcount_data_array = extract_sleep_from_fitbit_step_count(fitbit_data_folder_path,
                                                                                   ground_truth_folder_path,
                                                                                   sleep_timeline_data_folder_path)
        
        sleep_summary_data_array = extract_sleep_from_fitbit_sleep_summary(fitbit_data_folder_path,
                                                                           ground_truth_folder_path,
                                                                           sleep_timeline_data_folder_path)
    
        combined_sleep_data_array = combine_step_count_sleep_and_sleep_summary(ground_truth_folder_path,sleep_timeline_stepcount_data_array,
                                                                               sleep_summary_data_array,
                                                                               sleep_timeline_data_folder_path)
    
        extract_work_day_sleep_pattern_from_om_signal(combined_sleep_data_array, om_signal_start_end_recording_path, sleep_routine_work_folder_path)
    else:
        sleep_summary_data_array = extract_sleep_from_fitbit_sleep_summary(fitbit_data_folder_path,
                                                                           ground_truth_folder_path,
                                                                           sleep_timeline_data_folder_path)
    
        extract_work_day_sleep_pattern_from_om_signal(sleep_summary_data_array, om_signal_start_end_recording_path,
                                                      sleep_routine_work_folder_path)
