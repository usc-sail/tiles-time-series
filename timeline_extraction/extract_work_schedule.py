#!/usr/bin/env python3

import os
import sys
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
id_csv = 'IDs.csv'
job_shift_csv = 'Job_Shift.csv'
recording_start_end_csv = 'OM_Signal_Start_End.csv'

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

# valid recording threshold
valid_om_signal_recording_gap_hour = 5


def extract_nurse_om_signal_recording_start_end(ground_truth_folder_path, job_shift_folder_path,
                                                om_signal_data_folder_path, output_data_folder_path,
                                                valid_om_signal_recording_gap_hour):
    """
    Extract OM signal start recording time and end recording time

    Parameters
    ----------
    ground_truth_folder_path: str
        ground truth folder
        
    job_shift_folder_path: str
        job shift type folder
        
    om_signal_data_folder_path: str
        om signal preprocessed data folder
        
    output_data_folder_path: str
        output folder path
    
    Returns
    -------
    None, just save under output_data_folder_path

    """
    
    # list om_signal files
    file_name_array = os.listdir(om_signal_data_folder_path)
    
    # read id
    id_data_df = pd.read_csv(os.path.join(ground_truth_folder_path, id_csv))
    
    # read job shift
    job_shift_df = pd.read_csv(os.path.join(job_shift_folder_path, job_shift_csv))
    
    work_start_end_time = []
    
    # if we have extracted the feature before
    for file_name in file_name_array:
        
        # read user id and work shift_type
        user_id = id_data_df.loc[id_data_df['OMuser_id'] == file_name.split('_omsignal.csv')[0]]['user_id'].values[0]
        work_shift = job_shift_df.loc[job_shift_df['uid'] == user_id]['job_shift'].values[0]
        
        print('Extract om signal recording start and end time line for ' + user_id)
        
        # read om_signal
        omsignal_timestamp_df = pd.read_csv(os.path.join(om_signal_data_folder_path, file_name))['Timestamp']
        
        # init start and end datetime
        start_recording_datetime = datetime.datetime.strptime(omsignal_timestamp_df[0], date_time_format)
        end_recording_datetime = datetime.datetime.strptime(omsignal_timestamp_df[0], date_time_format)
        
        current_datetime = datetime.datetime.strptime(omsignal_timestamp_df[0], date_time_format)
        last_datetime = datetime.datetime.strptime(omsignal_timestamp_df[0],date_time_format)
        
        # if user id is not null
        if len(user_id) is not 0:
            # iterate row in om_signal_timestamp_df
            for row in omsignal_timestamp_df:
                
                current_datetime = datetime.datetime.strptime(row, date_time_format)
                delta_time = current_datetime - last_datetime
                
                # if delta is larger than 5 hours, we think there is a new recording
                if delta_time.seconds > 60 * 60 * valid_om_signal_recording_gap_hour:
                    end_recording_datetime = last_datetime
                    
                    if (end_recording_datetime - start_recording_datetime).seconds > 60 * 120:
                        
                        # work shift to determine work date
                        if work_shift == 1:
                            work_date = end_recording_datetime.strftime(date_only_date_time_format)
                        else:
                            if end_recording_datetime.year is 2018:
                                if start_recording_datetime.hour < 6:
                                    work_date = (end_recording_datetime - datetime.timedelta(days=1)).strftime(date_only_date_time_format)
                                else:
                                    work_date = start_recording_datetime.strftime(date_only_date_time_format)
                            else:
                                work_date = (end_recording_datetime - datetime.timedelta(days=1)).strftime(date_only_date_time_format)
                        
                        work_start_end_time.append([user_id, work_shift, work_date,
                                                    start_recording_datetime.strftime(date_time_format)[:-3],
                                                    end_recording_datetime.strftime(date_time_format)[:-3]])
                        
                        # format work_start_end_time_df
                        work_start_end_time_df = pd.DataFrame(work_start_end_time,
                                                              columns=['user_id', 'work_shift_type',
                                                                       'recording_date', 'start_recording_time', 'end_recording_time'])
                        
                        work_start_end_time_df.to_csv(os.path.join(output_data_folder_path, recording_start_end_csv), index=False)
                    
                    # set new start recording time to current time
                    start_recording_datetime = current_datetime
                
                # update last datetime
                last_datetime = current_datetime
                

if __name__ == '__main__':
    
    # Define the parser
    parser = argparse.ArgumentParser(description='Parse read folders and output folders.')
    parser.add_argument('-i', '--input_data_directory', type=str, required=False,
                        help='Directory with source data.')
    parser.add_argument('-o', '--output_directory', type=str, required=False,
                        help='File with processed data.')
    parser.add_argument('-v', '--valid_om_signal_recording_gap_hour', type=str, required=False,
                        help='minimum gap between different recording')
    args = parser.parse_args()

    # if we have these parser information, then read them
    if args.input_data_directory is not None: main_data_directory = args.input_data_directory
    if args.output_directory is not None: output_data_folder_path = args.output_directory
    if args.valid_om_signal_recording_gap_hour is not None: valid_om_signal_recording_gap_hour = int(args.valid_om_signal_recording_gap_hour)

    # omsignal folder path
    om_signal_data_folder_path = get_omsignal_data_folder(main_data_directory)

    # ground_truth path
    ground_truth_folder_path = get_ground_truth_folder(main_data_directory)

    # job_shift path
    job_shift_folder_path = get_job_shift_folder(main_data_directory)

    # om signal start and end recording time
    om_signal_start_end_recording_path = get_om_signal_start_end_recording_folder(output_data_folder_path)

    # Create output folder if not exist
    if os.path.exists(output_data_folder_path) is False: os.mkdir(output_data_folder_path)
    
    extract_nurse_om_signal_recording_start_end(ground_truth_folder_path,
                                                job_shift_folder_path,
                                                om_signal_data_folder_path,
                                                om_signal_start_end_recording_path,
                                                valid_om_signal_recording_gap_hour)
