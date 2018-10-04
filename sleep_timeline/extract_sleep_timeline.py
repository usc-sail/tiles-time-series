#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import argparse

# add util into the file path, so we can import helper functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from files import *
from load_data_basic import *

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
# save_type = 'combined'
save_type = 'sleep_summary'


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


def extract_sleep_from_fitbit_sleep_summary(fitbit_data_folder_path, main_data_folder_path, output_data_folder_path):
    """
    Read Sleep Summary for each participant

    Parameters
    ----------
    fitbit_data_folder_path: str
        Fitbit data folder.
        
    main_data_folder_path: str
        main data folder, like id.
        
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
    # id_data_df = pd.read_csv(os.path.join(ground_truth_folder_path, id_csv))
    # id_data_df = pd.read_csv(os.path.join(main_data_folder_path, 'keck_wave2/id-mapping', 'mitreids.csv'))
    id_data_df = getParticipantInfo(main_data_directory)
    
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
            user_id = id_data_df.loc[id_data_df['ParticipantID'] == participant_id]['MitreID'].values[0]
            sleep_summary = pd.read_csv(os.path.join(output_data_folder_path, participant_id + '_Sleep_Summary.csv'))
        
            print('Read data for ' + user_id)
        
            frame_data = {}
            frame_data['user_id'] = user_id
            frame_data['participant_id'] = participant_id
            frame_data['sleep_data'] = sleep_summary

            sleep_summary_data_array.append(frame_data)
    else:
        for individual_summary_data in summary_data_array:
            user_id = id_data_df.loc[id_data_df['ParticipantID'] == individual_summary_data[0]]['MitreID'].values[0]
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
        
        parser.add_argument('-t', '--save_type', type=str, required=False, help='Save data type.')
        parser.add_argument('-i', '--main_data_directory', type=str, required=False, help='Directory with source data.')
        parser.add_argument('-o', '--sleep_directory', type=str, required=False, help='File with processed data.')
        
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
    fitbit_data_folder_path = os.path.join(main_data_directory, 'keck_wave_all/2_raw_csv_data/fitbit')

    # ground_truth path
    # ground_truth_folder_path = get_ground_truth_folder(main_data_directory)

    # om signal start and end recording time
    # om_signal_start_end_recording_path = get_om_signal_start_end_recording_folder(sleep_directory)

    # sleep timeline
    sleep_timeline_data_folder_path = os.path.join(sleep_directory, 'sleep_timeline')
    
    # Create folder if not exist
    if os.path.exists(sleep_timeline_data_folder_path) is False: os.mkdir(sleep_timeline_data_folder_path)

    if 'sleep_summary' in save_type:
        extract_sleep_from_fitbit_sleep_summary(fitbit_data_folder_path, main_data_directory, sleep_timeline_data_folder_path)