"""
This is script is modified based on Karel Mundnich's script: days_at_work.py
Script is modified by Tiantian Feng
"""

import os, errno
import glob
import argparse
import numpy as np
import pandas as pd
from dateutil import rrule
from datetime import datetime, timedelta

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


def getDataFrame(file):
    
    # Read and prepare owl data per participant
    data = pd.read_csv(file, index_col=0)
    # Drop last line for now
    data = data.drop(data.index[len(data) - 1])
    
    data.index = pd.to_datetime(data.index)
    
    return data


def constructRecordingFrame(index, recording_time_df, start_recording_time, end_recording_time):
    
    # if duration if larger than 5 hours, it is a valid recording
    if (end_recording_time - start_recording_time).total_seconds() > 3600 * 1:
        # Construct frame data
        frame_df = pd.DataFrame([index.strftime(date_only_date_time_format),
                                 start_recording_time.strftime(date_time_format)[:-3],
                                 end_recording_time.strftime(date_time_format)[:-3]],
                                index=['date', 'start_recording_time', 'end_recording_time']).transpose()
        
        recording_time_df = frame_df if len(recording_time_df) is 0 else recording_time_df.append(frame_df)

    return recording_time_df


def constructSurveyFrame(index, recording_time_df, start_recording_time, end_recording_time,
                         workBeforeSurvey=0, surveyAtWork=0):
    # if duration if larger than 5 hours, it is a valid recording
    if (end_recording_time - start_recording_time).total_seconds() > 3600 * 1:
        # Construct frame data
        frame_df = pd.DataFrame([index.strftime(date_only_date_time_format),
                                 index.strftime(date_time_format)[:-3],
                                 start_recording_time.strftime(date_time_format)[:-3],
                                 end_recording_time.strftime(date_time_format)[:-3],
                                 workBeforeSurvey, surveyAtWork],
                                 index=['date', 'survey_time',
                                        'start_recording_time', 'end_recording_time',
                                        'workBeforeSurvey', 'surveyAtWork']).transpose()
        
        recording_time_df = frame_df if len(recording_time_df) is 0 else recording_time_df.append(frame_df)
    
    return recording_time_df


def getWorkingOnlySensorRecordingTimeline(recording_timeline_directory, data_directory, days_at_work_df, stream):
    
    # Read participant id array
    participant_id_array = days_at_work_df.columns.values
    
    for participant_id in participant_id_array:
    
        print('Start processing for participant' + '(' + stream + ')' + ': ' + participant_id)
        
        # Read the data and days of work for participant
        if stream == 'owl_in_one':
            file_name = os.path.join(data_directory, participant_id + '_bleProximity.csv')
        elif stream == 'omsignal':
            file_name = os.path.join(data_directory, participant_id + '_omsignal.csv')
        
        if os.path.exists(file_name):
            data_df = getDataFrame(file_name)
            data_df = data_df.sort_index()
            participant_days_at_work_df = days_at_work_df[participant_id].to_frame()
            
            # Save data frame
            recording_time_df = []
            
            for index, is_recording in participant_days_at_work_df.iterrows():
                # There is an recording
                if is_recording.values[0] == 1:
                    
                    # Initialize start and end date
                    start_date, end_date = index, index + timedelta(days=1)
                    date_at_work_data_df = data_df[start_date.strftime(date_time_format) : end_date.strftime(date_time_format)]
                    if len(date_at_work_data_df) > 0:
                        datetime_at_work = pd.to_datetime(date_at_work_data_df.index)
                        
                        start_recording_time = datetime_at_work[0]
                        last_recording_time = datetime_at_work[0]
                        
                        for time in datetime_at_work:
                            current_recording_time = time
                            
                            if (current_recording_time - last_recording_time).total_seconds() > 3600 * 4:
                                
                                # Construct frame and append
                                recording_time_df = constructRecordingFrame(index, recording_time_df, start_recording_time, last_recording_time)
                                start_recording_time = current_recording_time
                            last_recording_time = current_recording_time
        
                        # Construct frame and append
                        recording_time_df = constructRecordingFrame(index, recording_time_df, start_recording_time, last_recording_time)
                        
            if len(recording_time_df) > 0:
                last_row = []
                save_df = pd.DataFrame()
                
                for index, row in recording_time_df.iterrows():
                    if len(last_row) == 0:
                        last_row = row
                    else:
                        last_end_time = datetime.strptime(last_row['end_recording_time'], date_time_format)
                        current_start_time = datetime.strptime(row['start_recording_time'], date_time_format)
                        if (current_start_time - last_end_time).total_seconds() > 3600 * 2:
                            save_df = last_row.to_frame().transpose() if len(save_df) == 0 else save_df.append(last_row.to_frame().transpose())
                            last_row = row
                        else:
                            last_row['end_recording_time'] = row['end_recording_time']
    
                save_df.to_csv(os.path.join(recording_timeline_directory, participant_id + '.csv'), index=False)
            print('End processing for participant: ' + participant_id)
        else:
            print('File not found!')


def getStartEndTime(index, shift='day'):
    
    if shift == 'day':
        start_work_date = index.replace(hour=7, minute=0, second=0)
        end_work_date = index.replace(hour=19, minute=0, second=0)
    else:
        start_work_date = (index - timedelta(days=1)).replace(hour=19, minute=0, second=0)
        end_work_date = index.replace(hour=7, minute=0, second=0)
        
    return start_work_date, end_work_date


def getWorkTimeline(recording_timeline_directory, data_directory, main_data_directory):
    
    # Read ID
    IDs = pd.read_csv(os.path.join(main_data_directory, 'id-mapping', 'mitreids.csv'), index_col=1)
    participantIDs = list(IDs['participant_id'])
    
    # We need to see when nurse is responding to the survey and based on that decide whether people worked that day
    MGT = pd.read_csv(os.path.join(main_data_directory, 'ground_truth', 'MGT', 'MGT.csv'), index_col=2)
    MGT.index = pd.to_datetime(MGT.index)

    for participant_id in participantIDs:
        
        print('Start processing for participant (ground_truth): ' + participant_id)
        
        # Get the user id
        user_id = IDs.loc[IDs['participant_id'] == participant_id].index.values[0]
        MGT_per_participant = MGT.loc[MGT['uid'] == user_id]
        
        MGT_job_per_participant = MGT_per_participant.loc[MGT_per_participant['survey_type'] == 'job']

        work_time_df = pd.DataFrame()
        
        # Wave1 survey
        for index, row in MGT_job_per_participant.iterrows():
            try:
                if row['location_mgt'] == 2.0 or row['itp_mgt'] > -1:  # At work when answering the survey according to MGT, question 1
                    # Day shift nurse
                    if 17 < index.hour < 24:
                        start_work_date, end_work_date = getStartEndTime(index, shift='day')
                    # Night shift nurse
                    else:
                        start_work_date, end_work_date = getStartEndTime(index, shift='night')

                    work_time_df = constructSurveyFrame(index, work_time_df, start_work_date, end_work_date)
                    
            except KeyError:
                print('Participant ' + row['uid'] + ' is not in participant list from IDs.csv.')

        if len(work_time_df) > 0:
            work_time_df = work_time_df.sort_values('date')
            work_time_df.to_csv(os.path.join(recording_timeline_directory, participant_id + '.csv'), index=False)
        print('End processing for participant: ' + participant_id)


def getAllTimeSensorRecordingTimeline(recording_timeline_directory, data_directory, participant_id_job_shift_df, stream):
    
    # Read ID
    IDs = pd.read_csv(os.path.join(main_data_directory, 'id-mapping', 'mitreids.csv'), index_col=1)
    participantIDs = list(IDs['participant_id'])

    if stream == 'fitbit':
        for participant_id in participantIDs:
    
            print('Start processing for participant (fitbit recording timeline): ' + participant_id)
            file_name = os.path.join(data_directory, participant_id + '_heartRate.csv')
            
            if os.path.exists(file_name) is True:
                
                data_df = getDataFrame(file_name)
                data_df.index = pd.to_datetime(data_df.index)
                data_df = data_df.sort_index()
                
                if len(data_df) > 0:
                    last_timestamp = pd.to_datetime(data_df.index.values[0])
                    start_recording_timestamp = pd.to_datetime(data_df.index.values[0])
                    
                    recording_timeline = pd.DataFrame()
                    
                    for index, row in data_df.iterrows():
                        if (pd.to_datetime(index) - last_timestamp).total_seconds() > 3600 * 1.5:
                            frame = pd.DataFrame(index=[start_recording_timestamp.strftime(date_only_date_time_format)], columns=['date', 'start_recording_time', 'end_recording_time'])
                            frame['date'] = start_recording_timestamp.strftime(date_only_date_time_format)
                            frame['start_recording_time'] = start_recording_timestamp.strftime(date_time_format)[:-3]
                            frame['end_recording_time'] = last_timestamp.strftime(date_time_format)[:-3]
                            start_recording_timestamp = index
    
                            recording_timeline = recording_timeline.append(frame)
                        
                        last_timestamp = index
                    
                    if (pd.to_datetime(last_timestamp) - pd.to_datetime(start_recording_timestamp)).total_seconds() > 3600:
                        frame = pd.DataFrame(index=[start_recording_timestamp.strftime(date_only_date_time_format)],
                                             columns=['date', 'start_recording_time', 'end_recording_time'])
                        frame['date'] = start_recording_timestamp.strftime(date_only_date_time_format)
                        frame['start_recording_time'] = start_recording_timestamp.strftime(date_time_format)[:-3]
                        frame['end_recording_time'] = last_timestamp.strftime(date_time_format)[:-3]
                        
                        recording_timeline = recording_timeline.append(frame)
    
                    if len(recording_timeline) > 0:
                        recording_timeline = recording_timeline.sort_values('start_recording_time')
                        recording_timeline.to_csv(os.path.join(recording_timeline_directory, participant_id + '.csv'), index=False)
                        
                    print('Finish processing for participant (fitbit recording timeline): ' + participant_id)
                
    elif stream == 'realizd':
        for participant_id in participantIDs:
    
            print('Start processing for participant (RealizD recording timeline): ' + participant_id)
            file_name = os.path.join(data_directory, participant_id + '_realizd.csv')
        
            if os.path.exists(file_name) is True:
    
                recording_timeline = pd.DataFrame()
                
                data_df = getDataFrame(file_name)
                data_df.index = pd.to_datetime(data_df.index)
                data_df = data_df.sort_index()
                
                for index, row in data_df.iterrows():
                    frame = pd.DataFrame(index=[index.strftime(date_only_date_time_format)], columns=['date', 'start_recording_time', 'end_recording_time'])
                    frame['date'] = index.strftime(date_only_date_time_format)
                    frame['start_recording_time'] = index.strftime(date_time_format)[:-3]
                    # frame['end_recording_time'] = row['TimestampEnd']
                    frame['end_recording_time'] = (timedelta(seconds=int(row['SecondsOnPhone'])) + index).strftime(date_time_format)[:-3]
                    recording_timeline = recording_timeline.append(frame)

                if len(recording_timeline) > 0:
                    recording_timeline = recording_timeline.sort_values('start_recording_time')
                    recording_timeline.to_csv(os.path.join(recording_timeline_directory, participant_id + '.csv'), index=False)

            print('Finish processing for participant (RealizD recording timeline): ' + participant_id)


if __name__ == "__main__":
    
    DEBUG = 1
    
    if DEBUG == 0:
        """
            Parse the args:
            1. main_data_directory: directory to store keck data
            2. output_directory: main output directory
    
        """
        parser = argparse.ArgumentParser(description='Create a dataframe of worked days.')
        parser.add_argument('-i', '--main_data_directory', type=str, required=True,
                            help='Directory for data.')
        parser.add_argument('-o', '--output_directory', type=str, required=True,
                            help='Directory for output.')
        
        # stream_types = ['omsignal', 'owl_in_one', 'ground_truth']
        
        args = parser.parse_args()
    
        """
            days_at_work_directory = '../output/days_at_work'
            main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
            recording_timeline_directory = '../output/recording_timeline'
        """
    
        main_data_directory = os.path.join(os.path.expanduser(os.path.normpath(args.main_data_directory)), 'keck_wave_all/2_preprocessed_data')
        days_at_work_directory = os.path.join(os.path.expanduser(os.path.normpath(args.output_directory)), 'days_at_work')
        recording_timeline_directory = os.path.join(os.path.expanduser(os.path.normpath(args.output_directory)), 'recording_timeline')
        
        print('main_data_directory: ' + main_data_directory)
        print('days_at_work_directory: ' + days_at_work_directory)
        print('recording_timeline_directory: ' + recording_timeline_directory)
    
    else:
        days_at_work_directory = '../output/days_at_work'
        # main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
        main_data_directory = '../../data/keck_wave_all'
        recording_timeline_directory = '../output/recording_timeline'

    # stream_types = ['fitbit']
    # stream_types = ['realizd', 'tiles_app_surveys', 'omsignal', 'owl_in_one', 'ground_truth']
    # stream_types = ['owl_in_one', 'ground_truth']
    stream_types = ['owl_in_one']
    
    # participant_id_job_shift_df = getParticipantIDJobShift(main_data_directory)
    participant_id_job_shift_df = pd.DataFrame()
    
    if os.path.exists(recording_timeline_directory) is False: os.mkdir(recording_timeline_directory)
    
    for stream in stream_types:
        
        # Recording can happen anytime during data collection
        if stream == 'fitbit' or stream == 'realizd':
            if os.path.exists(os.path.join(recording_timeline_directory, stream)) is False: os.mkdir(os.path.join(recording_timeline_directory, stream))
            getAllTimeSensorRecordingTimeline(os.path.join(recording_timeline_directory, stream), os.path.join(main_data_directory, '2_raw_csv_data', stream), participant_id_job_shift_df, stream)
        
        # Recording only happens during work
        else:
            # Read days at work dataframe
            days_at_work_df = getDataFrame(os.path.join(days_at_work_directory, stream + '_days_at_work.csv'))
            
            if stream == 'omsignal' or stream == 'owl_in_one':
                if os.path.exists(os.path.join(recording_timeline_directory, stream)) is False: os.mkdir(os.path.join(recording_timeline_directory, stream))
                getWorkingOnlySensorRecordingTimeline(os.path.join(recording_timeline_directory, stream), os.path.join(main_data_directory, '2_raw_csv_data', stream), days_at_work_df, stream)
            
            elif stream == 'tiles_app_surveys':
                if os.path.exists(os.path.join(recording_timeline_directory, stream)) is False: os.mkdir(os.path.join(recording_timeline_directory, stream))
                getWorkTimeline(os.path.join(recording_timeline_directory, stream), os.path.join(main_data_directory, '2_raw_csv_data', stream), main_data_directory)

            elif stream == 'ground_truth':
                if os.path.exists(os.path.join(recording_timeline_directory, stream)) is False: os.mkdir(os.path.join(recording_timeline_directory, stream))
                getWorkTimeline(os.path.join(recording_timeline_directory, stream), os.path.join(main_data_directory, stream), main_data_directory)

    print('End extracting recording timeline for sensors or app use!')