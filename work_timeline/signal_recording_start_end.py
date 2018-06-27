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


def getParticipantIDJobShift(main_data_directory):
    participant_id_job_shift_df = []
    
    # job shift
    job_shift_df = pd.read_csv(os.path.join('../../data/keck_wave1/2_preprocessed_data/', 'job shift/Job_Shift.csv'))
    
    # read id
    id_data_df = pd.read_csv(os.path.join('../../data/keck_wave1/2_preprocessed_data/', 'ground_truth/IDs.csv'))
    
    for index, id_data in id_data_df.iterrows():
        # get job shift and participant id
        job_shift = job_shift_df.loc[job_shift_df['uid'] == id_data['user_id']]['job_shift'].values[0]
        participant_id = id_data['OMuser_id']
        
        frame_df = pd.DataFrame(job_shift, index=['job_shift'], columns=[participant_id]).transpose()
        
        participant_id_job_shift_df = frame_df if len(
            participant_id_job_shift_df) == 0 else participant_id_job_shift_df.append(frame_df)

    return participant_id_job_shift_df


def getDataFrame(file):
    
    # Read and prepare owl data per participant
    data = pd.read_csv(file, index_col=0)
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


def getRecordingTimeline(recording_timeline_directory, data_directory, days_at_work_df, stream):
    
    # Read participant id array
    participant_id_array = days_at_work_df.columns.values
    
    for participant_id in participant_id_array:
    
        print('Start processing for participant' + '(' + stream + ')' + ': ' + participant_id)
        
        # Read the data and days of work for participant
        if stream == 'owl_in_one':
            file_name = os.path.join(data_directory, participant_id + '_bleProximity.csv')
        elif stream == 'omsignal':
            file_name = os.path.join(data_directory, participant_id + '_omsignal.csv')
           
        data_df = getDataFrame(file_name)
        participant_days_at_work_df = days_at_work_df[participant_id].to_frame()
        
        # Save data frame
        recording_time_df = []
        
        for index, is_recording in participant_days_at_work_df.iterrows():
            # There is an recording
            if is_recording.values[0] == 1:
                
                # Initialize start and end date
                start_date, end_date = index, index + timedelta(days=1)
                date_at_work_data_df = data_df[start_date.strftime(date_time_format) : end_date.strftime(date_time_format)]
                datetime_at_work = pd.to_datetime(date_at_work_data_df.index)
                
                start_recording_time = datetime_at_work[0]
                last_recording_time = datetime_at_work[0]
                
                for time in datetime_at_work:
                    current_recording_time = time
                    
                    if (current_recording_time - last_recording_time).total_seconds() > 3600 * 2:
                        
                        # Construct frame and append
                        recording_time_df = constructRecordingFrame(index, recording_time_df,
                                                                    start_recording_time,
                                                                    last_recording_time)
                        start_recording_time = current_recording_time
                    last_recording_time = current_recording_time

                # Construct frame and append
                recording_time_df = constructRecordingFrame(index, recording_time_df,
                                                            start_recording_time,
                                                            last_recording_time)
        if len(recording_time_df) > 0:
            last_row = []
            save_df = []
            
            for index, row in recording_time_df.iterrows():
                if len(last_row) == 0:
                    last_row = row
                else:
                    last_end_time = datetime.strptime(last_row['end_recording_time'], date_time_format)
                    current_start_time = datetime.strptime(row['start_recording_time'], date_time_format)
                    if (current_start_time - last_end_time).total_seconds() >  3600 * 2:
                        save_df = last_row.to_frame().transpose() if len(save_df) == 0 else save_df.append(last_row.to_frame().transpose())
                        last_row = row
                    else:
                        last_row['end_recording_time'] = row['end_recording_time']

            save_df.to_csv(os.path.join(recording_timeline_directory, participant_id + '.csv'), index=False)
        print('End processing for participant: ' + participant_id)


def getWorkTimeline(recording_timeline_directory, data_directory, participant_id_job_shift_df):
    
    # We need to see when nurse is responding to the survey and based on that decide whether people worked that day
    MGT = pd.read_csv(os.path.join(data_directory, 'MGT.csv'), index_col=2)
    IDs = pd.read_csv(os.path.join(data_directory, 'IDs.csv'), index_col=1)
    IDs.columns = ['Evidation_id']
    IDs.index.names = ['MITRE_id']

    MGT.index = pd.to_datetime(MGT.index)

    participantIDs = sorted(list(IDs['Evidation_id'].unique()))

    for participant_id in participantIDs:
        
        print('Start processing for participant (ground_truth): ' + participant_id)
        
        # Get the job shift
        participant_job_shift = participant_id_job_shift_df.loc[participant_id]['job_shift']

        # Get the job shift
        user_id = IDs.loc[IDs['Evidation_id'] == participant_id].index.values[0]
        MGT_per_participant = MGT.loc[MGT['uid'] == user_id]
        
        MGT_job_per_participant = MGT_per_participant.loc[MGT_per_participant['survey_type'] == 'job']

        work_time_df = []
        
        for index, row in MGT_job_per_participant.iterrows():
            try:
                if row['location_mgt'] == 2.0:  # At work when answering the survey according to MGT, question 1
                    # Day shift nurse
                    if 17 < index.hour < 24:
                        start_work_date = index.replace(hour=7, minute=0, second=0)
                        end_work_date = index.replace(hour=19, minute=0, second=0)
                    # Night shift nurse
                    else:
                        start_work_date = index.replace(hour=19, minute=0, second=0)
                        end_work_date = (index + timedelta(days=1)).replace(hour=7, minute=0, second=0)

                    work_time_df = constructRecordingFrame(index, work_time_df, start_work_date, end_work_date)
                    
            except KeyError:
                print('Participant ' + row['uid'] + ' is not in participant list from IDs.csv.')
            
        if len(work_time_df) > 0:
            work_time_df = work_time_df.sort_values('date')
            work_time_df.to_csv(os.path.join(recording_timeline_directory, participant_id + '.csv'), index=False)
        print('End processing for participant: ' + participant_id)


if __name__ == "__main__":
    
    """
        Parse the args:
        1. main_data_directory: directory to store keck data
        2. days_at_work_directory: directory to store days at work data using different modalities
        3. recording_timeline_directory: directory to store timeline of recording
    """
    parser = argparse.ArgumentParser(description='Create a dataframe of worked days.')
    parser.add_argument('-i', '--main_data_directory', type=str, required=True,
                        help='Directory for data.')
    parser.add_argument('-d', '--days_at_work_directory', type=str, required=True,
                        help='Directory with days at work.')
    parser.add_argument('-r', '--recording_timeline_directory', type=str, required=True,
                        help='Directory with recording timeline.')
    
    stream_types = ['omsignal', 'owl_in_one', 'ground_truth']
    
    """
    args = parser.parse_args()

    main_data_directory = os.path.expanduser(os.path.normpath(args.main_data_directory))
    days_at_work_directory = os.path.expanduser(os.path.normpath(args.days_at_work_directory))
    recording_timeline_directory = os.path.expanduser(os.path.normpath(args.recording_timeline_directory))
    
    print('main_data_directory: ' + main_data_directory)
    print('days_at_work_directory: ' + days_at_work_directory)
    print('recording_timeline_directory: ' + recording_timeline_directory)
    """
    
    days_at_work_directory = '../output/days_at_work'
    main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
    recording_timeline_directory = '../output/recording_timeline'

    participant_id_job_shift_df = getParticipantIDJobShift(main_data_directory)
    
    if os.path.exists(recording_timeline_directory) is False: os.mkdir(recording_timeline_directory)
    
    for stream in stream_types:
        
        # Read days at work dataframe
        days_at_work_df = getDataFrame(os.path.join(days_at_work_directory, stream + '_days_at_work.csv'))
        
        if stream == 'omsignal' or stream == 'owl_in_one':
            if os.path.exists(os.path.join(recording_timeline_directory, stream)) is False: os.mkdir(os.path.join(recording_timeline_directory, stream))
            getRecordingTimeline(os.path.join(recording_timeline_directory, stream), os.path.join(main_data_directory, stream), days_at_work_df, stream)
        
        elif stream == 'ground_truth':
            if os.path.exists(os.path.join(recording_timeline_directory, stream)) is False: os.mkdir(os.path.join(recording_timeline_directory, stream))
            getWorkTimeline(os.path.join(recording_timeline_directory, stream), os.path.join(main_data_directory, stream), participant_id_job_shift_df)