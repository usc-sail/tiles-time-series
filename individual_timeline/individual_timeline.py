import os, errno
import argparse
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from load_data_basic import getParticipantIDJobShift, getParticipantID
from load_data_basic import getParticipantStartTime, getParticipantEndTime

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

# sleep after work duration thereshold
sleep_after_work_duration_threshold = 12


def getExpectedStartEndWorkFromRecording(shift_type, start_recording, end_recording):
    
    # index is the start recording time
    if 4 <= start_recording.hour <= 14 and shift_type == 1:
        work_start_time = start_recording.replace(hour=7, minute=0, second=0, microsecond=0)
        work_end_time = start_recording.replace(hour=19, minute=0, second=0, microsecond=0)
    elif 0 <= start_recording.hour < 4 and shift_type == 0:
        work_start_time = (start_recording - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
        work_end_time = start_recording.replace(hour=7, minute=0, second=0, microsecond=0)
    elif shift_type == 0:
        work_start_time = start_recording.replace(hour=19, minute=0, second=0, microsecond=0)
        work_end_time = (start_recording + timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
    else:
        if 15 < end_recording.hour <= 22:
            work_start_time = end_recording.replace(hour=7, minute=0, second=0, microsecond=0)
            work_end_time = end_recording.replace(hour=19, minute=0, second=0, microsecond=0)
        elif 0 < end_recording.hour <= 10:
            work_start_time = (end_recording - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
            work_end_time = end_recording.replace(hour=7, minute=0, second=0, microsecond=0)
        else:
            work_start_time = start_recording
            work_end_time = end_recording
    
    return work_start_time.strftime(date_time_format)[:-3], work_end_time.strftime(date_time_format)[:-3]


def getTimelineDataFrame(timeline_directory, participant_id, stream):
    timeline = []
    timeline_header = ['start_recording_time', 'end_recording_time', 'data_source']
    
    if stream == 'sleep':
        try:
            if os.path.isfile(os.path.join(timeline_directory, participant_id + '_Sleep_Summary.csv')) is True:
                timeline = pd.read_csv(os.path.join(timeline_directory, participant_id + '_Sleep_Summary.csv'))[['SleepBeginTimestamp', 'SleepEndTimestamp', 'data_source']]
                timeline.columns = timeline_header
                timeline['type'] = 'sleep'
                
        except ValueError:
            pass
    else:
        try:
            if os.path.isfile(os.path.join(timeline_directory, stream, participant_id + '.csv')) is True:
                timeline = pd.read_csv(os.path.join(timeline_directory, stream, participant_id + '.csv'))[timeline_header[0:2]]
                timeline['type'] = stream
                timeline['data_source'] = 0
        except ValueError:
            pass
    
    return timeline


def getFullTimeline(individual_timeline):
    
    colunms = ['date', 'start_recording_time', 'end_recording_time',
               'type', 'data_source',
               'expected_start_work_time', 'expected_end_work_time',
               'arrive_before_work', 'leave_after_work',
               'time_to_work_after_sleep', 'time_to_sleep_after_work',
               'is_sleep_before_work', 'is_sleep_after_work', 'work_status',
               'duration_in_seconds']

    start_time = getParticipantStartTime()
    end_time = getParticipantEndTime()

    individual_timeline.index = pd.to_datetime(individual_timeline['start_recording_time'])
    individual_timeline = individual_timeline.drop('start_recording_time', axis=1)
    # individual_timeline = individual_timeline.drop(columns=['start_recording_time'])
    individual_timeline = individual_timeline[start_time:end_time]

    individual_timeline_df = pd.DataFrame()

    if len(individual_timeline) > 0:
    
        last_work_end_time = None
        last_sleep_end_time = None
        last_sleep_end_time_index = None
        
        # Iterate rows in timeline, add more information
        for index, row in individual_timeline.iterrows():
        
            frame = pd.DataFrame(index=[index], columns=colunms)
        
            duration = pd.to_datetime(row['end_recording_time']) - index
        
            frame['duration_in_seconds'] = int(duration.total_seconds())
            frame['start_recording_time'] = index.strftime(date_time_format)[:-3]
            frame['end_recording_time'] = row['end_recording_time']
            frame['type'] = row['type']
            frame['shift_type'] = row['shift_type']
            frame['date'] = index.date()
            frame['data_source'] = row['data_source']
            
            # row type sleep
            if row['type'] == 'sleep':
                if last_work_end_time is not None:
                    if (index - last_work_end_time).total_seconds() < 3600 * sleep_after_work_duration_threshold:
                        frame['is_sleep_after_work'] = 1
                        frame['time_to_sleep_after_work'] = (index - last_work_end_time).total_seconds() / 3600
            
                last_sleep_end_time = pd.to_datetime(row['end_recording_time'])
                last_sleep_end_time_index = index
                
            # row type recording
            elif row['type'] == 'omsignal' or row['type'] == 'owl_in_one' or row['type'] == 'ground_truth':
                
                frame['work_status'] = 1

                if row['type'] == 'ground_truth':
                    work_start_time = index.strftime(date_time_format)
                    work_end_time = row['end_recording_time']
                    
                else:
                    work_start_time, work_end_time = getExpectedStartEndWorkFromRecording(row['shift_type'], index, pd.to_datetime(row['end_recording_time']))

                frame['expected_start_work_time'] = work_start_time
                frame['expected_end_work_time'] = work_end_time

                work_start_time = pd.to_datetime(work_start_time)
                work_end_time = pd.to_datetime(work_end_time)
                
                if row['type'] == 'owl_in_one':
                    if (work_start_time - index).total_seconds() > 0:
                        frame['arrive_before_work'] = int((work_start_time - index).total_seconds())
                    if (pd.to_datetime(row['end_recording_time']) - work_end_time).total_seconds() > 0:
                        frame['leave_after_work'] = int((pd.to_datetime(row['end_recording_time']) - work_end_time).total_seconds())

                
                if last_sleep_end_time is not None:
                    if (index - last_sleep_end_time).total_seconds() < 3600 * sleep_after_work_duration_threshold:
                        individual_timeline_df.loc[last_sleep_end_time_index, 'is_sleep_before_work'] = 1
                        individual_timeline_df.loc[last_sleep_end_time_index, 'time_to_work_after_sleep'] = (work_start_time - last_sleep_end_time).total_seconds() / 3600
                    
                    last_work_end_time = work_end_time
            
            individual_timeline_df = individual_timeline_df.append(frame)
            
    return individual_timeline_df


def main(main_data_directory, recording_timeline_directory, sleep_timeline_directory, individual_timeline_directory):
    
    stream_types = ['sleep', 'omsignal', 'owl_in_one', 'ground_truth']

    job_shift = getParticipantIDJobShift(main_data_directory)
    
    IDs = getParticipantID(main_data_directory)
    
    for user_id in IDs.index:
        participant_id = IDs.loc[user_id].values[0]
        
        print('Start Processing (Individual timeline): ' + participant_id)

        individual_timeline = []
        for stream in stream_types:
            if stream == 'sleep':
                timeline = getTimelineDataFrame(sleep_timeline_directory, participant_id, stream)
            else:
                timeline = getTimelineDataFrame(recording_timeline_directory, participant_id, stream)
            
            if len(timeline) != 0:
                if len(individual_timeline) == 0:
                    individual_timeline = timeline
                else:
                    individual_timeline = individual_timeline.append(timeline, ignore_index=True)
        individual_timeline['shift_type'] = job_shift.loc[participant_id].values[0]
        
        if len(individual_timeline) != 0:
            individual_timeline = individual_timeline.sort_values('start_recording_time')
            individual_timeline = getFullTimeline(individual_timeline)
            individual_timeline.to_csv(os.path.join(individual_timeline_directory, participant_id + '.csv'), index=False)
            

if __name__ == "__main__":
    
    DEBUG = 1
    
    if DEBUG == 0:
        """
            Parse the args:
            1. main_data_directory: directory to store keck data
            2. sleep_timeline_directory: directory to store sleep timeline
            3. recording_timeline_directory: directory to store timeline of recording
        """
        parser = argparse.ArgumentParser(description='Create a dataframe of worked days.')
        parser.add_argument('-i', '--main_data_directory', type=str, required=True,
                            help='Directory for data.')
        parser.add_argument('-s', '--sleep_timeline_directory', type=str, required=True,
                            help='Directory with days at work.')
        parser.add_argument('-r', '--recording_timeline_directory', type=str, required=True,
                            help='Directory with recording timeline.')
        parser.add_argument('-t', '--individual_timeline_directory', type=str, required=True,
                            help='Directory with recording timeline.')
        
        args = parser.parse_args()
    
        main_data_directory = os.path.expanduser(os.path.normpath(args.main_data_directory))
        sleep_timeline_directory = os.path.expanduser(os.path.normpath(args.sleep_timeline_directory))
        recording_timeline_directory = os.path.expanduser(os.path.normpath(args.recording_timeline_directory))
        individual_timeline_directory = os.path.expanduser(os.path.normpath(args.individual_timeline_directory))
    
    else:
      
        main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
        recording_timeline_directory = '../output/recording_timeline'
        sleep_timeline_directory = '../output/sleep_timeline'
        individual_timeline_directory = '../output/individual_timeline'
    

    print('main_data_directory: ' + main_data_directory)
    print('sleep_timeline_directory: ' + sleep_timeline_directory)
    print('recording_timeline_directory: ' + recording_timeline_directory)
    print('individual_timeline_directory: ' + individual_timeline_directory)
    
    if os.path.exists(individual_timeline_directory) is False: os.mkdir(individual_timeline_directory)

    participant_id_job_shift_df = getParticipantIDJobShift(main_data_directory)

    main(main_data_directory, recording_timeline_directory, sleep_timeline_directory, individual_timeline_directory)