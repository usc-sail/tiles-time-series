import os, errno
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from dateutil import rrule
from datetime import datetime, timedelta

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from load_data_basic import getParticipantIDJobShift, getParticipantID
from load_data_basic import getParticipantStartTime, getParticipantEndTime


def main(main_data_directory, individual_timeline_directory):

    daily_timeline = ['work_start', 'work_end',
                      'sleep_start_main', 'sleep_end_main',
                      'is_main_sleep_before_work', 'is_main_sleep_after_work',
                      'sleep_start_sub', 'sleep_end_sub',
                      'ground_truth_start', 'ground_truth_end',
                      'omsignal_start_recording', 'omsignal_end_recording',
                      'owl_in_one_start_recording', 'owl_in_one_end_recording', 'work_status']
    
    start_time = getParticipantStartTime()
    end_time = getParticipantEndTime()

    data_collection_dates = [start_time + timedelta(i) for i in range((end_time - start_time).days + 1)]
    
    IDs = getParticipantID(main_data_directory)

    for user_id in IDs.index:
        participant_id = IDs.loc[user_id].values[0]

        print('Start Processing (Individual timeline by Date): ' + participant_id)
    
        # Read timeline data
        individual_timeline = pd.read_csv(os.path.join(individual_timeline_directory, participant_id + '.csv'), index_col=0)
        individual_timeline.index = pd.to_datetime(individual_timeline.index)
        individual_timeline = individual_timeline[start_time:end_time]
        
        individual_timeline_by_date = pd.DataFrame(np.nan, index=data_collection_dates,
                                                   columns=daily_timeline)
        
        if len(individual_timeline) > 0:
            
            last_work_end_time = None
            last_sleep_end_time = None
            last_sleep_end_time_index = None
            
            for index, row in individual_timeline.iterrows():
                
                if row['type'] == 'sleep':
                    if individual_timeline_by_date['sleep_start_main'][index.date()] != np.nan:
                        individual_timeline_by_date['sleep_start_main'][index.date()] = index.strftime(date_time_format)[:-3]
                        individual_timeline_by_date['sleep_end_main'][index.date()] = row['end_recording_time']

                        if last_work_end_time != None:
                            if (index - last_work_end_time).total_seconds() < 3600 * 8:
                                individual_timeline_by_date['is_main_sleep_after_work'][index.date()] = 1
                        
                    else:
                        duration_of_sleep = index - pd.to_datetime(row['end_recording_time'])
                        duration_of_existing_sleep = pd.to_datetime(individual_timeline_by_date['sleep_end_main'][index.date()]) - pd.to_datetime(individual_timeline_by_date['sleep_start_main'][index.date()])
                        
                        if duration_of_sleep.total_seconds() > duration_of_existing_sleep.total_seconds():
                            individual_timeline_by_date['sleep_start_sub'][index.date()] = individual_timeline_by_date['sleep_start_main'][index.date()]
                            individual_timeline_by_date['sleep_end_sub'][index.date()] = individual_timeline_by_date['sleep_end_main'][index.date()]
                            
                            individual_timeline_by_date['sleep_start_main'][index.date()] = index.strftime(date_time_format)[:-3]
                            individual_timeline_by_date['sleep_end_main'][index.date()] = row['end_recording_time']

                            if last_work_end_time != None:
                                if (index - last_work_end_time).total_seconds() < 3600 * 8:
                                    individual_timeline_by_date['is_main_sleep_after_work'][index.date()] = 1
    
                    last_sleep_end_time = pd.to_datetime(row['end_recording_time'])
                    last_sleep_end_time_index = index.date()

                elif row['type'] == 'omsignal':
                    individual_timeline_by_date['omsignal_start_recording'][index.date()] = index.strftime(date_time_format)[:-3]
                    individual_timeline_by_date['omsignal_end_recording'][index.date()] = row['end_recording_time']
                    
                elif row['type'] == 'owl_in_one':
                    individual_timeline_by_date['owl_in_one_start_recording'][index.date()] = index.strftime(date_time_format)[:-3]
                    individual_timeline_by_date['owl_in_one_end_recording'][index.date()] = row['end_recording_time']
                    individual_timeline_by_date['work_status'][index.date()] = 1
                    
                elif row['type'] == 'ground_truth':
                    individual_timeline_by_date['ground_truth_start'][index.date()] = index.strftime(date_time_format)[:-3]
                    individual_timeline_by_date['ground_truth_end'][index.date()] = row['end_recording_time']
                    
                    individual_timeline_by_date['work_start'][index.date()] = index.strftime(date_time_format)[:-3]
                    individual_timeline_by_date['work_end'][index.date()] = row['end_recording_time']
                    
                    individual_timeline_by_date['work_status'][index.date()] = 1
                    
                    if last_sleep_end_time != None:
                        if (index - last_sleep_end_time).total_seconds() < 3600 * 8:
                            individual_timeline_by_date['is_main_sleep_before_work'][last_sleep_end_time_index] = 1
                        
                    last_work_end_time = pd.to_datetime(row['end_recording_time'])
                    
                    # If it is night shift nurse, she/he also worked for next day
                    if index.hour == 19:
                        individual_timeline_by_date['work_status'][index.date() + timedelta(days=1)] = 1
                
                # If type is recording data, we need to do some conditions, when artifacts happened
                if row['type'] == 'omsignal' or row['type'] == 'owl_in_one':
                    
                    individual_timeline_by_date['work_status'][index.date()] = 1
                    
                    # recording happened during midnight, must be night shift nurse, she/he also worked for last day
                    if 0 < index.hour < 4:
                        individual_timeline_by_date['work_start'][index.date()] = (index - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
                        individual_timeline_by_date['work_end'][index.date()] = index.replace(hour=7, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
                        individual_timeline_by_date['work_status'][index - timedelta(days=1)] = 1
                    else:
                        individual_timeline_by_date['work_start'][index.date()] = index.replace(hour=7, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
                        individual_timeline_by_date['work_end'][index.date()] = index.replace(hour=19, minute=0,second=0, microsecond=0).strftime(date_time_format)[:-3]
                    
                    if last_sleep_end_time != None:
                        if (index - last_sleep_end_time).total_seconds() < 3600 * 8:
                            individual_timeline_by_date['is_main_sleep_before_work'][last_sleep_end_time_index] = 1

                    last_work_end_time = pd.to_datetime(individual_timeline_by_date['work_end'][index.date()])
                    
            # At the end, we want to see if the sleep is before or after work
            """
            for i in range(len(individual_timeline_by_date.index)):
                date = individual_timeline_by_date.index[i]
                data = individual_timeline_by_date[individual_timeline_by_date.index[i]]
                # If there is sleep data
                if len(data['sleep_end']) > 0 and len(data['work_start']) > 0:
                    if (data['work_start'] - data['sleep_start']).total_seconds() > 60 * 10:
                        individual_timeline_by_date[1][index.date()] = 1
            """
            individual_timeline_by_date.to_csv(os.path.join(individual_timeline_directory, participant_id + '_By_Date.csv'), index_label='Date')

def main_df(main_data_directory, individual_timeline_directory):
    
    colunms = ['date', 'start_recording_time', 'end_recording_time', 'type',
               'is_sleep_before_work', 'is_sleep_after_work', 'work_status',
               'duration_in_seconds']
    
    start_time = getParticipantStartTime()
    end_time = getParticipantEndTime()
    
    data_collection_dates = [start_time + timedelta(i) for i in range((end_time - start_time).days + 1)]
    
    IDs = getParticipantID(main_data_directory)
    
    for user_id in IDs.index:
        
        participant_id = IDs.loc[user_id].values[0]
        
        print('Start Processing (Individual timeline by Date): ' + participant_id)
        
        # Read timeline data
        individual_timeline = pd.read_csv(os.path.join(individual_timeline_directory, participant_id + '.csv'), index_col=0)
        individual_timeline.index = pd.to_datetime(individual_timeline.index)
        individual_timeline = individual_timeline[start_time:end_time]
        
        if len(individual_timeline) > 0:
            
            last_work_end_time = None
            last_sleep_end_time = None
            last_sleep_end_time_index = None

            individual_timeline_df = pd.DataFrame()
            
            for index, row in individual_timeline.iterrows():
    
                frame = pd.DataFrame(index=[index], columns=colunms)
    
                duration = pd.to_datetime(row['end_recording_time']) - index
                
                frame['duration_in_seconds'] = duration.total_seconds()
                frame['start_recording_time'] = index.strftime(date_time_format)[:-3]
                frame['end_recording_time'] = row['end_recording_time']
                frame['type'] = row['type']
                frame['date'] = index.date()

                if row['type'] == 'sleep':
                    if last_work_end_time != None:
                        if (index - last_work_end_time).total_seconds() < 3600 * 8:
                            frame['is_sleep_after_work'] = 1
                    
                    last_sleep_end_time = pd.to_datetime(row['end_recording_time'])
                    last_sleep_end_time_index = index
                
                elif row['type'] == 'omsignal' or row['type'] == 'owl_in_one' or row['type'] == 'ground_truth':
                    frame['work_status'] = 1
                    if last_sleep_end_time != None:
                        if (index - last_sleep_end_time).total_seconds() < 3600 * 8:
                            individual_timeline_df.loc[last_sleep_end_time_index, 'is_sleep_before_work'] = 1
                    last_work_end_time = pd.to_datetime(row['end_recording_time'])

                individual_timeline_df = individual_timeline_df.append(frame)

            individual_timeline_df.to_csv(os.path.join(individual_timeline_directory, participant_id + '_By_Date.csv'), index_label='Date')


if __name__ == "__main__":
    
    """
        Parse the args:
        1. main_data_directory: directory to store keck data
        2. sleep_timeline_directory: directory to store sleep timeline
        3. recording_timeline_directory: directory to store timeline of recording
    
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
    
    """
    main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
    recording_timeline_directory = '../output/recording_timeline'
    sleep_timeline_directory = '../output/sleep_timeline'
    individual_timeline_directory = '../output/individual_timeline'

    print('main_data_directory: ' + main_data_directory)
    print('recording_timeline_directory: ' + recording_timeline_directory)
    
    if os.path.exists(individual_timeline_directory) is False: os.mkdir(individual_timeline_directory)
    
    participant_id_job_shift_df = getParticipantIDJobShift(main_data_directory)

    main_df(main_data_directory, individual_timeline_directory)