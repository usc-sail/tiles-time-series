import os, errno
import argparse
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from load_data_basic import getParticipantIDJobShift, getParticipantInfo
from load_data_basic import getParticipantStartTime, getParticipantEndTime

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

# sleep after work duration thereshold
sleep_after_work_duration_threshold = 12
sleep_after_sleep_duration_threshold = 2


def getExpectedStartEndWorkFromRecording(shift_type, start_recording, end_recording):
    
    # index is the start recording time
    if 4 <= start_recording.hour <= 14:
        work_start_time = start_recording.replace(hour=7, minute=0, second=0, microsecond=0)
        work_end_time = start_recording.replace(hour=19, minute=0, second=0, microsecond=0)
    elif 0 <= start_recording.hour < 4:
        work_start_time = (start_recording - timedelta(days=1)).replace(hour=19, minute=0, second=0, microsecond=0)
        work_end_time = start_recording.replace(hour=7, minute=0, second=0, microsecond=0)
    elif 19 <= start_recording.hour < 24:
        work_start_time = start_recording.replace(hour=19, minute=0, second=0, microsecond=0)
        work_end_time = (start_recording + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
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
    timeline = pd.DataFrame()
    # timeline_header = ['start_recording_time', 'end_recording_time', 'is_main_sleep', 'time_in_bed', 'data_source']
    timeline_header = ['start_recording_time', 'end_recording_time',
                       'SleepMinutesAwake', 'SleepMinutesStageDeep',
                       'SleepMinutesStageLight', 'SleepMinutesStageRem',
                       'SleepMinutesStageWake', 'SleepEfficiency',
                       'data_source']
    
    if stream == 'sleep':
        try:
            if os.path.isfile(os.path.join(timeline_directory, participant_id + '_Sleep_Summary.csv')) is True:
                timeline = pd.read_csv(os.path.join(timeline_directory,
                                                    participant_id + '_Sleep_Summary.csv'))[['SleepBeginTimestamp', 'SleepEndTimestamp',
                                                                                             # 'SleepMain_Sleep', 'SleepTime_In_Bed',
                                                                                             'SleepMinutesAwake',
                                                                                             'SleepMinutesStageDeep',
                                                                                             'SleepMinutesStageLight',
                                                                                             'SleepMinutesStageRem',
                                                                                             'SleepMinutesStageWake',
                                                                                             'SleepEfficiency',
                                                                                             'data_source']]
                timeline.columns = timeline_header
                timeline['type'] = 'sleep'
                
        except ValueError:
            pass
    else:
        try:
            if os.path.isfile(os.path.join(timeline_directory, stream, participant_id + '.csv')) is True:
                timeline = pd.read_csv(os.path.join(timeline_directory, stream, participant_id + '.csv'))[timeline_header[0:2]]
                timeline['type'] = stream
                # timeline['is_main_sleep'] = np.nan
                # timeline['time_in_bed'] = np.nan
                timeline['SleepMinutesAwake'] = np.nan
                timeline['SleepMinutesStageDeep'] = np.nan
                timeline['SleepMinutesStageLight'] = np.nan
                timeline['SleepMinutesStageRem'] = np.nan
                timeline['SleepMinutesStageWake'] = np.nan
                timeline['SleepEfficiency'] = np.nan
                timeline['data_source'] = 0
                
        except ValueError:
            pass
    
    return timeline


def getFullTimeline(individual_timeline):
    
    colunms = ['date', 'start_recording_time', 'end_recording_time',
               'type', 'fitbit_unused_time_since_last_use',
               'expected_start_work_time', 'expected_end_work_time',
               'arrive_before_work', 'leave_after_work',
               'time_to_work_after_sleep', 'time_to_sleep_after_work',
               'is_sleep_before_work', 'is_sleep_after_work', 'is_sleep_after_sleep',
               # 'is_main_sleep', 'time_in_bed',
               'is_sleep_transition_before_work', 'is_sleep_transition_after_work',
               'SleepMinutesAwake', 'SleepMinutesStageDeep',
               'SleepMinutesStageLight', 'SleepMinutesStageRem',
               'SleepMinutesStageWake', 'SleepEfficiency',
               'work_status', 'duration_in_seconds',
               'job_survey_timestamp', 'health_survey_timestamp', 'personality_survey_timestamp',
               'itp_mgt', 'irb_mgt', 'ocb_mgt', 'cwb_mgt',
               'neu_mgt', 'con_mgt', 'ext_mgt', 'agr_mgt', 'ope_mgt',
               'pos_af_mgt', 'neg_af_mgt', 'anxiety_mgt', 'stress_mgt',
               'alcohol_mgt', 'tobacco_mgt', 'exercise_mgt', 'sleep_mgt',
               'interaction_mgt', 'activity_mgt', 'location_mgt', 'event_mgt', 'work_mgt']

    start_time = getParticipantStartTime()
    end_time = getParticipantEndTime()

    individual_timeline.index = pd.to_datetime(individual_timeline['start_recording_time'])
    individual_timeline = individual_timeline.drop('start_recording_time', axis=1)
    individual_timeline = individual_timeline[start_time:end_time]

    individual_timeline_df = pd.DataFrame()

    if len(individual_timeline) > 0:
    
        last_work_end_time = None
        last_sleep_df = None
        last_sleep_end_time_index = None
        
        last_fitbit_use_end_time = None
        
        # Iterate rows in timeline, add more information
        for index, row in individual_timeline.iterrows():
        
            frame = pd.DataFrame(index=[index], columns=colunms)
        
            duration = pd.to_datetime(row['end_recording_time']) - index

            # Initialize frame value to np.nan
            frame['duration_in_seconds'] = int(duration.total_seconds())
            frame['start_recording_time'] = index.strftime(date_time_format)[:-3]
            frame['end_recording_time'] = row['end_recording_time']
            frame['type'] = row['type']
            frame['shift_type'] = row['shift_type']
            frame['date'] = index.date()
            
            frame['is_sleep_after_work'] = np.nan
            frame['is_sleep_before_work'] = np.nan
            frame['is_sleep_after_sleep'] = np.nan
            frame['is_sleep_adaption'] = np.nan

            frame['is_sleep_transition_before_work'] = np.nan
            frame['is_sleep_transition_after_work'] = np.nan

            frame['SleepMinutesAwake'] = row['SleepMinutesAwake']
            frame['SleepMinutesStageDeep'] = row['SleepMinutesStageDeep']
            frame['SleepMinutesStageLight'] = row['SleepMinutesStageLight']
            frame['SleepMinutesStageRem'] = row['SleepMinutesStageRem']
            frame['SleepMinutesStageWake'] = row['SleepMinutesStageWake']
            frame['SleepEfficiency'] = row['SleepEfficiency']
            
            # row type sleep case
            # add frame['is_sleep_after_work'] and frame['is_sleep_after_sleep']
            if row['type'] == 'sleep':
                if last_work_end_time is not None:
                    if (index - last_work_end_time).total_seconds() < 3600 * sleep_after_work_duration_threshold:
                        frame['is_sleep_after_work'] = 1
                        frame['time_to_sleep_after_work'] = int((index - last_work_end_time).total_seconds())
                        
                if last_sleep_df is not None:
                    if (index - pd.to_datetime(last_sleep_df['end_recording_time'].values[0])).total_seconds() < 3600 * sleep_after_sleep_duration_threshold:
                        frame['is_sleep_after_sleep'] = 1
                    
                last_sleep_df = frame
                last_sleep_end_time_index = index

            # row type fitbit case
            # add frame['fitbit_unused_time_since_last_use']
            elif row['type'] == 'fitbit':
                if last_fitbit_use_end_time is not None:
                    frame['fitbit_unused_time_since_last_use'] = int((index - last_fitbit_use_end_time).total_seconds())
                last_fitbit_use_end_time = pd.to_datetime(row['end_recording_time'])
            
            # row type recording
            # Detect when participant is at work
            elif row['type'] == 'omsignal' or row['type'] == 'owl_in_one' or row['type'] == 'ground_truth':
                
                frame['work_status'] = 1

                if row['type'] == 'ground_truth':
                    work_start_time = index.strftime(date_time_format)[:-3]
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
                
                if last_sleep_df is not None:
                    if (index - pd.to_datetime(last_sleep_df['end_recording_time'].values[0])).total_seconds() < 3600 * sleep_after_work_duration_threshold:
                        individual_timeline_df.loc[last_sleep_end_time_index, 'is_sleep_before_work'] = 1
                        individual_timeline_df.loc[last_sleep_end_time_index, 'time_to_work_after_sleep'] = int((work_start_time - pd.to_datetime(last_sleep_df['end_recording_time'].values[0])).total_seconds())
                        
                        # If last sleep is only before work, then it is transition
                        # Assign it 'is_sleep_transition_before_work' to 1
                        if last_sleep_df['is_sleep_after_work'].values[0] != 1 and row['shift_type'] == 2:
                            individual_timeline_df.loc[last_sleep_end_time_index, 'is_sleep_transition_before_work'] = 1
                    
                last_work_end_time = work_end_time
            
            individual_timeline_df = individual_timeline_df.append(frame)
            
    return individual_timeline_df


def getSleepAdaptionTimeline(individual_timeline):
    
    individual_timeline_df = pd.DataFrame()
    last_sleep_df = None
    
    for index, timeline in individual_timeline.iterrows():
        
        if timeline['type'] == 'sleep':
    
            if timeline['is_sleep_after_work'] == 1 and timeline['is_sleep_before_work'] != 1 and timeline['shift_type'] == 2:
                timeline['is_sleep_transition_after_work'] = 1
            
            if last_sleep_df is not None:
                
                if last_sleep_df['is_sleep_before_work'] != 1 and last_sleep_df['is_sleep_after_work'] == 1 and timeline['shift_type'] == 2:
                    if (index - pd.to_datetime(last_sleep_df['end_recording_time'])).total_seconds() < 3600 * sleep_after_sleep_duration_threshold:
                        timeline['is_sleep_transition_after_work'] = 1
                
                if last_sleep_df['start_recording_time'] == timeline['start_recording_time']:
                    if (pd.to_datetime(last_sleep_df['end_recording_time']) - pd.to_datetime(timeline['end_recording_time'])).total_seconds() < 0:
                        individual_timeline_df = individual_timeline_df[individual_timeline_df['start_recording_time'] != last_sleep_df['start_recording_time']]
                        individual_timeline_df = individual_timeline_df.append(timeline)
                else:
                    individual_timeline_df = individual_timeline_df.append(timeline)
            else:
                individual_timeline_df = individual_timeline_df.append(timeline)
                
            last_sleep_df = timeline
            
        elif timeline['type'] == 'omsignal' or timeline['type'] == 'owl_in_one' or timeline['type'] == 'ground_truth':
            individual_timeline_df = individual_timeline_df.append(timeline)
            
        else:
            individual_timeline_df = individual_timeline_df.append(timeline)

    individual_timeline_df = individual_timeline_df[individual_timeline.columns]
    
    return individual_timeline_df
    

def combineDailySurvey(main_data_directory, participant_id, user_id, individual_timeline):
    
    survey_data = pd.read_csv(os.path.join(main_data_directory, 'keck_wave2/ground_truth/MGT.csv'), index_col=0)
    
    print('Read Participant Survey: ' + participant_id)

    daily_survey_job = ['itp_mgt', 'irb_mgt', 'ocb_mgt', 'cwb_mgt' ]
    daily_survey_health = ['alcohol_mgt', 'tobacco_mgt', 'exercise_mgt', 'sleep_mgt']
    daily_survey_personality = ['neu_mgt', 'con_mgt', 'ext_mgt', 'agr_mgt', 'ope_mgt']
    daily_survey_common = ['pos_af_mgt', 'neg_af_mgt', 'anxiety_mgt', 'stress_mgt',
                           'interaction_mgt', 'activity_mgt', 'location_mgt',
                           'event_mgt', 'work_mgt']
    
    individual_timeline_df = pd.DataFrame()

    if user_id in survey_data.index:
        survey_data = survey_data.loc[user_id]

        for index, timeline in individual_timeline.iterrows():
        
            if timeline['type'] == 'sleep':
                for survey_index, survey in survey_data.iterrows():
                    
                    survey_time = pd.to_datetime(survey['timestamp'])
                    start_recording_time = pd.to_datetime(timeline['start_recording_time'])
                    
                    start_diff = (survey_time - start_recording_time).total_seconds()
                    
                    if 0 < start_diff < 24 * 3600:
                        
                        if survey['survey_type'] == 'job':
                            for colunm in daily_survey_job:
                                timeline[colunm] = survey[colunm]
                            timeline['job_survey_timestamp'] = survey_time.strftime(date_time_format)[:-3]
                            
                        elif survey['survey_type'] == 'health':
                            for colunm in daily_survey_health:
                                timeline[colunm] = survey[colunm]
                            timeline['health_survey_timestamp'] = survey_time.strftime(date_time_format)[:-3]
                            
                        elif survey['survey_type'] == 'personality':
                            for colunm in daily_survey_personality:
                                timeline[colunm] = survey[colunm]
                            timeline['personality_survey_timestamp'] = survey_time.strftime(date_time_format)[:-3]

                        for colunm in daily_survey_common:
                            timeline[colunm] = survey[colunm]
                            
            individual_timeline_df = individual_timeline_df.append(timeline)
    else:
        
        for index, timeline in individual_timeline.iterrows():
            if timeline['type'] == 'sleep':
                for colunm in daily_survey_job:
                    timeline[colunm] = np.nan
                for colunm in daily_survey_health:
                    timeline[colunm] = np.nan
                for colunm in daily_survey_common:
                    timeline[colunm] = np.nan
                for colunm in daily_survey_personality:
                    timeline[colunm] = np.nan
                
                timeline['job_survey_timestamp'] = np.nan
                timeline['health_survey_timestamp'] = np.nan
                timeline['personality_survey_timestamp'] = np.nan
            
            individual_timeline_df = individual_timeline_df.append(timeline)

    individual_timeline_df = individual_timeline_df[individual_timeline.columns]
    
    return individual_timeline_df


def main(main_data_directory, recording_timeline_directory, sleep_timeline_directory, individual_timeline_directory):
    
    # stream_types = ['sleep', 'omsignal', 'owl_in_one', 'tiles_app_surveys', 'fitbit', 'realizd']
    stream_types = ['sleep', 'omsignal', 'owl_in_one', 'ground_truth', 'fitbit', 'realizd']
    # stream_types = ['sleep', 'omsignal', 'owl_in_one', 'ground_truth', 'fitbit']
    # stream_types = ['sleep', 'omsignal']
    
    # Get participant ID
    participant_info = getParticipantInfo(main_data_directory)
    participant_info = participant_info.set_index('MitreID')
    
    # Read timeline for each participant
    for user_id in participant_info.index:
        
        participant_id = participant_info.loc[user_id]['ParticipantID']
        
        if len(participant_info.loc[user_id]) > 0:
            if participant_info.loc[user_id]['Shift'] == 'Day shift':
                shift_type = 1
            else:
                shift_type = 2

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
                    
                individual_timeline['shift_type'] = 1 if participant_info.loc[user_id]['Shift'] == 'Day shift' else 2
        
        if len(individual_timeline) != 0:
            individual_timeline = individual_timeline.sort_values('start_recording_time')
            individual_timeline = getFullTimeline(individual_timeline)
            individual_timeline = getSleepAdaptionTimeline(individual_timeline)
            individual_daily_survey = combineDailySurvey(main_data_directory, participant_id, user_id, individual_timeline)
            
            if shift_type == 1:
                if os.path.exists(os.path.join(individual_timeline_directory, 'day')) is False:
                    os.mkdir(os.path.join(individual_timeline_directory, 'day'))
                individual_daily_survey.to_csv(os.path.join(individual_timeline_directory, 'day',
                                                            participant_id + '.csv'), index=False)
            elif shift_type == 2:
                if os.path.exists(os.path.join(individual_timeline_directory, 'night')) is False:
                    os.mkdir(os.path.join(individual_timeline_directory, 'night'))
                individual_daily_survey.to_csv(os.path.join(individual_timeline_directory, 'night',
                                                            participant_id + '.csv'), index=False)
            else:
                if os.path.exists(os.path.join(individual_timeline_directory, 'unknown')) is False:
                    os.mkdir(os.path.join(individual_timeline_directory, 'unknown'))
                individual_daily_survey.to_csv(os.path.join(individual_timeline_directory, 'unknown',
                                                            participant_id + '.csv'), index=False)
            
            
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
      
        # main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
        main_data_directory = '../../data'
        recording_timeline_directory = '../output/recording_timeline'
        sleep_timeline_directory = '../output/sleep_timeline'
        individual_timeline_directory = '../output/individual_timeline'
    
    print('main_data_directory: ' + main_data_directory)
    print('sleep_timeline_directory: ' + sleep_timeline_directory)
    print('recording_timeline_directory: ' + recording_timeline_directory)
    print('individual_timeline_directory: ' + individual_timeline_directory)
    
    if os.path.exists(individual_timeline_directory) is False: os.mkdir(individual_timeline_directory)

    # participant_id_job_shift_df = getParticipantIDJobShift(main_data_directory)

    main(main_data_directory, recording_timeline_directory, sleep_timeline_directory, individual_timeline_directory)