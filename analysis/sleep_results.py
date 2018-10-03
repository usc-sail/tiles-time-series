import os, errno
import argparse
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from scipy.stats import kurtosis
from scipy.stats.mstats import moment
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from load_data_basic import getParticipantIDJobShift, getParticipantInfo, read_MGT, read_pre_study_info
from load_data_basic import getParticipantStartTime, getParticipantEndTime

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

# sleep after work duration thereshold
sleep_after_work_duration_threshold = 12
sleep_after_sleep_duration_threshold = 2

# sleep duration minute
sleep_duration_max_in_minute = 840
before_sleep_duration_max_in_minute = 120
after_sleep_duration_max_in_minute  = 120
minute_offset = 5

output_colunms = [ 'participant_id', 'sleep_transition_number', 'night_shift_type',
                   'nurse_years', 'general_health', 'life_satisfaction',
                   'nap_proxy', 'switcher_sleep', 'night_stay', 'incomplete_shifter',
                   'date', 'start_recording_time', 'end_recording_time',
                   'sleep_transition', 'type', 'shift_type',
                   'is_sleep_before_work', 'is_sleep_after_work', 'is_sleep_after_sleep',
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
                   'interaction_mgt', 'activity_mgt', 'location_mgt', 'event_mgt', 'work_mgt',
                   'sleep_heart_rate_max', 'sleep_heart_rate_min',
                   'sleep_heart_rate_mean', 'sleep_heart_rate_std',
                   'sleep_heart_rate_percentile_10', 'sleep_heart_rate_percentile_90',
                   'sleep_heart_rate_kurtosis', 'sleep_heart_rate_moment',
                   'before_sleep_heart_rate_max', 'before_sleep_heart_rate_min',
                   'before_sleep_heart_rate_mean', 'before_sleep_heart_rate_std',
                   'before_sleep_heart_rate_percentile_10', 'before_sleep_heart_rate_percentile_90',
                   'before_sleep_heart_rate_kurtosis', 'before_sleep_heart_rate_moment',
                   'after_sleep_heart_rate_max', 'after_sleep_heart_rate_min',
                   'after_sleep_heart_rate_mean', 'after_sleep_heart_rate_std',
                   'after_sleep_heart_rate_percentile_10', 'after_sleep_heart_rate_percentile_90',
                   'after_sleep_heart_rate_kurtosis', 'after_sleep_heart_rate_moment']


def getDataFrame(file):
    # Read and prepare owl data per participant
    data = pd.read_csv(file, index_col=0)
    data = data.drop(data.index[len(data) - 1])
    data.index = pd.to_datetime(data.index)
    
    return data


def get_sleep_survey_data(individual_timeline_directory):
    
    colunms = ['date', 'start_recording_time', 'end_recording_time', 'type', 'shift_type',
               'is_sleep_before_work', 'is_sleep_after_work', 'is_sleep_after_sleep',
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

    output_colunms = ['participant_id', 'sleep_transition_number', 'night_shift_type',
                      'nurse_years', 'general_health', 'life_satisfaction',
                      'nap_proxy', 'switcher_sleep', 'night_stay', 'incomplete_shifter',
                      'date', 'start_recording_time', 'end_recording_time',
                      'sleep_transition', 'type', 'shift_type',
                      'is_sleep_before_work', 'is_sleep_after_work', 'is_sleep_after_sleep',
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
                      'interaction_mgt', 'activity_mgt', 'location_mgt', 'event_mgt', 'work_mgt',
                      'sleep_heart_rate_max', 'sleep_heart_rate_min',
                      'sleep_heart_rate_mean', 'sleep_heart_rate_std',
                      'sleep_heart_rate_percentile_10', 'sleep_heart_rate_percentile_90',
                      'sleep_heart_rate_kurtosis', 'sleep_heart_rate_moment',
                      'before_sleep_heart_rate_max', 'before_sleep_heart_rate_min',
                      'before_sleep_heart_rate_mean', 'before_sleep_heart_rate_std',
                      'before_sleep_heart_rate_percentile_10', 'before_sleep_heart_rate_percentile_90',
                      'before_sleep_heart_rate_kurtosis', 'before_sleep_heart_rate_moment',
                      'after_sleep_heart_rate_max', 'after_sleep_heart_rate_min',
                      'after_sleep_heart_rate_mean', 'after_sleep_heart_rate_std',
                      'after_sleep_heart_rate_percentile_10', 'after_sleep_heart_rate_percentile_90',
                      'after_sleep_heart_rate_kurtosis', 'after_sleep_heart_rate_moment']

    shift_type = ['day', 'night', 'unknown']
    
    # Get participant ID
    # IDs = getParticipantID(main_data_directory)
    participant_info = getParticipantInfo(main_data_directory)
    participant_info = participant_info.set_index('MitreID')
    
    # Read Pre-Study info
    PreStudyInfo = read_pre_study_info(main_data_directory)
    
    final_df = pd.DataFrame()
    
    # Heart rate data
    # Assume the duration is 600 minute maximum
    heart_rate_df = pd.DataFrame()

    # Append colunms
    for i in range(sleep_duration_max_in_minute):
        output_colunms.append(str('minute_' + str(i)))
        
    for i in range(before_sleep_duration_max_in_minute):
        output_colunms.append(str('before_minute_' + str(i)))

    for i in range(after_sleep_duration_max_in_minute):
        output_colunms.append(str('after_minute_' + str(i)))

    # Read final df
    if os.path.exists(os.path.join('output', 'sleep_survey_full.csv')) is True:
        final_df = pd.read_csv(os.path.join('output', 'sleep_survey_full.csv'))

    else:
        # Read timeline for each participant
        for user_id in participant_info.index:
            
            participant_id = participant_info.loc[user_id]['ParticipantID']
            shift = 1 if participant_info.loc[user_id]['Shift'] == 'Day shift' else 2
            wave_number = participant_info.loc[user_id]['Wave']

            user_pre_study_info = PreStudyInfo.loc[PreStudyInfo['uid'] == user_id]
            
            if wave_number != 3:
                print('Start Processing (Individual timeline): ' + participant_id)
    
                file_name = os.path.join(main_data_directory, 'keck_wave2/3_preprocessed_data/fitbit', participant_id + '_heartRate.csv')
                fitbit_data_df = pd.DataFrame()
                
                if os.path.exists(file_name) is True:
                    fitbit_data_df = getDataFrame(file_name)
                    fitbit_data_df.index = pd.to_datetime(fitbit_data_df.index)
                    fitbit_data_df = fitbit_data_df.sort_index()
                    
                
                timeline = pd.read_csv(os.path.join(individual_timeline_directory, shift_type[shift-1], participant_id + '.csv'))
                timeline = timeline.loc[timeline['type'] == 'sleep']
                
                if len(timeline) > 0:
        
                    timeline_df = pd.DataFrame()
        
                    timeline = timeline[colunms]
                    timeline['participant_id'] = participant_id
                    
                    timeline['sleep_transition'] = np.nan
                    timeline['sleep_heart_rate_max'] = np.nan
                    timeline['sleep_heart_rate_min'] = np.nan
                    timeline['sleep_heart_rate_mean'] = np.nan
                    timeline['sleep_heart_rate_std'] = np.nan
                    timeline['sleep_heart_rate_percentile_10'] = np.nan
                    timeline['sleep_heart_rate_percentile_90'] = np.nan
                    timeline['sleep_heart_rate_kurtosis'] = np.nan
                    timeline['sleep_heart_rate_moment'] = np.nan
    
                    timeline['before_sleep_heart_rate_max'] = np.nan
                    timeline['before_sleep_heart_rate_min'] = np.nan
                    timeline['before_sleep_heart_rate_mean'] = np.nan
                    timeline['before_sleep_heart_rate_std'] = np.nan
                    timeline['before_sleep_heart_rate_percentile_10'] = np.nan
                    timeline['before_sleep_heart_rate_percentile_90'] = np.nan
                    timeline['before_sleep_heart_rate_kurtosis'] = np.nan
                    timeline['before_sleep_heart_rate_moment'] = np.nan
    
                    timeline['after_sleep_heart_rate_max'] = np.nan
                    timeline['after_sleep_heart_rate_min'] = np.nan
                    timeline['after_sleep_heart_rate_mean'] = np.nan
                    timeline['after_sleep_heart_rate_std'] = np.nan
                    timeline['after_sleep_heart_rate_percentile_10'] = np.nan
                    timeline['after_sleep_heart_rate_percentile_90'] = np.nan
                    timeline['after_sleep_heart_rate_kurtosis'] = np.nan
                    timeline['after_sleep_heart_rate_moment'] = np.nan
    
                    timeline['sleep_transition'] = np.nan
                    timeline['night_shift_type'] = np.nan
                    timeline['sleep_transition_number'] = np.nan
                    timeline['nap_proxy'] = np.nan
                    timeline['switcher_sleep'] = np.nan
                    timeline['night_stay'] = np.nan
                    timeline['incomplete_shifter'] = np.nan
                    
                    timeline['nurse_years'] = user_pre_study_info['nurse_years_pre-study'].values[0] if len(user_pre_study_info['nurse_years_pre-study']) > 0 else np.nan
                    timeline['general_health'] = user_pre_study_info['general_health_pre-study'].values[0] if len(user_pre_study_info['general_health_pre-study']) > 0 else np.nan
                    timeline['life_satisfaction'] = user_pre_study_info['life_satisfaction_pre-study'].values[0] if len(user_pre_study_info['life_satisfaction_pre-study']) > 0 else np.nan
    
                    for i in range(sleep_duration_max_in_minute):
                        timeline['minute_' + str(i)] = np.nan
                        
                    if len(fitbit_data_df) > 0:
                        for index, row in timeline.iterrows():
                            
                            print('Process: ' + pd.to_datetime(row['start_recording_time']).strftime(date_only_date_time_format))
                            
                            # Initialize start and end recording time
                            start_recording_time = row['start_recording_time']
                            end_recording_time = row['end_recording_time']
    
                            # decide which type of switch the participant is
                            # 3: Switcher sleep
                            # 5: Incomplete switcher
                            # 2: Nap Proxy
                            # 4: No Sleep
                            if row['shift_type'] == 2 and row['is_sleep_transition_before_work'] == 1:
                                start_recording_datetime = pd.to_datetime(row['start_recording_time'])
                                end_recording_datetime = pd.to_datetime(row['end_recording_time'])
        
                                if 2 <= start_recording_datetime.hour <= 18:
                                    if (end_recording_datetime - start_recording_datetime).total_seconds() > 3600 * 6:
                                        row['sleep_transition'] = 5
                                    else:
                                        row['sleep_transition'] = 2
                                elif start_recording_datetime.hour <= 1 or start_recording_datetime.hour >= 20:
                                    if (end_recording_datetime - start_recording_datetime).total_seconds() > 3600 * 6:
                                        row['sleep_transition'] = 3
                                    else:
                                        row['sleep_transition'] = 4
                                else:
                                    row['sleep_transition'] = 4
    
                            # Fitbit stat during the sleep
                            # heart rate: mean, min, max, std, etc.
                            fitbit_data_df['HeartRate'] = fitbit_data_df['HeartRate'].apply(pd.to_numeric)
                            fitbit_row_df = pd.DataFrame()
                            fitbit_row_df = fitbit_data_df[start_recording_time:end_recording_time]
    
                            end_recording_time_after_sleep = (pd.to_datetime(end_recording_time) + timedelta(hours=2)).strftime(date_time_format)[:-3]
                            start_recording_time_before_sleep = (pd.to_datetime(start_recording_time) - timedelta(hours=2)).strftime(date_time_format)[:-3]
    
                            fitbit_row_before_sleep_df = fitbit_data_df[start_recording_time_before_sleep:start_recording_time]
                            fitbit_row_after_sleep_df  = fitbit_data_df[end_recording_time:end_recording_time_after_sleep]
                            
                            if len(fitbit_row_df) > 150:
                                row['sleep_heart_rate_max'] = np.max(np.array(fitbit_row_df, dtype=int))
                                row['sleep_heart_rate_min'] = np.min(np.array(fitbit_row_df, dtype=int))
                                row['sleep_heart_rate_mean'] = np.mean(np.array(fitbit_row_df, dtype=int))
                                row['sleep_heart_rate_std'] = np.std(np.array(fitbit_row_df, dtype=int))
                                row['sleep_heart_rate_percentile_10'] = np.percentile(np.array(fitbit_row_df, dtype=int), 10)
                                row['sleep_heart_rate_percentile_90'] = np.percentile(np.array(fitbit_row_df, dtype=int), 90)
                                row['sleep_heart_rate_kurtosis'] = kurtosis(np.array(fitbit_row_df, dtype=int))[0]
                                row['sleep_heart_rate_moment'] = moment(np.array(fitbit_row_df, dtype=int))[0]
                                
                                # We want to get minute level HR as well
                                # heart_rate_frame_df = pd.DataFrame(columns=heart_rate_df_col)
                                start_time = pd.to_datetime(start_recording_time).replace(second=0)
                                
                                for i in range(0, sleep_duration_max_in_minute, minute_offset):
                                    minute_start = pd.to_datetime(start_time) + timedelta(minutes=i)
                                    minute_end = pd.to_datetime(start_time) + timedelta(minutes=i+1)
                                    heart_rate_minute_data = pd.DataFrame()
                                    
                                    if (pd.to_datetime(end_recording_time) - minute_end).total_seconds() > 0:
                                        heart_rate_minute_data = fitbit_data_df[minute_start:minute_end]
                                    
                                    if len(heart_rate_minute_data) > 0:
                                        heart_rate_minute_data = heart_rate_minute_data.loc[(heart_rate_minute_data['HeartRate'] <= 180) & (heart_rate_minute_data['HeartRate'] >= 30)]
                                    
                                    row['minute_' + str(i)] = np.mean(fitbit_data_df[minute_start:minute_end].values) if len(heart_rate_minute_data) > 0 else np.nan
                            
                            # Fitbit data before sleep
                            if len(fitbit_row_before_sleep_df) > 100:
                                row['before_sleep_heart_rate_max'] = np.max(np.array(fitbit_row_before_sleep_df))
                                row['before_sleep_heart_rate_min'] = np.min(np.array(fitbit_row_before_sleep_df))
                                row['before_sleep_heart_rate_mean'] = np.mean(np.array(fitbit_row_before_sleep_df))
                                row['before_sleep_heart_rate_std'] = np.std(np.array(fitbit_row_before_sleep_df))
                                row['before_sleep_heart_rate_percentile_10'] = np.percentile(np.array(fitbit_row_before_sleep_df), 10)
                                row['before_sleep_heart_rate_percentile_90'] = np.percentile(np.array(fitbit_row_before_sleep_df), 90)
                                row['before_sleep_heart_rate_kurtosis'] = kurtosis(np.array(fitbit_row_before_sleep_df))[0]
                                row['before_sleep_heart_rate_moment'] = moment(np.array(fitbit_row_before_sleep_df), moment=1)[0]
    
                                # We want to get minute level HR as well
                                # heart_rate_frame_df = pd.DataFrame(columns=heart_rate_df_col)
                                start_time = pd.to_datetime(start_recording_time_before_sleep).replace(second=0)
    
                                for i in range(before_sleep_duration_max_in_minute):
                                    minute_start = pd.to_datetime(start_time) + timedelta(minutes=i)
                                    minute_end = pd.to_datetime(start_time) + timedelta(minutes=i+1)
                                    heart_rate_minute_data = pd.DataFrame()
        
                                    if (pd.to_datetime(start_recording_time) - minute_end).total_seconds() > 0:
                                        heart_rate_minute_data = fitbit_row_before_sleep_df[minute_start:minute_end]
        
                                    if len(heart_rate_minute_data) > 0:
                                        heart_rate_minute_data = heart_rate_minute_data.loc[(heart_rate_minute_data['HeartRate'] <= 180) & (heart_rate_minute_data['HeartRate'] >= 30)]
        
                                    row['before_minute_' + str(i)] = np.mean(heart_rate_minute_data[minute_start:minute_end].values) if len(heart_rate_minute_data) > 0 else np.nan
    
                            if len(fitbit_row_after_sleep_df) > 100:
                                row['after_sleep_heart_rate_max'] = np.max(np.array(fitbit_row_after_sleep_df))
                                row['after_sleep_heart_rate_min'] = np.min(np.array(fitbit_row_after_sleep_df))
                                row['after_sleep_heart_rate_mean'] = np.mean(np.array(fitbit_row_after_sleep_df))
                                row['after_sleep_heart_rate_std'] = np.std(np.array(fitbit_row_after_sleep_df))
                                row['after_sleep_heart_rate_percentile_10'] = np.percentile(np.array(fitbit_row_after_sleep_df), 10)
                                row['after_sleep_heart_rate_percentile_90'] = np.percentile(np.array(fitbit_row_after_sleep_df), 90)
                                row['after_sleep_heart_rate_kurtosis'] = kurtosis(np.array(fitbit_row_after_sleep_df))[0]
                                row['after_sleep_heart_rate_moment'] = moment(np.array(fitbit_row_after_sleep_df), moment=1)[0]
    
                                # We want to get minute level HR as well
                                # heart_rate_frame_df = pd.DataFrame(columns=heart_rate_df_col)
                                start_time = pd.to_datetime(end_recording_time).replace(second=0)
    
                                for i in range(after_sleep_duration_max_in_minute):
                                    minute_start = pd.to_datetime(start_time) + timedelta(minutes=i)
                                    minute_end = pd.to_datetime(start_time) + timedelta(minutes=i+1)
                                    heart_rate_minute_data = pd.DataFrame()
        
                                    if (pd.to_datetime(end_recording_time_after_sleep) - minute_end).total_seconds() > 0:
                                        heart_rate_minute_data = fitbit_row_after_sleep_df[minute_start:minute_end]
        
                                    if len(heart_rate_minute_data) > 0:
                                        heart_rate_minute_data = heart_rate_minute_data.loc[(heart_rate_minute_data['HeartRate'] <= 180) & (heart_rate_minute_data['HeartRate'] >= 30)]
        
                                    row['after_minute_' + str(i)] = np.mean(heart_rate_minute_data[minute_start:minute_end].values) if len(heart_rate_minute_data) > 0 else np.nan
    
                            timeline_df = timeline_df.append(row)
                    else:
                        timeline_df = timeline
                    
                    if len(timeline_df) > 0:
                        number_of_transition = len(timeline_df.loc[timeline_df['sleep_transition'] > 0])
                        if number_of_transition > 0:
                            timeline_df['sleep_transition_number'] = number_of_transition
                            
                            night_shift_distribution = [len(timeline_df.loc[timeline_df['sleep_transition'] == 2]),
                                                        len(timeline_df.loc[timeline_df['sleep_transition'] == 3]),
                                                        len(timeline_df.loc[timeline_df['sleep_transition'] == 4]),
                                                        len(timeline_df.loc[timeline_df['sleep_transition'] == 5])]
                            
                            timeline_df['nap_proxy'] = night_shift_distribution[0]
                            timeline_df['switcher_sleep'] = night_shift_distribution[1]
                            timeline_df['night_stay'] = night_shift_distribution[2]
                            timeline_df['incomplete_shifter'] = night_shift_distribution[3]
                            timeline_df['night_shift_type'] = np.argmax(night_shift_distribution) + 2
                            
                            timeline_df = timeline_df[output_colunms]
                        
                    final_df = final_df.append(timeline_df)
    
                final_df = final_df[output_colunms]
                final_df.to_csv(os.path.join('../output', 'sleep_survey_full.csv'), index=False)
                
        # final_df = final_df[output_colunms]
        # final_df.to_csv(os.path.join('output', 'sleep_survey_full.csv'), index=False)
    
    return final_df
    
    
def construct_year_frame(data, index=''):
    
    frame = pd.DataFrame(index=[index])

    frame['number_of_sleep'] = len(data)
    frame['duration_mean'] = np.mean(np.array(data['duration_in_seconds']))
    frame['duration_std'] = np.std(np.array(data['duration_in_seconds']))
    frame['sleep_efficiency_mean'] = np.mean(np.array(data['SleepEfficiency']))
    frame['sleep_efficiency_std'] = np.std(np.array(data['SleepEfficiency']))
    frame['sleep_heart_rate_mean_mean'] = np.mean(data['sleep_heart_rate_mean'])
    frame['sleep_heart_rate_mean_std'] = np.std(data['sleep_heart_rate_mean'])

    frame['sleep_awake_mean'] = np.mean(data['SleepMinutesAwake'])
    frame['sleep_awake_std'] = np.std(data['SleepMinutesAwake'])
    frame['sleep_light_mean'] = np.mean(data['SleepMinutesStageLight'])
    frame['sleep_light_std'] = np.std(data['SleepMinutesStageLight'])
    frame['sleep_deep_mean'] = np.mean(data['SleepMinutesStageDeep'])
    frame['sleep_deep_std'] = np.std(data['SleepMinutesStageDeep'])
    frame['sleep_rem_mean'] = np.mean(data['SleepMinutesStageRem'])
    frame['sleep_rem_std'] = np.std(data['SleepMinutesStageRem'])
    
    return frame


if __name__ == "__main__":
    
    DEBUG = 1
    
    ANALYSIS = 'health'
    
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
        
        main_data_directory = '../../data/'
        recording_timeline_directory = '../output/recording_timeline'
        sleep_timeline_directory = '../output/sleep_timeline'
        individual_timeline_directory = '../output/individual_timeline'
    
    print('main_data_directory: ' + main_data_directory)
    print('sleep_timeline_directory: ' + sleep_timeline_directory)
    print('recording_timeline_directory: ' + recording_timeline_directory)
    print('individual_timeline_directory: ' + individual_timeline_directory)

    # Read sleep data
    data = get_sleep_survey_data(individual_timeline_directory)

    # Read some basic information
    # Read participant info
    participant_info = getParticipantInfo(main_data_directory)
    participant_info = participant_info.set_index('MitreID')
    
    # Read MGT
    MGT = read_MGT(main_data_directory)
    # Read Pre-Study info
    PreStudyInfo = read_pre_study_info(main_data_directory)

    # Day data
    day_data = data.loc[data['shift_type'] == 1]
    day_workday_data = day_data.loc[(day_data['is_sleep_before_work'] == 1) & (day_data['is_sleep_after_work'] == 1)]
    day_off_day_data = day_data.loc[(day_data['is_sleep_before_work'] != 1) & (day_data['is_sleep_after_work'] != 1)]
    day_transition_day_data = day_data.loc[(day_data['is_sleep_transition_before_work'] == 1) | (day_data['is_sleep_transition_after_work'] == 1)]

    # Night data
    night_data = data.loc[data['shift_type'] == 2]
    night_workday_data = night_data.loc[(night_data['is_sleep_before_work'] == 1) & (night_data['is_sleep_after_work'] == 1)]
    night_off_day_data = night_data.loc[(night_data['is_sleep_before_work'] != 1) & (night_data['is_sleep_after_work'] != 1)]
    night_transition_day_data = night_data.loc[(night_data['is_sleep_transition_before_work'] == 1) | (night_data['is_sleep_transition_after_work'] == 1)]

    day_sleep_duration = day_workday_data['duration_in_seconds'] / 3600
    night_sleep_duration = night_workday_data['duration_in_seconds'] / 3600

    day_sleep_duration = day_off_day_data['duration_in_seconds'] / 3600
    night_sleep_duration = night_off_day_data['duration_in_seconds'] / 3600
    
    stat, p = ttest_ind(day_sleep_duration, night_sleep_duration)

    # We want to see nurse year and sleep stat
    if ANALYSIS == 'year':
        
        data_year = pd.DataFrame()
        
        # Nurse year
        # 1. < 5 year; 2. 5 - 15 year; 3. > 15 year
        stat_df = pd.DataFrame()

        # Day shift
        stat_df = stat_df.append(construct_year_frame(day_data.loc[day_data['nurse_years'] < 5], index='five_year_and_less_day'))
        stat_df = stat_df.append(construct_year_frame(day_data.loc[(day_data['nurse_years'] >= 5) & (day_data['nurse_years'] < 15)], index='five_year_and_fifteen_year_day'))
        stat_df = stat_df.append(construct_year_frame(day_data.loc[(day_data['nurse_years'] >= 15)], index='fifteen_year_and_more_day'))

        # Day shift only work-days
        stat_df = stat_df.append(construct_year_frame(day_workday_data.loc[day_workday_data['nurse_years'] < 5], index='five_year_and_less_day_workday'))
        stat_df = stat_df.append(construct_year_frame(day_workday_data.loc[(day_workday_data['nurse_years'] >= 5) & (day_workday_data['nurse_years'] < 15)], index='five_year_and_fifteen_year_day_workday'))
        stat_df = stat_df.append(construct_year_frame(day_workday_data.loc[(day_workday_data['nurse_years'] >= 15)], index='fifteen_year_and_more_day_workday'))

        # Day shift only off-days
        stat_df = stat_df.append(construct_year_frame(day_off_day_data.loc[day_off_day_data['nurse_years'] < 5], index='five_year_and_less_day_off_day'))
        stat_df = stat_df.append(construct_year_frame(day_off_day_data.loc[(day_off_day_data['nurse_years'] >= 5) & (day_off_day_data['nurse_years'] < 15)], index='five_year_and_fifteen_year_day_off_day'))
        stat_df = stat_df.append(construct_year_frame(day_off_day_data.loc[(day_off_day_data['nurse_years'] >= 15)], index='fifteen_year_and_more_day_off_day'))

        # Day shift only trans-days
        stat_df = stat_df.append(construct_year_frame(day_transition_day_data.loc[day_transition_day_data['nurse_years'] < 5], index='five_year_and_less_day_trans_day'))
        stat_df = stat_df.append(construct_year_frame(day_transition_day_data.loc[(day_transition_day_data['nurse_years'] >= 5) & (day_transition_day_data['nurse_years'] < 15)], index='five_year_and_fifteen_year_day_trans_day'))
        stat_df = stat_df.append(construct_year_frame(day_transition_day_data.loc[(day_transition_day_data['nurse_years'] >= 15)], index='fifteen_year_and_more_day_trans_day'))

        # Night shift
        stat_df = stat_df.append(construct_year_frame(night_data.loc[night_data['nurse_years'] < 5], index='five_year_and_less_night'))
        stat_df = stat_df.append(construct_year_frame(night_data.loc[(night_data['nurse_years'] >= 5) & (night_data['nurse_years'] < 15)], index='five_year_and_fifteen_year_night'))
        stat_df = stat_df.append(construct_year_frame(night_data.loc[(night_data['nurse_years'] >= 15)], index='fifteen_year_and_more_night'))

        # Night shift only work-days
        stat_df = stat_df.append(construct_year_frame(night_workday_data.loc[night_workday_data['nurse_years'] < 5], index='five_year_and_less_night_workday'))
        stat_df = stat_df.append(construct_year_frame(night_workday_data.loc[(night_workday_data['nurse_years'] >= 5) & (night_workday_data['nurse_years'] < 15)],index='five_year_and_fifteen_year_night_workday'))
        stat_df = stat_df.append(construct_year_frame(night_workday_data.loc[(night_workday_data['nurse_years'] >= 15)], index='fifteen_year_and_more_night_workday'))

        # Night shift only off-days
        stat_df = stat_df.append(construct_year_frame(night_off_day_data.loc[night_off_day_data['nurse_years'] < 5], index='five_year_and_less_night_off_day'))
        stat_df = stat_df.append(construct_year_frame(night_off_day_data.loc[(night_off_day_data['nurse_years'] >= 5) & (night_off_day_data['nurse_years'] < 15)],index='five_year_and_fifteen_year_night_off_day'))
        stat_df = stat_df.append(construct_year_frame(night_off_day_data.loc[(night_off_day_data['nurse_years'] >= 15)], index='fifteen_year_and_more_night_off_day'))

        # Night shift only trans-days
        stat_df = stat_df.append(construct_year_frame(night_transition_day_data.loc[night_transition_day_data['nurse_years'] < 5], index='five_year_and_less_night_trans_day'))
        stat_df = stat_df.append(construct_year_frame(night_transition_day_data.loc[(night_transition_day_data['nurse_years'] >= 5) & (night_transition_day_data['nurse_years'] < 15)], index='five_year_and_fifteen_year_night_trans_day'))
        stat_df = stat_df.append(construct_year_frame(night_transition_day_data.loc[(night_transition_day_data['nurse_years'] >= 15)], index='fifteen_year_and_more_night_trans_day'))

        print('Done Year Processing')
    
    elif ANALYSIS == 'health':
    
        # Nurse health
        # 1. <= 60; 2. 60 - 85 year; 3. > 85
        stat_df = pd.DataFrame()

        # Day shift
        stat_df = stat_df.append(construct_year_frame(day_data.loc[day_data['general_health'] <= 60], index='poor_health_day'))
        stat_df = stat_df.append(construct_year_frame(day_data.loc[(day_data['general_health'] > 60) & (day_data['general_health'] <= 85)], index='mid_health_day'))
        stat_df = stat_df.append(construct_year_frame(day_data.loc[(day_data['general_health'] > 85)], index='good_health_day'))

        # Day shift workday
        stat_df = stat_df.append(construct_year_frame(day_workday_data.loc[day_workday_data['general_health'] <= 60], index='poor_health_day_workday'))
        stat_df = stat_df.append(construct_year_frame(day_workday_data.loc[(day_workday_data['general_health'] > 60) & (day_workday_data['general_health'] <= 85)], index='mid_health_day_workday'))
        stat_df = stat_df.append(construct_year_frame(day_workday_data.loc[(day_workday_data['general_health'] > 85)], index='good_health_day_workday'))

        # Day shift off-day
        stat_df = stat_df.append(construct_year_frame(day_off_day_data.loc[day_off_day_data['general_health'] <= 60], index='poor_health_day_off_day'))
        stat_df = stat_df.append(construct_year_frame(day_off_day_data.loc[(day_off_day_data['general_health'] > 60) & (day_off_day_data['general_health'] <= 85)], index='mid_health_day_off_day'))
        stat_df = stat_df.append(construct_year_frame(day_off_day_data.loc[(day_off_day_data['general_health'] > 85)], index='good_health_day_off_day'))

        # Night shift
        stat_df = stat_df.append(construct_year_frame(night_data.loc[night_data['general_health'] <= 60], index='poor_health_night'))
        stat_df = stat_df.append(construct_year_frame(night_data.loc[(night_data['general_health'] > 60) & (night_data['general_health'] <= 85)], index='mid_health_night'))
        stat_df = stat_df.append(construct_year_frame(night_data.loc[(night_data['general_health'] > 85)], index='good_health_night'))

        # Night shift workday
        stat_df = stat_df.append(construct_year_frame(night_workday_data.loc[night_workday_data['general_health'] <= 60], index='poor_health_night_workday'))
        stat_df = stat_df.append(construct_year_frame(night_workday_data.loc[(night_workday_data['general_health'] > 60) & (night_workday_data['general_health'] <= 85)], index='mid_health_night_workday'))
        stat_df = stat_df.append(construct_year_frame(night_workday_data.loc[(night_workday_data['general_health'] > 85)], index='good_health_night_workday'))

        # Night shift off-day
        stat_df = stat_df.append(construct_year_frame(night_off_day_data.loc[night_off_day_data['general_health'] <= 60], index='poor_health_night_off_day'))
        stat_df = stat_df.append(construct_year_frame(night_off_day_data.loc[(night_off_day_data['general_health'] > 60) & (night_off_day_data['general_health'] <= 85)], index='mid_health_night_off_day'))
        stat_df = stat_df.append(construct_year_frame(night_off_day_data.loc[(night_off_day_data['general_health'] > 85)], index='good_health_night_off_day'))

        print('Done Health Processing')
    # data = data.loc[data['shift_type'] == 2]
    # data = data.loc[data['is_sleep_transition_before_work'] == 1]
    
    
    # Less adaptive
    # data = data.loc[(data['sleep_transition'] == 2) | (data['sleep_transition'] == 4)]

    # More adaptive
    # data = data.loc[(data['sleep_transition'] == 3) | (data['sleep_transition'] == 5)]

    # data = data.loc[(data['sleep_transition'] == 2) | (data['sleep_transition'] == 3) |
    #                (data['sleep_transition'] == 4) | (data['sleep_transition'] == 5)]

    less_adaptive_data = data.loc[(data['night_shift_type'] == 2) | (data['night_shift_type'] == 4)]
    more_adaptive_data = data.loc[(data['night_shift_type'] == 3) | (data['night_shift_type'] == 5)]

    colunms = [ 'itp_mgt', 'irb_mgt', 'ocb_mgt', 'cwb_mgt',
                'neu_mgt', 'con_mgt', 'ext_mgt', 'agr_mgt', 'ope_mgt',
                'pos_af_mgt', 'neg_af_mgt', 'anxiety_mgt', 'stress_mgt',
                'alcohol_mgt', 'tobacco_mgt', 'exercise_mgt', 'sleep_mgt']

    less_adaptive_data_df = less_adaptive_data[colunms]
    less_adaptive_data_df_corr = less_adaptive_data_df.corr(method='pearson')

    more_adaptive_data_df = more_adaptive_data[colunms]
    more_adaptive_data_df_corr = more_adaptive_data_df.corr(method='pearson')
    
    # participant_id_list = data['participant_id'].unique().tolist()
    
    # for participant in participant_id_list:
    #    participant_data = data.loc[data['participant_id'] == participant]
    #    print(participant_data)
    
    print(len(data))
    
    # data = data.loc[data['is_sleep_before_work'] != 1]
    # data = data.loc[data['is_sleep_after_work'] != 1]

    # data = data.loc[data['is_sleep_after_work'] == 1]
    # data = data.loc[data['is_sleep_before_work'] == 1]
    # data = data.loc[data['is_sleep_after_work'] != 1]

    # 1. Regular
    # get some correlation
    colunms = ['duration_in_seconds',
               'SleepMinutesAwake', 'SleepMinutesStageDeep',
               'SleepMinutesStageLight', 'SleepMinutesStageRem',
               'SleepMinutesStageWake', 'SleepEfficiency',
               'sleep_heart_rate_max', 'sleep_heart_rate_min',
               'sleep_heart_rate_mean', 'sleep_heart_rate_std',
               'sleep_heart_rate_percentile_10', 'sleep_heart_rate_percentile_90',
               'sleep_heart_rate_kurtosis', 'sleep_heart_rate_moment',
               # 'before_sleep_heart_rate_max', 'before_sleep_heart_rate_min',
               # 'before_sleep_heart_rate_mean', 'before_sleep_heart_rate_std',
               # 'before_sleep_heart_rate_percentile_10', 'before_sleep_heart_rate_percentile_90',
               # 'before_sleep_heart_rate_kurtosis', 'before_sleep_heart_rate_moment',
               # 'after_sleep_heart_rate_max', 'after_sleep_heart_rate_min',
               # 'after_sleep_heart_rate_mean', 'after_sleep_heart_rate_std',
               # 'after_sleep_heart_rate_percentile_10', 'after_sleep_heart_rate_percentile_90',
               # 'after_sleep_heart_rate_kurtosis', 'after_sleep_heart_rate_moment',
               'pos_af_mgt', 'neg_af_mgt', 'anxiety_mgt', 'stress_mgt']
    reg_data = data.loc[data['sleep_heart_rate_max'] > -1]
    reg_data = reg_data[colunms]

    reg_corr_pearson = reg_data.corr(method='pearson')
    reg_corr_kendall = reg_data.corr(method='kendall')
    reg_corr_spearman = reg_data.corr(method='spearman')

    reg_corr_pearson.to_csv(os.path.join('output', 'reg_corr.csv'), index=True)
    
    # 2. Personality
    colunms = ['duration_in_seconds',
               'SleepMinutesAwake', 'SleepMinutesStageDeep',
               'SleepMinutesStageLight', 'SleepMinutesStageRem',
               'SleepMinutesStageWake', 'SleepEfficiency',
               'sleep_heart_rate_max', 'sleep_heart_rate_min',
               'sleep_heart_rate_mean', 'sleep_heart_rate_std',
               'sleep_heart_rate_percentile_10', 'sleep_heart_rate_percentile_90',
               'sleep_heart_rate_kurtosis', 'sleep_heart_rate_moment',
               'neu_mgt', 'con_mgt', 'ext_mgt', 'agr_mgt', 'ope_mgt']
    
    personality_data = data.loc[data['neu_mgt'] > -1]
    personality_data = personality_data.loc[personality_data['sleep_heart_rate_max'] > -1]
    personality_data = personality_data[colunms]
    
    personality_corr_pearson = personality_data.corr(method='pearson')
    personality_corr_kendall = personality_data.corr(method='kendall')
    personality_corr_spearman = personality_data.corr(method='spearman')

    personality_corr_pearson.to_csv(os.path.join('output', 'personality_corr.csv'), index=True)

    # 3. Health
    colunms = ['duration_in_seconds',
               'SleepMinutesAwake', 'SleepMinutesStageDeep',
               'SleepMinutesStageLight', 'SleepMinutesStageRem',
               'SleepMinutesStageWake', 'SleepEfficiency',
               'sleep_heart_rate_max', 'sleep_heart_rate_min',
               'sleep_heart_rate_mean', 'sleep_heart_rate_std',
               'sleep_heart_rate_percentile_10', 'sleep_heart_rate_percentile_90',
               'sleep_heart_rate_kurtosis', 'sleep_heart_rate_moment',
               'alcohol_mgt', 'tobacco_mgt', 'exercise_mgt', 'sleep_mgt']
    
    health_data = data.loc[data['alcohol_mgt'] > -1]
    health_data = health_data.loc[health_data['sleep_heart_rate_max'] > -1]
    health_data = health_data[colunms]
    
    health_corr_pearson = health_data.corr(method='pearson')
    health_corr_kendall = health_data.corr(method='kendall')
    health_corr_spearman = health_data.corr(method='spearman')

    health_corr_pearson.to_csv(os.path.join('output', 'health_corr.csv'), index=True)

    # 4. Job
    colunms = ['duration_in_seconds',
               'SleepMinutesAwake', 'SleepMinutesStageDeep',
               'SleepMinutesStageLight', 'SleepMinutesStageRem',
               'SleepMinutesStageWake', 'SleepEfficiency',
               'sleep_heart_rate_max', 'sleep_heart_rate_min',
               'sleep_heart_rate_mean', 'sleep_heart_rate_std',
               'sleep_heart_rate_percentile_10', 'sleep_heart_rate_percentile_90',
               'sleep_heart_rate_kurtosis', 'sleep_heart_rate_moment',
               'itp_mgt', 'irb_mgt', 'ocb_mgt', 'cwb_mgt']
    
    job_data = data.loc[data['itp_mgt'] > -1]
    job_data = job_data.loc[job_data['sleep_heart_rate_max'] > -1]

    job_data = job_data[colunms]
    
    job_corr_pearson = job_data.corr(method='pearson')
    job_corr_kendall = job_data.corr(method='kendall')
    job_corr_spearman = job_data.corr(method='spearman')

    job_corr_pearson.to_csv(os.path.join('output', 'job_corr.csv'), index=True)
    
    print(data)
