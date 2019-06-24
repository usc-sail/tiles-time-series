"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import numpy as np
import pandas as pd
import pickle
import preprocess
from scipy import stats


def nurse_sleep(sleep_dict, participant_id_shift_dict, uid, shift_str):
    sleep_summary_df = sleep_dict['summary']
    sleep_summary_df = sleep_summary_df.sort_index()
    if len(sleep_summary_df) > 3:
        participant_id_shift_dict[uid] = {}
        participant_id_shift_dict[uid]['shift_data'] = []
        participant_id_shift_dict[uid]['shift_duration'] = []
        participant_id_shift_dict[uid]['shift_efficiency'] = []

        shift_df = pd.DataFrame()
        duration_list, efficiency_list, num_sleep_list = [], [], []
        last_time = pd.to_datetime(list(sleep_summary_df['SleepBeginTimestamp'])[0])
        
        new_shift_frame = True
        for i in range(len(sleep_summary_df)):
            current_df = sleep_summary_df.iloc[i, :]
            
            cond1 = current_df['sleep_before_work'] == 1
            cond2 = current_df['sleep_after_work'] == 1

            # Last sleep
            time_gap = pd.to_datetime(current_df['SleepBeginTimestamp']) - pd.to_datetime(last_time)
            con3 = (time_gap.total_seconds() / 3600) < 30
            if shift_str == 'day':
                con4 = (time_gap.total_seconds() / 3600) < 8
            else:
                con4 = (time_gap.total_seconds() / 3600) < 12
            
            cond_equal = (time_gap.total_seconds() / 3600) < 0.1
            con0 = (time_gap.total_seconds() / 3600) < 8

            # shift start, first day
            if cond1 and not cond2:
                
                if new_shift_frame == False:
                    shift_df = pd.DataFrame()
                    duration_list, efficiency_list, num_sleep_list = [], [], []
                    shift_df = shift_df.append(current_df)
                    duration_list.append(current_df['duration'])
                    efficiency_list.append(current_df['SleepEfficiency'])
                    num_sleep_list.append(1)
                    new_shift_frame = True
                    '''
                    if len(shift_df) > 2:
                        participant_id_shift_dict[uid]['shift'].append(shift_df)
                        participant_id_shift_dict[uid]['shift_duration'].append(duration_list)
                    '''
                else:
                    # already start
                    if cond_equal and new_shift_frame and len(shift_df) > 0:
                        shift_df.loc[list(shift_df.index)[-1], :] = current_df
                        duration_list[-1] = current_df['duration']
                    elif len(shift_df) == 1 and con4:
                        shift_df = shift_df.append(current_df)
                        duration_list[-1] += current_df['duration']
                        num_sleep_list[-1] += 1
                        efficiency_list[-1] += current_df['SleepEfficiency']

                last_time = pd.to_datetime(current_df['SleepBeginTimestamp'])
            
            elif cond1 and cond2:
                # shift in between
                if cond_equal and new_shift_frame and len(shift_df) > 0:
                    shift_df.loc[list(shift_df.index)[-1], :] = current_df
                    duration_list[-1] = current_df['duration']
                elif con3:
                    if len(shift_df) != 0 and new_shift_frame:
                        shift_df = shift_df.append(current_df)
                        if con0:
                            duration_list[-1] += current_df['duration']
                            num_sleep_list[-1] += 1
                            efficiency_list[-1] += current_df['SleepEfficiency']
                        else:
                            duration_list.append(current_df['duration'])
                            efficiency_list.append(current_df['SleepEfficiency'])
                            num_sleep_list.append(1)
                else:
                    '''
                    if len(shift_df) > 2 and new_shift_frame:
                        participant_id_shift_dict[uid]['shift_data'].append(shift_df)
                        participant_id_shift_dict[uid]['shift_duration'].append(duration_list)
                        participant_id_shift_dict[uid]['shift_efficiency'].append(list(np.divide(np.array(efficiency_list), np.array(num_sleep_list))))
                    '''
                    new_shift_frame = False
                    shift_df = pd.DataFrame()
                    duration_list, efficiency_list, num_sleep_list = [], [], []
                last_time = pd.to_datetime(current_df['SleepBeginTimestamp'])
                    
            elif not cond1 and cond2:
                if cond_equal and new_shift_frame and len(shift_df) > 0:
                    shift_df.loc[list(shift_df.index)[-1], :] = current_df
                    duration_list[-1] = current_df['duration']
                elif con3 and new_shift_frame:
                    shift_df = shift_df.append(current_df)
                    duration_list.append(current_df['duration'])
                    efficiency_list.append(current_df['SleepEfficiency'])
                    num_sleep_list.append(1)
                    
                if len(shift_df) > 2 and new_shift_frame:
                    if i != len(sleep_summary_df) - 1:
                        
                        next_df = sleep_summary_df.iloc[i+1, :]
                        cond5 = next_df['sleep_before_work'] == 1
                        cond6 = next_df['sleep_after_work'] == 1
                        time_gap = pd.to_datetime(next_df['SleepBeginTimestamp']) - pd.to_datetime(current_df['SleepEndTimestamp'])
                        cond7 = (time_gap.total_seconds() / 3600) < 8
                        
                        if not cond5 and cond6 and cond7:
                            shift_df = shift_df.append(current_df)
                            duration_list[-1] += current_df['duration']
                            num_sleep_list[-1] += 1
                            efficiency_list[-1] += current_df['SleepEfficiency']
                    
                    if np.nanmin(duration_list) > 3:
                        participant_id_shift_dict[uid]['shift_data'].append(shift_df)
                        participant_id_shift_dict[uid]['shift_duration'].append(duration_list)
                        participant_id_shift_dict[uid]['shift_efficiency'].append(list(np.divide(np.array(efficiency_list), np.array(num_sleep_list))))

                new_shift_frame = False
                shift_df = pd.DataFrame()
                duration_list, efficiency_list, num_sleep_list = [], [], []
                last_time = pd.to_datetime(current_df['SleepBeginTimestamp'])
            
            else:
                new_shift_frame = False
                shift_df = pd.DataFrame()
                duration_list, efficiency_list, num_sleep_list = [], [], []
                

def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
    psqi_raw_igtb = load_data_basic.read_PSQI_Raw(tiles_data_path)
    
    # mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    if os.path.exists(os.path.join(os.path.dirname(__file__), 'sleep_dev.pkl')) is True:
        pkl_file = open(os.path.join(os.path.dirname(__file__), 'sleep_dev.pkl'), 'rb')
        participant_id_shift_dict = pickle.load(pkl_file)
        
        day_participant_shift_dict, night_participant_shift_dict = {}, {}
        day_duration_list, night_duration_list = [], []
        day_slope_list, night_slope_list = [], []
        
        day_df, night_df = pd.DataFrame(), pd.DataFrame()

        for uid in list(participant_id_shift_dict.keys()):
            if participant_id_shift_dict[uid]['shift'] == 'day' and len(participant_id_shift_dict[uid]['shift_duration']) != 0:
                tmp_list = []
                
                for duration in participant_id_shift_dict[uid]['shift_duration']:
                    if len(duration) == 3:
                        tmp_list.append(duration)
                    elif len(duration) >= 4:
                        duration = [duration[0], np.mean(np.array(duration[1:-1])), duration[-1]]
                        # duration = [duration[0], (duration[1] + duration[2]) / 2, duration[3]]
                        tmp_list.append(duration)
                
                if len(tmp_list) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, 3), list(np.nanmean(np.array(tmp_list), axis=0)))
                    day_duration_list.append(list(np.nanmean(np.array(tmp_list), axis=0)))
                    day_slope_list.append(slope)

                    tmp_df = pd.DataFrame(index=[uid])
                    tmp_df['slope'] = slope
                    for col in igtb_cols:
                        tmp_df[col] = igtb_df.loc[[uid], :][col][0]
                    for col in list(psqi_raw_igtb.columns):
                        tmp_df[col] = psqi_raw_igtb.loc[[uid], :][col][0]
                    day_df = day_df.append(tmp_df)

            elif participant_id_shift_dict[uid]['shift'] == 'night' and len(participant_id_shift_dict[uid]['shift_duration']) != 0:
                tmp_list = []
                for duration in participant_id_shift_dict[uid]['shift_duration']:
                    if len(duration) == 3:
                        tmp_list.append(duration)
                    elif len(duration) == 4:
                        duration = [duration[0], (duration[1] + duration[2]) / 2, duration[3]]
                        tmp_list.append(duration)

                if len(tmp_list) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(0, 3), list(np.nanmean(np.array(tmp_list), axis=0)))
                    night_duration_list.append(list(np.nanmean(np.array(tmp_list), axis=0)))
                    night_slope_list.append(slope)

                    tmp_df = pd.DataFrame(index=[uid])
                    tmp_df['slope'] = slope
                    for col in igtb_cols:
                        tmp_df[col] = igtb_df.loc[[uid], :][col][0]
                    for col in list(psqi_raw_igtb.columns):
                        tmp_df[col] = psqi_raw_igtb.loc[[uid], :][col][0]

                    night_df = night_df.append(tmp_df)
        
        num_day_increase = len(np.where(np.array(day_slope_list) > 0.1)[0]) / len(day_slope_list)
        num_day_unchange = len(np.where(np.abs(day_slope_list) <= 0.1)[0]) / len(day_slope_list)
        num_day_decrease = len(np.where(np.array(day_slope_list) < -0.1)[0]) / len(day_slope_list)

        num_night_increase = len(np.where(np.array(night_slope_list) > 0.1)[0]) / len(night_slope_list)
        num_night_unchange = len(np.where(np.abs(night_slope_list) <= 0.1)[0]) / len(night_slope_list)
        num_night_decrease = len(np.where(np.array(night_slope_list) < -0.1)[0]) / len(night_slope_list)
        print()
        
    else:
        participant_id_shift_dict = {}
        for idx, participant_id in enumerate(top_participant_id_list):
            print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
            
            sleep_path = os.path.join(data_config.sleep_path, participant_id + '.pkl')
            if os.path.exists(sleep_path) is False:
                continue
            
            pkl_file = open(os.path.join(data_config.sleep_path, participant_id + '.pkl'), 'rb')
            sleep_dict = pickle.load(pkl_file)

            nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
            primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            job_str = 'nurse' if nurse == 1 and 'Dialysis' not in primary_unit else 'non_nurse'
            shift_str = 'day' if shift == 'Day shift' else 'night'
            
            uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
            nurse_sleep(sleep_dict, participant_id_shift_dict, uid, shift_str)
            
            if uid in list(participant_id_shift_dict.keys()):
                participant_id_shift_dict[uid]['job'] = job_str
                participant_id_shift_dict[uid]['shift'] = shift_str

    output = open(os.path.join(os.path.dirname(__file__), 'sleep_dev.pkl'), 'wb')
    pickle.dump(participant_id_shift_dict, output)


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)