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
from datetime import timedelta
import collections


def nurse_sleep(sleep_dict, participant_id_shift_dict, uid, shift_str):
    sleep_summary_df = sleep_dict['summary']
    sleep_summary_df = sleep_summary_df.sort_index()
    if len(sleep_summary_df) > 3:
        participant_id_shift_dict[uid] = {}
        participant_id_shift_dict[uid]['trans_data'] = []
        participant_id_shift_dict[uid]['trans_type'] = []
        # participant_id_shift_dict[uid]['shift_efficiency'] = []

        shift_df = pd.DataFrame()
        duration_list, efficiency_list, num_sleep_list = [], [], []
        
        cond1 = (sleep_summary_df['sleep_before_work_nearest'] == 1)
        cond2 = (sleep_summary_df['sleep_after_work'] == 0)
        sleep_before_work_df = sleep_summary_df.loc[(cond1) & (cond2)]
        
        if len(sleep_before_work_df) > 0:
            for i in range(len(sleep_before_work_df)):
                current_df = sleep_before_work_df.iloc[i, :]
                
                current_time = pd.to_datetime(current_df['SleepBeginTimestamp'])
                valid_start_time = (current_time - timedelta(days=1)).strftime(load_data_basic.date_time_format)[:-3]
                valid_end_time = (current_time + timedelta(minutes=1)).strftime(load_data_basic.date_time_format)[:-3]
                
                sleep_trans_df = sleep_summary_df[valid_start_time:valid_end_time]
                
                if len(sleep_trans_df) > 0:
                    # Night shift
                    if shift_str == 'night':
                        
                        if len(sleep_trans_df) == 2:
                            cond_nap_1 = 10 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][1]).hour < 18
                            
                            cond4 = 20 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0]).hour < 24
                            cond5 = 0 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0]).hour < 8
                            cond_nap_2 = cond4 or cond5
                            
                            time_gap = pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][1]) - pd.to_datetime(sleep_trans_df['SleepEndTimestamp'][0])
                            cond_gap = time_gap.total_seconds() / 3600 < 2
                            
                            # If two sleep is near, consider as one sleep
                            if cond_gap:
                                cond6 = 18 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0]).hour < 24
                                cond7 = 0 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0]).hour < 3

                                cond_switch_and_no_sleep = cond6 or cond7
                                time_span = pd.to_datetime(sleep_trans_df['SleepEndTimestamp'][1]) - pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0])
                                cond_span = time_span.total_seconds() / 3600 < 8
                                if cond_switch_and_no_sleep and cond_span:
                                    participant_id_shift_dict[uid]['trans_type'].append('no_sleep')
                                    participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                                elif cond_switch_and_no_sleep and not cond_span:
                                    participant_id_shift_dict[uid]['trans_type'].append('switch')
                                    participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                                else:
                                    participant_id_shift_dict[uid]['trans_type'].append('incomplete')
                                    participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                            else:
                                if cond_nap_1 and cond_nap_2:
                                    participant_id_shift_dict[uid]['trans_type'].append('nap')
                                    participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                        if len(sleep_trans_df) == 1:
                            cond_incomplete = 3 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0]).hour < 12
                            
                            cond4 = 18 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0]).hour < 24
                            cond5 = 0 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0]).hour < 3
                            cond_switch_and_no_sleep = cond4 or cond5
                            
                            cond_switch_duration = sleep_trans_df['duration'][0] > 8
                            cond_nap = 10 <= pd.to_datetime(sleep_trans_df['SleepBeginTimestamp'][0]).hour < 18
                            if cond_incomplete and not cond_switch_duration:
                                participant_id_shift_dict[uid]['trans_type'].append('incomplete')
                                participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                            elif cond_switch_and_no_sleep and not cond_switch_duration:
                                participant_id_shift_dict[uid]['trans_type'].append('no_sleep')
                                participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                            elif cond_switch_and_no_sleep and cond_switch_duration:
                                participant_id_shift_dict[uid]['trans_type'].append('switch')
                                participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                            elif cond_nap:
                                participant_id_shift_dict[uid]['trans_type'].append('nap')
                                participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                            
                    else:
                        current_time = pd.to_datetime(current_df['SleepBeginTimestamp'])
                        valid_start_time = (current_time - timedelta(hours=36)).strftime(load_data_basic.date_time_format)[:-3]
                        valid_end_time = (current_time + timedelta(minutes=1)).strftime(load_data_basic.date_time_format)[:-3]
    
                        sleep_trans_df = sleep_summary_df[valid_start_time:valid_end_time]

                        duration_list, start_list = [], []
                        for i in range(len(sleep_trans_df)):
                            current_df = sleep_trans_df.iloc[i, :]
                            if current_df['sleep_after_work'] == 0:
                                if current_df['duration'] > 4:
                                    duration_list.append(current_df['duration'])
                                    hour = pd.to_datetime(current_df['SleepBeginTimestamp']).hour + pd.to_datetime(current_df['SleepBeginTimestamp']).minute / 60
                                    
                                    if hour > 12:
                                        hour = hour - 24
                                    start_list.append(hour)
                        
                        if len(start_list) >= 2:
                            diff = np.abs(start_list[-1] - start_list[-2])
                        
                            if diff < 1:
                                participant_id_shift_dict[uid]['trans_type'].append('con')
                                participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                            else:
                                participant_id_shift_dict[uid]['trans_type'].append('non_con')
                                participant_id_shift_dict[uid]['trans_data'].append(sleep_trans_df)
                    

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
    
    if os.path.exists(os.path.join(os.path.dirname(__file__), 'sleep_trans.pkl')) is True:
        pkl_file = open(os.path.join(os.path.dirname(__file__), 'sleep_trans.pkl'), 'rb')
        participant_id_shift_dict = pickle.load(pkl_file)
        
        day_participant_shift_dict, night_participant_shift_dict = {}, {}
        day_duration_list, night_duration_list = [], []
        day_slope_list, night_slope_list = [], []
        
        day_df, night_df = pd.DataFrame(), pd.DataFrame()
        
        for uid in list(participant_id_shift_dict.keys()):
            if participant_id_shift_dict[uid]['shift'] == 'night' and len(participant_id_shift_dict[uid]['trans_type']) > 0:
                night_participant_shift_dict[uid] = participant_id_shift_dict[uid]
                
                row_df = pd.DataFrame(0, index=[uid], columns=['nap', 'incomplete', 'switch', 'no_sleep'])
                
                trans_dict = collections.Counter(participant_id_shift_dict[uid]['trans_type'])
                for key in list(trans_dict.keys()):
                    freq = trans_dict[key] / len(participant_id_shift_dict[uid]['trans_type'])
                    row_df[key] = freq
                row_df['incomplete_switch'] = row_df['switch'][0] + row_df['incomplete'][0]
                
                for col in igtb_cols:
                    row_df[col] = igtb_df.loc[[uid], :][col][0]
                    
                for col in list(psqi_raw_igtb.columns):
                    row_df[col] = psqi_raw_igtb.loc[[uid], :][col][0]
                row_df['nurseyears'] = str(igtb_df.loc[[uid], :]['nurseyears'][0])
                
                if row_df['nurseyears'][0] == 'nan':
                    row_df['nurseyears'][0] = np.nan
                else:
                    row_df['nurseyears'][0] = int(row_df['nurseyears'][0])
                
                night_df = night_df.append(row_df)
            elif participant_id_shift_dict[uid]['shift'] == 'day' and len(participant_id_shift_dict[uid]['trans_type']) > 0:
                day_participant_shift_dict[uid] = participant_id_shift_dict[uid]
                row_df = pd.DataFrame(0, index=[uid], columns=['con', 'non_con'])
    
                trans_dict = collections.Counter(participant_id_shift_dict[uid]['trans_type'])
                for key in list(trans_dict.keys()):
                    freq = trans_dict[key] / len(participant_id_shift_dict[uid]['trans_type'])
                    row_df[key] = freq

                for col in igtb_cols:
                    row_df[col] = igtb_df.loc[[uid], :][col][0]
                    
                for col in list(psqi_raw_igtb.columns):
                    row_df[col] = psqi_raw_igtb.loc[[uid], :][col][0]
                    
                row_df['nurseyears'] = str(igtb_df.loc[[uid], :]['nurseyears'][0])
    
                if row_df['nurseyears'][0] == 'nan':
                    row_df['nurseyears'][0] = np.nan
                else:
                    row_df['nurseyears'][0] = int(row_df['nurseyears'][0])
    
                day_df = day_df.append(row_df)
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
            if job_str == 'nurse':
                nurse_sleep(sleep_dict, participant_id_shift_dict, uid, shift_str)
                
                if uid in list(participant_id_shift_dict.keys()):
                    participant_id_shift_dict[uid]['job'] = job_str
                    participant_id_shift_dict[uid]['shift'] = shift_str

    output = open(os.path.join(os.path.dirname(__file__), 'sleep_trans.pkl'), 'wb')
    pickle.dump(participant_id_shift_dict, output)


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)