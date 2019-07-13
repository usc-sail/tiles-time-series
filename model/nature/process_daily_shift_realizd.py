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


def process_shift_realizd(data_config, data_df, sleep_df, days_at_work_df, igtb_df, participant_id):
    
    nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
    primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
    shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
    job_str = 'nurse' if nurse == 1 and 'Dialysis' not in primary_unit else 'non_nurse'
    shift_str = 'day' if shift == 'Day shift' else 'night'

    if len(data_df) < 1000 or len(sleep_df) < 5 or job_str is 'non_nurse' or days_at_work_df is None:
        return None
    
    date_list = pd.to_datetime(days_at_work_df.index)
    change_list = np.where(np.array(list((date_list[1:] - date_list[:-1]).days)) > 1)[0]
    change_array = np.zeros([len(change_list) + 1, 2])
    change_array[1:, 0] = change_list + 1
    change_array[:-1, 1] = change_list + 1
    change_array[-1, 1] = len(date_list)
    
    shift_dict = {}
    
    for i, change in enumerate(change_array):
        date_shift_list = list(days_at_work_df.index)[int(change[0]):int(change[1])]
        if len(date_shift_list) < 2:
            continue
            
        shift_df = pd.DataFrame()
        start_str = date_shift_list[0]
        end_str = date_shift_list[-1]
        
        if shift_str == 'day':
            start_str = pd.to_datetime(start_str).replace(hour=7).strftime(load_data_basic.date_time_format)[:-3]
            end_str = pd.to_datetime(end_str).replace(hour=7).strftime(load_data_basic.date_time_format)[:-3]
        else:
            start_str = pd.to_datetime(start_str).replace(hour=19).strftime(load_data_basic.date_time_format)[:-3]
            end_str = pd.to_datetime(end_str).replace(hour=19).strftime(load_data_basic.date_time_format)[:-3]

        days = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days + 1
        
        # One day before the shift
        pre_start_str = (pd.to_datetime(start_str) - timedelta(days=1)).strftime(load_data_basic.date_time_format)[:-3]
        pre_end_str = start_str
        day_raw_data = data_df[pre_start_str:pre_end_str]
        sleep_raw_data = sleep_df[pre_start_str:pre_end_str]
        row_df = pd.DataFrame(index=[pre_start_str])

        if len(sleep_raw_data) > 0:
            if np.nansum(sleep_raw_data['duration']) > 2:
                row_df['duration'] = np.nansum(sleep_raw_data['duration'])
                row_df['SleepEfficiency'] = np.nanmean(sleep_raw_data['SleepEfficiency'])

        if len(day_raw_data) > 0:
            row_df['frequency'] = len(day_raw_data)
            row_df['total_time'] = np.sum(np.array(day_raw_data))
            row_df['mean_time'] = np.mean(np.array(day_raw_data))
            row_df['above_1min'] = len(np.where(np.array(day_raw_data) > 60)[0])
            row_df['less_than_1min'] = len(np.where(np.array(day_raw_data) <= 60)[0])
        else:
            row_df['frequency'] = 0
            row_df['total_time'] = 0
            row_df['mean_time'] = 0
            row_df['above_1min'] = 0
            row_df['less_than_1min'] = 0

        # Inter duration
        inter_df = pd.DataFrame()
        for j in range(len(day_raw_data)):
            time_df = day_raw_data.iloc[j, :]
            time_row_df = pd.DataFrame(index=[list(day_raw_data.index)[j]])
            time_row_df['start'] = list(day_raw_data.index)[j]
            time_row_df['end'] = (pd.to_datetime(list(day_raw_data.index)[j]) + timedelta(seconds=int(time_df['SecondsOnPhone']))).strftime(load_data_basic.date_time_format)[:-3]
            inter_df = inter_df.append(time_row_df)
        inter_duration_list = []
        if len(inter_df) > 3:
            start_list = list(pd.to_datetime(inter_df['start']))
            end_list = list(pd.to_datetime(inter_df['end']))
            for j in range(len(day_raw_data) - 1):
                inter_time = (start_list[j + 1] - end_list[j]).total_seconds()
                # if inter time is larger than 4 hours, we assume it is sleep
                if inter_time > 3600 * 4:
                    continue
                inter_duration_list.append(inter_time)
        row_df['mean_inter'] = np.mean(inter_duration_list)
        
        shift_df = shift_df.append(row_df)

        for day in range(days):
            day_start_str = (pd.to_datetime(start_str) + timedelta(days=day)).strftime(load_data_basic.date_time_format)[:-3]
            day_end_str = (pd.to_datetime(start_str) + timedelta(days=day+1)).strftime(load_data_basic.date_time_format)[:-3]
            
            work_start_str = day_start_str
            work_end_str = (pd.to_datetime(start_str) + timedelta(days=day, hours=12)).strftime(load_data_basic.date_time_format)[:-3]
            
            off_start_str = (pd.to_datetime(start_str) + timedelta(days=day, hours=12)).strftime(load_data_basic.date_time_format)[:-3]
            off_end_str = day_end_str
            
            day_raw_data = data_df[day_start_str:day_end_str]
            shift_raw_data = data_df[work_start_str:work_end_str]
            off_raw_data = data_df[off_start_str:off_end_str]
            sleep_raw_data = sleep_df[day_start_str:day_end_str]

            row_df = pd.DataFrame(index=[day_start_str])
            
            if len(sleep_raw_data) > 0:
                if np.nansum(sleep_raw_data['duration']) > 2:
                    row_df['duration'] = np.nansum(sleep_raw_data['duration'])
                    row_df['SleepEfficiency'] = np.nanmean(sleep_raw_data['SleepEfficiency'])
            
            if len(day_raw_data) > 0:
                row_df['frequency'] = len(day_raw_data)
                row_df['total_time'] = np.sum(np.array(day_raw_data))
                row_df['mean_time'] = np.mean(np.array(day_raw_data))
                row_df['above_1min'] = len(np.where(np.array(day_raw_data) > 60)[0])
                row_df['less_than_1min'] = len(np.where(np.array(day_raw_data) <= 60)[0])
            else:
                row_df['frequency'] = 0
                row_df['total_time'] = 0
                row_df['mean_time'] = 0
                row_df['above_1min'] = 0
                row_df['less_than_1min'] = 0

            # Inter duration
            inter_df = pd.DataFrame()
            for j in range(len(day_raw_data)):
                time_df = day_raw_data.iloc[j, :]
                time_row_df = pd.DataFrame(index=[list(day_raw_data.index)[j]])
                time_row_df['start'] = list(day_raw_data.index)[j]
                time_row_df['end'] = (pd.to_datetime(list(day_raw_data.index)[j]) + timedelta(seconds=int(time_df['SecondsOnPhone']))).strftime(load_data_basic.date_time_format)[:-3]
                inter_df = inter_df.append(time_row_df)
            inter_duration_list = []
            if len(inter_df) > 3:
                start_list = list(pd.to_datetime(inter_df['start']))
                end_list = list(pd.to_datetime(inter_df['end']))
                for j in range(len(day_raw_data) - 1):
                    inter_time = (start_list[j + 1] - end_list[j]).total_seconds()
                    # if inter time is larger than 4 hours, we assume it is sleep
                    if inter_time > 3600 * 4:
                        continue
                    inter_duration_list.append(inter_time)
                row_df['mean_inter'] = np.mean(inter_duration_list)
            
            '''
            if len(shift_raw_data) > 0:
                row_df['shift_frequency'] = len(shift_raw_data)
                row_df['shift_total_time'] = np.sum(np.array(shift_raw_data))
                row_df['shift_mean_time'] = np.mean(np.array(shift_raw_data))
                row_df['shift_above_1min'] = len(np.where(np.array(shift_raw_data) > 60)[0])
                row_df['shift_less_than_1min'] = len(np.where(np.array(shift_raw_data) <= 60)[0])
            else:
                row_df['shift_frequency'] = 0
                row_df['shift_total_time'] = 0
                row_df['shift_mean_time'] = 0
                row_df['shift_above_1min'] = 0
                row_df['shift_less_than_1min'] = 0
                
            if len(off_raw_data) > 0:
                row_df['off_frequency'] = len(off_raw_data)
                row_df['off_total_time'] = np.sum(np.array(off_raw_data))
                row_df['off_mean_time'] = np.mean(np.array(off_raw_data))
                # row_df['off_std_time'] = np.std(np.array(off_raw_data))
                row_df['off_above_1min'] = len(np.where(np.array(off_raw_data) > 60)[0])
                row_df['off_less_than_1min'] = len(np.where(np.array(off_raw_data) <= 60)[0])
            else:
                row_df['off_frequency'] = 0
                row_df['off_total_time'] = 0
                row_df['off_mean_time'] = 0
                # row_df['off_std_time'] = 0
                row_df['off_above_1min'] = 0
                row_df['off_less_than_1min'] = 0
            '''
            shift_df = shift_df.append(row_df)

        # After the shift
        after_start_str = (pd.to_datetime(end_str) + timedelta(days=1)).strftime(load_data_basic.date_time_format)[:-3]
        after_end_str = (pd.to_datetime(end_str) + timedelta(days=2)).strftime(load_data_basic.date_time_format)[:-3]
        day_raw_data = data_df[after_start_str:after_end_str]
        sleep_raw_data = sleep_df[after_start_str:after_end_str]

        row_df = pd.DataFrame(index=[after_start_str])

        if len(sleep_raw_data) > 0:
            if np.nansum(sleep_raw_data['duration']) > 2:
                row_df['duration'] = np.nansum(sleep_raw_data['duration'])
                row_df['SleepEfficiency'] = np.nanmean(sleep_raw_data['SleepEfficiency'])

        if len(day_raw_data) > 0:
            row_df['frequency'] = len(day_raw_data)
            row_df['total_time'] = np.sum(np.array(day_raw_data))
            row_df['mean_time'] = np.mean(np.array(day_raw_data))
            row_df['above_1min'] = len(np.where(np.array(day_raw_data) > 60)[0])
            row_df['less_than_1min'] = len(np.where(np.array(day_raw_data) <= 60)[0])
        else:
            row_df['frequency'] = 0
            row_df['total_time'] = 0
            row_df['mean_time'] = 0
            row_df['above_1min'] = 0
            row_df['less_than_1min'] = 0

        # Inter duration
        inter_df = pd.DataFrame()
        for j in range(len(day_raw_data)):
            time_df = day_raw_data.iloc[j, :]
            time_row_df = pd.DataFrame(index=[list(day_raw_data.index)[j]])
            time_row_df['start'] = list(day_raw_data.index)[j]
            time_row_df['end'] = (pd.to_datetime(list(day_raw_data.index)[j]) + timedelta(seconds=int(time_df['SecondsOnPhone']))).strftime(load_data_basic.date_time_format)[:-3]
            inter_df = inter_df.append(time_row_df)
        inter_duration_list = []
        if len(inter_df) > 3:
            start_list = list(pd.to_datetime(inter_df['start']))
            end_list = list(pd.to_datetime(inter_df['end']))
            for j in range(len(day_raw_data) - 1):
                inter_time = (start_list[j + 1] - end_list[j]).total_seconds()
                # if inter time is larger than 4 hours, we assume it is sleep
                if inter_time > 3600 * 4:
                    continue
                inter_duration_list.append(inter_time)
        row_df['mean_inter'] = np.mean(inter_duration_list)
        shift_df = shift_df.append(row_df)

        shift_dict[start_str] = {}
        shift_dict[start_str]['data_length'] = len(shift_df)
        shift_dict[start_str]['data'] = shift_df
    
    return shift_dict


def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    chi_data_config = config.Config()
    chi_data_config.readChiConfigFile(config_path)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    load_data_path.load_chi_preprocess_path(chi_data_config, process_data_path)
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
    psqi_raw_igtb = load_data_basic.read_PSQI_Raw(tiles_data_path)
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    for idx, participant_id in enumerate(top_participant_id_list[:]):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
        job_str = 'nurse' if nurse == 1 else 'non_nurse'
        shift_str = 'day' if shift == 'Day shift' else 'night'
        
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        days_at_work_detail_df = load_sensor_data.read_preprocessed_days_at_work_detailed(data_config.days_at_work_path, participant_id)

        if os.path.exists(os.path.join(data_config.sleep_path, participant_id + '.pkl')) is False:
            continue

        pkl_file = open(os.path.join(data_config.sleep_path, participant_id + '.pkl'), 'rb')
        participant_sleep_dict = pickle.load(pkl_file)

        sleep_df = participant_sleep_dict['summary'].sort_index()
        realizd_raw_df = load_sensor_data.read_realizd(os.path.join(tiles_data_path, '2_raw_csv_data/realizd/'), participant_id)

        if realizd_raw_df is None:
            continue

        shift_dict = process_shift_realizd(data_config, realizd_raw_df, sleep_df, days_at_work_detail_df, igtb_df, participant_id)

        output = open(os.path.join(data_config.phone_usage_path, participant_id + '.pkl'), 'wb')
        pickle.dump(shift_dict, output)


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)