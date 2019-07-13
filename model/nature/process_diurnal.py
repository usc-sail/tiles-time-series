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


def process_diurnal_realizd(data_config, data_df, days_at_work_df, igtb_df, participant_id):
    process_df = pd.DataFrame()
    
    if len(data_df) < 700 or len(days_at_work_df) < 5:
        return None
    
    days = (pd.to_datetime(data_df.index[-1]).date() - pd.to_datetime(data_df.index[0]).date()).days
    start_str = pd.to_datetime(data_df.index[0]).date()
    
    for i in range(days):
    
        date_start_str = (pd.to_datetime(start_str) + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3]
        row_df = pd.DataFrame(index=[date_start_str])
    
        for diurnal_type in ['night', 'morning', 'afternoon', 'evening']:
            
            if diurnal_type == 'night':
                offset = 0
            elif diurnal_type == 'morning':
                offset = 6
            elif diurnal_type == 'afternoon':
                offset = 12
            else:
                offset = 18
            
            diurnal_start_str = (pd.to_datetime(date_start_str) + timedelta(hours=offset)).strftime(load_data_basic.date_time_format)[:-3]
            diurnal_end_str = (pd.to_datetime(date_start_str) + timedelta(hours=offset+6)).strftime(load_data_basic.date_time_format)[:-3]
            raw_df = data_df[diurnal_start_str:diurnal_end_str]
        
            if len(raw_df) > 0:
                row_df[diurnal_type + '_frequency'] = len(raw_df)
                row_df[diurnal_type + '_total_time'] = np.sum(np.array(raw_df))
                row_df[diurnal_type + '_mean_time'] = np.mean(np.array(raw_df))
                row_df[diurnal_type + '_less_than_1min'] = len(np.where(np.array(raw_df) <= 60)[0])
                row_df[diurnal_type + '_above_1min'] = len(np.where((np.array(raw_df) > 60))[0])
            else:
                row_df[diurnal_type + '_frequency'] = 0
                row_df[diurnal_type + '_total_time'] = 0
                row_df[diurnal_type + '_mean_time'] = np.nan
                row_df[diurnal_type + '_less_than_1min'] = 0
                row_df[diurnal_type + '_above_1min'] = 0

            if len(raw_df) == 0:
                row_df[diurnal_type + '_total_time'] = np.nan
                continue

            inter_df = pd.DataFrame()
            for j in range(len(raw_df)):
                time_df = raw_df.iloc[j, :]
                time_row_df = pd.DataFrame(index=[list(raw_df.index)[j]])
                time_row_df['start'] = list(raw_df.index)[j]
                time_row_df['end'] = (pd.to_datetime(list(raw_df.index)[j]) + timedelta(seconds=int(time_df['SecondsOnPhone']))).strftime(load_data_basic.date_time_format)[:-3]
                inter_df = inter_df.append(time_row_df)
            
            if len(inter_df) == 0:
                row_df[diurnal_type + '_total_time'] = np.nan
            else:
                inter_duration_list = []
                start_list = list(pd.to_datetime(inter_df['start']))
                end_list = list(pd.to_datetime(inter_df['end']))
                for j in range(len(raw_df) - 1):
                    inter_time = (start_list[j + 1] - end_list[j]).total_seconds()
                    # if inter time is larger than 4 hours, we assume it is sleep
                    if inter_time > 3600 * 4:
                        continue
                    inter_duration_list.append(inter_time)
                row_df[diurnal_type + '_mean_inter'] = np.mean(inter_duration_list)
        
        if days_at_work_df.loc[date_start_str, 'work'] == 1:
            row_df['work'] = 1
        else:
            row_df['work'] = 0
        
        process_df = process_df.append(row_df)
    
    work_df = process_df.loc[process_df['work'] == 1]
    off_df = process_df.loc[process_df['work'] == 0]
    
    process_df.to_csv(os.path.join(data_config.phone_usage_path, participant_id + '_diurnal.csv.gz'), compression='gzip')
    participant_df = pd.DataFrame(index=[participant_id])
    
    for col in list(process_df.columns):
        if 'work' not in col:
            participant_df[col + '_mean'] = np.nanmean(np.array(process_df[col]))
            participant_df['work_' + col + '_mean'] = np.nanmean(np.array(work_df[col]))
            participant_df['off_' + col + '_mean'] = np.nanmean(np.array(off_df[col]))
    
    igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
    for col in igtb_cols:
        participant_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]
    
    return participant_df


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
    
    valid_data = 0

    # if os.path.exists(os.path.join(data_config.phone_usage_path, 'dianual_summary.csv.gz')) is True:
    #    all_df = pd.read_csv(os.path.join(data_config.phone_usage_path, 'dianual_summary.csv.gz'), index_col=0)

    all_df = pd.DataFrame()

    for idx, participant_id in enumerate(top_participant_id_list[:]):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        days_at_work_df = load_sensor_data.read_preprocessed_days_at_work(data_config.days_at_work_path, participant_id)
        realizd_raw_df = load_sensor_data.read_realizd(os.path.join(tiles_data_path, '2_raw_csv_data/realizd/'),participant_id)
        
        nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
        job_str = 'nurse' if nurse == 1 else 'non_nurse'
        shift_str = 'day' if shift == 'Day shift' else 'night'
        
        if job_str == 'nurse' and len(realizd_raw_df) > 700:
            valid_data = valid_data + 1
        
        if realizd_raw_df is None or days_at_work_df is None:
            continue
        
        if len(realizd_raw_df) < 700 or len(days_at_work_df) < 5:
            continue
        
        participant_df = process_diurnal_realizd(data_config, realizd_raw_df, days_at_work_df, igtb_df, participant_id)
        
        if participant_df is not None:
            all_df = all_df.append(participant_df)
            all_df = all_df.loc[:, list(participant_df.columns)]
    
    print(valid_data)
    all_df.to_csv(os.path.join(data_config.phone_usage_path, 'dianual_summary.csv.gz'), compression='gzip')


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)