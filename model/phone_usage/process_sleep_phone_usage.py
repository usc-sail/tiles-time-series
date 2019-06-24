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
    
    # mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    num_point_per_day = chi_data_config.num_point_per_day
    offset = 1440 / num_point_per_day
    window = chi_data_config.window
    
    for idx, participant_id in enumerate(top_participant_id_list[:]):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
        job_str = 'nurse' if nurse == 1 and 'Dialysis' not in primary_unit else 'non_nurse'
        shift_str = 'day' if shift == 'Day shift' else 'night'

        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        
        realizd_df = load_sensor_data.read_preprocessed_realizd(data_config.realizd_sensor_dict['preprocess_path'], participant_id)
        fitbit_df = load_sensor_data.read_preprocessed_fitbit(data_config.fitbit_sensor_dict['preprocess_path'], participant_id)
        days_at_work_df = load_sensor_data.read_preprocessed_days_at_work(data_config.days_at_work_path, participant_id)
        
        if realizd_df is not None and fitbit_df is not None:
            
            dates_range = (pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days
            start_time = pd.to_datetime(realizd_df.index[0])
            
            daily_df = pd.DataFrame(index=[np.arange(0, num_point_per_day)], columns=['Phone', 'HeartRatePPG', 'StepCount'])
            daily_array = np.zeros([dates_range, num_point_per_day, 3])
            
            daily_array[:, :, :] = np.nan
            daily_df.loc[:, :] = np.nan
            days_list = [(pd.to_datetime(realizd_df.index[0]).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3] for i in range(dates_range)]
            
            if days_at_work_df is None:
                continue

            days_at_work_list = []
            days_at_work_df = days_at_work_df.dropna()
            days_at_work_array = np.zeros([dates_range, 1])
            for day in list(days_at_work_df.index):
                if day in days_list: days_at_work_list.append(day)
                
            days_at_work_list = list(set(days_at_work_list))
            days_at_work_list.sort()
            
            days_at_work_daily_df = pd.DataFrame(index=[np.arange(0, num_point_per_day)], columns=['Phone', 'HeartRatePPG', 'StepCount'])
            days_off_work_daily_df = pd.DataFrame(index=[np.arange(0, num_point_per_day)], columns=['Phone', 'HeartRatePPG', 'StepCount'])

            for i in range(dates_range):
                dates_str = (pd.to_datetime(realizd_df.index[0]) + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3]

                start_str = (start_time + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3]
                end_str = (start_time + timedelta(days=i+1)).strftime(load_data_basic.date_time_format)[:-3]

                realizd_day_df = realizd_df[start_str:end_str]
                fitbit_day_df = fitbit_df[start_str:end_str]
                day_usage = np.nansum(np.array(realizd_day_df))
                fitbit_len = len(fitbit_day_df.dropna())
                
                for j in range(num_point_per_day):
                    start_str = (start_time + timedelta(days=i) + timedelta(minutes=j*offset-int(window/2))).strftime(load_data_basic.date_time_format)[:-3]
                    end_str = (start_time + timedelta(days=i) + timedelta(minutes=j*offset+int(window/2))).strftime(load_data_basic.date_time_format)[:-3]
                    
                    if day_usage > 300:
                        data_df = realizd_df[start_str:end_str]
                        daily_array[i, j, 0] = np.nanmean(np.array(data_df))
                    
                    if fitbit_len > 360:
                        data_df = fitbit_df[start_str:end_str]
                        if len(data_df['HeartRatePPG'].dropna()) > window / 2:
                            daily_array[i, j, 1] = np.nanmean(data_df['HeartRatePPG'])
                        if len(data_df['StepCount'].dropna()) > window / 2:
                            daily_array[i, j, 2] = np.nanmean(data_df['StepCount'].dropna().astype(int))
                
                if dates_str in days_at_work_list:
                    days_at_work_array[i] = 1
                else:
                    days_at_work_array[i] = 0

            daily_df.loc[:, :] = np.nanmean(daily_array, axis=0)

            days_at_work_daily_array = daily_array[np.where(days_at_work_array == 1)[0]]
            days_off_work_daily_array = daily_array[np.where(days_at_work_array == 0)[0]]
            days_at_work_daily_df.loc[:, :] = np.nanmean(days_at_work_daily_array, axis=0)
            days_off_work_daily_df.loc[:, :] = np.nanmean(days_off_work_daily_array, axis=0)
            
            participant_id_shift_dict = {}
            participant_id_shift_dict['days_list'] = days_list
            participant_id_shift_dict['daily_array'] = daily_array
            participant_id_shift_dict['daily_data'] = daily_df

            participant_id_shift_dict['days_at_work'] = days_at_work_array

            participant_id_shift_dict['days_at_work_daily_array'] = days_at_work_daily_array
            participant_id_shift_dict['days_off_daily_array'] = days_off_work_daily_array

            participant_id_shift_dict['days_at_work_daily_data'] = days_at_work_daily_df
            participant_id_shift_dict['days_off_work_daily_data'] = days_off_work_daily_df

            output = open(os.path.join(chi_data_config.save_path, participant_id + '.pkl'), 'wb')
            pickle.dump(participant_id_shift_dict, output)


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)