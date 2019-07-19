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
import dpmm

import dpgmm_gibbs
from vdpgmm import VDPGMM

from pybgmm.prior import NIW
from pybgmm.igmm import PCRPMM
from datetime import timedelta
from dpkmeans import dpmeans
from collections import Counter

from sklearn import mixture
import pickle


def return_agg_daily_average(chi_data_config, realizd_df, fitbit_df, days_at_work_list):
    # Basic setting
    num_point_per_day = chi_data_config.num_point_per_day
    offset = 1440 / num_point_per_day
    
    dates_range = (pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days
    start_time = pd.to_datetime(realizd_df.index[0]).replace(hour=0, minute=0)
    
    data_array = np.zeros([dates_range, num_point_per_day, 3])
    data_array[:, :, :] = np.nan

    days_at_work_array = np.zeros([dates_range, 1])
    
    # Iterate dates
    for i in range(dates_range):
        dates_str = (pd.to_datetime(realizd_df.index[0]) + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3]
    
        day_start_str = (start_time + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3]
        day_end_str = (start_time + timedelta(days=i+1) - timedelta(seconds=1)).strftime(load_data_basic.date_time_format)[:-3]
        
        realizd_day_df = realizd_df[day_start_str:day_end_str]
        fitbit_day_df = fitbit_df[day_start_str:day_end_str]
        fitbit_len = len(fitbit_day_df.dropna())

        day_usage = np.nansum(np.array(realizd_day_df))
        
        time_list = [(start_time + timedelta(days=i) + timedelta(minutes=j * offset)).strftime(load_data_basic.date_time_format)[:-3] for j in range(num_point_per_day)]
        daily_df = pd.DataFrame(index=time_list, columns=['SecondsOnPhone', 'HeartRatePPG', 'StepCount'])

        if day_usage < 300:
            continue

        if fitbit_len < 120:
            continue

        realizd_time_list = list(realizd_day_df.index)
        daily_df.loc[realizd_time_list, 'SecondsOnPhone'] = realizd_day_df.loc[realizd_time_list,'SecondsOnPhone']
        daily_df.loc[realizd_time_list, 'HeartRatePPG'] = fitbit_day_df.loc[realizd_time_list, 'HeartRatePPG']
        daily_df.loc[realizd_time_list, 'StepCount'] = fitbit_day_df.loc[realizd_time_list, 'StepCount']

        data_array[i, :, :] = np.array(daily_df)

        if dates_str in days_at_work_list:
            days_at_work_array[i] = 1
        else:
            days_at_work_array[i] = 0
    
    daily_array = np.nanmean(data_array, axis=0)
    inds = np.where(np.isnan(daily_array))
    daily_array[inds] = np.take(np.nanmean(daily_array, axis=0), inds[1])
    
    days_at_work_daily_array = np.nanmean(data_array[np.where(days_at_work_array == 1)[0]], axis=0)
    inds = np.where(np.isnan(days_at_work_daily_array))
    days_at_work_daily_array[inds] = np.take(np.nanmean(days_at_work_daily_array, axis=0), inds[1])
    
    days_off_work_daily_array = np.nanmean(data_array[np.where(days_at_work_array == 0)[0]], axis=0)
    inds = np.where(np.isnan(days_off_work_daily_array))
    days_off_work_daily_array[inds] = np.take(np.nanmean(days_off_work_daily_array, axis=0), inds[1])
    
    return daily_array, days_at_work_daily_array, days_off_work_daily_array


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
    load_data_path.load_chi_clustering_path(chi_data_config, process_data_path, agg='week')
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')

    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    num_point_per_day = chi_data_config.num_point_per_day
    offset = 1440 / num_point_per_day
    window = chi_data_config.window
    
    for idx, participant_id in enumerate(top_participant_id_list[0:]):

        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        if os.path.exists(os.path.join(chi_data_config.clustering_save_path, participant_id + '.pkl')) is False:
            continue

        pkl_file = open(os.path.join(chi_data_config.clustering_save_path, participant_id + '.pkl'), 'rb')
        week_dict = pickle.load(pkl_file)

        agg_dis_array = week_dict['agg_dis']
        
        week_list = [key_str for key_str in list(week_dict.keys()) if len(week_dict[key_str]) == 2]
        dis_array = np.zeros([len(week_list),
                              week_dict[week_list[0]]['dis'].shape[0],
                              week_dict[week_list[0]]['dis'].shape[1]])
        
        for i, week in enumerate(week_list):
            if len(week_dict[week]) == 2:
                dis = week_dict[week]['dis']
                dis_array[i, :, :] = dis
            
        print()
        

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)