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


def cluster_data(data, data_config, iter=100):
    cluster_df = data.copy()
    data_df = data.copy()
    data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())
    
    if data_config.cluster_dict['cluster_method'] == 'collapsed_gibbs':
        dpgmm = dpmm.DPMM(n_components=20, alpha=float(data_config.cluster_dict['cluster_alpha']))
        dpgmm.fit_collapsed_Gibbs(np.array(data_df))
        cluster_id = dpgmm.predict(np.array(data_df))
    elif data_config.cluster_dict['cluster_method'] == 'gibbs':
        cluster_id = dpgmm_gibbs.DPMM(np.array(data_df), alpha=float(data_config.cluster_dict['cluster_alpha']), iter=iter, K=50)
    elif data_config.cluster_dict['cluster_method'] == 'vdpgmm':
        model = VDPGMM(T=20, alpha=float(data_config.cluster_dict['cluster_alpha']), max_iter=iter)
        model.fit(np.array(data_df))
        cluster_id = model.predict(np.array(data_df))
    elif data_config.cluster_dict['cluster_method'] == 'dpkmeans':
        dp = dpmeans(np.array(data_df))
        cluster_id, obj, em_time = dp.fit(np.array(data_df))
    elif data_config.cluster_dict['cluster_method'] == 'pcrpmm':
        # Model parameters
        alpha = float(data_config.cluster_dict['cluster_alpha'])
        K = 100  # initial number of components
        n_iter = 300
        
        D = np.array(data_df).shape[1]
        
        # Intialize prior
        covar_scale = np.var(np.array(data_df))
        # covar_scale = np.median(LA.eigvals(np.cov(np.array(col_data_df).T)))
        mu_scale = np.amax(np.array(data_df)) - covar_scale
        m_0 = np.mean(np.array(data_df), axis=0)
        k_0 = covar_scale ** 2 / mu_scale ** 2
        # k_0 = 1. / 20
        v_0 = D + 3
        S_0 = covar_scale ** 2 * v_0 * np.eye(D)
        # S_0 = 1. * np.eye(D)
        
        prior = NIW(m_0, k_0, v_0, S_0)
        
        ## Setup PCRPMM
        pcrpmm = PCRPMM(np.array(data_df), prior, alpha, save_path=None, assignments="rand", K=K)
        
        ## Perform collapsed Gibbs sampling
        pcrpmm.collapsed_gibbs_sampler(n_iter, n_power=float(data_config.cluster_dict['power']), num_saved=1)
        cluster_id = pcrpmm.components.assignments
    else:
        dpgmm = mixture.BayesianGaussianMixture(n_components=20, max_iter=500, covariance_type='full').fit(np.array(data_df))
        cluster_id = dpgmm.predict(np.array(data_df))
    
    print(Counter(cluster_id))
    cluster_df.loc[:, 'cluster'] = cluster_id

    return cluster_df


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

        realizd_df = load_sensor_data.read_preprocessed_realizd(data_config.realizd_sensor_dict['preprocess_path'], participant_id)
        fitbit_df = load_sensor_data.read_preprocessed_fitbit(data_config.fitbit_sensor_dict['preprocess_path'], participant_id)
        days_at_work_df = load_sensor_data.read_preprocessed_days_at_work(data_config.days_at_work_path, participant_id)
        
        if realizd_df is None or fitbit_df is None:
            continue
            
        if np.nansum(np.array(realizd_df['SecondsOnPhone'])) < 14000:
            continue
            
        # Days at work feature
        if days_at_work_df is None:
            continue

        realizd_df.loc[realizd_df['SecondsOnPhone'] > 60].loc[:, 'SecondsOnPhone'] = 60

        days_at_work_list = []
        dates_range = (pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days
        days_list = [(pd.to_datetime(realizd_df.index[0]).replace(hour=0, minute=0, second=0,microsecond=0) + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3] for i in range(dates_range)]

        days_at_work_df = days_at_work_df.dropna()
        days_at_work_array = np.zeros([dates_range, 1])
        for day in list(days_at_work_df.index):
            if day in days_list: days_at_work_list.append(day)

        days_at_work_list = list(set(days_at_work_list))
        days_at_work_list.sort()
        
        # Aggregated
        agg_daily_array, agg_days_at_work_daily_array, agg_days_off_work_daily_array = return_agg_daily_average(chi_data_config, realizd_df, fitbit_df, days_at_work_list)
        
        week_range = int((pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days / 7)
        start_time = pd.to_datetime(realizd_df.index[0]).replace(hour=0, minute=0)
        
        final_data_df = pd.DataFrame()
        
        week_dict = {}
        
        if week_range < 6:
            continue
        
        for i in range(week_range):
    
            time_list = [(start_time + timedelta(weeks=i) + timedelta(minutes=j * offset)).strftime(load_data_basic.date_time_format)[:-3] for j in range(num_point_per_day)]
            week_df = pd.DataFrame(index=time_list, columns=['SecondsOnPhone', 'HeartRatePPG', 'StepCount'])

            week_array = np.zeros([7, num_point_per_day, 3])
            week_array[:, :, :] = np.nan

            for j in range(7):
                
                day_start_str = (start_time + timedelta(weeks=i, days=j)).strftime(load_data_basic.date_time_format)[:-3]
                day_end_str = (start_time + timedelta(weeks=i, days=j+1) - timedelta(seconds=1)).strftime(load_data_basic.date_time_format)[:-3]

                realizd_day_df = realizd_df[day_start_str:day_end_str]
                fitbit_day_df = fitbit_df[day_start_str:day_end_str]
                day_usage = np.nansum(np.array(realizd_day_df))
                fitbit_len = len(fitbit_day_df.dropna())
            
                if day_usage < 300:
                    continue

                if fitbit_len < 120:
                    continue

                realizd_time_list = list(realizd_day_df.index)
                daily_df = pd.DataFrame(index=realizd_time_list, columns=['SecondsOnPhone', 'HeartRatePPG', 'StepCount'])

                daily_df.loc[realizd_time_list, 'SecondsOnPhone'] = realizd_day_df.loc[realizd_time_list, 'SecondsOnPhone']
                daily_df.loc[realizd_time_list, 'HeartRatePPG'] = fitbit_day_df.loc[realizd_time_list, 'HeartRatePPG']
                daily_df.loc[realizd_time_list, 'StepCount'] = fitbit_day_df.loc[realizd_time_list, 'StepCount']
    
                week_array[j, :, :] = np.array(daily_df).astype(float)
            
            agg_week_array = np.nanmean(week_array, axis=0)
            inds = np.where(np.isnan(agg_week_array))

            week_dict['week_' + str(i)] = {}
            if len(inds[0]) < 720:
                agg_week_array[inds] = agg_daily_array[inds]
                
                week_df.loc[:, :] = agg_week_array
                final_data_df = final_data_df.append(week_df)
                week_dict['week_' + str(i)]['data'] = week_df
            else:
                week_dict['week_' + str(i)]['data'] = pd.DataFrame()
        
        if len(final_data_df) > 1440 * 5:
            final_data_df = cluster_data(final_data_df, chi_data_config, iter=300)

            unique_cluster_list = list(set(list(np.array(final_data_df['cluster']))))
            agg_dis_array = np.zeros([48, len(unique_cluster_list)])
            
            # Get distribution
            for key_str in list(week_dict.keys()):
                if len(week_dict[key_str]['data']) == 0:
                    continue
                
                time_list = list(week_dict[key_str]['data'].index)
                week_dict[key_str]['data'].loc[time_list, 'cluster'] = final_data_df.loc[time_list, 'cluster']

                week_distribution = np.zeros([48, len(unique_cluster_list)])
                week_data_df = week_dict[key_str]['data']
                week_cluster_array = np.array(week_data_df['cluster'])

                for i in range(48):
                    tmp_cluster_array = week_cluster_array[i * 30:(i + 1) * 30]
                    counter_dict = Counter(tmp_cluster_array)
    
                    for cluster_id in list(counter_dict.keys()):
                        week_distribution[i, int(cluster_id)] = counter_dict[cluster_id]

                    week_dict[key_str]['dis'] = week_distribution
                    agg_dis_array += week_distribution
            week_dict['agg_dis'] = agg_dis_array
            
            output = open(os.path.join(chi_data_config.clustering_save_path, participant_id + '.pkl'), 'wb')
            pickle.dump(week_dict, output)
            

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)