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
from scipy.stats import entropy


def KL(P,Q):
    """
    Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0.
    """
    epsilon = 0.00001
    
    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon
    
    divergence = np.sum(P*np.log(P/Q))
    return divergence


def return_agg_daily_average(chi_data_config, realizd_df, fitbit_df):
    # Basic setting
    num_point_per_day = chi_data_config.num_point_per_day
    
    dates_range = (pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days
    data_array = np.zeros([dates_range, num_point_per_day, 4])
    data_array[:, :, :] = np.nan

    # Iterate dates
    for i in range(dates_range):
    
        daily_array = read_day_data(realizd_df, fitbit_df, i)
        if daily_array is not None:
            data_array[i, :, :] = daily_array

    daily_array = np.nanmean(data_array, axis=0)
    inds = np.where(np.isnan(daily_array))
    daily_array[inds] = np.take(np.nanmean(daily_array, axis=0), inds[1])
    
    return daily_array


def cluster_data(data, data_config, iter=100):
    cluster_df = data.copy()
    data_df = data.copy()
    min_array, max_array = np.array(data_df.min()), np.array(data_df.max())
    data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())
    
    if data_config.cluster_dict['cluster_method'] == 'collapsed_gibbs':
        dpgmm = dpmm.DPMM(n_components=20, alpha=float(data_config.cluster_dict['cluster_alpha']))
        dpgmm.fit_collapsed_Gibbs(np.array(data_df))
        cluster_id = dpgmm.predict(np.array(data_df))
        model = dpgmm
    elif data_config.cluster_dict['cluster_method'] == 'gibbs':
        cluster_id = dpgmm_gibbs.DPMM(np.array(data_df), alpha=float(data_config.cluster_dict['cluster_alpha']), iter=iter, K=50)
        model = dpgmm_gibbs
    elif data_config.cluster_dict['cluster_method'] == 'vdpgmm':
        vdpgmm = VDPGMM(T=20, alpha=float(data_config.cluster_dict['cluster_alpha']), max_iter=iter)
        vdpgmm.fit(np.array(data_df))
        cluster_id = vdpgmm.predict(np.array(data_df))
        model = vdpgmm
    elif data_config.cluster_dict['cluster_method'] == 'dpkmeans':
        dp = dpmeans(np.array(data_df))
        cluster_id, obj, em_time = dp.fit(np.array(data_df))
        model = dp
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
        model = pcrpmm
    else:
        dpgmm = mixture.BayesianGaussianMixture(n_components=10, max_iter=1000, covariance_type='full').fit(np.array(data_df))
        cluster_id = dpgmm.predict(np.array(data_df))
        model = dpgmm
    
    print(Counter(cluster_id))
    cluster_df.loc[:, 'cluster'] = cluster_id
    unique_cluster_list = list(set(list(cluster_id)))

    return model, unique_cluster_list, min_array, max_array


def clustering_data(chi_data_config, agg, realizd_df, fitbit_df, agg_daily_array):
    num_point_per_day = chi_data_config.num_point_per_day
    offset = 1440 / num_point_per_day
    
    dates_range = int((pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days / agg)
    start_time = pd.to_datetime(realizd_df.index[0]).replace(hour=0, minute=0)
    final_data_df = pd.DataFrame()
    
    if dates_range < 4:
        return None, None, None, None
    
    for i in range(dates_range):
        
        time_list = [(start_time + timedelta(days=i*agg, minutes=j*offset)).strftime(load_data_basic.date_time_format)[:-3] for j in range(num_point_per_day)]
        dates_df = pd.DataFrame(index=time_list, columns=['SecondsOnPhone', 'NumberOfTime', 'HeartRatePPG', 'StepCount'])
        
        dates_array = np.zeros([agg, num_point_per_day, 4])
        dates_array[:, :, :] = np.nan
        
        for j in range(agg):
            daily_array = read_day_data(realizd_df, fitbit_df, i*agg+j)
            
            if daily_array is not None:
                dates_array[j, :, :] = np.array(daily_array).astype(float)
        
        agg_dates_array = np.nanmean(dates_array, axis=0)
        inds = np.where(np.isnan(agg_dates_array))
        
        if len(inds[0]) < 720:
            agg_dates_array[inds] = agg_daily_array[inds]
            dates_df.loc[:, :] = agg_dates_array
            final_data_df = final_data_df.append(dates_df)
            
    if len(final_data_df) > 1440 * 4:
        physio_model, physio_cluster_list, min_array, max_array = cluster_data(final_data_df[['SecondsOnPhone', 'NumberOfTime']], chi_data_config, iter=300)
        return physio_model, physio_cluster_list, min_array, max_array
    else:
        return None, None, None, None
    
        
def read_day_data(realizd_df, fitbit_df, days):
    start_time = pd.to_datetime(realizd_df.index[0]).replace(hour=0, minute=0)
    day_start_str = (start_time + timedelta(days=days)).strftime(load_data_basic.date_time_format)[:-3]
    day_end_str = (start_time + timedelta(days=days + 1) - timedelta(seconds=1)).strftime(load_data_basic.date_time_format)[:-3]
    
    # Read data
    realizd_day_df = realizd_df[day_start_str:day_end_str]
    fitbit_day_df = fitbit_df[day_start_str:day_end_str]
    day_usage = np.nansum(np.array(realizd_day_df))
    fitbit_len = len(fitbit_day_df.dropna())
    
    if day_usage < 300:
        return None
    
    if fitbit_len < 120:
        return None
    
    realizd_time_list = list(realizd_day_df.index)
    daily_df = pd.DataFrame(index=realizd_time_list, columns=['SecondsOnPhone', 'NumberOfTime', 'HeartRatePPG', 'StepCount'])
    
    daily_df.loc[realizd_time_list, 'SecondsOnPhone'] = realizd_day_df.loc[realizd_time_list, 'SecondsOnPhone']
    daily_df.loc[realizd_time_list, 'NumberOfTime'] = realizd_day_df.loc[realizd_time_list, 'NumberOfTime']
    daily_df.loc[realizd_time_list, 'HeartRatePPG'] = fitbit_day_df.loc[realizd_time_list, 'HeartRatePPG']
    daily_df.loc[realizd_time_list, 'StepCount'] = fitbit_day_df.loc[realizd_time_list, 'StepCount']
    
    return np.array(daily_df).astype(float)


def cal_shuffle_dist(agg, sliding, num_point_per_day, realizd_df, fitbit_df, agg_daily_array,
                     realizd_model, realizd_cluster_list, min_array, max_array):
    
    dates_range = int(((pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days - agg) / sliding)
    dist_dict = {}
    
    for dates in range(dates_range):
        dist_dict[dates] = {}
        
        for shuffle_idx in range(100):
            if shuffle_idx % 20 == 0:
                print('days: %d, shuffle: %d' % (dates * sliding, shuffle_idx))
            dates_range_array = np.arange(dates * sliding, dates * sliding + agg + sliding, 1)
            np.random.shuffle(dates_range_array)
            first_group = dates_range_array[:agg]
            second_group = dates_range_array[sliding:]
    
            first_group_data_array = np.zeros([agg, num_point_per_day, 4])
            second_group_data_array = np.zeros([agg, num_point_per_day, 4])

            first_group_data_array[:, :, :] = np.nan
            second_group_data_array[:, :, :] = np.nan
        
            for j, days in enumerate(first_group):
        
                first_group_day_array = read_day_data(realizd_df, fitbit_df, int(days))
                second_group_day_array = read_day_data(realizd_df, fitbit_df, int(second_group[j]))
                
                if first_group_day_array is not None: first_group_data_array[j, :, :] = first_group_day_array
                if second_group_day_array is not None: second_group_data_array[j, :, :] = second_group_day_array
    
            agg_first_array = np.nanmean(first_group_data_array, axis=0)
            agg_second_array = np.nanmean(second_group_data_array, axis=0)
            first_inds, second_inds = np.where(np.isnan(agg_first_array)), np.where(np.isnan(agg_second_array))
            
            agg_first_array[first_inds] = agg_daily_array[first_inds]
            agg_second_array[second_inds] = agg_daily_array[second_inds]
    
            agg_first_array = (agg_first_array[:, :2] - min_array) / (max_array - min_array)
            agg_second_array = (agg_second_array[:, :2] - min_array) / (max_array - min_array)
    
            first_cluster_array = realizd_model.predict(agg_first_array)
            second_cluster_array = realizd_model.predict(agg_second_array)
    
            first_distribution, second_distribution = np.zeros([48, len(realizd_cluster_list)]), np.zeros([48, len(realizd_cluster_list)])
    
            kl_realizd_dist_array = np.zeros([1, 48])
            
            for i in range(48):
                tmp_cluster_array = first_cluster_array[i * 30:(i + 1) * 30]
                counter_dict = Counter(tmp_cluster_array)
                for cluster_id in list(counter_dict.keys()):
                    if cluster_id in realizd_cluster_list:
                        first_distribution[i, realizd_cluster_list.index(cluster_id)] = counter_dict[cluster_id]
    
                tmp_cluster_array = second_cluster_array[i * 30:(i + 1) * 30]
                counter_dict = Counter(tmp_cluster_array)
                for cluster_id in list(counter_dict.keys()):
                    if cluster_id in realizd_cluster_list:
                        second_distribution[i, realizd_cluster_list.index(cluster_id)] = counter_dict[cluster_id]
    
                epsilon = 0.00001
    
                first_pdf, second_pdf = first_distribution[i, :], second_distribution[i, :]
                first_pdf, second_pdf = first_pdf / np.sum(first_pdf), second_pdf / np.sum(second_pdf)
                kl_realizd_dist = entropy(first_pdf + epsilon, second_pdf + epsilon) + entropy(second_pdf + epsilon, first_pdf + epsilon)
                kl_realizd_dist = kl_realizd_dist / 2
                kl_realizd_dist_array[0, i] = kl_realizd_dist

            dist_dict[dates][shuffle_idx] = {}
            dist_dict[dates][shuffle_idx]['first_realizd'] = first_cluster_array
            dist_dict[dates][shuffle_idx]['second_realizd'] = second_cluster_array
            dist_dict[dates][shuffle_idx]['realizd_dist'] = kl_realizd_dist_array
    
    return dist_dict


def cal_regular_dist(agg, sliding, num_point_per_day, realizd_df, fitbit_df,
                     agg_daily_array, realizd_model, realizd_cluster_list, min_array, max_array):
    dates_range = int(((pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days - agg) / sliding)
    
    dist_dict = {}
    for dates in range(dates_range):
        dates_range_array = np.arange(dates * sliding, dates * sliding + agg + sliding, 1)
        first_group = dates_range_array[:agg]
        second_group = dates_range_array[sliding:]
        
        first_group_data_array = np.zeros([agg, num_point_per_day, 4])
        second_group_data_array = np.zeros([agg, num_point_per_day, 4])

        first_group_data_array[:, :, :] = np.nan
        second_group_data_array[:, :, :] = np.nan
        
        for j, days in enumerate(first_group):
            
            first_group_day_array = read_day_data(realizd_df, fitbit_df, int(days))
            second_group_day_array = read_day_data(realizd_df, fitbit_df, int(second_group[j]))
            
            if first_group_day_array is not None: first_group_data_array[j, :, :] = first_group_day_array
            if second_group_day_array is not None: second_group_data_array[j, :, :] = second_group_day_array
        
        agg_first_array = np.nanmean(first_group_data_array, axis=0)
        agg_second_array = np.nanmean(second_group_data_array, axis=0)
        first_inds, second_inds = np.where(np.isnan(agg_first_array)), np.where(np.isnan(agg_second_array))
        
        agg_first_array[first_inds] = agg_daily_array[first_inds]
        agg_second_array[second_inds] = agg_daily_array[second_inds]

        agg_first_array = (agg_first_array[:, :2] - min_array) / (max_array - min_array)
        agg_second_array = (agg_second_array[:, :2] - min_array) / (max_array - min_array)
        
        first_cluster_array = realizd_model.predict(agg_first_array)
        second_cluster_array = realizd_model.predict(agg_second_array)
        
        first_distribution, second_distribution = np.zeros([48, len(realizd_cluster_list)]), np.zeros([48, len(realizd_cluster_list)])

        kl_realizd_dist_array = np.zeros([1, 48])
        for i in range(48):
            
            tmp_cluster_array = first_cluster_array[i * 30:(i + 1) * 30]
            counter_dict = Counter(tmp_cluster_array)
            for cluster_id in list(counter_dict.keys()):
                if cluster_id in realizd_cluster_list:
                    first_distribution[i, realizd_cluster_list.index(cluster_id)] = counter_dict[cluster_id]
            
            tmp_cluster_array = second_cluster_array[i * 30:(i + 1) * 30]
            counter_dict = Counter(tmp_cluster_array)
            for cluster_id in list(counter_dict.keys()):
                if cluster_id in realizd_cluster_list:
                    second_distribution[i, realizd_cluster_list.index(cluster_id)] = counter_dict[cluster_id]

            epsilon = 0.00001
            
            first_pdf, second_pdf = first_distribution[i, :], second_distribution[i, :]
            first_pdf, second_pdf = first_pdf / np.sum(first_pdf), second_pdf / np.sum(second_pdf)
            kl_realizd_dist = entropy(first_pdf + epsilon, second_pdf + epsilon) + entropy(second_pdf + epsilon, first_pdf + epsilon)
            kl_realizd_dist = kl_realizd_dist / 2
            kl_realizd_dist_array[0, i] = kl_realizd_dist

        dist_dict[dates] = {}
        dist_dict[dates]['first_realizd'] = first_cluster_array
        dist_dict[dates]['second_realizd'] = second_cluster_array
        dist_dict[dates]['realizd_dist'] = kl_realizd_dist_array
    
    return dist_dict


def activity_curve_change(chi_data_config, agg, sliding, realizd_df, fitbit_df, agg_daily_array,
                          realizd_model, realizd_cluster_list, min_array, max_array):
    num_point_per_day = chi_data_config.num_point_per_day
    dates_range = int((pd.to_datetime(realizd_df.index[-1]) - pd.to_datetime(realizd_df.index[0])).days / sliding)
    
    if dates_range < 20:
        return None
    
    dist_dict = {}
    
    reg_dist_dict = cal_regular_dist(agg, sliding, num_point_per_day, realizd_df, fitbit_df, agg_daily_array,
                                     realizd_model, realizd_cluster_list, min_array, max_array)
    
    shuffle_dist_dict = cal_shuffle_dist(agg, sliding, num_point_per_day, realizd_df, fitbit_df, agg_daily_array,
                                         realizd_model, realizd_cluster_list, min_array, max_array)

    dist_dict['regular'] = reg_dist_dict
    dist_dict['shuffle'] = shuffle_dist_dict

    return dist_dict
    

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
    agg, sliding = 10, 2
    
    load_data_path.load_chi_preprocess_path(chi_data_config, process_data_path)
    load_data_path.load_chi_activity_curve_path(chi_data_config, process_data_path, agg=10, sliding=2)
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')

    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list[100:150]):

        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        # Read data first
        realizd_df = load_sensor_data.read_preprocessed_realizd(data_config.realizd_sensor_dict['preprocess_path'], participant_id)
        fitbit_df = load_sensor_data.read_preprocessed_fitbit(data_config.fitbit_sensor_dict['preprocess_path'], participant_id)
        
        if realizd_df is None or fitbit_df is None:
            continue
            
        if np.nansum(np.array(realizd_df['SecondsOnPhone'])) < 14000:
            continue

        realizd_df.loc[realizd_df['SecondsOnPhone'] > 60].loc[:, 'SecondsOnPhone'] = 60
        
        # Aggregated globally
        agg_daily_array = return_agg_daily_average(chi_data_config, realizd_df, fitbit_df)
        
        # Clustering
        realizd_model, realizd_cluster_list, min_array, max_array = clustering_data(chi_data_config, agg, realizd_df, fitbit_df, agg_daily_array)
        
        if realizd_model is None:
            continue
        
        # Calculate change score
        dist_dict = activity_curve_change(chi_data_config, agg, sliding, realizd_df, fitbit_df,
                                          agg_daily_array, realizd_model, realizd_cluster_list, min_array, max_array)

        output = open(os.path.join(chi_data_config.activity_curve_path, participant_id + '_realizd.pkl'), 'wb')
        pickle.dump(dist_dict, output)
        

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)