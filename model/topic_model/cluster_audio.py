"""
Filter the data
"""
from __future__ import print_function

from sklearn import mixture
import os
import sys
import pandas as pd
import numpy as np
import dpmm
from theano import tensor as tt
import pymc3 as pm
import random
from collections import Counter
import dpgmm_gibbs

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser

from filter import Filter

SEED = 5132290 # from random.org

np.random.seed(SEED)


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
    mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    alpha = 1
    if os.path.exists(os.path.join('result')) is False:
        os.mkdir(os.path.join('result'))
        
    if os.path.exists(os.path.join('result', 'clustering')) is False:
        os.mkdir(os.path.join('result', 'clustering'))

    if os.path.exists(os.path.join('result', 'clustering', 'alpha_' + str(alpha))) is False:
        os.mkdir(os.path.join('result', 'clustering', 'alpha_' + str(alpha)))

    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # Read id
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
        
        # Read other sensor data, the aim is to detect whether people workes during a day
        if len(os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id))) < 3:
            continue
        file_list = [file for file in os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) if 'utterance' not in file]
        
        raw_audio_df = pd.DataFrame()
        utterance_df = pd.DataFrame()
        
        for file in file_list:
            tmp_raw_audio_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file), index_col=0)
            raw_audio_df = raw_audio_df.append(tmp_raw_audio_df)
            tmp_raw_audio_df = tmp_raw_audio_df.drop(columns=['F0_sma'])
            
            if os.path.exists(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'utterance_' + file)) is True:
                day_utterance_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'utterance_' + file), index_col=0)
                utterance_df = utterance_df.append(day_utterance_df)
                continue
            
            time_diff = pd.to_datetime(list(tmp_raw_audio_df.index)[1:]) - pd.to_datetime(list(tmp_raw_audio_df.index)[:-1])
            time_diff = list(time_diff.total_seconds())

            change_point_start_list = [0]
            change_point_end_list = list(np.where(np.array(time_diff) > 1)[0])

            [change_point_start_list.append(change_point_end + 1) for change_point_end in change_point_end_list]
            change_point_end_list.append(len(tmp_raw_audio_df.index) - 1)

            time_start_end_list = []
            for i, change_point_end in enumerate(change_point_end_list):
                if 10 < change_point_end - change_point_start_list[i] < 10 * 100:
                    time_start_end_list.append([list(tmp_raw_audio_df.index)[change_point_start_list[i]], list(tmp_raw_audio_df.index)[change_point_end]])
            
            day_utterance_df = pd.DataFrame()
            for time_start_end in time_start_end_list:
                start_time = (pd.to_datetime(time_start_end[0])).strftime(load_data_basic.date_time_format)[:-3]
                end_time = (pd.to_datetime(time_start_end[1])).strftime(load_data_basic.date_time_format)[:-3]
                tmp_utterance_raw_df = tmp_raw_audio_df[start_time:end_time]
                tmp_utterance_df = pd.DataFrame(index=[list(tmp_utterance_raw_df.index)[0]])

                tmp_utterance_df['start'] = start_time
                tmp_utterance_df['end'] = end_time
                # tmp_utterance_df['duration'] = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds()
                for col in list(tmp_utterance_raw_df.columns):
                    tmp_utterance_df[col + '_mean'] = np.mean(np.array(tmp_utterance_raw_df[col]))
                    tmp_utterance_df[col + '_std'] = np.std(np.array(tmp_utterance_raw_df[col]))

                day_utterance_df = day_utterance_df.append(tmp_utterance_df)
            
            day_utterance_df.to_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'utterance_' + file), compression='gzip')
            utterance_df = utterance_df.append(day_utterance_df)

        utterance_norm_df = utterance_df.drop(columns=['start', 'end'])
        raw_audio_df_norm = (raw_audio_df - raw_audio_df.mean()) / raw_audio_df.std()
        utterance_norm_df = (utterance_norm_df - utterance_norm_df.mean()) / utterance_norm_df.std()

        # dpmm_model = dpmm.DPMM(n_components=-1, alpha=100)  # -1, 1, 2, 5
        # -1 means that we initialize with 1 cluster per point
        # dpmm_model.fit_collapsed_Gibbs(np.array(utterance_norm_df))
        # utterance_cluster_id = dpmm_model.predict(np.array(utterance_norm_df))
        if os.path.exists(os.path.join('result', 'clustering', 'alpha_' + str(alpha), participant_id)) is False:
            os.mkdir(os.path.join('result', 'clustering', 'alpha_' + str(alpha), participant_id))
        
        '''
        utterance_cluster_id = dpgmm_gibbs.DPMM(np.array(utterance_norm_df), alpha=alpha, iter=300, K=50)
        utterance_df.loc[:, 'cluster'] = utterance_cluster_id
        utterance_df.to_csv(os.path.join('result', 'clustering', 'alpha_' + str(alpha), participant_id, 'utterance_cluster_without_duration.csv.gz'), compression='gzip')
        '''
        
        '''
        raw_audio_cluster_id = dpgmm_gibbs.DPMM(np.array(raw_audio_df_norm), alpha=alpha, iter=100, K=50)
        raw_audio_df.loc[:, 'cluster'] = raw_audio_cluster_id
        raw_audio_df.to_csv(os.path.join('result', 'clustering', 'alpha_' + str(alpha), participant_id, 'raw_cluster.csv.gz'), compression='gzip')
        '''
        '''
        dpgmm = dpmm.DPMM(n_components=30, alpha=alpha)  # -1, 1, 2, 5
        # n_components is the number of initial clusters (at random, TODO k-means init)
        # -1 means that we initialize with 1 cluster per point
        dpgmm.fit_collapsed_Gibbs(np.array(utterance_norm_df))
        utterance_cluster_id = dpgmm.predict(np.array(utterance_norm_df))
        utterance_df.loc[:, 'cluster'] = utterance_cluster_id
        utterance_df.to_csv(os.path.join('result', 'clustering', 'alpha_' + str(alpha), participant_id, 'utterance_cluster_without_duration_collapsed_Gibbs.csv.gz'), compression='gzip')
        '''

        dpgmm = dpmm.DPMM(n_components=30, alpha=alpha)  # -1, 1, 2, 5
        # n_components is the number of initial clusters (at random, TODO k-means init)
        # -1 means that we initialize with 1 cluster per point
        dpgmm.fit_collapsed_Gibbs(np.array(raw_audio_df_norm))
        raw_audio_cluster_id = dpgmm.predict(np.array(raw_audio_df_norm))
        raw_audio_df.loc[:, 'cluster'] = raw_audio_cluster_id
        raw_audio_df.to_csv(os.path.join('result', 'clustering', 'alpha_' + str(alpha), participant_id, 'raw_cluster_collapsed_Gibbs.csv.gz'), compression='gzip')


        '''
        # Fit a Dirichlet process Gaussian mixture using five components
        utterance_dpgmm = mixture.BayesianGaussianMixture(n_components=50, covariance_type='full').fit(np.array(utterance_norm_df))
        utterance_cluster_id = utterance_dpgmm.predict(np.array(utterance_norm_df))
        utterance_df.loc[:, 'cluster'] = utterance_cluster_id
        utterance_df.to_csv(os.path.join('result', 'clustering', 'alpha_' + str(alpha), participant_id, 'utterance_cluster_without_duration.csv.gz'), compression='gzip')

        raw_audio_dpgmm = mixture.BayesianGaussianMixture(n_components=50, covariance_type='full').fit(np.array(raw_audio_df_norm))
        raw_audio_cluster_id = raw_audio_dpgmm.predict(np.array(raw_audio_df_norm))
        raw_audio_df.loc[:, 'cluster'] = raw_audio_cluster_id
        raw_audio_df.to_csv(os.path.join('result', 'clustering', 'alpha_' + str(alpha), participant_id, 'raw_cluster.csv.gz'), compression='gzip')
        '''

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining

if __name__ == '__main__':
    
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)

'''
K = 30

with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1., alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))

    tau = pm.Gamma('tau', 1., 1., shape=K)
    lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    mu = pm.Normal('mu', 0, tau=lambda_ * tau, shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau, observed=np.array(raw_audio_df_norm))

with model:
    trace = pm.sample(1000, random_seed=SEED)
# dpgmm = mixture.BayesianGaussianMixture(n_components=5, covariance_type='full').fit(np.array(raw_audio_df_norm))
# cluster_id = dpgmm.predict(np.array(raw_audio_df_norm))
'''

'''
n_samples = 50

# Generate random sample, two components
np.random.seed(0)

# 2, 10-dimensional Gaussians
C = np.eye(10)
for i in range(100):
    C[random.randint(0,9)][random.randint(0,9)] = random.random()
X = np.r_[np.dot(np.random.randn(n_samples, 10), C), .7 * np.random.randn(n_samples, 10) + np.array([-6, 3, 0, 5, -8, 0, 0, 0, -3, -2])]

# 2, 5-dimensional Gaussians
# C = np.eye(5)
# for i in xrange(25):
#    C[random.randint(0,4)][random.randint(0,4)] = random.random()
# X = np.r_[np.dot(np.random.randn(n_samples, 5), C),
#          .7 * np.random.randn(n_samples, 5) + np.array([-6, 3, 5, -8, -2])]

from sklearn import mixture

dpmm = dpmm.DPMM(n_components=-1)  # -1, 1, 2, 5
# n_components is the number of initial clusters (at random, TODO k-means init)
# -1 means that we initialize with 1 cluster per point
dpmm.fit_collapsed_Gibbs(X)
dpmm.predict(X)
'''

# raw_audio_cluster_id = dpmm.dpmm(np.array(raw_audio_df_norm))

'''
dpmm_model = dpmm.DPMM(n_components=-1)  # -1, 1, 2, 5
# -1 means that we initialize with 1 cluster per point
dpmm_model.fit_collapsed_Gibbs(np.array(raw_audio_df_norm))
dpmm_model.predict(np.array(raw_audio_df_norm))
'''