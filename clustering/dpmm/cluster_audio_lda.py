"""
Cluster the audio data
"""
from __future__ import print_function

from sklearn import mixture
import os
import sys
import pandas as pd
import numpy as np
import dpmm
from collections import Counter
import dpgmm_gibbs
from vdpgmm import VDPGMM

from pybgmm.prior import NIW
from pybgmm.igmm import PCRPMM
from datetime import timedelta

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SEED = 5132290  # from random.org
np.random.seed(SEED)


def lda_audio(data_df, cluster_df, data_config, participant_id, lda_components='auto'):
    data_cluster_path = data_config.audio_sensor_dict['clustering_path']
    
    X = np.array(data_df)
    y = np.array(cluster_df['cluster'])
    
    if lda_components == 'auto':
        if len(np.unique(y)) < data_df.shape[1]:
            lda_array = LinearDiscriminantAnalysis(n_components=None).fit(X, y).transform(X)
        else:
            lda_array = LinearDiscriminantAnalysis(n_components=data_df.shape[1]).fit(X, y).transform(X)
    else:
        lda_array = LinearDiscriminantAnalysis(n_components=int(lda_components)).fit(X, y).transform(X)
    
    lda_df = pd.DataFrame(lda_array, index=list(cluster_df.index))
    if os.path.exists(os.path.join(data_cluster_path, participant_id)) is False:
        os.mkdir(os.path.join(data_cluster_path, participant_id))
    lda_df.to_csv(os.path.join(data_cluster_path, participant_id, data_config.audio_sensor_dict['lda_path']), compression='gzip')

    return lda_df


def cluster_audio(data_df, data_config, participant_id, iter=100, cluster_name='utterance_cluster'):
    
    if data_config.audio_sensor_dict['cluster_method'] == 'collapsed_gibbs':
        dpgmm = dpmm.DPMM(n_components=20, alpha=float(data_config.audio_sensor_dict['cluster_alpha']))
        dpgmm.fit_collapsed_Gibbs(np.array(data_df))
        cluster_id = dpgmm.predict(np.array(data_df))
    elif data_config.audio_sensor_dict['cluster_method'] == 'gibbs':
        cluster_id = dpgmm_gibbs.DPMM(np.array(data_df), alpha=float(data_config.audio_sensor_dict['cluster_alpha']), iter=iter, K=50)
    elif data_config.audio_sensor_dict['cluster_method'] == 'vdpgmm':
        model = VDPGMM(T=20, alpha=float(data_config.audio_sensor_dict['cluster_alpha']), max_iter=1000)
        model.fit(np.array(data_df))
        cluster_id = model.predict(np.array(data_df))
    elif data_config.audio_sensor_dict['cluster_method'] == 'pcrpmm':
        # Model parameters
        alpha = float(data_config.audio_sensor_dict['cluster_alpha'])
        K = 100  # initial number of components
        n_iter = 300
        
        D = np.array(data_df).shape[1]
        
        # Generate data
        mu_scale = 1
        covar_scale = 1
        
        # Intialize prior
        m_0 = np.zeros(D)
        k_0 = covar_scale ** 2 / mu_scale ** 2
        v_0 = D + 3
        S_0 = covar_scale ** 2 * v_0 * np.eye(D)
        prior = NIW(m_0, k_0, v_0, S_0)
        
        ## Setup PCRPMM
        pcrpmm = PCRPMM(np.array(data_df), prior, alpha, save_path=None, assignments="rand", K=K)
        
        ## Perform collapsed Gibbs sampling
        pcrpmm.collapsed_gibbs_sampler(n_iter, n_power=1.1, num_saved=1)
        cluster_id = pcrpmm.components.assignments
    else:
        dpgmm = mixture.BayesianGaussianMixture(n_components=20, covariance_type='full').fit(np.array(data_df))
        cluster_id = dpgmm.predict(np.array(data_df))
    
    print(Counter(cluster_id))
    cluster_df = data_df.copy()
    cluster_df.loc[:, 'cluster'] = cluster_id

    return cluster_df
    
    
def read_feature_data_df(data_config, participant_id):
    filter_data_path = data_config.audio_sensor_dict['filter_path']
    
    # Read only relevant data
    config_cond = 'pause_threshold_' + str(data_config.audio_sensor_dict['pause_threshold']) + '_' + data_config.audio_sensor_dict['audio_feature']
    file_list = [file for file in os.listdir(os.path.join(filter_data_path, participant_id)) if data_config.audio_sensor_dict['cluster_data'] in file and config_cond in file]
    
    data_df = pd.DataFrame()
    
    for file in file_list:
        tmp_raw_audio_df = pd.read_csv(os.path.join(filter_data_path, participant_id, file), index_col=0)
        if len(tmp_raw_audio_df) < 3:
            continue
        
        # if cluster raw_audio
        if data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
            tmp_raw_audio_df = tmp_raw_audio_df.drop(columns=['F0_sma'])
            data_df = data_df.append(tmp_raw_audio_df)
        # if cluster utterance
        elif data_config.audio_sensor_dict['cluster_data'] == 'utterance':
            if os.path.exists(os.path.join(filter_data_path, participant_id, file)) is True:
                day_utterance_df = pd.read_csv(os.path.join(filter_data_path, participant_id, file), index_col=0)
                data_df = data_df.append(day_utterance_df)
        
        elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
            if os.path.exists(os.path.join(filter_data_path, participant_id, file)) is True:
                day_minute_df = pd.read_csv(os.path.join(filter_data_path, participant_id, file), index_col=0)
                data_df = data_df.append(day_minute_df)
                
    return data_df


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
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    filter_data_path = data_config.audio_sensor_dict['filter_path']
    
    for idx, participant_id in enumerate(top_participant_id_list[:]):
        
        print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # Read other sensor data, the aim is to detect whether people workes during a day
        folder_exist_cond = os.path.exists(os.path.join(filter_data_path, participant_id))
        if folder_exist_cond is False:
            continue
        enough_data_cond = len(os.listdir(os.path.join(filter_data_path, participant_id))) < 3
        if enough_data_cond:
            continue
        
        # Read data
        data_df = read_feature_data_df(data_config, participant_id)
        if len(data_df) == 0:
            continue

        # if cluster utterance
        if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
            cluster_name = 'utterance_cluster'
        elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
            cluster_name = 'minute_cluster'
        # process audio feature for cluster
        elif data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
            cluster_name = 'raw_audio_cluster'
        else:
            cluster_name = 'raw_audio_cluster'
            
        # Cluster audio
        data_df = (data_df - data_df.mean()) / data_df.std()
        cluster_df = cluster_audio(data_df, data_config, participant_id, iter=100, cluster_name=cluster_name)
        data_cluster_path = data_config.audio_sensor_dict['clustering_path']

        if os.path.exists(os.path.join(data_cluster_path, participant_id)) is False:
            os.mkdir(os.path.join(data_cluster_path, participant_id))
        cluster_df.to_csv(os.path.join(data_cluster_path, participant_id, cluster_name + '.csv.gz'), compression='gzip')

        # LDA transform
        lda_df = lda_audio(data_df, cluster_df, data_config, participant_id, lda_components=data_config.audio_sensor_dict['lda_num'])

        # cluster lda feature
        final_cluster_df = cluster_audio(lda_df, data_config, participant_id, iter=100, cluster_name=cluster_name)
        if os.path.exists(os.path.join(data_cluster_path, participant_id)) is False:
            os.mkdir(os.path.join(data_cluster_path, participant_id))
        final_cluster_df.to_csv(os.path.join(data_cluster_path, participant_id, data_config.audio_sensor_dict['lda_clustering_path']), compression='gzip')
        
        print(participant_id + ': success')
        

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)
