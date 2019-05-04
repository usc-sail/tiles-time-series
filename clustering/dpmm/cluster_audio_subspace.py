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

from scipy.stats import invwishart, invgamma, wishart

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


def cluster_audio(data_df, data_config, iter=100):
	cluster_df = data_df.copy()
	
	# ['F0final_sma', 'pcm_intensity_sma', 'pcm_loudness_sma', 'shimmerLocal_sma', 'jitterLocal_sma']
	# process_col_list = ['F0final_sma', 'pcm_loudness_sma', 'jitter', 'duration', 'F0_sma', 'logHNR_sma',
	# 					'spectral', 'logHNR_sma', 'audspecRasta_lengthL1norm_sma']
	process_col_list = ['F0_sma', 'duration', 'pcm_loudness_sma',
						'pcm_fftMag_spectralCentroid_sma', 'pcm_fftMag_spectralEntropy_sma',
						'audspecRasta_lengthL1norm_sma', 'audspec_lengthL1norm_sma']
	for process_col in process_col_list:
		if 'jitter' in process_col:
			col_data_df = data_df[['shimmerLocal_sma_mean', 'shimmerLocal_sma_std', 'jitterLocal_sma_mean', 'jitterLocal_sma_std']]
		elif 'duration' in process_col and 'duration' in data_config.audio_sensor_dict['audio_feature']:
			col_data_df = data_df[['mean_segment', 'foreground_ratio']]
		elif 'duration' in process_col:
			if 'duration' in data_config.audio_sensor_dict['audio_feature']:
				col_data_df = data_df[['num_segment', 'mean_segment', 'foreground_ratio']]
			else:
				col_data_df = pd.DataFrame()
		else:
			col_data_df = data_df[[process_col + '_mean', process_col + '_std']]
		'''
		elif 'spectral' in process_col:
			col_data_df = data_df[['pcm_fftMag_spectralCentroid_sma_mean', 'pcm_fftMag_spectralCentroid_sma_std',
			                       'pcm_fftMag_spectralEntropy_sma_mean', 'pcm_fftMag_spectralEntropy_sma_std']]
		'''
			
		if len(col_data_df) == 0:
			continue
			
		if data_config.audio_sensor_dict['cluster_method'] == 'collapsed_gibbs':
			dpgmm = dpmm.DPMM(n_components=20, alpha=float(data_config.audio_sensor_dict['cluster_alpha']))
			dpgmm.fit_collapsed_Gibbs(np.array(col_data_df))
			cluster_id = dpgmm.predict(np.array(col_data_df))
		elif data_config.audio_sensor_dict['cluster_method'] == 'gibbs':
			cluster_id = dpgmm_gibbs.DPMM(np.array(col_data_df), alpha=float(data_config.audio_sensor_dict['cluster_alpha']), iter=iter, K=50)
		elif data_config.audio_sensor_dict['cluster_method'] == 'vdpgmm':
			model = VDPGMM(T=20, alpha=float(data_config.audio_sensor_dict['cluster_alpha']), max_iter=300)
			model.fit(np.array(col_data_df))
			cluster_id = model.predict(np.array(col_data_df))
		elif data_config.audio_sensor_dict['cluster_method'] == 'pcrpmm':
			# Model parameters
			alpha = float(data_config.audio_sensor_dict['cluster_alpha'])
			K = 100  # initial number of components
			n_iter = 300
	
			D = np.array(col_data_df).shape[1]
	
			# Generate data
			mu_scale = 1
			covar_scale = 1
	
			# Intialize prior
			m_0 = np.mean(np.array(col_data_df), axis=0)
			k_0 = covar_scale ** 2 / mu_scale ** 2
			v_0 = D + 3
			S_0 = covar_scale ** 2 * v_0 * np.eye(D)

			prior = NIW(m_0, k_0, v_0, S_0)
	
			## Setup PCRPMM
			pcrpmm = PCRPMM(np.array(col_data_df), prior, alpha, save_path=None, assignments="rand", K=K)
	
			## Perform collapsed Gibbs sampling
			pcrpmm.collapsed_gibbs_sampler(n_iter, n_power=1.1, num_saved=1)
			cluster_id = pcrpmm.components.assignments
		else:
			dpgmm = mixture.BayesianGaussianMixture(n_components=20, max_iter=500, covariance_type='full').fit(np.array(col_data_df))
			cluster_id = dpgmm.predict(np.array(col_data_df))

		print(Counter(cluster_id))
		cluster_df.loc[:, process_col + '_cluster'] = cluster_id
	
	return cluster_df


def read_feature_data_df(data_config, participant_id):
	filter_data_path = data_config.audio_sensor_dict['filter_path']
	audio_feature = data_config.audio_sensor_dict['audio_feature']
	pause_threshold = str(data_config.audio_sensor_dict['pause_threshold'])

	# Read only relevant data
	if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
		file_name = 'pause_threshold_' + pause_threshold + '_' + audio_feature
	else:
		file_name = audio_feature
	
	file_list = [file for file in os.listdir(os.path.join(filter_data_path, participant_id)) if data_config.audio_sensor_dict['cluster_data'] in file and file_name in file]
	file_list.sort()
	
	# Read data df
	data_df = pd.DataFrame()
	for file in file_list:
		day_df = pd.read_csv(os.path.join(filter_data_path, participant_id, file), index_col=0)
		data_df = data_df.append(day_df)
	return data_df


def enable_cluster_participant(data_config, participant_id):
	# Read basic
	filter_data_path = data_config.audio_sensor_dict['filter_path']
	audio_feature = data_config.audio_sensor_dict['audio_feature']
	pause_threshold = str(data_config.audio_sensor_dict['pause_threshold'])
	
	# Read other sensor data, the aim is to detect whether people workes during a day
	folder_exist_cond = os.path.exists(os.path.join(filter_data_path, participant_id))
	if folder_exist_cond is False:
		return False
	
	# Read only relevant data
	if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
		file_name = 'pause_threshold_' + pause_threshold + '_' + audio_feature
	else:
		file_name = audio_feature
	file_list = [file for file in os.listdir(os.path.join(filter_data_path, participant_id)) if
				 data_config.audio_sensor_dict['cluster_data'] in file and file_name in file]
	
	enough_data_cond = len(file_list) < 5
	if enough_data_cond:
		return False
	
	return True


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
		if enable_cluster_participant(data_config, participant_id) is False:
			continue

		# Read data
		data_df = read_feature_data_df(data_config, participant_id)
		print('data shape: ', data_df.shape)
		if len(data_df) == 0:
			continue

		# process cluster name
		if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
			cluster_name = 'utterance_cluster'
		elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
			cluster_name = 'minute_cluster'
		elif data_config.audio_sensor_dict['cluster_data'] == 'snippet':
			cluster_name = 'snippet_cluster'
		elif data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
			cluster_name = 'raw_audio_cluster'
		else:
			cluster_name = 'snippet_cluster'

		# Cluster audio
		data_df = (data_df - data_df.mean()) / data_df.std()
		cluster_df = cluster_audio(data_df, data_config, iter=200)
		
		data_cluster_path = data_config.audio_sensor_dict['clustering_path']

		if os.path.exists(os.path.join(data_cluster_path, participant_id)) is False:
			os.mkdir(os.path.join(data_cluster_path, participant_id))
		cluster_df.to_csv(os.path.join(data_cluster_path, participant_id, cluster_name + '_subspace.csv.gz'), compression='gzip')

		print(participant_id + ': success')


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
