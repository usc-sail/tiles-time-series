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
from theano import tensor as tt
import pymc3 as pm
import random
from collections import Counter
import dpgmm_gibbs
from vdpgmm import VDPGMM
from sklearn import preprocessing
from sklearn import datasets

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser

SEED = 5132290 # from random.org

np.random.seed(SEED)


def cluster_audio(data_df, data_config, participant_id, iter=100, cluster_name='utterance_cluster'):
	data_cluster_path = data_config.audio_sensor_dict['clustering_path']

	if data_config.audio_sensor_dict['cluster_method'] == 'collapsed_gibbs':
		dpgmm = dpmm.DPMM(n_components=30, alpha=float(data_config.audio_sensor_dict['cluster_alpha']))
		dpgmm.fit_collapsed_Gibbs(np.array(data_df))
		cluster_id = dpgmm.predict(np.array(data_df))
	elif data_config.audio_sensor_dict['cluster_method'] == 'gibbs':
		cluster_id = dpgmm_gibbs.DPMM(np.array(data_df), alpha=float(data_config.audio_sensor_dict['cluster_alpha']), iter=iter, K=50)
	elif data_config.audio_sensor_dict['cluster_method'] == 'vdpgmm':
		model = VDPGMM(T=50, alpha=float(data_config.audio_sensor_dict['cluster_alpha']), max_iter=1000)
		model.fit(np.array(data_df))
		cluster_id = model.predict(np.array(data_df))
	else:
		dpgmm = mixture.BayesianGaussianMixture(n_components=50, covariance_type='full').fit(np.array(data_df))
		cluster_id = dpgmm.predict(np.array(data_df))

	data_df.loc[:, 'cluster'] = cluster_id
	if os.path.exists(os.path.join(data_cluster_path, participant_id)) is False:
		os.mkdir(os.path.join(data_cluster_path, participant_id))
	data_df.to_csv(os.path.join(data_cluster_path, participant_id, cluster_name + '.csv.gz'), compression='gzip')


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

	for idx, participant_id in enumerate(top_participant_id_list):

		print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read id
		uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]

		# Read other sensor data, the aim is to detect whether people workes during a day
		if len(os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id))) < 5:
			continue
		file_list = [file for file in os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) if 'utterance' not in file]

		raw_audio_df, utterance_df = pd.DataFrame(), pd.DataFrame()

		for file in file_list:
			tmp_raw_audio_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file), index_col=0)
			tmp_raw_audio_df = tmp_raw_audio_df.drop(columns=['F0_sma'])

			# if cluster raw_audio
			if data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
				raw_audio_df = raw_audio_df.append(tmp_raw_audio_df)
			# if cluster utterance
			elif data_config.audio_sensor_dict['cluster_data'] == 'utterance':
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
					tmp_utterance_df['duration'] = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds()
					for col in list(tmp_utterance_raw_df.columns):
						tmp_utterance_df[col + '_mean'] = np.mean(np.array(tmp_utterance_raw_df[col]))
						tmp_utterance_df[col + '_std'] = np.std(np.array(tmp_utterance_raw_df[col]))

					day_utterance_df = day_utterance_df.append(tmp_utterance_df)

				day_utterance_df.to_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'utterance_' + file), compression='gzip')
				utterance_df = utterance_df.append(day_utterance_df)

		# process audio feature for cluster
		if data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
			raw_audio_df_norm = (raw_audio_df - raw_audio_df.mean()) / raw_audio_df.std()
			cluster_audio(raw_audio_df_norm, data_config, participant_id, iter=100, cluster_name='raw_audio_cluster')
		# if cluster utterance
		elif data_config.audio_sensor_dict['cluster_data'] == 'utterance':
			utterance_norm_df = utterance_df.drop(columns=['start', 'end', 'duration'])
			utterance_norm_df = (utterance_norm_df - utterance_norm_df.mean()) / utterance_norm_df.std()
			cluster_audio(utterance_norm_df, data_config, participant_id, iter=100, cluster_name='utterance_cluster')


if __name__ == '__main__':
	
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'ticc' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)