"""
Top level classes for the preprocess model.
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own pyspark path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'preprocess')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import segmentation
import load_sensor_data, load_data_path, load_data_basic
import numpy as np
import pandas as pd
import argparse

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

import warnings
warnings.filterwarnings("ignore")


import dpmm

import dpgmm_gibbs
from vdpgmm import VDPGMM

from pybgmm.prior import NIW
from pybgmm.igmm import PCRPMM
from datetime import timedelta
from dpkmeans import dpmeans
from collections import Counter
from sklearn import mixture


def cluster_data(data, data_config, iter=200):
	cluster_df = data.copy()
	data_df = data.copy()
	# data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())

	if data_config.fitbit_sensor_dict['cluster_method'] == 'collapsed_gibbs':
		dpgmm = dpmm.DPMM(n_components=10, alpha=float(data_config.fitbit_sensor_dict['cluster_alpha']))
		dpgmm.fit_collapsed_Gibbs(np.array(data_df))
		cluster_id = dpgmm.predict(np.array(data_df))
	elif data_config.fitbit_sensor_dict['cluster_method'] == 'gibbs':
		cluster_id = dpgmm_gibbs.DPMM(np.array(data_df), alpha=float(data_config.fitbit_sensor_dict['cluster_alpha']), iter=iter, K=50)
	elif data_config.fitbit_sensor_dict['cluster_method'] == 'vdpgmm':
		model = VDPGMM(T=20, alpha=float(data_config.fitbit_sensor_dict['cluster_alpha']), max_iter=iter)
		model.fit(np.array(data_df))
		cluster_id = model.predict(np.array(data_df))
	elif data_config.fitbit_sensor_dict['cluster_method'] == 'dpkmeans':
		dp = dpmeans(np.array(data_df))
		cluster_id, obj, em_time = dp.fit(np.array(data_df))
	elif data_config.fitbit_sensor_dict['cluster_method'] == 'pcrpmm':
		# Model parameters
		alpha = float(data_config.fitbit_sensor_dict['cluster_alpha'])
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
		pcrpmm.collapsed_gibbs_sampler(n_iter, n_power=float(data_config.fitbit_sensor_dict['power']), num_saved=1)
		cluster_id = pcrpmm.components.assignments
	else:
		dpgmm = mixture.BayesianGaussianMixture(n_components=10, max_iter=500, covariance_type='full').fit(np.array(data_df))
		cluster_id = dpgmm.predict(np.array(data_df))

	print(Counter(cluster_id))
	cluster_df.loc[:, 'cluster'] = cluster_id

	return cluster_df


def main(tiles_data_path, config_path, experiement):
	###########################################################
	# 1. Create Config
	###########################################################
	process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))

	data_config = config.Config()
	data_config.readConfigFile(config_path, experiement)

	# Load preprocess folder
	# Load all data path according to config file
	load_data_path.load_all_available_path(data_config, process_data_path,
										   preprocess_data_identifier='preprocess',
										   segmentation_data_identifier='segmentation',
										   filter_data_identifier='filter_data',
										   clustering_data_identifier='clustering')

	# Load Fitbit summary folder
	fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')

	###########################################################
	# 2. Get participant id list
	###########################################################
	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	# data_cols = ['duration', 'HeartRatePPG_mean', 'StepCount_mean', 'HeartRatePPG_cov', 'StepCount_cov', 'cov']
	data_cols = ['HeartRatePPG_mean', 'StepCount_mean', 'HeartRatePPG_cov', 'StepCount_cov', 'cov']

	for idx, participant_id in enumerate(top_participant_id_list):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# read shift type
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]

		if nurse != 1:
			continue

		save_participant_folder = os.path.join(data_config.fitbit_sensor_dict['segmentation_path'], participant_id)
		if not os.path.exists(save_participant_folder):
			continue

		###########################################################
		# 4.1 Read days at work data
		###########################################################
		days_at_work_df = load_sensor_data.read_preprocessed_days_at_work(data_config.days_at_work_path, participant_id)

		###########################################################
		# Read Fitbit data that is associated with these days
		###########################################################
		fitbit_workday_dict = load_sensor_data.read_preprocessed_fitbit_during_work(data_config, participant_id, days_at_work_df, shift)
		file_list = os.listdir(save_participant_folder)
		final_df = pd.DataFrame()

		for file_name in file_list:
			if 'shift' not in file_name:
				continue

			seg_data_df = pd.read_csv(os.path.join(save_participant_folder, file_name), index_col=0)

			start_str, end_str = list(seg_data_df['start'])[0], list(seg_data_df['end'])[-1]
			fitbit_work_data_df = fitbit_workday_dict[start_str]
			fitbit_work_data_df = fitbit_work_data_df.fillna(fitbit_work_data_df.mean())

			# iterate over the segments, compute mean, std, duration as feature
			start_list, end_list = list(seg_data_df['start']), list(seg_data_df['end'])

			for i, start_str in enumerate(start_list):
				end_str = end_list[i]

				fitbit_seg_data_df = fitbit_work_data_df[start_str:end_str]
				hr_array, step_array = np.array(fitbit_seg_data_df['HeartRatePPG']), np.array(fitbit_seg_data_df['StepCount'])

				row_df = pd.DataFrame(index=[start_str])
				row_df['start'] = start_str
				row_df['end'] = end_str

				row_df['duration'] = len(fitbit_seg_data_df)
				row_df['HeartRatePPG_mean'] = np.nanmean(hr_array)
				row_df['StepCount_mean'] = np.nanmean(step_array)

				cov = np.cov(np.array(fitbit_seg_data_df).T)
				row_df['HeartRatePPG_cov'] = cov[0, 0]
				row_df['StepCount_cov'] = cov[1, 1]
				row_df['cov'] = cov[0, 1]

				final_df = final_df.append(row_df)

		if len(final_df) < 100:
			continue

		norm_df = final_df.loc[:, data_cols]
		norm_df = (norm_df - norm_df.mean()) / norm_df.std()
		# final_df.loc[:, data_cols] = norm_df.loc[:, data_cols]

		if len(norm_df) > 100:
			cluster_df = cluster_data(norm_df, data_config, iter=200)
			final_df.loc[list(cluster_df.index), 'cluster'] = cluster_df['cluster']

			for file_name in file_list:
				if 'shift' not in file_name:
					continue

				if 'DS' in file_name:
					continue

				seg_data_df = pd.read_csv(os.path.join(save_participant_folder, file_name), index_col=0)
				save_seg_df = final_df.loc[list(seg_data_df.index), :]

				save_clustering_participant_folder = os.path.join(data_config.fitbit_sensor_dict['clustering_path'], participant_id)
				if not os.path.exists(save_clustering_participant_folder):
					os.mkdir(save_clustering_participant_folder)

				save_seg_df.to_csv(os.path.join(save_clustering_participant_folder, file_name), compression='gzip')


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
	parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
	parser.add_argument("--experiement", required=False, help="Experiement name")

	args = parser.parse_args()

	tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
	experiement = 'dpmm' if args.config is None else args.config

	main(tiles_data_path, config_path, experiement)