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

from sklearn.feature_selection import mutual_info_classif

import dpmm
import dpgmm_gibbs
from vdpgmm import VDPGMM

from pybgmm.prior import NIW
from pybgmm.igmm import PCRPMM
from datetime import timedelta
from dpkmeans import dpmeans
from collections import Counter
from sklearn import mixture

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


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

	igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]

	###########################################################
	# 2. Get participant id list
	###########################################################
	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	final_mi_df = pd.DataFrame()

	for idx, participant_id in enumerate(top_participant_id_list):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# read shift type
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]

		if nurse != 1:
			continue

		save_participant_folder = os.path.join(data_config.fitbit_sensor_dict['clustering_path'], participant_id)
		if not os.path.exists(save_participant_folder):
			continue

		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
		primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
		supervise = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].supervise[0]
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		job_str = 'nurse' if nurse == 1 else 'non_nurse'
		shift_str = 'day' if shift == 'Day shift' else 'night'
		children = str(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].children[0])
		age = str(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].age[0])

		children_data = int(children) if len(children) == 1 and children != ' ' else np.nan
		age = float(age) if age != 'nan' else np.nan
		supervise_str = 'Supervise' if supervise == 1 else 'Non-Supervise'

		icu_str = 'non_icu'

		for unit in icu_list:
			if unit in primary_unit:
				icu_str = 'icu'

		if 'ICU' in primary_unit:
			icu_str = 'icu'

		###########################################################
		# Read Fitbit data that is associated with these days
		###########################################################
		file_list = os.listdir(save_participant_folder)

		# Get all unique cluster
		all_df = pd.DataFrame()
		for file_name in file_list:
			if 'shift' not in file_name:
				continue
			cluster_data_df = pd.read_csv(os.path.join(save_participant_folder, file_name), index_col=0)
			all_df = all_df.append(cluster_data_df)

		owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
		if len(all_df) < 200:
			continue

		if owl_in_one_df is None:
			continue

		unique_loc_list = list(owl_in_one_df.columns)

		mutual_list = ['cluster'] + unique_loc_list
		mutual_df = pd.DataFrame()

		if 'lounge' not in unique_loc_list:
			continue

		for file_name in file_list:

			if '_ratio' not in file_name:
				continue

			ratio_data_df = pd.read_csv(os.path.join(save_participant_folder, file_name), index_col=0)
			ratio_data_df = ratio_data_df[mutual_list]
			mutual_df = mutual_df.append(ratio_data_df)

		if len(mutual_df) < 200:
			continue

		unique_loc_list = ['lounge', 'med', 'ns', 'pat']

		X = np.array(np.array(mutual_df[unique_loc_list]))
		y = np.array(np.array(mutual_df['cluster']))

		mi_array = mutual_info_classif(X, y, discrete_features=True)
		row_df = pd.DataFrame(index=[participant_id])
		for i, room in enumerate(unique_loc_list):
			row_df[room] = mi_array[i]

		row_df['job'] = job_str
		row_df['icu'] = icu_str
		row_df['shift'] = shift_str
		row_df['supervise'] = supervise_str
		row_df['children'] = children_data
		row_df['age'] = age

		for col in igtb_cols:
			row_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]

		final_mi_df = final_mi_df.append(row_df)

	print()

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