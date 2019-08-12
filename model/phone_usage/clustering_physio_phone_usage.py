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
from dpkmeans import dpmeans
from collections import Counter

from sklearn import mixture
import pickle

import warnings
warnings.filterwarnings('ignore')


def cluster_data(data, data_config, iter=100):
	data_df = data.copy().dropna()
	data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())

	if data_config.cluster_dict['cluster_method'] == 'collapsed_gibbs':
		dpgmm = dpmm.DPMM(n_components=20, alpha=float(data_config.cluster_dict['cluster_alpha']))
		dpgmm.fit_collapsed_Gibbs(np.array(data_df))
		cluster_id = dpgmm.predict(np.array(data_df))
	elif data_config.cluster_dict['cluster_method'] == 'gibbs':
		cluster_id = dpgmm_gibbs.DPMM(np.array(data_df), alpha=float(data_config.cluster_dict['cluster_alpha']), iter=iter, K=50)
	elif data_config.cluster_dict['cluster_method'] == 'vdpgmm':
		vdpgmm = VDPGMM(T=100, alpha=float(data_config.cluster_dict['cluster_alpha']), max_iter=iter)
		vdpgmm.fit(np.array(data_df).astype(float))
		cluster_id = vdpgmm.predict(np.array(data_df))
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
		dpgmm = mixture.BayesianGaussianMixture(n_components=10, max_iter=1000, covariance_type='full').fit(np.array(data_df))
		cluster_id = dpgmm.predict(np.array(data_df))

	print(Counter(cluster_id))
	cluster_df = pd.DataFrame(cluster_id, index=list(data_df.index), columns=['cluster'])

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
	load_data_path.load_chi_clustering_path(chi_data_config, process_data_path, agg='all')

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read data first
		realizd_df = load_sensor_data.read_preprocessed_realizd(data_config.realizd_sensor_dict['preprocess_path'], participant_id)
		fitbit_df = load_sensor_data.read_preprocessed_fitbit(data_config.fitbit_sensor_dict['preprocess_path'], participant_id)

		if realizd_df is None or fitbit_df is None:
			continue

		if np.nansum(np.array(realizd_df['SecondsOnPhone'])) < 14000:
			continue

		realizd_df.loc[realizd_df['SecondsOnPhone'] > 60].loc[:, 'SecondsOnPhone'] = 60

		# Cluster data
		realizd_cluster_df = cluster_data(realizd_df, chi_data_config)
		fitbit_cluster_df = cluster_data(fitbit_df, chi_data_config)

		# Data dict
		data_dict = {}
		data_dict['realizd'] = realizd_cluster_df
		data_dict['fitbit'] = fitbit_cluster_df

		output = open(os.path.join(chi_data_config.clustering_save_path, participant_id + '.pkl'), 'wb')
		pickle.dump(data_dict, output)


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)