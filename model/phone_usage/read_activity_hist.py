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

import scipy.stats
from scipy import stats

from datetime import timedelta
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import grangercausalitytests

import warnings
warnings.filterwarnings('ignore')

# Read basic
agg, sliding = 6, 3

def calculate_change_scores(regular_dist_array, empirical_dist_array):

	# Calculate p value for each distribution
	p_array = np.zeros([1, regular_dist_array.shape[0]])

	# Iterate
	for i in range(empirical_dist_array.shape[1]):
		p_array[0, i] = len(np.where(empirical_dist_array[:, i] >= regular_dist_array[i])[0] * 1.1) / empirical_dist_array.shape[0]

	# Calculate change score
	reject_array, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_array[0], alpha=0.5, method='fdr_bh')
	change_score = np.sum(reject_array)

	return change_score


def read_dist_array(dist_dict):

	if len(dist_dict) == 0:
		return None, None
	shuffle_dict = dist_dict['shuffle']

	empirical_dist_array = np.zeros([len(shuffle_dict.keys()), len(shuffle_dict[0]['dist'])])
	regular_dist_array = dist_dict['data']['dist']

	# Iterate dates range
	for shuffle_idx in list(shuffle_dict.keys()):
		# Data dict
		empirical_dist_array[shuffle_idx, :] = shuffle_dict[shuffle_idx]['dist']

	return regular_dist_array, empirical_dist_array


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
	load_data_path.load_chi_activity_curve_path(chi_data_config, process_data_path, experiment='chi', agg=agg, sliding=sliding)

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		if os.path.exists(os.path.join(chi_data_config.activity_curve_path, participant_id + '.pkl')) is False:
			continue

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		data_dict = np.load(os.path.join(chi_data_config.activity_curve_path, participant_id + '.pkl'), allow_pickle=True)

		# Read data first
		realizd_dist_dict = data_dict['realizd']
		fitbit_dist_dict = data_dict['fitbit']

		# Read empirical distance matrix for realizd
		realizd_change_score_array = np.zeros([1, len(realizd_dist_dict.keys())])

		# Calculate change score for realizd
		for day in list(realizd_dist_dict.keys()):
			regular_dist_array, empirical_dist_array = read_dist_array(realizd_dist_dict[day])
			if regular_dist_array is not None:
				realizd_change_score_array[0, day] = calculate_change_scores(regular_dist_array, empirical_dist_array)

		# Read empirical distance matrix for Fitbit
		fitbit_change_score_array = np.zeros([1, len(fitbit_dist_dict.keys())])

		# Calculate change score for Fitbit
		for day in list(realizd_dist_dict.keys()):
			regular_dist_array, empirical_dist_array = read_dist_array(fitbit_dist_dict[day])
			if regular_dist_array is not None:
				fitbit_change_score_array[0, day] = calculate_change_scores(regular_dist_array, empirical_dist_array)

		final_dict = {}
		final_dict['realizd'] = realizd_change_score_array
		final_dict['fitbit'] = fitbit_change_score_array
		final_dict['pearson_r'] = np.corrcoef(fitbit_change_score_array[0], realizd_change_score_array[0])[0, 1]
		final_dict['spearsman_r'] = scipy.stats.spearmanr(fitbit_change_score_array[0], realizd_change_score_array[0])[0]

		print('pearson_r: %.3f' % (final_dict['pearson_r']))
		print('spearsman_r: %.3f' % (final_dict['spearsman_r']))

		if scipy.stats.spearmanr(fitbit_change_score_array[0], realizd_change_score_array[0])[0] > -1:

			# Fitbit -> realizd
			dist_array = np.zeros([len(fitbit_change_score_array[0]), 2])
			dist_array[:, 0] = fitbit_change_score_array[0]
			dist_array[:, 1] = realizd_change_score_array[0]

			test_stats = grangercausalitytests(dist_array, maxlag=3, verbose=False)
			ssr_chi2test = test_stats[2][0]['ssr_chi2test']

			print('chi test: %.3f' % (ssr_chi2test[1]))
			final_dict['chi fitbit->realizd'] = ssr_chi2test[1]

			# realizd -> Fitbit
			dist_array = np.zeros([len(fitbit_change_score_array[0]), 2])
			dist_array[:, 0] = realizd_change_score_array[0]
			dist_array[:, 1] = fitbit_change_score_array[0]

			test_stats = grangercausalitytests(dist_array, maxlag=3, verbose=False)
			ssr_chi2test = test_stats[2][0]['ssr_chi2test']

			print('chi test: %.3f' % (ssr_chi2test[1]))
			final_dict['chi realizd->fitbit'] = ssr_chi2test[1]

		else:
			final_dict['chi fitbit->realizd'] = np.nan
			final_dict['chi realizd->fitbit'] = np.nan


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)