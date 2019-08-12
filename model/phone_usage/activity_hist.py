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

from collections import Counter
import pickle
from scipy.stats import entropy

from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')

# Read basic
agg, sliding = 8, 3


def read_data(chi_data_config, data_df, start_str, end_str, agg, sliding):
	# Unique list
	unique_list = list(set(list(np.array(data_df['cluster']))))

	# Get basic data
	num_of_point = int(1440 / chi_data_config.window)
	window = chi_data_config.window
	offset = chi_data_config.offset

	sec_data_df = data_df[start_str:end_str]
	data_array = np.zeros([agg+sliding, num_of_point, len(unique_list)])
	data_array[:, :] = np.nan

	if len(sec_data_df) < 720 * 3:
		return None

	# Get compared data
	for i in range(agg+sliding):
		day_start_str = (pd.to_datetime(start_str) + timedelta(days=i)).strftime(load_data_basic.date_time_format)[:-3]

		for j in range(num_of_point):
			# Get start and end time
			tmp_start_str = (pd.to_datetime(day_start_str) + timedelta(minutes=j*offset)).strftime(load_data_basic.date_time_format)[:-3]
			tmp_end_str = (pd.to_datetime(day_start_str) + timedelta(minutes=j*offset+window)).strftime(load_data_basic.date_time_format)[:-3]

			tmp_df = sec_data_df[tmp_start_str:tmp_end_str]

			if len(tmp_df) == 0:
				continue

			# Set data
			counter_dict = Counter(np.array(tmp_df['cluster']))
			for cluster_id in list(counter_dict.keys()):
				data_array[i, j, unique_list.index(cluster_id)] = counter_dict[cluster_id]

	return data_array


def generate_shuffle_pdf(data_array, agg, shuffle=False):
	# Shuffle
	idx_array = np.arange(0, data_array.shape[0])

	if shuffle is True:
		np.random.shuffle(idx_array)

	# Shuffled array
	first_array = data_array[idx_array[:agg], :, :].copy()
	second_array = data_array[idx_array[agg:], :, :].copy()

	# Aggregation array
	first_sum_array = np.nansum(first_array, axis=0)
	second_sum_array = np.nansum(second_array, axis=0)

	# Imputation
	inds = np.where(np.isnan(first_sum_array))
	first_sum_array[inds] = np.take(np.nanmean(first_sum_array, axis=0), inds[1])
	inds = np.where(np.isnan(second_sum_array))
	second_sum_array[inds] = np.take(np.nanmean(second_sum_array, axis=0), inds[1])

	# Plus one filter
	first_sum_array[:, :] = first_sum_array[:, :] + 1
	second_sum_array[:, :] = second_sum_array[:, :] + 1

	# pdf array
	first_pdf = np.divide(first_sum_array, np.sum(first_sum_array, axis=1).reshape([data_array.shape[1], 1]))
	second_pdf = np.divide(second_sum_array, np.sum(second_sum_array, axis=1).reshape([data_array.shape[1], 1]))

	# Filter array
	first_filter_pdf, second_filter_pdf = np.zeros(first_pdf.shape), np.zeros(first_pdf.shape)

	for i in range(data_array.shape[1]):
		for j in range(data_array.shape[2]):
			if i == 0:
				first_filter_pdf[i, j] = (first_pdf[i, j] + first_pdf[i + 1, j] + first_pdf[-1, j]) / 3
				second_filter_pdf[i, j] = (second_pdf[i, j] + second_pdf[i + 1, j] + second_pdf[-1, j]) / 3
			elif i == data_array.shape[1] - 1:
				first_filter_pdf[i, j] = (first_pdf[i, j] + first_pdf[i - 1, j] + first_pdf[0, j]) / 3
				second_filter_pdf[i, j] = (second_pdf[i, j] + second_pdf[i - 1, j] + second_pdf[0, j]) / 3
			else:
				first_filter_pdf[i, j] = np.nanmean(first_pdf[i - 1:i + 2, j])
				second_filter_pdf[i, j] = np.nanmean(second_pdf[i - 1:i + 2, j])

	first_filter_pdf = np.divide(first_filter_pdf, np.sum(first_filter_pdf, axis=1).reshape([data_array.shape[1], 1]))
	second_filter_pdf = np.divide(second_filter_pdf, np.sum(second_filter_pdf, axis=1).reshape([data_array.shape[1], 1]))

	return first_filter_pdf, second_filter_pdf


def calculateDist(first_array, second_array):
	# Dist array
	kl_dist_array = np.zeros([1, first_array.shape[0]])
	for j in range(first_array.shape[0]):
		first_pdf, second_pdf = first_array[j, :], second_array[j, :]
		kl_dist = entropy(first_pdf, second_pdf) + entropy(second_pdf, first_pdf)
		kl_dist = kl_dist / 2
		kl_dist_array[0, j] = kl_dist

	return kl_dist_array[0]


def cal_activity_distribution(chi_data_config, data_df, start_str, end_str):

	dates_range = int(((pd.to_datetime(end_str) - pd.to_datetime(start_str)).days - agg) / sliding) - 1
	start_time = pd.to_datetime(start_str).replace(hour=0, minute=0)

	if dates_range < 10:
		return

	# Final data
	data_dict = {}

	# Iterate dates range
	for i in range(dates_range):

		# Data dict
		data_dict[i] = {}

		# Read start and end str
		dates_start_str = (start_time + timedelta(days=sliding*i)).strftime(load_data_basic.date_time_format)[:-3]
		dates_end_str = (start_time + timedelta(days=sliding*(i+1)+agg)).strftime(load_data_basic.date_time_format)[:-3]

		# Read array
		data_array = read_data(chi_data_config, data_df, dates_start_str, dates_end_str, agg, sliding)
		if data_array is None:
			continue

		# Read basic
		first_pdf_array, second_pdf_array = generate_shuffle_pdf(data_array, agg, shuffle=False)
		dist_array = calculateDist(first_pdf_array, second_pdf_array)

		data_dict[i]['data'] = {}
		data_dict[i]['data']['dist'] = dist_array

		# data_dict[i]['data']['first'] = first_pdf_array
		# data_dict[i]['data']['second'] = second_pdf_array

		data_dict[i]['shuffle'] = {}
		# Get shuffle array
		for j in range(500):
			if j % 20 == 0:
				print('Process week num: %d week; Shuffle time: %d' % (i + 1, j))

			first_pdf_array, second_pdf_array = generate_shuffle_pdf(data_array, agg, shuffle=True)
			dist_array = calculateDist(first_pdf_array, second_pdf_array)

			data_dict[i]['shuffle'][j] = {}
			data_dict[i]['shuffle'][j]['dist'] = dist_array
			# data_dict[i]['shuffle'][j]['first'] = first_pdf_array
			# data_dict[i]['shuffle'][j]['second'] = second_pdf_array

	return data_dict


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

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		if os.path.exists(os.path.join(chi_data_config.clustering_save_path, participant_id + '.pkl')) is False:
			continue

		data_dict = np.load(os.path.join(chi_data_config.clustering_save_path, participant_id + '.pkl'), allow_pickle=True)

		# Read data first
		realizd_cluster_df = data_dict['realizd']
		fitbit_cluster_df = data_dict['fitbit']

		start_str = pd.to_datetime(realizd_cluster_df.index[0]).strftime(load_data_basic.date_time_format)[:-3]
		end_str = (pd.to_datetime(realizd_cluster_df.index[-1]) + timedelta(days=1)).strftime(load_data_basic.date_time_format)[:-3]

		# Cal dist
		realizd_data_dict = cal_activity_distribution(chi_data_config, realizd_cluster_df, start_str, end_str)
		fitbit_data_dict = cal_activity_distribution(chi_data_config, fitbit_cluster_df, start_str, end_str)

		if realizd_data_dict is None or fitbit_data_dict is None:
			continue

		# Final data
		final_dict = {}
		final_dict['realizd'] = realizd_data_dict
		final_dict['fitbit'] = fitbit_data_dict

		output = open(os.path.join(chi_data_config.activity_curve_path, participant_id + '.pkl'), 'wb')
		pickle.dump(final_dict, output)


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)