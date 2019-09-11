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
agg, sliding = 6, 2
threshold = 0.2

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def calculate_change_array(regular_dist_array, empirical_dist_array):

	# Calculate p value for each distribution
	p_array = np.zeros([1, regular_dist_array.shape[0]])

	# Iterate
	for i in range(empirical_dist_array.shape[1]):
		p_array[0, i] = len(np.where(empirical_dist_array[:, i] >= regular_dist_array[i])[0]) / empirical_dist_array.shape[0]

	# Calculate change score
	# reject_array, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_array[0], alpha=0.5, method='fdr_bh')
	reject_array = p_array[0] <= 0.1

	return p_array[0], np.array(reject_array).astype(int), np.nansum(p_array[0] <= 0.1)


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


def readBasicDataDf(igtb_df, participant_id):

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

	row_df = pd.DataFrame(index=[participant_id])
	row_df['job'] = job_str
	row_df['icu'] = icu_str
	row_df['shift'] = shift_str
	row_df['supervise'] = supervise_str
	row_df['children'] = children_data
	row_df['age'] = age

	return row_df


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
	igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	final_df = pd.DataFrame()
	fitbit_p_val_df, realizd_p_val_df = pd.DataFrame(), pd.DataFrame()

	if os.path.exists(os.path.join(chi_data_config.activity_curve_path, 'result2.csv.gz')) is False:
		for idx, participant_id in enumerate(top_participant_id_list[:]):

			if os.path.exists(os.path.join(chi_data_config.activity_curve_path, participant_id + '.pkl')) is False:
				continue

			print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

			data_dict = np.load(os.path.join(chi_data_config.activity_curve_path, participant_id + '.pkl'), allow_pickle=True)

			# Read data first
			realizd_dist_dict = data_dict['realizd']
			fitbit_dist_dict = data_dict['fitbit']

			# Read empirical distance matrix for realizd
			realizd_change_score_array = np.zeros([len(realizd_dist_dict.keys()), int(1440 / chi_data_config.offset)])
			realizd_p_array = np.zeros([len(realizd_dist_dict.keys()), int(1440 / chi_data_config.offset)])
			realizd_cs_array = np.zeros([1, len(realizd_dist_dict.keys())])

			# Calculate change score for realizd
			for day in list(realizd_dist_dict.keys()):
				regular_dist_array, empirical_dist_array = read_dist_array(realizd_dist_dict[day])

				if regular_dist_array is not None:
					# calculate_change_array = calculate_change_scores(regular_dist_array, empirical_dist_array)
					realizd_p_array[day, :], realizd_change_score_array[day, :], realizd_cs_array[0, day] = calculate_change_array(regular_dist_array, empirical_dist_array)

			# Read empirical distance matrix for Fitbit
			fitbit_change_score_array = np.zeros([len(realizd_dist_dict.keys()), int(1440 / chi_data_config.offset)])
			fitbit_p_array = np.zeros([len(realizd_dist_dict.keys()), int(1440 / chi_data_config.offset)])

			# Calculate change score for Fitbit
			for day in list(realizd_dist_dict.keys()):
				regular_dist_array, empirical_dist_array = read_dist_array(fitbit_dist_dict[day])
				if regular_dist_array is not None:
					# fitbit_change_score_array[0, day] = calculate_change_array(regular_dist_array, empirical_dist_array)
					fitbit_p_array[day, :], fitbit_change_score_array[day, :], fitbit_cs = calculate_change_array(regular_dist_array, empirical_dist_array)

			sync = np.array(fitbit_change_score_array == realizd_change_score_array).astype(int)
			sync = np.nansum(sync, axis=1)
			final_dict = {}
			final_dict['realizd'] = realizd_change_score_array
			final_dict['fitbit'] = fitbit_change_score_array
			final_dict['mean'] = np.nanmedian(sync)
			final_dict['std'] = np.nanstd(sync)
			final_dict['p'] = np.corrcoef(fitbit_p_array.flatten(), realizd_p_array.flatten())

			row_df = readBasicDataDf(igtb_df, participant_id)

			row_df['mean'] = np.nanmean(sync)
			row_df['rea_mean'] = np.nanmean(realizd_cs_array)
			row_df['std'] = np.nanstd(sync)
			row_df['p'] = np.corrcoef(fitbit_p_array.flatten(), realizd_p_array.flatten())[0, 1]

			tmp = np.abs(realizd_p_array - fitbit_p_array)[:-2, :]
			tmp = np.nanmean(tmp, axis=1)
			# tmp = tmp[tmp ] = np.nan
			row_df['diff'] = np.nanmean(tmp)

			for col in igtb_cols:
				row_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]
			final_df = final_df.append(row_df)

			fitbit_p_val_row_df, realizd_p_val_row_df = readBasicDataDf(igtb_df, participant_id), readBasicDataDf(igtb_df, participant_id)

			for i, val in enumerate(np.nanmean(fitbit_p_array, axis=0)):
				fitbit_p_val_row_df[i] = val
			for i, val in enumerate(np.nanmean(realizd_p_array, axis=0)):
				realizd_p_val_row_df[i] = val

			realizd_p_val_df = realizd_p_val_df.append(realizd_p_val_row_df)
			fitbit_p_val_df = fitbit_p_val_df.append(fitbit_p_val_row_df)

		final_df = final_df.dropna()
		final_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'result1.csv.gz'), compression='gzip')
		realizd_p_val_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'result_realizd.csv.gz'), compression='gzip')
		fitbit_p_val_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'result1_fitbit.csv.gz'), compression='gzip')

	else:
		final_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'result1.csv.gz'), index_col=0)
		realizd_p_val_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'result_realizd.csv.gz'), index_col=0)
		fitbit_p_val_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'result1_fitbit.csv.gz'), index_col=0)

	add_cols = ['Pain', 'LifeSatisfaction', 'General_Health',
	            'Flexbility', 'Inflexbility', 'Perceivedstress',
	            'energy_fatigue', 'energy', 'fatigue', 'Engage']

	for participant_id in list(final_df.index):
		for col in add_cols:
			data_str = str(igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0])
			if len(data_str) == 0:
				final_df.loc[participant_id, col] = np.nan
				continue

			if 'a' in data_str or ' ' in data_str:
				final_df.loc[participant_id, col] = np.nan
			else:
				final_df.loc[participant_id, col] = float(data_str)

	nurse_df = final_df.loc[final_df['job'] == 'nurse']

	print()
	compare_method_list = ['icu', 'supervise', 'children', 'age']
	for compare_method in compare_method_list:
		if compare_method == 'icu':
			first_df = nurse_df.loc[nurse_df[compare_method] == 'icu']
			second_df = nurse_df.loc[nurse_df[compare_method] == 'non_icu']
			cond_list = ['icu', 'non-icu']
		elif compare_method == 'supervise':
			first_df = nurse_df.loc[nurse_df[compare_method] == 'Supervise']
			second_df = nurse_df.loc[nurse_df[compare_method] == 'Non-Supervise']
			cond_list = ['Supervise', 'Non-Supervise']
		elif compare_method == 'children':
			first_df = nurse_df.loc[nurse_df[compare_method] > 0]
			second_df = nurse_df.loc[nurse_df[compare_method] == 0]
			cond_list = ['With child', 'Without Child']
		else:
			first_df = nurse_df.loc[nurse_df[compare_method] >= np.nanmedian(nurse_df[compare_method])]
			second_df = nurse_df.loc[nurse_df[compare_method] < np.nanmedian(nurse_df[compare_method])]
			cond_list = ['Above Median Age', 'Below Median Age']

		stat, p = stats.ttest_ind(first_df['mean'].dropna(), second_df['mean'].dropna(), equal_var=False)
		print('%s: mean = %.2f, std = %.2f' % (cond_list[0], np.mean(first_df['mean']), np.std(first_df['mean'])))
		print('%s: mean = %.2f, std = %.2f' % (cond_list[1], np.mean(second_df['mean']), np.std(second_df['mean'])))
		print('K-S test for %s' % 'mean')
		print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))

		'''
		stat, p = stats.ttest_ind(first_df['p'].dropna(), second_df['p'].dropna(), equal_var=False)
		print('%s: p = %.2f, std = %.2f' % (cond_list[0], np.mean(first_df['p']), np.std(first_df['p'])))
		print('%s: p = %.2f, std = %.2f' % (cond_list[1], np.mean(second_df['p']), np.std(second_df['p'])))
		print('K-S test for %s' % 'mean')
		print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))
		'''
	nurse_realizd_p_df = realizd_p_val_df.loc[realizd_p_val_df['job'] == 'nurse']
	compare_method_list = ['icu', 'supervise', 'children', 'age']
	for compare_method in compare_method_list:
		if compare_method == 'icu':
			first_df = nurse_realizd_p_df.loc[nurse_realizd_p_df[compare_method] == 'icu']
			second_df = nurse_realizd_p_df.loc[nurse_realizd_p_df[compare_method] == 'non_icu']
			cond_list = ['icu', 'non-icu']
		elif compare_method == 'supervise':
			first_df = nurse_realizd_p_df.loc[nurse_realizd_p_df[compare_method] == 'Supervise']
			second_df = nurse_realizd_p_df.loc[nurse_realizd_p_df[compare_method] == 'Non-Supervise']
			cond_list = ['Supervise', 'Non-Supervise']
		elif compare_method == 'children':
			first_df = nurse_realizd_p_df.loc[nurse_realizd_p_df[compare_method] > 0]
			second_df = nurse_realizd_p_df.loc[nurse_realizd_p_df[compare_method] == 0]
			cond_list = ['With child', 'Without Child']
		else:
			first_df = nurse_realizd_p_df.loc[nurse_realizd_p_df[compare_method] >= np.nanmedian(realizd_p_val_df[compare_method])]
			second_df = nurse_realizd_p_df.loc[nurse_realizd_p_df[compare_method] < np.nanmedian(realizd_p_val_df[compare_method])]
			cond_list = ['Above Median Age', 'Below Median Age']

		print()
		# tmp1 =

	ana_cols = ['itp_igtb', 'irb_igtb', 'ocb_igtb', 'stai_igtb', 'pos_af_igtb', 'neg_af_igtb', 'energy_fatigue', 'Pain']
	from scipy.stats.stats import pearsonr

	r_row_df = pd.DataFrame(index=[str(agg) + '_r'])
	p_row_df = pd.DataFrame(index=[str(agg) + '_p'])

	corr_df = pd.DataFrame()
	for col in ana_cols:
		data_df = nurse_df[[col, 'mean']].dropna()

		r, p = stats.pearsonr(np.array(data_df[col]), np.array(data_df['mean']))

		r_row_df[col] = r
		p_row_df[col] = p

	corr_df = corr_df.append(r_row_df)
	corr_df = corr_df.append(p_row_df)

	print()


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)