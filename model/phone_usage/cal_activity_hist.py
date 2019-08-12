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
threshold = 0.3

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


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


def readBasicDataDf(igtb_df, participant_id):

	nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
	primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
	shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
	job_str = 'nurse' if nurse == 1 else 'non_nurse'
	shift_str = 'day' if shift == 'Day shift' else 'night'

	icu_str = 'non_icu'

	for unit in icu_list:
		if unit in primary_unit:
			icu_str = 'icu'

	if 'ICU' in primary_unit:
		icu_str = 'icu'

	if job_str == 'non_nurse':
		if 'lab' in primary_unit or 'Lab' in primary_unit:
			icu_str = 'lab'

	row_df = pd.DataFrame(index=[participant_id])
	row_df['job'] = job_str
	row_df['icu'] = icu_str
	row_df['shift'] = shift_str

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

	if os.path.exists(os.path.join(chi_data_config.activity_curve_path, 'result.csv.gz')) is False:
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

			row_df = readBasicDataDf(igtb_df, participant_id)

			row_df['p'] = final_dict['pearson_r']

			for col in igtb_cols:
				row_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]

			final_df = final_df.append(row_df)
			final_df = final_df.dropna()
			final_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'result.csv.gz'), compression='gzip')
	else:
		final_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'result.csv.gz'), index_col=0)

	add_cols = ['Emotional_Wellbeing', 'Pain', 'LifeSatisfaction', 'General_Health',
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

	non_nurse_df = final_df.loc[final_df['job'] == 'non_nurse']
	lab_df = non_nurse_df.loc[non_nurse_df['icu'] == 'lab']

	nurse_df = final_df.loc[final_df['job'] == 'nurse']
	day_nurse_df = nurse_df.loc[nurse_df['shift'] == 'day']
	night_nurse_df = nurse_df.loc[nurse_df['shift'] == 'night']
	icu_nurse_df = nurse_df.loc[nurse_df['icu'] == 'icu']
	non_icu_nurse_df = nurse_df.loc[nurse_df['icu'] == 'non_icu']

	len0 = len(non_nurse_df.loc[(non_nurse_df['p'] > threshold)])
	len1 = len(lab_df.loc[(lab_df['p'] > threshold)])
	len2 = len(nurse_df.loc[(nurse_df['p'] > threshold)])

	print('\nOverall: %d\n' % (len0 + len2))
	len3 = len(day_nurse_df.loc[(day_nurse_df['p'] > threshold)])
	len4 = len(night_nurse_df.loc[(night_nurse_df['p'] > threshold)])
	len5 = len(icu_nurse_df.loc[(icu_nurse_df['p'] > threshold)])
	len6 = len(non_icu_nurse_df.loc[(non_icu_nurse_df['p'] > threshold)])

	print('\nnon_nurse (%d): %.2f' % (len(non_nurse_df), len0 / len(non_nurse_df) * 100))
	print('lab (%d): %.2f\n' % (len(lab_df), len1 / len(lab_df) * 100))

	print('nurse (%d): %.2f\n' % (len(nurse_df), len2 / len(nurse_df) * 100))
	print('day nurse (%d): %.2f' % (len(day_nurse_df), len3 / len(day_nurse_df) * 100))
	print('night nurse (%d): %.2f' % (len(night_nurse_df), len4 / len(night_nurse_df) * 100))
	print('icu nurse (%d): %.2f' % (len(icu_nurse_df), len5 / len(icu_nurse_df) * 100))
	print('non_icu nurse (%d): %.2f\n' % (len(non_icu_nurse_df), len6 / len(non_icu_nurse_df) * 100))

	data1_df = final_df.loc[(final_df['p'] > threshold)]
	data2_df = final_df.loc[(final_df['p'] <= threshold)]

	'''
	data1_df = final_df.loc[(final_df['chi_p'] > 0.05)]
	data2_df = final_df.loc[(final_df['chi_p'] <= 0.05)]
	'''
	for col in igtb_cols + add_cols:
		print(col)
		# stat, p = stats.ks_2samp(data1_df[col].dropna(), data2_df[col].dropna())
		stat, p = stats.ttest_ind(data1_df[col].dropna(), data2_df[col].dropna(), equal_var=False)
		print('High sync: mean = %.2f, std = %.2f' % (np.mean(data1_df[col]), np.std(data1_df[col])))
		print('Low sync: mean = %.2f, std = %.2f' % (np.mean(data2_df[col]), np.std(data2_df[col])))
		print('K-S test for %s' % col)
		print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)