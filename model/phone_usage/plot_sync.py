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

import seaborn as sns
sns.set(style='whitegrid')

import warnings
warnings.filterwarnings('ignore')

# Read basic
agg, sliding = 7, 2
threshold = 0.2

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


def calculate_change_array(regular_dist_array, empirical_dist_array):

	# Calculate p value for each distribution
	p_array = np.zeros([1, regular_dist_array.shape[0]])

	# Iterate
	for i in range(empirical_dist_array.shape[1]):
		p_array[0, i] = len(np.where(empirical_dist_array[:, i] >= regular_dist_array[i])[0]) / empirical_dist_array.shape[0]

	# Calculate change score
	reject_array, pvals_corrected, alphacSidak, alphacBonf = multipletests(p_array[0], alpha=0.5, method='fdr_bh')

	return p_array[0], np.array(reject_array).astype(int)


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


def readBasicDataDf(igtb_df, participant_id, median_age=None):

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
	supervise_str = 'Manager' if supervise == 1 else 'Non-Manager'

	icu_str = 'Non-ICU'

	for unit in icu_list:
		if unit in primary_unit:
			icu_str = 'ICU'

	if 'ICU' in primary_unit:
		icu_str = 'ICU'

	row_df = pd.DataFrame(index=[participant_id])
	row_df['job'] = job_str
	row_df['icu'] = icu_str
	row_df['shift'] = shift_str
	row_df['supervise'] = supervise_str
	row_df['children'] = 'Have child' if children_data > 0 else 'Don\'t Have child'
	if median_age is None:
		row_df['age'] = age
	else:
		row_df['age'] = '>= Med. Age' if age >= median_age else '< Med. Age'
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
	plt_realizd_df, plt_fitbit_df = pd.DataFrame(), pd.DataFrame()
	fitbit_p_val_df, realizd_p_val_df = pd.DataFrame(), pd.DataFrame()

	plt_corr_df = pd.DataFrame()
	plt_diff_df = pd.DataFrame()

	# Get median age first
	if os.path.exists(os.path.join(chi_data_config.activity_curve_path, 'age.csv.gz')) is False:
		for idx, participant_id in enumerate(top_participant_id_list[:]):
			if os.path.exists(os.path.join(chi_data_config.activity_curve_path, participant_id + '.pkl')) is False:
				continue

			print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

			row_df = readBasicDataDf(igtb_df, participant_id)
			final_df = final_df.append(row_df)
		final_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'age.csv.gz'), compression='gzip')
	else:
		final_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'age.csv.gz'), index_col=0)
	nurse_df = final_df.loc[final_df['job'] == 'nurse']
	median_age = np.nanmedian(nurse_df['age'])

	# plot_realizd
	enable_process = True
	if os.path.exists(os.path.join(chi_data_config.activity_curve_path, 'plot_realizd.csv.gz')) is False or enable_process:
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

			# Calculate change score for realizd
			for day in list(realizd_dist_dict.keys()):
				regular_dist_array, empirical_dist_array = read_dist_array(realizd_dist_dict[day])

				if regular_dist_array is not None:
					# calculate_change_array = calculate_change_scores(regular_dist_array, empirical_dist_array)
					realizd_p_array[day, :], realizd_change_score_array[day, :] = calculate_change_array(regular_dist_array, empirical_dist_array)

			# Read empirical distance matrix for Fitbit
			fitbit_change_score_array = np.zeros([len(realizd_dist_dict.keys()), int(1440 / chi_data_config.offset)])
			fitbit_p_array = np.zeros([len(realizd_dist_dict.keys()), int(1440 / chi_data_config.offset)])

			# Calculate change score for Fitbit
			for day in list(realizd_dist_dict.keys()):
				regular_dist_array, empirical_dist_array = read_dist_array(fitbit_dist_dict[day])
				if regular_dist_array is not None:
					# fitbit_change_score_array[0, day] = calculate_change_array(regular_dist_array, empirical_dist_array)
					fitbit_p_array[day, :], fitbit_change_score_array[day, :] = calculate_change_array(regular_dist_array, empirical_dist_array)

			fitbit_p_val_row_df, realizd_p_val_row_df = readBasicDataDf(igtb_df, participant_id), readBasicDataDf(igtb_df, participant_id)
			corr_p_row_df = readBasicDataDf(igtb_df, participant_id)

			for i, val in enumerate(np.nanmean(fitbit_p_array, axis=0)):
				fitbit_p_val_row_df[i] = val
				row_df = readBasicDataDf(igtb_df, participant_id, median_age=median_age)
				row_df['time'] = i
				row_df['change'] = val
				plt_fitbit_df = plt_fitbit_df.append(row_df)

			for i, val in enumerate(np.nanmean(realizd_p_array, axis=0)):
				realizd_p_val_row_df[i] = val
				row_df = readBasicDataDf(igtb_df, participant_id, median_age=median_age)
				row_df['time'] = i
				row_df['change'] = val
				plt_realizd_df = plt_realizd_df.append(row_df)

			for i in range(len(np.nanmean(realizd_p_array, axis=0))):
				val = np.corrcoef(fitbit_p_array[:, i], realizd_p_array[:, i])[0, 1]
				row_df = readBasicDataDf(igtb_df, participant_id, median_age=median_age)
				row_df['time'] = i
				row_df['change'] = val
				plt_corr_df = plt_corr_df.append(row_df)

			for i, val in enumerate(np.nanmean(fitbit_p_array, axis=0) - np.nanmean(realizd_p_array, axis=0)):
				row_df = readBasicDataDf(igtb_df, participant_id, median_age=median_age)
				row_df['time'] = i
				row_df['change'] = np.abs(val)
				plt_diff_df = plt_diff_df.append(row_df)

			realizd_p_val_df = realizd_p_val_df.append(realizd_p_val_row_df)
			fitbit_p_val_df = fitbit_p_val_df.append(fitbit_p_val_row_df)

		realizd_p_val_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'result_realizd.csv.gz'), compression='gzip')
		fitbit_p_val_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'result1_fitbit.csv.gz'), compression='gzip')
		plt_realizd_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'plot_realizd.csv.gz'), compression='gzip')
		plt_fitbit_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'plot_fitbit.csv.gz'), compression='gzip')
		plt_corr_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'plot_corr.csv.gz'), compression='gzip')
		plt_diff_df.to_csv(os.path.join(chi_data_config.activity_curve_path, 'plot_diff.csv.gz'), compression='gzip')

	else:
		plt_realizd_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'plot_realizd.csv.gz'), index_col=0)
		plt_fitbit_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'plot_fitbit.csv.gz'), index_col=0)
		plt_corr_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'plot_corr.csv.gz'), index_col=0)
		plt_diff_df = pd.read_csv(os.path.join(chi_data_config.activity_curve_path, 'plot_diff.csv.gz'), index_col=0)

	plt_realizd_df = plt_realizd_df.loc[plt_realizd_df['job'] == 'nurse']
	plt_fitbit_df = plt_fitbit_df.loc[plt_fitbit_df['job'] == 'nurse']
	plt_diff_df = plt_diff_df.loc[plt_diff_df['job'] == 'nurse']
	compare_method_list = ['icu', 'supervise', 'children', 'age']

	fig = plt.figure(figsize=(14, 8))
	axes = fig.subplots(nrows=4, ncols=2)

	for i, compare_method in enumerate(compare_method_list):
		if compare_method == 'icu':
			cond_list = ['ICU', 'Non-ICU']
			hue_str = 'husl'
		elif compare_method == 'supervise':
			cond_list = ['Manager', 'Non-Manager']
			hue_str = 'seismic'
		elif compare_method == 'children':
			cond_list = ['With child', 'Without Child']
			hue_str = 'Dark2'
		else:
			cond_list = ['Above Median Age', 'Below Median Age']
			hue_str = 'plasma'

		sns.lineplot(x="time", y='change', dashes=False, marker="o", hue=compare_method, data=plt_diff_df, palette=hue_str, ax=axes[i][0])
		sns.lineplot(x="time", y='change', dashes=False, marker="o", hue=compare_method, data=plt_fitbit_df, palette=hue_str, ax=axes[i][1])

		for j in range(2):
			# axes[i].set_xticklabels(plot_list, fontdict={'fontweight': 'bold', 'fontsize': 20})
			axes[i][j].yaxis.set_tick_params(size=0)
			axes[i][j].grid(linestyle='--')
			axes[i][j].grid(False, axis='y')

			handles, labels = axes[i][j].get_legend_handles_labels()
			axes[i][j].legend(handles=handles[1:], labels=labels[1:], prop={'size': 12, 'weight': 'bold'})

			axes[i][j].set_ylim([-0.25, 0.25])
			axes[i][j].set_xlim([0, 23])

			axes[i][j].set_xlabel('')
			axes[i][j].set_ylabel('')
			axes[i][j].set_xticks(range(0, 23, 4))
			axes[i][j].tick_params(axis='x', labelsize=15)

			for tick in axes[i][j].yaxis.get_major_ticks():
				tick.label1.set_fontsize(14)
				tick.label1.set_fontweight('bold')

			for tick in axes[i][j].xaxis.get_major_ticks():
				tick.label1.set_fontsize(14)
				tick.label1.set_fontweight('bold')

		axes[i][1].set_yticklabels([])
		axes[i][0].tick_params(axis='y', labelsize=15)

		axes[i][0].set_ylabel('Change \n Significance', fontweight='bold', fontsize=14)

	# plt.tight_layout()
	axes[3][0].set_xlabel('Time interval in a day', fontweight='bold', fontsize=14)
	axes[3][1].set_xlabel('Time interval in a day', fontweight='bold', fontsize=14)

	plt.subplots_adjust(top=0.965, bottom=0.07, left=0.06, right=0.985, hspace=0.25, wspace=0.05)
	plt.figtext(0.265, 0.985, 'Smartphone Interaction Routine Change', ha='center', va='center', fontsize=15, fontweight='bold')
	plt.figtext(0.765, 0.985, 'Physio Routine Change', ha='center', va='center', fontsize=15, fontweight='bold')

	plt.show()

	print()


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)