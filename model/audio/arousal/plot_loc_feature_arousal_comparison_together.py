"""
Filter the data
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import pandas as pd
import numpy as np
from datetime import timedelta
import pickle
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']

def main(tiles_data_path, config_path, experiment):

	# Create Config
	process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, 'data'))

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

	# feat_list = ['F0_sma', 'fft', 'pcm_intensity_sma'] # pcm_loudness_sma, pcm_intensity_sma, pcm_RMSenergy_sma, pcm_zcr_sma
	# 'shift', 'supervise', 'language', 'gender', 'icu'
	num_of_sec = 4
	compare_method = 'shift'

	# final_df = pd.DataFrame()
	data_df = pd.DataFrame()

	loc_list = ['pat', 'ns', 'lounge', 'unknown']

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read id
		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
		supervise = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].supervise[0]
		language = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].language[0]
		primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		gender_str = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Sex[0]

		job_str = 'nurse' if nurse == 1 else 'non_nurse'
		shift_str = 'day' if shift == 'Day shift' else 'night'
		supervise_str = 'Supervise' if supervise == 1 else 'Non-Supervise'
		language_str = 'English' if language == 1 else 'Non-English'
		if job_str == 'non_nurse':
			continue
		icu_str = 'non_icu'
		for unit in icu_list:
			if unit in primary_unit:
				icu_str = 'icu'
		if 'ICU' in primary_unit:
			icu_str = 'icu'

		if os.path.exists(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec), participant_id + '.pkl')) is False:
			continue

		arousal_loc_dict = np.load(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec), participant_id + '.pkl'), allow_pickle=True)
		daily_dict, loc_dict, global_dict = arousal_loc_dict['daily'], arousal_loc_dict['loc'], arousal_loc_dict['global']

		if 'lounge' not in list(loc_dict[0]):
			continue

		for loc in loc_list:

			if 'unknown' in loc:
				continue

			for i in range(num_of_sec):
				row_df = pd.DataFrame(index=[participant_id])
				row_df['job'] = job_str
				row_df['icu'] = icu_str
				row_df['shift'] = shift_str
				row_df['supervise'] = supervise_str
				row_df['language'] = language_str
				row_df['gender'] = gender_str
				row_df['time'] = i
				row_df['loc'] = loc

				if loc_dict[i][loc]['pitch_arousal_ratio'] < 5:
					row_df['Log-Pitch Pos/Neg Arousal Ratio'] = loc_dict[i][loc]['pitch_arousal_ratio']
				else:
					row_df['Log-Pitch Pos/Neg Arousal Ratio'] = np.nan

				if loc_dict[i][loc]['intensity_arousal_ratio'] < 5:
					row_df['Intensity Pos/Neg Arousal Ratio'] = loc_dict[i][loc]['intensity_arousal_ratio']
				else:
					row_df['Log-Pitch Pos/Neg Arousal Ratio'] = np.nan

				if loc_dict[i][loc]['arousal_ratio'] < 5:
					row_df['Combined Pos/Neg Arousal Ratio'] = loc_dict[i][loc]['arousal_ratio']
				else:
					row_df['Log-Pitch Pos/Neg Arousal Ratio'] = np.nan

				row_df['Speaking Probability'] = loc_dict[i][loc]['speak_rate']

				data_df = data_df.append(row_df)

	nurse_df = data_df.loc[data_df['job'] == 'nurse']
	if compare_method == 'shift':
		cond_list = ['day', 'night']
		title_list = ['Day Shift Nurse', 'Night Shift Nurse']
	elif compare_method == 'supervise':
		cond_list = ['Supervise', 'Non-Supervise']
		title_list = ['Nurse Manager', 'Non-Nurse Manager']
	elif compare_method == 'language':
		cond_list = ['English', 'Non-English']
		title_list = ['Native Language: English', 'Native Language: Non-English']
	elif compare_method == 'gender':
		cond_list = ['Male', 'Female']
		title_list = ['Male Nurse', 'Female Nurse']
	else:
		cond_list = ['icu', 'non_icu']
		title_list = ['ICU Nurse', 'Non-ICU Nurse']

	# fig = plt.figure(figsize=(30, 8))
	# axes = fig.subplots(nrows=2, ncols=4)

	if num_of_sec == 6:
		fig, big_axes = plt.subplots(figsize=(34, 12), nrows=2, ncols=1)
	else:
		fig, big_axes = plt.subplots(figsize=(34, 12), nrows=2, ncols=1)

	for row, big_ax in enumerate(big_axes, start=0):
		big_ax.set_title(' ', fontsize=40, fontweight='bold')

		# Turn off axis lines and ticks of the big subplot
		# obs alpha is 0 in RGBA string!
		big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
		# removes the white frame
		big_ax._frameon = False

	if compare_method == 'shift':
		first_data_df = nurse_df.loc[nurse_df['shift'] == cond_list[0]]
		second_data_df = nurse_df.loc[nurse_df['shift'] == cond_list[1]]
	elif compare_method == 'supervise':
		first_data_df = nurse_df.loc[nurse_df['supervise'] == cond_list[0]]
		second_data_df = nurse_df.loc[nurse_df['supervise'] == cond_list[1]]
	elif compare_method == 'language':
		first_data_df = nurse_df.loc[nurse_df['language'] == cond_list[0]]
		second_data_df = nurse_df.loc[nurse_df['language'] == cond_list[1]]
	elif compare_method == 'gender':
		first_data_df = nurse_df.loc[nurse_df['gender'] == cond_list[0]]
		second_data_df = nurse_df.loc[nurse_df['gender'] == cond_list[1]]
	else:
		first_data_df = nurse_df.loc[nurse_df['icu'] == cond_list[0]]
		second_data_df = nurse_df.loc[nurse_df['icu'] == cond_list[1]]

	data_cols = ['Log-Pitch Pos/Neg Arousal Ratio', 'Intensity Pos/Neg Arousal Ratio', 'Combined Pos/Neg Arousal Ratio', 'Speaking Probability']

	'''
	sns.boxplot(x="time", y=data_cols[0], hue='loc', data=first_data_df, palette="seismic", ax=axes[0][0])
	sns.boxplot(x="time", y=data_cols[1], hue='loc', data=first_data_df, palette="seismic", ax=axes[0][1])
	sns.boxplot(x="time", y=data_cols[2], hue='loc', data=first_data_df, palette="seismic", ax=axes[0][2])
	sns.boxplot(x="time", y=data_cols[3], hue='loc', data=first_data_df, palette="seismic", ax=axes[0][3])
	sns.boxplot(x="time", y=data_cols[0], hue='loc', data=second_data_df, palette="seismic", ax=axes[1][0])
	sns.boxplot(x="time", y=data_cols[1], hue='loc', data=second_data_df, palette="seismic", ax=axes[1][1])
	sns.boxplot(x="time", y=data_cols[2], hue='loc', data=second_data_df, palette="seismic", ax=axes[1][2])
	sns.boxplot(x="time", y=data_cols[3], hue='loc', data=second_data_df, palette="seismic", ax=axes[1][3])
	'''

	for i in range(2):
		for j in range(4):
			axes = fig.add_subplot(2, 4, i * 4 + j + 1)

			if i == 0:
				sns.boxplot(x="time", y=data_cols[j], hue='loc', data=first_data_df, palette="seismic", ax=axes)
			else:
				sns.boxplot(x="time", y=data_cols[j], hue='loc', data=second_data_df, palette="seismic", ax=axes)

			axes.set_xlim([-0.75, num_of_sec - 0.25])
			axes.set_xlabel('')
			axes.set_ylabel('')
			axes.set_xticks(range(num_of_sec))

			if num_of_sec == 4:
				plot_list = ['0-3', '3-6', '6-9', '9-12']
			elif num_of_sec == 6:
				plot_list = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12']
			elif num_of_sec == 3:
				plot_list = ['0-4', '4-8', '8-12']
			else:
				# plot_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
				plot_list = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10-11', '11-12']

			for time in range(num_of_sec):
				if i == 0:
					tmp_df = first_data_df.loc[first_data_df['time'] == time]
				else:
					tmp_df = second_data_df.loc[second_data_df['time'] == time]

				ns_df = tmp_df.loc[tmp_df['loc'] == 'ns'].dropna()
				pat_df = tmp_df.loc[tmp_df['loc'] == 'pat'].dropna()
				lounge_df = tmp_df.loc[tmp_df['loc'] == 'lounge'].dropna()

				data_col = data_cols[j]

				ns_df = ns_df.loc[ns_df[data_col] < 5]
				pat_df = pat_df.loc[pat_df[data_col] < 5]
				lounge_df = lounge_df.loc[lounge_df[data_col] < 5]

				stats_value, p = stats.f_oneway(np.array(ns_df[data_col]), np.array(pat_df[data_col]), np.array(lounge_df[data_col]))

				if p < 0.01:
					plot_list[time] = plot_list[time] + '\n(p<0.01)'
				else:
					plot_list[time] = plot_list[time] + '\n(p=' + str(p)[:4] + ')'

			plt.rcParams["font.weight"] = "bold"
			plt.rcParams['axes.labelweight'] = 'bold'

			if num_of_sec == 6:
				for tick in axes.yaxis.get_major_ticks():
					tick.label1.set_fontsize(17)
					tick.label1.set_fontweight('bold')
			else:
				for tick in axes.yaxis.get_major_ticks():
					tick.label1.set_fontsize(23)
					tick.label1.set_fontweight('bold')

			axes.yaxis.set_tick_params(size=1)
			axes.grid(linestyle='--')
			axes.grid(False, axis='y')

			handles, labels = axes.get_legend_handles_labels()

			if num_of_sec == 6:
				axes.set_xticklabels(plot_list, fontdict={'fontweight': 'bold', 'fontsize': 17})

				axes.legend(handles=handles[0:], labels=labels[0:], prop={'size': 14}, loc='upper right')
				axes.tick_params(axis="y", labelsize=18)
				axes.set_title(data_cols[j], fontweight='bold', fontsize=18)

				if j == 3:
					axes.set_ylim([-0.26, 1.75])
				else:
					axes.set_ylim([-1.5, 5.5])
			else:
				axes.set_xticklabels(plot_list, fontdict={'fontweight': 'bold', 'fontsize': 23.5})

				axes.legend(handles=handles[0:], labels=labels[0:], prop={'size': 20}, loc='upper right')
				axes.tick_params(axis="y", labelsize=23)
				axes.set_title(data_cols[j], fontweight='bold', fontsize=23.5)

				if j == 3:
					axes.set_ylim([-0.26, 1.75])
				else:
					axes.set_ylim([-1.0, 5.0])

	# plt.suptitle(cond_str, fontsize=14, fontweight='bold')
	plt.tight_layout()

	if num_of_sec == 6:
		plt.figtext(0.5, 0.98, title_list[0], ha='center', va='center', fontsize=20, fontweight='bold')
		plt.figtext(0.5, 0.5, title_list[1], ha='center', va='center', fontsize=20, fontweight='bold')
	else:
		plt.figtext(0.51, 0.975, title_list[0], ha='center', va='center', fontsize=25.5, fontweight='bold')
		plt.figtext(0.51, 0.485, title_list[1], ha='center', va='center', fontsize=25.5, fontweight='bold')

	# fig.subplots_adjust(top=0.88)
	# plt.show()

	if os.path.exists(os.path.join('plot')) is False:
		os.mkdir(os.path.join('plot'))

	if os.path.exists(os.path.join('plot', 'loc')) is False:
		os.mkdir(os.path.join('plot', 'loc'))

	if os.path.exists(os.path.join('plot', 'loc', compare_method)) is False:
		os.mkdir(os.path.join('plot', 'loc', compare_method))

	plt.savefig(os.path.join('plot', 'loc', compare_method + '_' + str(num_of_sec) + '.png'), dpi=300)
	# plt.close()
	# plt.savefig(os.path.join('plot', 'daily', 'offday_diurnal'), dpi=300)
	plt.close()


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)