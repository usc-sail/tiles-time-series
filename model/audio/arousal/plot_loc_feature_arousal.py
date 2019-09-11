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
	num_of_sec = 4
	compare_method = 'icu'
	alpha = 0.5

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
	elif compare_method == 'supervise':
		cond_list = ['Supervise', 'Non-Supervise']
	elif compare_method == 'language':
		cond_list = ['English', 'Non-English']
	elif compare_method == 'gender':
		cond_list = ['Male', 'Female']
	else:
		cond_list = ['icu', 'non_icu']

	for cond_str in cond_list:

		fig = plt.figure(figsize=(30, 4))
		axes = fig.subplots(nrows=1, ncols=4)

		if compare_method == 'shift':
			plot_data_df = nurse_df.loc[nurse_df['shift'] == cond_str]
		elif compare_method == 'supervise':
			plot_data_df = nurse_df.loc[nurse_df['supervise'] == cond_str]
		elif compare_method == 'language':
			plot_data_df = nurse_df.loc[nurse_df['language'] == cond_str]
		elif compare_method == 'gender':
			plot_data_df = nurse_df.loc[nurse_df['gender'] == cond_str]
		else:
			plot_data_df = nurse_df.loc[nurse_df['icu'] == cond_str]

		data_cols = ['Log-Pitch Pos/Neg Arousal Ratio', 'Intensity Pos/Neg Arousal Ratio',
		             'Combined Pos/Neg Arousal Ratio', 'Speaking Probability']
		sns.boxplot(x="time", y=data_cols[0], hue='loc', data=plot_data_df, palette="seismic", ax=axes[0])
		sns.boxplot(x="time", y=data_cols[1], hue='loc', data=plot_data_df, palette="seismic", ax=axes[1])
		sns.boxplot(x="time", y=data_cols[2], hue='loc', data=plot_data_df, palette="seismic", ax=axes[2])
		sns.boxplot(x="time", y=data_cols[3], hue='loc', data=plot_data_df, palette="seismic", ax=axes[3])

		for i in range(4):

			axes[i].set_xlim([-0.75, num_of_sec - 0.25])
			axes[i].set_xlabel('')
			axes[i].set_ylabel('')
			axes[i].set_xticks(range(num_of_sec))

			if num_of_sec == 4:
				plot_list = ['0-3', '3-6', '6-9', '9-12']
			elif num_of_sec == 6:
				plot_list = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12']
			elif num_of_sec == 3:
				plot_list = ['0-4', '4-8', '8-12']
			else:
				# plot_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
				plot_list = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6',
				             '6-7', '7-8', '8-9', '9-10', '10-11', '11-12']

			for time in range(num_of_sec):
				first_tmp_df = plot_data_df.loc[plot_data_df['time'] == time]

				ns_df = first_tmp_df.loc[first_tmp_df['loc'] == 'ns'].dropna()
				pat_df = first_tmp_df.loc[first_tmp_df['loc'] == 'pat'].dropna()
				lounge_df = first_tmp_df.loc[first_tmp_df['loc'] == 'lounge'].dropna()

				data_col = data_cols[i]

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

			axes[i].set_xticklabels(plot_list, fontdict={'fontweight': 'bold', 'fontsize': 14})
			# axes[i][j].set_yticklabels(axes[i][j].get_yticks(), fontdict={'fontweight': 'bold', 'fontsize': 14})
			# ax.yaxis.set_tick_params(labelsize=20)

			for tick in axes[i].yaxis.get_major_ticks():
				tick.label1.set_fontsize(14)
				tick.label1.set_fontweight('bold')

			axes[i].yaxis.set_tick_params(size=1)
			axes[i].grid(linestyle='--')
			axes[i].grid(False, axis='y')

			handles, labels = axes[i].get_legend_handles_labels()
			axes[i].legend(handles=handles[0:], labels=labels[0:], prop={'size': 10.5})
			# axes[i].lines[0].set_linestyle("--")
			# axes[i].lines[1].set_linestyle("--")
			axes[i].tick_params(axis="y", labelsize=14)

			axes[i].set_title(data_cols[i], fontweight='bold', fontsize=15)
			# axes[i][j].set_ylim([0.3, 0.6]) if i == 0 else axes[i][j].set_ylim([0.15, 0.55])
			axes[i].set_ylim([-1, 5])

		axes[3].set_ylim([-0.25, 1.25])

		# plt.suptitle(cond_str, fontsize=14, fontweight='bold')
		plt.tight_layout()
		# fig.subplots_adjust(top=0.88)
		# plt.show()

		if os.path.exists(os.path.join('plot')) is False:
			os.mkdir(os.path.join('plot'))

		if os.path.exists(os.path.join('plot', 'loc')) is False:
			os.mkdir(os.path.join('plot', 'loc'))

		if os.path.exists(os.path.join('plot', 'loc', cond_str)) is False:
			os.mkdir(os.path.join('plot', 'loc', cond_str))

		plt.savefig(os.path.join('plot', 'loc', cond_str, str(num_of_sec) + '.png'), dpi=300)
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