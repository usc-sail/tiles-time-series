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
	compare_list = ['shift', 'icu', 'gender', 'supervise', 'language']
	# compare_method = 'shift'
	feat_type = 'prob'

	# final_df = pd.DataFrame()
	data_df = pd.DataFrame()

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

		for i in range(num_of_sec):
			row_df = pd.DataFrame(index=[participant_id])
			row_df['job'] = job_str
			row_df['icu'] = icu_str
			row_df['shift'] = shift_str
			row_df['supervise'] = supervise_str
			row_df['language'] = language_str
			row_df['gender'] = gender_str
			row_df['time'] = i

			row_df['Log-Pitch Pos|Neg Arousal Ratio'] = global_dict[i]['pitch_arousal_ratio']
			row_df['Intensity Pos|Neg Arousal Ratio'] = global_dict[i]['intensity_arousal_ratio']
			row_df['Combined Pos|Neg Arousal Ratio'] = global_dict[i]['arousal_ratio']
			row_df['Speaking Probability'] = global_dict[i]['speak_rate']
			row_df['Mean Log-Pitch Arousal'] = global_dict[i]['pitch_arousal']
			row_df['Mean Intensity Arousal'] = global_dict[i]['intensity_arousal']
			row_df['Mean Combined Arousal'] = global_dict[i]['arousal']

			data_df = data_df.append(row_df)

	data_cols = ['Mean Log-Pitch Arousal', 'Mean Intensity Arousal', 'Mean Combined Arousal',
	             'Log-Pitch Pos|Neg Arousal Ratio', 'Intensity Pos|Neg Arousal Ratio', 'Combined Pos|Neg Arousal Ratio',
	             'Speaking Probability']

	for j, data_col in enumerate(data_cols):

		fig = plt.figure(figsize=(24, 4))
		axes = fig.subplots(nrows=1, ncols=5)

		for i, compare_method in enumerate(compare_list):

			nurse_df = data_df.loc[data_df['job'] == 'nurse']
			if compare_method == 'shift':
				first_data_df = nurse_df.loc[nurse_df['shift'] == 'day']
				second_data_df = nurse_df.loc[nurse_df['shift'] == 'night']
			elif compare_method == 'supervise':
				first_data_df = nurse_df.loc[nurse_df['supervise'] == 'Supervise']
				second_data_df = nurse_df.loc[nurse_df['supervise'] == 'Non-Supervise']
			elif compare_method == 'language':
				first_data_df = nurse_df.loc[nurse_df['language'] == 'English']
				second_data_df = nurse_df.loc[nurse_df['language'] == 'Non-English']
			elif compare_method == 'gender':
				first_data_df = nurse_df.loc[nurse_df['gender'] == 'Male']
				second_data_df = nurse_df.loc[nurse_df['gender'] == 'Female']
			else:
				first_data_df = nurse_df.loc[nurse_df['icu'] == 'icu']
				second_data_df = nurse_df.loc[nurse_df['icu'] == 'non_icu']

			first_analysis_df = first_data_df
			second_analysis_df = second_data_df
			plt_df = nurse_df
			sns.lineplot(x="time", y=data_cols[j], dashes=False, marker="o", hue=compare_method, data=plt_df, palette="seismic", ax=axes[i])

			if num_of_sec == 4:
				plot_list = ['0-3', '3-6', '6-9', '9-12']
			elif num_of_sec == 6:
				plot_list = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-12']
			elif num_of_sec == 3:
				plot_list = ['0-4', '4-8', '8-12']
			else:
				plot_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
				# plot_list = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6',
				#              '6-7', '7-8', '8-9', '9-10', '10-11', '11-12']

			for time in range(num_of_sec):
				first_tmp_df = first_analysis_df.loc[first_analysis_df['time'] == time]
				second_tmp_df = second_analysis_df.loc[second_analysis_df['time'] == time]

				data_col = data_cols[j]
				stat, p = stats.ks_2samp(first_tmp_df[data_col].dropna(), second_tmp_df[data_col].dropna())
				cohens_d = (np.nanmean(first_tmp_df[data_col]) - np.nanmean(second_tmp_df[data_col]))
				cohens_d = cohens_d / np.sqrt((np.nanstd(first_tmp_df[data_col]) ** 2 + np.nanstd(second_tmp_df[data_col] ** 2)) / 2)

				if cohens_d < 0:
					cohens_d = str(cohens_d)[:5]
				else:
					cohens_d = str(cohens_d)[:4]

				if num_of_sec != 12:
					if p < 0.01:
						plot_list[time] = plot_list[time] + '\n(p<0.01, \nd=' + cohens_d + ')'
					else:
						plot_list[time] = plot_list[time] + '\n(p=' + str(p)[:4] + ', \nd=' + cohens_d + ')'
				# sns.lineplot(x="time", y=data_cols[1], dashes=False, marker="o", hue=compare_method, data=plt_df, palette="seismic", ax=axes[1])
				# sns.lineplot(x="time", y=data_cols[2], dashes=False, marker="o", hue=compare_method, data=plt_df, palette="seismic", ax=axes[2])
				# sns.lineplot(x="time", y=data_cols[3], dashes=False, marker="o", hue=compare_method, data=plt_df, palette="seismic", ax=axes[3])
				axes[i].set_xticklabels(plot_list, fontdict={'fontweight': 'bold', 'fontsize': 12})

		for i in range(5):
			axes[i].set_xlim([-0.25, num_of_sec - 0.75])
			axes[i].set_xlabel('')
			axes[i].set_ylabel('')
			axes[i].set_xticks(range(num_of_sec))

			axes[i].yaxis.set_tick_params(size=1)
			axes[i].grid(linestyle='--')
			axes[i].grid(False, axis='y')

			handles, labels = axes[i].get_legend_handles_labels()
			axes[i].legend(handles=handles[1:], labels=labels[1:], prop={'size': 10.5})
			axes[i].lines[0].set_linestyle("--")
			axes[i].lines[1].set_linestyle("--")
			axes[i].tick_params(axis="y", labelsize=14)

			axes[i].set_title(compare_list[i], fontweight='bold', fontsize=14)

			if 'Ratio' in data_cols[j]:
				axes[i].set_ylim([0, 2])
			elif 'Mean' in data_cols[j]:
				axes[i].set_ylim([-0.25, 0.15])
			else:
				axes[i].set_ylim([0.1, 0.6])

		plt.suptitle(data_cols[j], fontsize=14, fontweight='bold')
		plt.tight_layout()
		fig.subplots_adjust(top=0.85)

		if os.path.exists(os.path.join('plot')) is False:
			os.mkdir(os.path.join('plot'))

		plt.savefig(os.path.join('plot', data_cols[j] + '_' + str(num_of_sec) + '.png'), dpi=300)
		plt.close()


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)