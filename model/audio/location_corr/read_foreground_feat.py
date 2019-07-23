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
import seaborn as sns
import matplotlib.pyplot as plt

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

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

	# pcm_loudness_sma, pcm_intensity_sma, pcm_RMSenergy_sma, pcm_zcr_sma, F0_sma
	# pcm_fftMag_fband250-650_sma, pcm_fftMag_fband1000-4000_sma, pcm_fftMag_spectralCentroid_sma
	feat = 'pcm_intensity_sma'

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read id
		uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
		position = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition)[0]
		shift = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift)[0]
		primary_unit = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit)[0]

		icu_str = 'non_icu'
		if 'ICU' in primary_unit:
			icu_str = 'icu'

		for unit in icu_list:
			if unit in primary_unit:
				icu_str = 'icu'

		shift_str = 'day' if shift == 'Day shift' else 'night'

		if os.path.exists(os.path.join('data', feat, participant_id + '.pkl')) is False:
			continue

		pkl_file = open(os.path.join('data', feat, participant_id + '.pkl'), 'rb')
		audio_length_part = pickle.load(pkl_file)

		if 'lounge' not in list(audio_length_part.keys()):
			continue

		if position == 1:

			fig = plt.figure(figsize=(8, 16))

			loc_list = ['lounge', 'med', 'ns', 'pat', 'unknown']
			axes = fig.subplots(nrows=len(loc_list))

			for i, loc in enumerate(loc_list):

				len_list = audio_length_part[loc]

				if feat == 'pcm_intensity_sma':
					len_list = np.log10(np.array(len_list) * (np.power(10, 12))) * 10
				# sns.kdeplot(len_list, shade=True, ax=axes[i])
				# bins = np.arange(0, 2.5, 0.05)
				# bins = np.arange(20, 500, 10)
				# bins = np.arange(100, 1000, 20)
				bins = np.arange(20, 80, 2)
				n, bins, patches = axes[i].hist(len_list, bins=bins, density=1, color=color_list[i])
				# n, bins, patches = axes[i].hist(len_list, density=1, color=color_list[i])
				axes[i].set_xlim([20, 80])
				# axes[i].set_xlim([100, 1000])
				# axes[i].set_xlim([20, 500])
				# axes[i].set_ylim([0, 0.25])
				# axes[i].set_xlim([0, 2.5])
				axes[i].set_title(loc)

				if i != len(loc_list) - 1:
					axes[i].set_xticklabels('')

			plt.tight_layout()

			if os.path.exists(os.path.join('plot')) is False:
				os.mkdir(os.path.join('plot'))

			if os.path.exists(os.path.join('plot', feat)) is False:
				os.mkdir(os.path.join('plot', feat))

			if os.path.exists(os.path.join('plot', feat, shift_str)) is False:
				os.mkdir(os.path.join('plot', feat, shift_str))

			if os.path.exists(os.path.join('plot', feat, icu_str)) is False:
				os.mkdir(os.path.join('plot', feat, icu_str))

			plt.savefig(os.path.join('plot', feat, shift_str, participant_id))
			plt.savefig(os.path.join('plot', feat, icu_str, participant_id))


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)