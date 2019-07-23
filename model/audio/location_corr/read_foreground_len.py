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

	threshold = 0.1

	for idx, participant_id in enumerate(top_participant_id_list[0:]):

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

		if os.path.exists(os.path.join('data', 'len', 'data_' + str(threshold), participant_id + '.pkl')) is False:
			continue

		pkl_file = open(os.path.join('data', 'len', 'data_' + str(threshold), participant_id + '.pkl'), 'rb')
		audio_length_part = pickle.load(pkl_file)

		if 'lounge' not in list(audio_length_part.keys()):
			continue

		if position == 1:

			fig = plt.figure(figsize=(8, 16))

			'''
			loc_list = list(audio_length_part.keys())
			
			if 'floor2' in list(audio_length_part.keys()):
				loc_list.remove('floor2')
			loc_list.sort()
			if 'floor2' in list(audio_length_part.keys()):
				audio_length_part['other_floor'] += audio_length_part['floor2']
			'''
			loc_list = ['lounge', 'med', 'ns', 'pat', 'unknown']
			axes = fig.subplots(nrows=len(loc_list))

			for i, loc in enumerate(loc_list):
				len_list = audio_length_part[loc]
				# sns.kdeplot(len_list, shade=True, ax=axes[i])
				bins = np.arange(0, 20, 1)
				n, bins, patches = axes[i].hist(len_list, bins=bins, density=1, color=color_list[i])
				axes[i].set_xlim([0, 20])
				axes[i].set_ylim([0, 0.3])
				axes[i].set_title(loc)

				if i != len(audio_length_part.keys()) - 1:
					axes[i].set_xticklabels('')

			plt.tight_layout()

			if os.path.exists(os.path.join('plot')) is False:
				os.mkdir(os.path.join('plot'))

			if os.path.exists(os.path.join('plot', 'len')) is False:
				os.mkdir(os.path.join('plot', 'len'))

			if os.path.exists(os.path.join('plot', 'len', 'data_' + str(threshold))) is False:
				os.mkdir(os.path.join('plot', 'len', 'data_' + str(threshold)))

			if os.path.exists(os.path.join('plot', 'len', 'data_' + str(threshold), shift_str)) is False:
				os.mkdir(os.path.join('plot', 'len', 'data_' + str(threshold), shift_str))

			if os.path.exists(os.path.join('plot', 'len', 'data_' + str(threshold), icu_str)) is False:
				os.mkdir(os.path.join('plot', 'len', 'data_' + str(threshold), icu_str))

			plt.savefig(os.path.join('plot', 'len', 'data_' + str(threshold), shift_str, participant_id))
			plt.savefig(os.path.join('plot', 'len', 'data_' + str(threshold), icu_str, participant_id))


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)