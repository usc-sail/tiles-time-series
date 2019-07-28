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

	feat_list = ['F0_sma', 'fft', 'pcm_intensity_sma'] # pcm_loudness_sma, pcm_intensity_sma, pcm_RMSenergy_sma, pcm_zcr_sma
	num_of_sec = 4
	window = int(12 / num_of_sec)

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read id
		uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		shift_str = 'day' if shift == 'Day shift' else 'night'

		arousal_loc_dict = np.load(os.path.join('data', 'loc', participant_id + '.pkl'), allow_pickle=True)
		unique_list = arousal_loc_dict['loc_list']

		arousal_dict, stat_dict = {}, {}
		for i in range(num_of_sec):
			stat_dict[i] = {}
			for loc in unique_list:
				stat_dict[i][loc] = {}
				stat_dict[i][loc]['sec_min'] = []
				stat_dict[i][loc]['stay_min'] = []
				stat_dict[i][loc]['speak_min'] = []
				stat_dict[i][loc]['pitch_arousal_list'] = []
				stat_dict[i][loc]['arousal_list'] = []

		stat_dict['global'] = {}
		for loc in unique_list:
			stat_dict['global'][loc] = {}
			stat_dict['global'][loc]['sec_min_list'] = []
			stat_dict['global'][loc]['stay_min_list'] = []
			stat_dict['global'][loc]['speak_min_list'] = []
			stat_dict['global'][loc]['pitch_arousal_list'] = []
			stat_dict['global'][loc]['arousal_list'] = []

		for data_str in list(arousal_loc_dict['data'].keys()):
			arousal_dict[data_str] = {}
			data_df = arousal_loc_dict['data'][data_str]

			if shift_str == 'day':
				start_str = pd.to_datetime(data_df.index[0]).replace(hour=7, minute=0, second=0)
			else:
				start_str = (pd.to_datetime(data_df.index[0]) - timedelta(hours=12)).replace(hour=19, minute=0, second=0)

			for i in range(num_of_sec):
				arousal_dict[data_str][i] = {}
				sec_start_str = (start_str + timedelta(hours=window*i)).strftime(load_data_basic.date_time_format)[:-3]
				sec_end_str = (start_str + timedelta(hours=window*i+window) - timedelta(minutes=1)).strftime(load_data_basic.date_time_format)[:-3]

				sec_df = data_df[sec_start_str:sec_end_str]
				if len(sec_df) < 60 * window / 2:
					for loc in unique_list:
						arousal_dict[data_str][i][loc] = {}
						arousal_dict[data_str][i][loc]['stay_min'] = np.nan
						arousal_dict[data_str][i][loc]['stay_rate'] = np.nan
						arousal_dict[data_str][i][loc]['speak_min'] = np.nan
						arousal_dict[data_str][i][loc]['speak_rate'] = np.nan
					continue

				for loc in unique_list:
					loc_df = sec_df.loc[sec_df['loc'] == loc]

					arousal_dict[data_str][i][loc] = {}
					arousal_dict[data_str][i][loc]['stay_min'] = len(loc_df)
					arousal_dict[data_str][i][loc]['stay_rate'] = len(loc_df) / len(sec_df)
					arousal_dict[data_str][i][loc]['arousal_list'] = []
					arousal_dict[data_str][i][loc]['pitch_arousal_list'] = []

					stat_dict[i][loc]['sec_min'].append(len(sec_df))
					stat_dict[i][loc]['stay_min'].append(len(loc_df))

					stat_dict['global'][loc]['sec_min_list'].append(len(sec_df))
					stat_dict['global'][loc]['stay_min_list'].append(len(loc_df))

					if len(loc_df) == 0:
						arousal_dict[data_str][i][loc]['speak_min'] = 0
						arousal_dict[data_str][i][loc]['speak_rate'] = 0
						arousal_dict[data_str][i][loc]['arousal_mean'] = np.nan

						stat_dict[i][loc]['speak_min'].append(0)
						stat_dict['global'][loc]['speak_min_list'].append(0)

					else:
						arousal_dict[data_str][i][loc]['speak_min'] = len(loc_df.dropna())
						arousal_dict[data_str][i][loc]['speak_rate'] = len(loc_df.dropna()) / len(loc_df)

						stat_dict[i][loc]['speak_min'].append(len(loc_df.dropna()))
						stat_dict['global'][loc]['speak_min_list'].append(len(loc_df.dropna()))

						if len(loc_df.dropna()) != 0:
							arousal_dict[data_str][i][loc]['arousal'] = np.nanmean(np.mean(np.array(loc_df[feat_list] * 2 - 1), axis=1))
							arousal_dict[data_str][i][loc]['pitch_arousal'] = np.nanmean(np.array(loc_df['F0_sma'] * 2 - 1))

							for value in np.nanmean(np.array(loc_df[feat_list].dropna()), axis=1):
								arousal_dict[data_str][i][loc]['arousal_list'].append(value * 2 - 1)
								stat_dict[i][loc]['arousal_list'].append(value * 2 - 1)
								stat_dict['global'][loc]['arousal_list'].append(value * 2 - 1)

							for value in np.array(loc_df['F0_sma'].dropna()):
								arousal_dict[data_str][i][loc]['pitch_arousal_list'].append(value * 2 - 1)
								stat_dict[i][loc]['pitch_arousal_list'].append(value * 2 - 1)
								stat_dict['global'][loc]['pitch_arousal_list'].append(value * 2 - 1)
						else:
							arousal_dict[data_str][i][loc]['arousal'] = np.nan
							arousal_dict[data_str][i][loc]['pitch_arousal'] = np.nan

		for loc in unique_list:
			stat_dict['global'][loc]['stay_rate'] = np.sum(np.array(stat_dict['global'][loc]['stay_min_list'])) / np.sum(np.array(stat_dict['global'][loc]['sec_min_list']))
			stat_dict['global'][loc]['speak_rate'] = np.sum(np.array(stat_dict['global'][loc]['speak_min_list'])) / np.sum(np.array(stat_dict['global'][loc]['stay_min_list']))
			stat_dict['global'][loc]['pitch_arousal_mean'] = np.nanmean(np.array(stat_dict['global'][loc]['pitch_arousal_list']))
			stat_dict['global'][loc]['arousal_mean'] = np.nanmean(np.array(stat_dict['global'][loc]['arousal_list']))

		for i in range(num_of_sec):
			for loc in unique_list:
				stat_dict[i][loc]['stay_rate'] = np.sum(np.array(stat_dict[i][loc]['stay_min'])) / np.sum(np.array(stat_dict[i][loc]['sec_min']))
				stat_dict[i][loc]['speak_rate'] = np.sum(np.array(stat_dict[i][loc]['speak_min'])) / np.sum(np.array(stat_dict[i][loc]['stay_min']))
				stat_dict[i][loc]['pitch_arousal_mean'] = np.nanmean(np.array(stat_dict[i][loc]['pitch_arousal_list']))
				stat_dict[i][loc]['arousal_mean'] = np.nanmean(np.array(stat_dict[i][loc]['arousal_list']))

		if os.path.exists(os.path.join('data')) is False:
			os.mkdir(os.path.join('data'))

		if os.path.exists(os.path.join('data', 'loc_arousal')) is False:
			os.mkdir(os.path.join('data', 'loc_arousal'))

		output = open(os.path.join('data', 'loc_arousal', participant_id + '.pkl'), 'wb')
		pickle.dump(arousal_loc_dict, output)


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)