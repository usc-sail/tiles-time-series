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

import warnings
warnings.filterwarnings("ignore")


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

		if os.path.exists(os.path.join('data', 'global', participant_id + '.pkl')) is False:
			continue

		arousal_loc_dict = np.load(os.path.join('data', 'global', participant_id + '.pkl'), allow_pickle=True)
		unique_list = arousal_loc_dict['loc_list']

		daily_dict, loc_dict, global_dict = {}, {}, {}
		global_dict['raw'] = pd.DataFrame()

		tmp_dict = {}
		tmp_dict['global'] = {}
		tmp_dict['global']['sec_min_list'] = []
		tmp_dict['global']['speak_min_list'] = []
		tmp_dict['global']['pitch_arousal_list'] = []
		tmp_dict['global']['intensity_arousal_list'] = []
		tmp_dict['global']['fft_arousal_list'] = []
		tmp_dict['global']['arousal_list'] = []

		for i in range(num_of_sec):
			loc_dict[i] = {}
			tmp_dict[i] = {}
			tmp_dict[i]['sec_min_list'] = []
			tmp_dict[i]['speak_min_list'] = []
			tmp_dict[i]['pitch_arousal_list'] = []
			tmp_dict[i]['intensity_arousal_list'] = []
			tmp_dict[i]['fft_arousal_list'] = []
			tmp_dict[i]['arousal_list'] = []

			for loc in unique_list:
				loc_dict[i][loc] = {}
				loc_dict[i][loc]['sec_min'] = []
				loc_dict[i][loc]['stay_min'] = []
				loc_dict[i][loc]['speak_min'] = []
				loc_dict[i][loc]['pitch_arousal_list'] = []
				loc_dict[i][loc]['intensity_arousal_list'] = []
				loc_dict[i][loc]['fft_arousal_list'] = []
				loc_dict[i][loc]['arousal_list'] = []

		loc_dict['global'] = {}
		for loc in unique_list:
			loc_dict['global'][loc] = {}
			loc_dict['global'][loc]['sec_min_list'] = []
			loc_dict['global'][loc]['stay_min_list'] = []
			loc_dict['global'][loc]['speak_min_list'] = []
			loc_dict['global'][loc]['pitch_arousal_list'] = []
			loc_dict['global'][loc]['intensity_arousal_list'] = []
			loc_dict['global'][loc]['fft_arousal_list'] = []
			loc_dict['global'][loc]['arousal_list'] = []

		for data_str in list(arousal_loc_dict['data'].keys()):
			daily_dict[data_str] = {}
			data_df = arousal_loc_dict['data'][data_str]

			if shift_str == 'day':
				start_str = pd.to_datetime(data_df.index[0]).replace(hour=7, minute=0, second=0)
			else:
				start_str = (pd.to_datetime(data_df.index[0]) - timedelta(hours=12)).replace(hour=19, minute=0, second=0)

			global_dict['raw'] = global_dict['raw'].append(data_df.dropna()[feat_list])

			for i in range(num_of_sec):
				daily_dict[data_str][i] = {}
				sec_start_str = (start_str + timedelta(hours=window*i)).strftime(load_data_basic.date_time_format)[:-3]
				sec_end_str = (start_str + timedelta(hours=window*i+window) - timedelta(minutes=1)).strftime(load_data_basic.date_time_format)[:-3]

				sec_df = data_df[sec_start_str:sec_end_str]
				if len(sec_df) < 60 * window / 2:
					for loc in unique_list:
						daily_dict[data_str][i][loc] = {}
						daily_dict[data_str][i][loc]['stay_min'] = np.nan
						daily_dict[data_str][i][loc]['stay_rate'] = np.nan
						daily_dict[data_str][i][loc]['speak_min'] = np.nan
						daily_dict[data_str][i][loc]['speak_rate'] = np.nan
					continue

				tmp_dict[i]['sec_min_list'].append(len(sec_df))
				tmp_dict['global']['sec_min_list'].append(len(sec_df))
				tmp_dict[i]['speak_min_list'].append(len(sec_df.dropna()))
				tmp_dict['global']['speak_min_list'].append(len(sec_df.dropna()))

				for value in np.nanmean(np.array(sec_df[['F0_sma', 'pcm_intensity_sma']].dropna()), axis=1):
					tmp_dict[i]['arousal_list'].append(value * 2 - 1)
					tmp_dict['global']['arousal_list'].append(value * 2 - 1)

				for value in np.array(sec_df['F0_sma'].dropna()):
					tmp_dict[i]['pitch_arousal_list'].append(value * 2 - 1)
					tmp_dict['global']['pitch_arousal_list'].append(value * 2 - 1)

				for value in np.array(sec_df['fft'].dropna()):
					tmp_dict[i]['fft_arousal_list'].append(value * 2 - 1)
					tmp_dict['global']['fft_arousal_list'].append(value * 2 - 1)

				for value in np.array(sec_df['pcm_intensity_sma'].dropna()):
					tmp_dict[i]['intensity_arousal_list'].append(value * 2 - 1)
					tmp_dict['global']['intensity_arousal_list'].append(value * 2 - 1)

				for loc in unique_list:
					loc_df = sec_df.loc[sec_df['loc'] == loc]

					daily_dict[data_str][i][loc] = {}
					daily_dict[data_str][i][loc]['stay_min'] = len(loc_df)
					daily_dict[data_str][i][loc]['stay_rate'] = len(loc_df) / len(sec_df)
					daily_dict[data_str][i][loc]['arousal_list'] = []
					daily_dict[data_str][i][loc]['pitch_arousal_list'] = []
					daily_dict[data_str][i][loc]['intensity_arousal_list'] = []
					daily_dict[data_str][i][loc]['fft_arousal_list'] = []

					loc_dict[i][loc]['sec_min'].append(len(sec_df))
					loc_dict[i][loc]['stay_min'].append(len(loc_df))

					loc_dict['global'][loc]['sec_min_list'].append(len(sec_df))
					loc_dict['global'][loc]['stay_min_list'].append(len(loc_df))

					if len(loc_df) == 0:
						daily_dict[data_str][i][loc]['speak_min'] = 0
						daily_dict[data_str][i][loc]['speak_rate'] = 0
						daily_dict[data_str][i][loc]['arousal_mean'] = np.nan

						loc_dict[i][loc]['speak_min'].append(0)
						loc_dict['global'][loc]['speak_min_list'].append(0)
					else:
						daily_dict[data_str][i][loc]['speak_min'] = len(loc_df.dropna())
						daily_dict[data_str][i][loc]['speak_rate'] = len(loc_df.dropna()) / len(loc_df)

						loc_dict[i][loc]['speak_min'].append(len(loc_df.dropna()))
						loc_dict['global'][loc]['speak_min_list'].append(len(loc_df.dropna()))

						if len(loc_df.dropna()) != 0:
							daily_dict[data_str][i][loc]['arousal'] = np.nanmean(np.mean(np.array(loc_df[['F0_sma', 'pcm_intensity_sma']]), axis=1))
							daily_dict[data_str][i][loc]['pitch_arousal'] = np.nanmean(np.array(loc_df['F0_sma']))

							for value in np.nanmean(np.array(loc_df[['F0_sma', 'pcm_intensity_sma']].dropna()), axis=1):
								daily_dict[data_str][i][loc]['arousal_list'].append(value * 2 - 1)
								loc_dict[i][loc]['arousal_list'].append(value * 2 - 1)
								loc_dict['global'][loc]['arousal_list'].append(value * 2 - 1)

							for value in np.array(loc_df['F0_sma'].dropna()):
								daily_dict[data_str][i][loc]['pitch_arousal_list'].append(value * 2 - 1)
								loc_dict[i][loc]['pitch_arousal_list'].append(value * 2 - 1)
								loc_dict['global'][loc]['pitch_arousal_list'].append(value * 2 - 1)

							for value in np.array(loc_df['fft'].dropna()):
								daily_dict[data_str][i][loc]['fft_arousal_list'].append(value * 2 - 1)
								loc_dict[i][loc]['fft_arousal_list'].append(value * 2 - 1)
								loc_dict['global'][loc]['fft_arousal_list'].append(value * 2 - 1)

							for value in np.array(loc_df['pcm_intensity_sma'].dropna()):
								daily_dict[data_str][i][loc]['intensity_arousal_list'].append(value * 2 - 1)
								loc_dict[i][loc]['intensity_arousal_list'].append(value * 2 - 1)
								loc_dict['global'][loc]['intensity_arousal_list'].append(value * 2 - 1)
						else:
							daily_dict[data_str][i][loc]['arousal'] = np.nan
							daily_dict[data_str][i][loc]['pitch_arousal'] = np.nan

		global_dict['speak_rate'] = np.nansum(np.array(tmp_dict['global']['speak_min_list'])) / np.nansum(np.array(tmp_dict['global']['sec_min_list']))
		global_dict['pitch_arousal'] = np.nanmean(np.array(tmp_dict['global']['pitch_arousal_list']))
		global_dict['pitch_arousal_90_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['pitch_arousal_list']), 90)
		global_dict['pitch_arousal_75_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['pitch_arousal_list']), 75)
		global_dict['pitch_arousal_25_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['pitch_arousal_list']), 25)
		global_dict['pitch_arousal_10_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['pitch_arousal_list']), 10)
		global_dict['pitch_arousal_std'] = np.nanstd(np.array(tmp_dict['global']['pitch_arousal_list']))
		global_dict['pitch_arousal_high'] = np.nanmean(np.array(tmp_dict['global']['pitch_arousal_list']) > 0)
		global_dict['pitch_arousal_low'] = np.nanmean(np.array(tmp_dict['global']['pitch_arousal_list']) <= 0)
		global_dict['pitch_arousal_ratio'] = global_dict['pitch_arousal_high'] / global_dict['pitch_arousal_low']

		global_dict['intensity_arousal'] = np.nanmean(np.array(tmp_dict['global']['intensity_arousal_list']))
		global_dict['intensity_arousal_90_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['intensity_arousal_list']), 90)
		global_dict['intensity_arousal_75_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['intensity_arousal_list']), 75)
		global_dict['intensity_arousal_25_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['intensity_arousal_list']), 25)
		global_dict['intensity_arousal_10_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['intensity_arousal_list']), 10)
		global_dict['intensity_arousal_std'] = np.nanstd(np.array(tmp_dict['global']['intensity_arousal_list']))
		global_dict['intensity_arousal_high'] = np.nanmean(np.array(tmp_dict['global']['intensity_arousal_list']) > 0)
		global_dict['intensity_arousal_low'] = np.nanmean(np.array(tmp_dict['global']['intensity_arousal_list']) <= 0)
		global_dict['intensity_arousal_ratio'] = global_dict['intensity_arousal_high'] / global_dict['intensity_arousal_low']

		global_dict['fft_arousal'] = np.nanmean(np.array(tmp_dict['global']['fft_arousal_list']))
		global_dict['fft_arousal_90_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['fft_arousal_list']), 90)
		global_dict['fft_arousal_75_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['fft_arousal_list']), 75)
		global_dict['fft_arousal_25_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['fft_arousal_list']), 25)
		global_dict['fft_arousal_10_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['fft_arousal_list']), 10)
		global_dict['fft_arousal_std'] = np.nanstd(np.array(tmp_dict['global']['fft_arousal_list']))
		global_dict['fft_arousal_high'] = np.nanmean(np.array(tmp_dict['global']['fft_arousal_list']) > 0)
		global_dict['fft_arousal_low'] = np.nanmean(np.array(tmp_dict['global']['fft_arousal_list']) <= 0)
		global_dict['fft_arousal_ratio'] = global_dict['fft_arousal_high'] / global_dict['fft_arousal_low']

		global_dict['arousal'] = np.nanmean(np.array(tmp_dict['global']['arousal_list']))
		global_dict['arousal_90_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['arousal_list']), 90)
		global_dict['arousal_75_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['arousal_list']), 75)
		global_dict['arousal_25_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['arousal_list']), 25)
		global_dict['arousal_10_percentile'] = np.nanpercentile(np.array(tmp_dict['global']['arousal_list']), 10)
		global_dict['arousal_std'] = np.nanstd(np.array(tmp_dict['global']['arousal_list']))
		global_dict['arousal_high'] = np.nanmean(np.array(tmp_dict['global']['arousal_list']) > 0)
		global_dict['arousal_low'] = np.nanmean(np.array(tmp_dict['global']['arousal_list']) <= 0)
		global_dict['arousal_ratio'] = global_dict['arousal_high'] / global_dict['arousal_low']

		for loc in unique_list:
			loc_dict['global'][loc]['stay_rate'] = np.nansum(np.array(loc_dict['global'][loc]['stay_min_list'])) / np.nansum(np.array(loc_dict['global'][loc]['sec_min_list']))
			loc_dict['global'][loc]['speak_rate'] = np.nansum(np.array(loc_dict['global'][loc]['speak_min_list'])) / np.nansum(np.array(loc_dict['global'][loc]['stay_min_list']))
			loc_dict['global'][loc]['speak_arousal'] = np.nansum(np.array(loc_dict['global'][loc]['speak_min_list'])) / np.nansum(np.array(loc_dict['global'][loc]['stay_min_list']))

			loc_dict['global'][loc]['pitch_arousal'] = np.nanmean(np.array(loc_dict['global'][loc]['pitch_arousal_list']))
			loc_dict['global'][loc]['pitch_arousal_90_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['pitch_arousal_list']), 90)
			loc_dict['global'][loc]['pitch_arousal_75_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['pitch_arousal_list']), 75)
			loc_dict['global'][loc]['pitch_arousal_25_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['pitch_arousal_list']), 25)
			loc_dict['global'][loc]['pitch_arousal_10_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['pitch_arousal_list']), 10)
			loc_dict['global'][loc]['pitch_arousal_std'] = np.nanstd(np.array(loc_dict['global'][loc]['pitch_arousal_list']))
			loc_dict['global'][loc]['pitch_arousal_high'] = np.nanmean(np.array(loc_dict['global'][loc]['pitch_arousal_list']) > 0)
			loc_dict['global'][loc]['pitch_arousal_low'] = np.nanmean(np.array(loc_dict['global'][loc]['pitch_arousal_list']) <= 0)
			loc_dict['global'][loc]['pitch_arousal_ratio'] = loc_dict['global'][loc]['pitch_arousal_high'] / loc_dict['global'][loc]['pitch_arousal_low']

			loc_dict['global'][loc]['intensity_arousal'] = np.nanmean(np.array(loc_dict['global'][loc]['intensity_arousal_list']))
			loc_dict['global'][loc]['intensity_arousal_90_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['intensity_arousal_list']), 90)
			loc_dict['global'][loc]['intensity_arousal_75_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['intensity_arousal_list']), 75)
			loc_dict['global'][loc]['intensity_arousal_25_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['intensity_arousal_list']), 25)
			loc_dict['global'][loc]['intensity_arousal_10_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['intensity_arousal_list']), 10)
			loc_dict['global'][loc]['intensity_arousal_std'] = np.nanstd(np.array(loc_dict['global'][loc]['intensity_arousal_list']))
			loc_dict['global'][loc]['intensity_arousal_high'] = np.nanmean(np.array(loc_dict['global'][loc]['intensity_arousal_list']) > 0)
			loc_dict['global'][loc]['intensity_arousal_low'] = np.nanmean(np.array(loc_dict['global'][loc]['intensity_arousal_list']) <= 0)
			loc_dict['global'][loc]['intensity_arousal_ratio'] = loc_dict['global'][loc]['intensity_arousal_high'] / loc_dict['global'][loc]['intensity_arousal_low']

			loc_dict['global'][loc]['fft_arousal'] = np.nanmean(np.array(loc_dict['global'][loc]['fft_arousal_list']))
			loc_dict['global'][loc]['fft_arousal_90_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['fft_arousal_list']), 90)
			loc_dict['global'][loc]['fft_arousal_75_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['fft_arousal_list']), 75)
			loc_dict['global'][loc]['fft_arousal_25_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['fft_arousal_list']), 25)
			loc_dict['global'][loc]['fft_arousal_10_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['fft_arousal_list']), 10)
			loc_dict['global'][loc]['fft_arousal_std'] = np.nanstd(np.array(loc_dict['global'][loc]['fft_arousal_list']))
			loc_dict['global'][loc]['fft_arousal_high'] = np.nanmean(np.array(loc_dict['global'][loc]['fft_arousal_list']) > 0)
			loc_dict['global'][loc]['fft_arousal_low'] = np.nanmean(np.array(loc_dict['global'][loc]['fft_arousal_list']) <= 0)
			loc_dict['global'][loc]['fft_arousal_ratio'] = loc_dict['global'][loc]['fft_arousal_high'] / loc_dict['global'][loc]['fft_arousal_low']

			loc_dict['global'][loc]['arousal'] = np.nanmean(np.array(loc_dict['global'][loc]['arousal_list']))
			loc_dict['global'][loc]['arousal_90_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['arousal_list']), 90)
			loc_dict['global'][loc]['arousal_75_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['arousal_list']), 75)
			loc_dict['global'][loc]['arousal_25_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['arousal_list']), 25)
			loc_dict['global'][loc]['arousal_10_percentile'] = np.nanpercentile(np.array(loc_dict['global'][loc]['arousal_list']), 10)
			loc_dict['global'][loc]['arousal_std'] = np.nanstd(np.array(loc_dict['global'][loc]['arousal_list']))
			loc_dict['global'][loc]['arousal_high'] = np.nanmean(np.array(loc_dict['global'][loc]['arousal_list']) > 0)
			loc_dict['global'][loc]['arousal_low'] = np.nanmean(np.array(loc_dict['global'][loc]['arousal_list']) <= 0)
			loc_dict['global'][loc]['arousal_ratio'] = loc_dict['global'][loc]['arousal_high'] / loc_dict['global'][loc]['arousal_low']

		for i in range(num_of_sec):
			global_dict[i] = {}
			global_dict[i]['speak_rate'] = np.nansum(np.array(tmp_dict[i]['speak_min_list'])) / np.nansum(np.array(tmp_dict[i]['sec_min_list']))

			global_dict[i]['pitch_arousal'] = np.nanmean(np.array(tmp_dict[i]['pitch_arousal_list']))
			global_dict[i]['pitch_arousal_90_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['pitch_arousal_list']), 90)
			global_dict[i]['pitch_arousal_75_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['pitch_arousal_list']), 75)
			global_dict[i]['pitch_arousal_25_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['pitch_arousal_list']), 25)
			global_dict[i]['pitch_arousal_10_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['pitch_arousal_list']), 10)
			global_dict[i]['pitch_arousal_std'] = np.nanstd(np.array(tmp_dict[i]['pitch_arousal_list']))
			global_dict[i]['pitch_arousal_high'] = np.nanmean(np.array(tmp_dict[i]['pitch_arousal_list']) > 0)
			global_dict[i]['pitch_arousal_low'] = np.nanmean(np.array(tmp_dict[i]['pitch_arousal_list']) <= 0)
			global_dict[i]['pitch_arousal_ratio'] = global_dict[i]['pitch_arousal_high'] / global_dict[i]['pitch_arousal_low']

			global_dict[i]['intensity_arousal'] = np.nanmean(np.array(tmp_dict[i]['intensity_arousal_list']))
			global_dict[i]['intensity_arousal_90_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['intensity_arousal_list']), 90)
			global_dict[i]['intensity_arousal_75_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['intensity_arousal_list']), 75)
			global_dict[i]['intensity_arousal_25_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['intensity_arousal_list']), 25)
			global_dict[i]['intensity_arousal_10_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['intensity_arousal_list']), 10)
			global_dict[i]['intensity_arousal_std'] = np.nanstd(np.array(tmp_dict[i]['intensity_arousal_list']))
			global_dict[i]['intensity_arousal_high'] = np.nanmean(np.array(tmp_dict[i]['intensity_arousal_list']) > 0)
			global_dict[i]['intensity_arousal_low'] = np.nanmean(np.array(tmp_dict[i]['intensity_arousal_list']) <= 0)
			global_dict[i]['intensity_arousal_ratio'] = global_dict[i]['intensity_arousal_high'] / global_dict[i]['intensity_arousal_low']

			global_dict[i]['fft_arousal'] = np.nanmean(np.array(tmp_dict[i]['fft_arousal_list']))
			global_dict[i]['fft_arousal_90_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['fft_arousal_list']), 90)
			global_dict[i]['fft_arousal_75_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['fft_arousal_list']), 75)
			global_dict[i]['fft_arousal_25_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['fft_arousal_list']), 25)
			global_dict[i]['fft_arousal_10_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['fft_arousal_list']), 10)
			global_dict[i]['fft_arousal_std'] = np.nanstd(np.array(tmp_dict[i]['fft_arousal_list']))
			global_dict[i]['fft_arousal_high'] = np.nanmean(np.array(tmp_dict[i]['fft_arousal_list']) > 0)
			global_dict[i]['fft_arousal_low'] = np.nanmean(np.array(tmp_dict[i]['fft_arousal_list']) <= 0)
			global_dict[i]['fft_arousal_ratio'] = global_dict[i]['fft_arousal_high'] / global_dict[i]['fft_arousal_low']

			global_dict[i]['arousal'] = np.nanmean(np.array(tmp_dict[i]['arousal_list']))
			global_dict[i]['arousal_90_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['arousal_list']), 90)
			global_dict[i]['arousal_75_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['arousal_list']), 75)
			global_dict[i]['arousal_25_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['arousal_list']), 25)
			global_dict[i]['arousal_10_percentile'] = np.nanpercentile(np.array(tmp_dict[i]['arousal_list']), 10)
			global_dict[i]['arousal_std'] = np.nanstd(np.array(tmp_dict[i]['arousal_list']))
			global_dict[i]['arousal_high'] = np.nanmean(np.array(tmp_dict[i]['arousal_list']) > 0)
			global_dict[i]['arousal_low'] = np.nanmean(np.array(tmp_dict[i]['arousal_list']) <= 0)
			global_dict[i]['arousal_ratio'] = global_dict[i]['arousal_high'] / global_dict[i]['arousal_low']

			for loc in unique_list:
				loc_dict[i][loc]['stay_rate'] = np.nansum(np.array(loc_dict[i][loc]['stay_min'])) / np.nansum(np.array(loc_dict[i][loc]['sec_min']))
				loc_dict[i][loc]['speak_rate'] = np.nansum(np.array(loc_dict[i][loc]['speak_min'])) / np.nansum(np.array(loc_dict[i][loc]['stay_min']))

				loc_dict[i][loc]['pitch_arousal'] = np.nanmean(np.array(loc_dict[i][loc]['pitch_arousal_list']))
				loc_dict[i][loc]['pitch_arousal_90_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['pitch_arousal_list']), 90)
				loc_dict[i][loc]['pitch_arousal_75_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['pitch_arousal_list']), 75)
				loc_dict[i][loc]['pitch_arousal_25_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['pitch_arousal_list']), 25)
				loc_dict[i][loc]['pitch_arousal_10_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['pitch_arousal_list']), 10)
				loc_dict[i][loc]['pitch_arousal_std'] = np.nanstd(np.array(loc_dict[i][loc]['pitch_arousal_list']))
				loc_dict[i][loc]['pitch_arousal_high'] = np.nanmean(np.array(loc_dict[i][loc]['pitch_arousal_list']) > 0)
				loc_dict[i][loc]['pitch_arousal_low'] = np.nanmean(np.array(loc_dict[i][loc]['pitch_arousal_list']) <= 0)
				loc_dict[i][loc]['pitch_arousal_ratio'] = loc_dict[i][loc]['pitch_arousal_high'] / loc_dict[i][loc]['pitch_arousal_low']

				loc_dict[i][loc]['intensity_arousal'] = np.nanmean(np.array(loc_dict[i][loc]['intensity_arousal_list']))
				loc_dict[i][loc]['intensity_arousal_90_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['intensity_arousal_list']), 90)
				loc_dict[i][loc]['intensity_arousal_75_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['intensity_arousal_list']), 75)
				loc_dict[i][loc]['intensity_arousal_25_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['intensity_arousal_list']), 25)
				loc_dict[i][loc]['intensity_arousal_10_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['intensity_arousal_list']), 10)
				loc_dict[i][loc]['intensity_arousal_std'] = np.nanstd(np.array(loc_dict[i][loc]['intensity_arousal_list']))
				loc_dict[i][loc]['intensity_arousal_high'] = np.nanmean(np.array(loc_dict[i][loc]['intensity_arousal_list']) > 0)
				loc_dict[i][loc]['intensity_arousal_low'] = np.nanmean(np.array(loc_dict[i][loc]['intensity_arousal_list']) <= 0)
				loc_dict[i][loc]['intensity_arousal_ratio'] = loc_dict[i][loc]['intensity_arousal_high'] / loc_dict[i][loc]['intensity_arousal_low']

				loc_dict[i][loc]['fft_arousal'] = np.nanmean(np.array(loc_dict[i][loc]['fft_arousal_list']))
				loc_dict[i][loc]['fft_arousal_90_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['fft_arousal_list']), 90)
				loc_dict[i][loc]['fft_arousal_75_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['fft_arousal_list']), 75)
				loc_dict[i][loc]['fft_arousal_25_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['fft_arousal_list']), 25)
				loc_dict[i][loc]['fft_arousal_10_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['fft_arousal_list']), 10)
				loc_dict[i][loc]['fft_arousal_std'] = np.nanstd(np.array(loc_dict[i][loc]['fft_arousal_list']))
				loc_dict[i][loc]['fft_arousal_high'] = np.nanmean(np.array(loc_dict[i][loc]['fft_arousal_list']) > 0)
				loc_dict[i][loc]['fft_arousal_low'] = np.nanmean(np.array(loc_dict[i][loc]['fft_arousal_list']) <= 0)
				loc_dict[i][loc]['fft_arousal_ratio'] = loc_dict[i][loc]['fft_arousal_high'] / loc_dict[i][loc]['fft_arousal_low']

				loc_dict[i][loc]['arousal'] = np.nanmean(np.array(loc_dict[i][loc]['arousal_list']))
				loc_dict[i][loc]['arousal_90_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['arousal_list']), 90)
				loc_dict[i][loc]['arousal_75_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['arousal_list']), 75)
				loc_dict[i][loc]['arousal_25_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['arousal_list']), 25)
				loc_dict[i][loc]['arousal_10_percentile'] = np.nanpercentile(np.array(loc_dict[i][loc]['arousal_list']), 10)
				loc_dict[i][loc]['arousal_std'] = np.nanstd(np.array(loc_dict[i][loc]['arousal_list']))
				loc_dict[i][loc]['arousal_high'] = np.nanmean(np.array(loc_dict[i][loc]['arousal_list']) > 0)
				loc_dict[i][loc]['arousal_low'] = np.nanmean(np.array(loc_dict[i][loc]['arousal_list']) <= 0)
				loc_dict[i][loc]['arousal_ratio'] = loc_dict[i][loc]['arousal_high'] / loc_dict[i][loc]['arousal_low']

		final_dict = {}
		final_dict['global'] = global_dict
		final_dict['loc'] = loc_dict
		final_dict['daily'] = daily_dict

		if os.path.exists(os.path.join('arousal')) is False:
			os.mkdir(os.path.join('arousal'))

		if os.path.exists(os.path.join('arousal', 'global')) is False:
			os.mkdir(os.path.join('arousal', 'global'))

		if os.path.exists(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec))) is False:
			os.mkdir(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec)))

		output = open(os.path.join('arousal', 'global', 'num_of_sec_' + str(num_of_sec), participant_id + '.pkl'), 'wb')
		pickle.dump(final_dict, output)


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)