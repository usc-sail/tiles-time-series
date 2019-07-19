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

color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']

def compare_feature(data_df, threshold, room1, room2, participant_type='icu'):

	if participant_type == 'icu':
		first_df = data_df.loc[data_df['icu'] == 'icu']
		second_df = data_df.loc[data_df['icu'] == 'non_icu']
		first_str, second_str = 'icu', 'non_icu'
	else:
		first_df = data_df.loc[data_df['shift'] == 'day']
		second_df = data_df.loc[data_df['shift'] == 'night']
		first_str, second_str = 'day', 'night'

	print('\n---------------------------------------------')
	print('p-value, %s-%s' % (room1, room2))

	# group 1
	print('---------------------------------------------')
	cond1 = first_df[room1 + '_' + room2 + '_p'] < 0.05
	cond2 = first_df[room1 + '_' + room2 + '_median_above_' + str(threshold)] == 1
	valid_len1 = len(first_df.loc[cond1 & cond2]) / len(first_df) * 100
	print('%s (%s - %s > %.1f): %.2f %%' % (first_str, room1, room2, threshold, valid_len1))
	cond2 = first_df[room1 + '_' + room2 + '_median_under_negative_' + str(threshold)] == 1
	valid_len2 = len(first_df.loc[cond1 & cond2]) / len(first_df) * 100
	print('%s (other): %.2f %%' % (first_str, 100 - valid_len1 - valid_len2))
	# cond2 = first_df[room1 + '_' + room2 + '_median_within_' + str(threshold)] == 1
	print('%s (-%.1f < %s - %s < %.1f): %.2f %%' % (first_str, threshold, room1, room2, threshold, 100 - valid_len1 - valid_len2))

	# group 2
	print('---------------------------------------------')
	cond1 = second_df[room1 + '_' + room2 + '_p'] < 0.05
	cond2 = second_df[room1 + '_' + room2 + '_median_above_' + str(threshold)] == 1
	valid_len1 = len(second_df.loc[cond1 & cond2]) / len(second_df) * 100
	print('%s (%s - %s > %.1f): %.2f %%' % (second_str, room1, room2, threshold, valid_len1))
	cond2 = second_df[room1 + '_' + room2 + '_median_under_negative_' + str(threshold)] == 1
	valid_len2 = len(second_df.loc[cond1 & cond2]) / len(second_df) * 100
	print('%s (%s - %s < -%.1f): %.2f %%' % (second_str, room1, room2, threshold, valid_len2))
	# cond2 = second_df[room1 + '_' + room2 + '_median_within_' + str(threshold)] == 1
	print('%s (other): %.2f %%' % (second_str, 100 - valid_len1 - valid_len2))
	print('---------------------------------------------\n')


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

	feat = 'pcm_loudness_sma'  # pcm_loudness_sma, pcm_intensity_sma, pcm_RMSenergy_sma, pcm_zcr_sma, F0_sma

	if feat == 'F0_sma':
		threshold = 5
	else:
		threshold = 0.05

	if os.path.exists(os.path.join('compare_' + feat + '.csv.gz')) is False:
		final_df = pd.DataFrame()
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

				# loc_list = ['lounge', 'med', 'ns', 'pat', 'unknown']
				row_df = pd.DataFrame(index=[participant_id])
				row_df['shift'] = shift_str
				row_df['icu'] = icu_str

				final_col = []
				for loc_first in ['pat', 'lounge', 'ns']:
					for loc_second in  ['pat', 'lounge', 'ns']:

						if loc_first == loc_second:
							continue

						if len(audio_length_part[loc_first]) < 1000 or len(audio_length_part[loc_second]) < 1000:
							continue

						first_data_array = audio_length_part[loc_first]
						second_data_array = audio_length_part[loc_second]

						stat, p = stats.ks_2samp(first_data_array, second_data_array)
						row_df[loc_first + '_' + loc_second + '_p'] = p
						row_df[loc_first + '_' + loc_second + '_stats'] = stat

				for col in ['lounge', 'ns', 'pat']:
					row_df[col + '_median'] = np.nanmedian(audio_length_part[col])
					row_df[col + '_mean'] = np.nanmean(audio_length_part[col])
					row_df[col + '_quantile25'] = np.nanquantile(audio_length_part[col], 0.25)
					row_df[col + '_quantile75'] = np.nanquantile(audio_length_part[col], 0.75)

				final_col = []
				for loc_first in ['pat', 'lounge', 'ns']:
					for loc_second in  ['pat', 'lounge', 'ns']:

						if loc_first == loc_second:
							continue

						if len(audio_length_part[loc_first]) < 1000 or len(audio_length_part[loc_second]) < 1000:
							continue

						loc = loc_first + '_' + loc_second
						final_col.append(loc)

						first_data_array = audio_length_part[loc_first]
						second_data_array = audio_length_part[loc_second]

						row_df[loc_first + '_' + loc_second + '_median_above_' + str(threshold)] = 0
						row_df[loc_first + '_' + loc_second + '_median_under_negative_' + str(threshold)] = 0
						row_df[loc_first + '_' + loc_second + '_median_within_' + str(threshold)] = 0
						row_df[loc_first + '_' + loc_second + '_mean_above_' + str(threshold)] = 0
						row_df[loc_first + '_' + loc_second + '_mean_under_negative_' + str(threshold)] = 0
						row_df[loc_first + '_' + loc_second + '_mean_within_' + str(threshold)] = 0

						if (np.nanmedian(np.array(first_data_array)) - np.nanmedian(np.array(second_data_array))) > threshold:
							row_df[loc_first + '_' + loc_second + '_median_above_' + str(threshold)] = 1
						elif (np.nanmedian(np.array(first_data_array)) - np.nanmedian(np.array(second_data_array))) < -threshold:
							row_df[loc_first + '_' + loc_second + '_median_under_negative_' + str(threshold)] = 1
						else:
							row_df[loc_first + '_' + loc_second + '_median_within_' + str(threshold)] = 1

						if (np.nanmean(np.array(first_data_array)) - np.nanmean(np.array(second_data_array))) > threshold:
							row_df[loc_first + '_' + loc_second + '_mean_above_' + str(threshold)] = 1
						elif (np.nanmean(np.array(first_data_array)) - np.nanmean(np.array(second_data_array))) < -threshold:
							row_df[loc_first + '_' + loc_second + '_mean_under_negative_' + str(threshold)] = 1
						else:
							row_df[loc_first + '_' + loc_second + '_mean_within_' + str(threshold)] = 1

				final_df = final_df.append(row_df)

		final_df.to_csv('compare_' + feat + '.csv.gz', compression='gzip')

	else:
		final_df = pd.read_csv('compare_' + feat + '.csv.gz', index_col=0)

	compare_feature(final_df, threshold, 'pat', 'ns', participant_type='icu')
	compare_feature(final_df, threshold, 'pat', 'lounge', participant_type='icu')
	compare_feature(final_df, threshold, 'ns', 'lounge', participant_type='icu')


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)