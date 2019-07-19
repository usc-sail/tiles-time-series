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

	feat = 'F0_sma'  # pcm_loudness_sma, pcm_intensity_sma, pcm_RMSenergy_sma, pcm_zcr_sma, F0_sma
	tmp_greater_than5, tmp_within5, tmp_less5 = 0, 0, 0

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

			loc_list = ['lounge', 'med', 'ns', 'pat', 'unknown']

			pat_longue_stat, pat_longue_p = stats.ks_2samp(audio_length_part['pat'], audio_length_part['lounge'])
			pat_ns_stat, pat_ns_p = stats.ks_2samp(audio_length_part['pat'], audio_length_part['ns'])

			print('\n\n')
			print('K-S test for %s' % feat)
			print('Statistics Patient-Lounge  = %.3f, p = %.3f' % (pat_longue_stat, pat_longue_p))
			print('Statistics Patient-NS = %.3f, p = %.3f' % (pat_ns_stat, pat_ns_p))
			print('Patient Room: mean = %.2f, std = %.2f' % (audio_length_part['pat'].dropna()), np.std(audio_length_part['pat'].dropna()))
			print('Lounge: mean = %.2f, std = %.2f' % (audio_length_part['lounge'].dropna()), np.std(audio_length_part['lounge'].dropna()))
			print('NS: mean = %.2f, std = %.2f' % (audio_length_part['ns'].dropna()), np.std(audio_length_part['ns'].dropna()))

			row_df = pd.DataFrame(index=[participant_id])
			row_df['shift'] = shift_str
			row_df['icu'] = icu_str

			row_df['Patient-Lounge-p'] = pat_longue_p
			row_df['Patient-NS-p'] = pat_ns_p
			row_df['Patient-Lounge-stat'] = pat_longue_stat
			row_df['Patient-NS-stat'] = pat_ns_stat

			final_df = final_df.append(row_df)

		final_df.to_csv('tmp.csv.gz', compression='gzip')

	'''
	if (np.nanmedian(np.array(audio_length_part['pat'])) - np.nanmedian(np.array(audio_length_part['lounge']))) > 5:
		tmp_greater_than5 += 1
	elif (np.nanmedian(np.array(audio_length_part['pat'])) - np.nanmedian(np.array(audio_length_part['lounge']))) < -5:
		tmp_less5 += 1
	else:
		tmp_within5 += 1

	for i, loc in enumerate(loc_list):

		print('loc: %s' % (loc))
		print(np.nanmedian(np.array(audio_length_part[loc])))
		print()
	'''
	'''
	print('tmp_greater_than5: %d' % tmp_greater_than5)
	print('tmp_less5: %d' % tmp_less5)
	print('tmp_within5: %d' % tmp_within5)
	'''


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)