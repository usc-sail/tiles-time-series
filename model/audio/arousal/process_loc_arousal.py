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
	'''
	['F0final_sma', 'jitterLocal_sma', 'jitterDDP_sma', 'shimmerLocal_sma', 'logHNR_sma', 'voiceProb_sma', 'F0_sma',
	 'F0env_sma', 'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma', 'pcm_RMSenergy_sma', 'pcm_zcr_sma',
	 'pcm_intensity_sma', 'pcm_loudness_sma', 'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
	 'pcm_fftMag_spectralRollOff25.0_sma', 'pcm_fftMag_spectralRollOff50.0_sma', 'pcm_fftMag_spectralRollOff75.0_sma',
	 'pcm_fftMag_spectralRollOff90.0_sma', 'pcm_fftMag_spectralFlux_sma', 'pcm_fftMag_spectralCentroid_sma',
	 'pcm_fftMag_spectralEntropy_sma', 'pcm_fftMag_spectralVariance_sma', 'pcm_fftMag_spectralSkewness_sma',
	 'pcm_fftMag_spectralKurtosis_sma', 'pcm_fftMag_spectralSlope_sma', 'pcm_fftMag_psySharpness_sma',
	 'pcm_fftMag_spectralHarmonicity_sma']
	 '''
	feat_list = ['F0_sma', 'fft', 'pcm_intensity_sma'] # pcm_loudness_sma, pcm_intensity_sma, pcm_RMSenergy_sma, pcm_zcr_sma

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read id
		uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
		participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]

		# Read other sensor data, the aim is to detect whether people workes during a day
		owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
		raw_audio_df = load_sensor_data.read_raw_audio(tiles_data_path, participant_id)

		# If we don't have fitbit data, no need to process it
		if raw_audio_df is None or owl_in_one_df is None:
			print('%s has no audio data' % participant_id)
			continue

		owl_in_one_off_time = list((pd.to_datetime(owl_in_one_df.index[1:]) - pd.to_datetime(owl_in_one_df.index[:-1])).total_seconds())
		owl_in_one_change_idx = np.where(np.array(owl_in_one_off_time) > 3600 * 4)[0]

		if len(owl_in_one_change_idx) < 5:
			continue

		start_end_list = [[owl_in_one_df.index[0], owl_in_one_df.index[owl_in_one_change_idx[0]]]]
		for i, owl_in_one_idx in enumerate(owl_in_one_change_idx[:-1]):
			start_end_list.append([owl_in_one_df.index[owl_in_one_change_idx[i]+1], owl_in_one_df.index[owl_in_one_change_idx[i+1]]])
		start_end_list.append([owl_in_one_df.index[owl_in_one_change_idx[-1] + 1], owl_in_one_df.index[-1]])

		unique_list = list(owl_in_one_df.columns)

		audio_loc = {}
		for loc_str in unique_list:
			audio_loc[loc_str] = {}
			for feat in feat_list:
				audio_loc[loc_str][feat] = []

		arousal_loc_dict = {}
		arousal_loc_dict['loc_list'] = unique_list
		arousal_loc_dict['data'] = {}

		valid_start_end = []
		for start_end in start_end_list:
			start, end = start_end[0], start_end[1]
			time_span = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 3600

			if time_span < 2:
				continue
			valid_start_end.append(start_end)

		if len(valid_start_end) < 10:
			continue

		tmp_index_list = list(raw_audio_df.loc[raw_audio_df['F0_sma'] == 0].index)
		raw_audio_df.loc[tmp_index_list, 'F0_sma'] = np.nan
		# log_pitch_array = np.log(raw_audio_df['F0_sma'])
		raw_audio_df.loc[:, 'F0_sma'] = np.array(np.log(raw_audio_df['F0_sma']))

		feat_data_array = np.divide(np.array(raw_audio_df['pcm_fftMag_fband1000-4000_sma']),
		                            np.array(raw_audio_df['pcm_fftMag_fband250-650_sma']))
		raw_audio_df.loc[:, 'fft'] = feat_data_array
		raw_audio_df = raw_audio_df[feat_list]

		log_pitch_array = np.array(raw_audio_df['F0_sma'].dropna())

		# Aggregate feature now
		for start_end in start_end_list:
			start, end = start_end[0], start_end[1]
			time_span = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 3600

			if time_span < 2:
				continue

			audio_sec_df = raw_audio_df[start:end]
			owl_in_one_sec_df = owl_in_one_df[start:end]

			if len(audio_sec_df) == 0:
				continue

			owl_sec_time_list = list(owl_in_one_sec_df.index)
			for j in range(len(owl_in_one_sec_df)):
				location_series = owl_in_one_sec_df.iloc[j, :]
				location = list(location_series[location_series == 1].index)[0]

				loc_time = pd.to_datetime(owl_sec_time_list[j])
				loc_time_str = loc_time.strftime(load_data_basic.date_time_format)[:-3]
				start_loc = (loc_time - timedelta(seconds=30)).strftime(load_data_basic.date_time_format)[:-3]
				end_loc = (loc_time + timedelta(seconds=30)).strftime(load_data_basic.date_time_format)[:-3]

				tmp_audio_df = audio_sec_df[start_loc:end_loc]

				if len(tmp_audio_df) < 10:
					continue
				
				for feat in feat_list:
					for feat_value in list(tmp_audio_df[feat]):
						if feat_value > 0:
							audio_loc[location][feat].append(feat_value)

		# Calculate arousal
		for start_end in start_end_list:
			start, end = start_end[0], start_end[1]
			time_span = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds() / 3600

			if time_span < 2:
				continue

			audio_sec_df = raw_audio_df[start:end]
			owl_in_one_sec_df = owl_in_one_df[start:end]

			day_arousal_df = pd.DataFrame(index=list(owl_in_one_sec_df.index), columns=feat_list + ['loc'])

			if len(audio_sec_df) == 0:
				continue

			owl_sec_time_list = list(owl_in_one_sec_df.index)
			for j in range(len(owl_in_one_sec_df)):
				location_series = owl_in_one_sec_df.iloc[j, :]
				location = list(location_series[location_series == 1].index)[0]

				loc_time = pd.to_datetime(owl_sec_time_list[j])
				loc_time_str = loc_time.strftime(load_data_basic.date_time_format)[:-3]
				start_loc = (loc_time - timedelta(seconds=30)).strftime(load_data_basic.date_time_format)[:-3]
				end_loc = (loc_time + timedelta(seconds=30)).strftime(load_data_basic.date_time_format)[:-3]

				tmp_audio_df = audio_sec_df[start_loc:end_loc]
				day_arousal_df.loc[loc_time_str, 'loc'] = location

				if len(tmp_audio_df.dropna()) < 10:
					continue

				for feat in feat_list:
					median_value = np.nanmedian(np.array(tmp_audio_df[feat]))
					percentage = np.nanmean(np.array(audio_loc[location][feat]) <= median_value)
					day_arousal_df.loc[loc_time_str, feat] = percentage

			arousal_loc_dict['data'][start] = day_arousal_df
				
		if os.path.exists(os.path.join('data')) is False:
			os.mkdir(os.path.
			         join('data'))

		if os.path.exists(os.path.join('data', 'loc')) is False:
			os.mkdir(os.path.join('data', 'loc'))

		output = open(os.path.join('data', 'loc', participant_id + '.pkl'), 'wb')
		pickle.dump(arousal_loc_dict, output)


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'audio_location' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)