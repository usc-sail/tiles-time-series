"""
Cluster the audio data
"""
from __future__ import print_function

from sklearn import mixture
import os
import sys
import pandas as pd
import numpy as np
import dpmm
from collections import Counter
import dpgmm_gibbs
from vdpgmm import VDPGMM

from pybgmm.prior import NIW
from pybgmm.igmm import PCRPMM
from datetime import timedelta

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import itertools
import operator

import config
import load_sensor_data, load_data_path, load_data_basic, parser

'''
[	'F0final_sma', 'jitterLocal_sma', 'jitterDDP_sma',
	'shimmerLocal_sma', 'logHNR_sma', 'voiceProb_sma',
	'F0env_sma', 'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
	'pcm_RMSenergy_sma', 'pcm_zcr_sma',
	'pcm_intensity_sma', 'pcm_loudness_sma',
	'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
	'pcm_fftMag_spectralRollOff25.0_sma', 'pcm_fftMag_spectralRollOff50.0_sma',
	'pcm_fftMag_spectralRollOff75.0_sma', 'pcm_fftMag_spectralRollOff90.0_sma',
	'pcm_fftMag_spectralFlux_sma', 'pcm_fftMag_spectralCentroid_sma',
	'pcm_fftMag_spectralEntropy_sma', 'pcm_fftMag_spectralVariance_sma',
	'pcm_fftMag_spectralSkewness_sma', 'pcm_fftMag_spectralKurtosis_sma',
	'pcm_fftMag_spectralSlope_sma', 'pcm_fftMag_psySharpness_sma',
	'pcm_fftMag_spectralHarmonicity_sma']
'jitterLocal_sma', 'shimmerLocal_sma',
'''

all_feature_list = ['F0final_sma', 'jitterLocal_sma', 'jitterDDP_sma',
					'shimmerLocal_sma', 'logHNR_sma', 'voiceProb_sma',
					'F0env_sma', 'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
					'pcm_RMSenergy_sma', 'pcm_zcr_sma',
					'pcm_intensity_sma', 'pcm_loudness_sma',
					'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
					'pcm_fftMag_spectralRollOff25.0_sma', 'pcm_fftMag_spectralRollOff50.0_sma',
					'pcm_fftMag_spectralRollOff75.0_sma', 'pcm_fftMag_spectralRollOff90.0_sma',
					'pcm_fftMag_spectralFlux_sma', 'pcm_fftMag_spectralCentroid_sma',
					'pcm_fftMag_spectralEntropy_sma', 'pcm_fftMag_spectralVariance_sma',
					'pcm_fftMag_spectralSkewness_sma', 'pcm_fftMag_spectralKurtosis_sma',
					'pcm_fftMag_spectralSlope_sma', 'pcm_fftMag_psySharpness_sma',
					'pcm_fftMag_spectralHarmonicity_sma']

prosodic_based_feature_list = ['F0final_sma', 'F0env_sma', 'logHNR_sma',
							   'pcm_zcr_sma', 'pcm_RMSenergy_sma',
							   'pcm_intensity_sma', 'pcm_loudness_sma',
						 	   'jitterLocal_sma', 'shimmerLocal_sma']

light_feature_list = ['F0final_sma', 'pcm_zcr_sma',
					  'jitterLocal_sma', 'shimmerLocal_sma', 'logHNR_sma',
					  'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
					  'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
					  'pcm_fftMag_spectralCentroid_sma', 'pcm_fftMag_spectralEntropy_sma']


medium_feature_list = ['F0final_sma', 'pcm_zcr_sma',
					   'jitterLocal_sma', 'shimmerLocal_sma', 'logHNR_sma',
					   'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
					   'pcm_RMSenergy_sma', 'pcm_intensity_sma', 'pcm_loudness_sma',
					   'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
					   'pcm_fftMag_spectralCentroid_sma', 'pcm_fftMag_spectralEntropy_sma']

non_energy_feature_list = ['F0final_sma', 'pcm_zcr_sma',
						   'jitterLocal_sma', 'shimmerLocal_sma',
						   'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
						   'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma']


spectral_feature_list = ['pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
						 'pcm_fftMag_spectralRollOff25.0_sma', 'pcm_fftMag_spectralRollOff50.0_sma',
						 'pcm_fftMag_spectralRollOff75.0_sma', 'pcm_fftMag_spectralRollOff90.0_sma',
						 'pcm_fftMag_spectralFlux_sma', 'pcm_fftMag_spectralCentroid_sma',
						 'pcm_fftMag_spectralEntropy_sma', 'pcm_fftMag_spectralVariance_sma',
						 'pcm_fftMag_spectralSkewness_sma', 'pcm_fftMag_spectralKurtosis_sma',
						 'pcm_fftMag_spectralSlope_sma', 'pcm_fftMag_psySharpness_sma',
						 'pcm_fftMag_spectralHarmonicity_sma']


with_duration_feature_list = ['F0final_sma', 'jitterLocal_sma', 'shimmerLocal_sma',
							  'pcm_RMSenergy_sma', 'pcm_zcr_sma',
							  'pcm_intensity_sma', 'pcm_loudness_sma', 'logHNR_sma',
							  'audspec_lengthL1norm_sma', 'audspecRasta_lengthL1norm_sma',
							  'pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',]


def extract_audio_feature(data_config, data_df, feature_list):

	time_diff = pd.to_datetime(list(data_df.index)[1:]) - pd.to_datetime(list(data_df.index)[:-1])
	time_diff = list(time_diff.total_seconds())

	# Make sure we are extract over each snippet
	change_point_start_list = [0]

	if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
		change_point_end_list = list(np.where(np.array(time_diff) > float(data_config.audio_sensor_dict['pause_threshold']))[0])
	elif data_config.audio_sensor_dict['cluster_data'] == 'snippet':
		change_point_end_list = list(np.where(np.array(time_diff) > 20)[0])
	else:
		change_point_end_list = list(np.where(np.array(time_diff) > 20)[0])

	[change_point_start_list.append(change_point_end + 1) for change_point_end in change_point_end_list]
	change_point_end_list.append(len(data_df.index) - 1)

	time_start_end_list = []
	for i, change_point_end in enumerate(change_point_end_list):
		if 10 < change_point_end - change_point_start_list[i] < 20 * 100:
			time_start_end_list.append([list(data_df.index)[change_point_start_list[i]], list(data_df.index)[change_point_end]])

	day_df = pd.DataFrame()

	for time_start_end in time_start_end_list:
		start_time = (pd.to_datetime(time_start_end[0])).strftime(load_data_basic.date_time_format)[:-3]
		end_time = (pd.to_datetime(time_start_end[1])).strftime(load_data_basic.date_time_format)[:-3]

		tmp_data_df = pd.DataFrame(index=[start_time])

		tmp_raw_data_df = data_df[start_time:end_time]
		tmp_raw_data_df = tmp_raw_data_df[feature_list]

		if 'with_duration' in data_config.audio_sensor_dict['audio_feature']:
			if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
				segments = [len(list(x[1])) for x in itertools.groupby(list(tmp_raw_data_df['F0final_sma']), lambda x: x == 0) if not x[0]]
			elif data_config.audio_sensor_dict['cluster_data'] == 'snippet':
				time_diff = pd.to_datetime(list(tmp_raw_data_df.index)[1:]) - pd.to_datetime(list(tmp_raw_data_df.index)[:-1])
				time_diff = list(time_diff.total_seconds())
				change_point_start_list = [0]
				change_point_end_list = list(np.where(np.array(time_diff) > 0.2)[0])
				[change_point_start_list.append(change_point_end + 1) for change_point_end in change_point_end_list]
				change_point_end_list.append(len(tmp_raw_data_df.index) - 1)

				segments = []
				for i, change_point_end in enumerate(change_point_end_list):
					segments.append(change_point_end - change_point_start_list[i])
			tmp_data_df['num_segment'] = len(segments)
			tmp_data_df['mean_segment'] = np.mean(segments)

			# Each snippet is 20s seconds
			tmp_data_df['foreground_ratio'] = len(tmp_raw_data_df) / (100 * 20)

		if len(tmp_raw_data_df.loc[tmp_raw_data_df['F0final_sma'] > 0]) == 0:
			continue
		if len(tmp_raw_data_df.loc[tmp_raw_data_df['jitterLocal_sma'] > 0]) == 0:
			continue
		if len(tmp_raw_data_df.loc[tmp_raw_data_df['shimmerLocal_sma'] > 0]) == 0:
			continue

		for col in list(tmp_raw_data_df.columns):
			if 'jitterLocal_sma' in col or 'shimmerLocal_sma' in col or 'F0final_sma' in col:
				tmp_data_df[col + '_mean'] = np.nanmean(np.array(tmp_raw_data_df.loc[tmp_raw_data_df[col] > 0][col]))
				tmp_data_df[col + '_std'] = np.nanstd(np.array(tmp_raw_data_df.loc[tmp_raw_data_df[col] > 0][col]))

				if 'with_duration' not in data_config.audio_sensor_dict['audio_feature']:
					tmp_data_df[col + '_quantile_10'] = np.nanquantile(np.array(tmp_raw_data_df.loc[tmp_raw_data_df[col] > 0][col]), 0.1)
					tmp_data_df[col + '_quantile_90'] = np.nanquantile(np.array(tmp_raw_data_df.loc[tmp_raw_data_df[col] > 0][col]), 0.9)

			elif 'logHNR_sma' in col:
				tmp_data_df[col + '_mean'] = np.nanmean(np.array(tmp_raw_data_df.loc[tmp_raw_data_df[col] > -75][col]))
				tmp_data_df[col + '_std'] = np.nanstd(np.array(tmp_raw_data_df.loc[tmp_raw_data_df[col] > -75][col]))

				if 'with_duration' not in data_config.audio_sensor_dict['audio_feature']:
					tmp_data_df[col + '_quantile_10'] = np.nanquantile(np.array(tmp_raw_data_df.loc[tmp_raw_data_df[col] > -75][col]), 0.1)
					tmp_data_df[col + '_quantile_90'] = np.nanquantile(np.array(tmp_raw_data_df.loc[tmp_raw_data_df[col] > -75][col]), 0.9)
			# elif 'pcm_fftMag_spectralCentroid_sma' in col or 'pcm_fftMag_spectralEntropy_sma' in col
			else:
				tmp_data_df[col + '_mean'] = np.nanmean(np.array(tmp_raw_data_df[col]))
				tmp_data_df[col + '_std'] = np.nanstd(np.array(tmp_raw_data_df[col]))
				if 'with_duration' not in data_config.audio_sensor_dict['audio_feature']:
					tmp_data_df[col + '_quantile_10'] = np.nanquantile(np.array(tmp_raw_data_df[col]), 0.1)
					tmp_data_df[col + '_quantile_90'] = np.nanquantile(np.array(tmp_raw_data_df[col]), 0.9)

		day_df = day_df.append(tmp_data_df)

	return day_df


def main(tiles_data_path, config_path, experiment, skip_preprocess=False):
	# Create Config
	process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))

	data_config = config.Config()
	data_config.readConfigFile(config_path, experiment)

	# Load all data path according to config file
	load_data_path.load_all_available_path(data_config, process_data_path,
										   preprocess_data_identifier='preprocess',
										   segmentation_data_identifier='segmentation',
										   filter_data_identifier='filter_data',
										   clustering_data_identifier='clustering')

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()
	
	if 'all' in data_config.audio_sensor_dict['audio_feature']:
		feature_list = all_feature_list
	elif 'prosodic' in data_config.audio_sensor_dict['audio_feature']:
		feature_list = prosodic_based_feature_list
	elif 'light' in data_config.audio_sensor_dict['audio_feature']:
		feature_list = light_feature_list
	elif 'spectral' in data_config.audio_sensor_dict['audio_feature']:
		feature_list = spectral_feature_list
	elif 'medium' in data_config.audio_sensor_dict['audio_feature'] :
		feature_list = medium_feature_list
	elif 'non_energy' in data_config.audio_sensor_dict['audio_feature'] :
		feature_list = non_energy_feature_list
	elif 'with_duration' in data_config.audio_sensor_dict['audio_feature']:
		feature_list = with_duration_feature_list
	else:
		feature_list = all_feature_list
		
	filter_path = data_config.audio_sensor_dict['filter_path']
	audio_feature = data_config.audio_sensor_dict['audio_feature']
	pause_threshold = str(data_config.audio_sensor_dict['pause_threshold'])
		
	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read other sensor data, the aim is to detect whether people workes during a day
		if os.path.exists(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) is False:
			continue

		if len(os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id))) < 3:
			continue
			
		file_list = [file for file in os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) if 'utterance' not in file and 'minute' not in file and 'snippet' not in file]

		for file in file_list:
			tmp_raw_audio_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file), index_col=0)
			if len(tmp_raw_audio_df) < 3:
				continue

			file_exist = False
			if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
				file_name = 'pause_threshold_' + pause_threshold + '_' + audio_feature + '_utterance_' + file
				if os.path.exists(os.path.join(filter_path, participant_id, file_name)) is True:
					file_exist = True
			else:
				file_name = audio_feature + '_snippet_' + file
				if os.path.exists(os.path.join(filter_path, participant_id, audio_feature + '_snippet_' + file)) is True:
					file_exist = True
			'''
			if file_exist == True:
				continue
			'''
			tmp_raw_audio_df = tmp_raw_audio_df.drop(columns=['F0_sma'])

			day_df = extract_audio_feature(data_config, tmp_raw_audio_df, feature_list)
			day_df.to_csv(os.path.join(filter_path, participant_id, file_name), compression='gzip')

		
if __name__ == '__main__':
	
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
