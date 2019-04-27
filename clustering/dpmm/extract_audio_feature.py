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
							   'audspec_lengthL1norm_sma',
							   'audspecRasta_lengthL1norm_sma',
							   'pcm_zcr_sma', 'pcm_RMSenergy_sma',
							   'pcm_intensity_sma', 'pcm_loudness_sma',
						 	   'jitterLocal_sma', 'shimmerLocal_sma']

light_feature_list = ['F0final_sma', 'pcm_zcr_sma',
					  'jitterLocal_sma', 'shimmerLocal_sma', 'logHNR_sma',
					  'audspec_lengthL1norm_sma',
					  'audspecRasta_lengthL1norm_sma',
					  'pcm_fftMag_spectralCentroid_sma', 'pcm_fftMag_spectralEntropy_sma',
					  'pcm_fftMag_spectralSkewness_sma', 'pcm_fftMag_spectralKurtosis_sma',
					  'pcm_fftMag_spectralSlope_sma']

spectral_feature_list = ['pcm_fftMag_fband250-650_sma', 'pcm_fftMag_fband1000-4000_sma',
						 'pcm_fftMag_spectralRollOff25.0_sma', 'pcm_fftMag_spectralRollOff50.0_sma',
						 'pcm_fftMag_spectralRollOff75.0_sma', 'pcm_fftMag_spectralRollOff90.0_sma',
						 'pcm_fftMag_spectralFlux_sma', 'pcm_fftMag_spectralCentroid_sma',
						 'pcm_fftMag_spectralEntropy_sma', 'pcm_fftMag_spectralVariance_sma',
						 'pcm_fftMag_spectralSkewness_sma', 'pcm_fftMag_spectralKurtosis_sma',
						 'pcm_fftMag_spectralSlope_sma', 'pcm_fftMag_psySharpness_sma',
						 'pcm_fftMag_spectralHarmonicity_sma']


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
	
	if data_config.audio_sensor_dict['audio_feature'] == 'all':
		feature_list = all_feature_list
	elif data_config.audio_sensor_dict['audio_feature'] == 'prosodic':
		feature_list = prosodic_based_feature_list
	elif data_config.audio_sensor_dict['audio_feature'] == 'light':
		feature_list = light_feature_list
	elif data_config.audio_sensor_dict['audio_feature'] == 'spectral':
		feature_list = spectral_feature_list
	else:
		feature_list = all_feature_list
		
	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read other sensor data, the aim is to detect whether people workes during a day
		if os.path.exists(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) is False:
			continue

		if len(os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id))) < 3:
			continue
			
		file_list = [file for file in os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) if 'utterance' not in file and 'minute' not in file]

		raw_audio_df, utterance_df, minute_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

		for file in file_list:
			tmp_raw_audio_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file), index_col=0)
			if len(tmp_raw_audio_df) < 3:
				continue
			
			tmp_raw_audio_df = tmp_raw_audio_df.drop(columns=['F0_sma'])

			# if cluster utterance
			if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
				if os.path.exists(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'utterance_' + file)) is True and skip_preprocess is True:
					day_utterance_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'utterance_' + file), index_col=0)
					utterance_df = utterance_df.append(day_utterance_df)
					continue

				time_diff = pd.to_datetime(list(tmp_raw_audio_df.index)[1:]) - pd.to_datetime(list(tmp_raw_audio_df.index)[:-1])
				time_diff = list(time_diff.total_seconds())

				change_point_start_list = [0]
				change_point_end_list = list(np.where(np.array(time_diff) > float(data_config.audio_sensor_dict['pause_threshold']))[0])

				[change_point_start_list.append(change_point_end + 1) for change_point_end in change_point_end_list]
				change_point_end_list.append(len(tmp_raw_audio_df.index) - 1)

				time_start_end_list = []
				for i, change_point_end in enumerate(change_point_end_list):
					if 10 < change_point_end - change_point_start_list[i] < 20 * 100:
						time_start_end_list.append([list(tmp_raw_audio_df.index)[change_point_start_list[i]], list(tmp_raw_audio_df.index)[change_point_end]])

				day_utterance_df = pd.DataFrame()
				for time_start_end in time_start_end_list:
					start_time = (pd.to_datetime(time_start_end[0])).strftime(load_data_basic.date_time_format)[:-3]
					end_time = (pd.to_datetime(time_start_end[1])).strftime(load_data_basic.date_time_format)[:-3]
					
					tmp_utterance_df = pd.DataFrame(index=[start_time])
					# tmp_utterance_df['duration'] = (pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds()
					
					tmp_utterance_raw_df = tmp_raw_audio_df[start_time:end_time]
					tmp_utterance_raw_df = tmp_utterance_raw_df[feature_list]
					
					full_length = len(tmp_utterance_raw_df)
					
					segments = [len(list(x[1])) for x in itertools.groupby(list(tmp_utterance_raw_df['F0final_sma']), lambda x: x == 0) if not x[0] ]
					# tmp_utterance_df['num_segment'] = len(segments)
					# tmp_utterance_df['mean_segment'] = np.mean(segments)
					
					tmp_utterance_raw_df = tmp_utterance_raw_df.loc[tmp_utterance_raw_df['F0final_sma'] > 0]
					if len(tmp_utterance_raw_df) == 0:
						continue
					
					# non_zero_f0_length = len(tmp_utterance_raw_df)
					# tmp_utterance_df['non_zero_f0_ratio'] = non_zero_f0_length / full_length
					
					for col in list(tmp_utterance_raw_df.columns):
						
						tmp_utterance_df[col + '_mean'] = np.mean(np.array(tmp_utterance_raw_df[col]))
						tmp_utterance_df[col + '_std'] = np.std(np.array(tmp_utterance_raw_df[col]))

					day_utterance_df = day_utterance_df.append(tmp_utterance_df)

				day_utterance_df.to_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'pause_threshold_' + str(data_config.audio_sensor_dict['pause_threshold']) + '_' + data_config.audio_sensor_dict['audio_feature'] + '_utterance_' + file), compression='gzip')
				utterance_df = utterance_df.append(day_utterance_df)
				
			elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
				if os.path.exists(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'minute_' + file)) is True and skip_preprocess is True:
					day_minute_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, 'minute_' + file), index_col=0)
					minute_df = minute_df.append(day_minute_df)
					continue
				
				time_start = pd.to_datetime(list(tmp_raw_audio_df.index)[0]).replace(second=0, microsecond=0).strftime(load_data_basic.date_time_format)[:-3]
				time_end = (pd.to_datetime(list(tmp_raw_audio_df.index)[-1]) + timedelta(minutes=1)).replace(second=0, microsecond=0).strftime(load_data_basic.date_time_format)[:-3]
				
				time_span = (pd.to_datetime(time_end) - pd.to_datetime(time_start)).total_seconds() / 60
				
				day_minute_df = pd.DataFrame()
				
				for offset in range(int(time_span)):
					start_time = (pd.to_datetime(time_start) + timedelta(minutes=offset-1)).strftime(load_data_basic.date_time_format)[:-3]
					end_time = (pd.to_datetime(time_start) + timedelta(minutes=offset+1)).strftime(load_data_basic.date_time_format)[:-3]
					index_time = (pd.to_datetime(time_start) + timedelta(minutes=offset)).strftime(load_data_basic.date_time_format)[:-3]
					
					tmp_minute_audio_data = tmp_raw_audio_df[start_time:end_time]
					tmp_minute_data_df = pd.DataFrame(index=[index_time])
					
					if len(tmp_minute_audio_data) < 10:
						continue
					
					tmp_minute_audio_data = tmp_minute_audio_data[feature_list]
					
					full_length = len(tmp_minute_audio_data)
					
					segments = [len(list(x[1])) for x in itertools.groupby(list(tmp_minute_audio_data['F0final_sma']), lambda x: x == 0) if not x[0]]
					# tmp_minute_data_df['num_segment'] = len(segments)
					# tmp_minute_data_df['mean_segment'] = np.mean(segments)
					
					tmp_minute_audio_data = tmp_minute_audio_data.loc[tmp_minute_audio_data['F0final_sma'] > 0]
					if len(tmp_minute_audio_data) == 0:
						continue
					
					non_zero_f0_length = len(tmp_minute_audio_data)
					
					tmp_minute_data_df['non_zero_f0_ratio'] = non_zero_f0_length / full_length
					
					for col in list(tmp_minute_audio_data.columns):
						
						tmp_minute_data_df[col + '_mean'] = np.mean(np.array(tmp_minute_audio_data[col]))
						tmp_minute_data_df[col + '_std'] = np.std(np.array(tmp_minute_audio_data[col]))
					
					day_minute_df = day_minute_df.append(tmp_minute_data_df)
				
				day_minute_df.to_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, data_config.audio_sensor_dict['audio_feature'] + '_minute_' + file), compression='gzip')
				minute_df = minute_df.append(day_minute_df)
		
		
if __name__ == '__main__':
	
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
