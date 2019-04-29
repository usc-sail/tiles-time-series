"""
Cluster the audio data
"""
from __future__ import print_function

from sklearn import mixture
import os
import sys
import pandas as pd
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser

SEED = 5132290  # from random.org

np.random.seed(SEED)


def lda_audio(data_df, cluster_df, data_config, participant_id, lda_components='auto'):

	data_cluster_path = data_config.audio_sensor_dict['clustering_path']

	X = np.array(data_df)
	y = np.array(cluster_df['cluster'])
	
	if lda_components == 'auto':
		if len(np.unique(y)) < data_df.shape[1]:
			lda_array = LinearDiscriminantAnalysis(n_components=None).fit(X, y).transform(X)
		else:
			lda_array = LinearDiscriminantAnalysis(n_components=data_df.shape[1]).fit(X, y).transform(X)
	else:
		lda_array = LinearDiscriminantAnalysis(n_components=int(lda_components)).fit(X, y).transform(X)

	lda_df = pd.DataFrame(lda_array, index=list(cluster_df.index))
	if os.path.exists(os.path.join(data_cluster_path, participant_id)) is False:
		os.mkdir(os.path.join(data_cluster_path, participant_id))
	lda_df.to_csv(os.path.join(data_cluster_path, participant_id, 'lda_' + str(lda_components) + '.csv.gz'), compression='gzip')
	

def main(tiles_data_path, config_path, experiment):
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

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read other sensor data, the aim is to detect whether people workes during a day
		if os.path.exists(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) is False:
			continue

		if len(os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id))) < 3:
			continue
		
		if data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
			file_list = [file for file in os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) if
						 'utterance' not in file and 'minute' not in file]
		else:
			config_cond = 'pause_threshold_' + str(data_config.audio_sensor_dict['pause_threshold']) + '_' + data_config.audio_sensor_dict['audio_feature']
			file_list = [file for file in os.listdir(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id)) if
						 data_config.audio_sensor_dict['cluster_data'] in file and config_cond in file]
		
		raw_audio_df, utterance_df, minute_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
		
		for file in file_list:
			tmp_raw_audio_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file), index_col=0)
			if len(tmp_raw_audio_df) < 3:
				continue
			
			# if cluster raw_audio
			if data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
				tmp_raw_audio_df = tmp_raw_audio_df.drop(columns=['F0_sma'])
				raw_audio_df = raw_audio_df.append(tmp_raw_audio_df)
			# if cluster utterance
			elif data_config.audio_sensor_dict['cluster_data'] == 'utterance':
				if os.path.exists(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file)) is True:
					day_utterance_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file), index_col=0)
					utterance_df = utterance_df.append(day_utterance_df)
			
			elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
				if os.path.exists(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file)) is True:
					day_minute_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['filter_path'], participant_id, file), index_col=0)
					minute_df = minute_df.append(day_minute_df)
		
		# process audio feature for cluster
		if data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
			raw_audio_df_norm = (raw_audio_df - raw_audio_df.mean()) / raw_audio_df.std()

		# if cluster utterance
		elif data_config.audio_sensor_dict['cluster_data'] == 'utterance':
			utterance_norm_df = utterance_df.copy()
			utterance_norm_df = (utterance_norm_df - utterance_norm_df.mean()) / utterance_norm_df.std()
			cluster_name = 'utterance_cluster'

			if os.path.exists(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_name + '.csv.gz')) is True:
				cluster_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_name + '.csv.gz'), index_col=0)
				lda_audio(utterance_norm_df, cluster_df, data_config, participant_id, lda_components=data_config.audio_sensor_dict['lda_num'])
				
		# if cluster utterance
		elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
			minute_norm_df = minute_df.copy()
			minute_norm_df = (minute_norm_df - minute_norm_df.mean()) / minute_norm_df.std()
			cluster_name = 'minute_cluster'
		
			if os.path.exists(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_name + '.csv.gz')) is True:
				cluster_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_name + '.csv.gz'), index_col=0)
				lda_audio(minute_norm_df, cluster_df, data_config, participant_id, lda_components=data_config.audio_sensor_dict['lda_num'])


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)

