"""
Cluster the audio data
"""
from __future__ import print_function

from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel, LdaModel
from gensim.corpora import Dictionary

from sklearn import mixture
import os
import sys
import pandas as pd
import numpy as np
import random
from collections import Counter
from datetime import timedelta
import pymc3 as pm
from theano import tensor as tt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser


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

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	data_cluster_path = data_config.audio_sensor_dict['clustering_path']

	final_corr_df = pd.DataFrame()

	for idx, participant_id in enumerate(top_participant_id_list):

		print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		participant_df = igtb_df.loc[igtb_df['ParticipantID'] == participant_id]

		lda_components = data_config.audio_sensor_dict['lda_num']
		cluster_file = 'lda_' + str(lda_components) + '_' + data_config.audio_sensor_dict['cluster_data'] + '_cluster.csv.gz'
		# Read other sensor data, the aim is to detect whether people workes during a day
		if os.path.exists(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_file)) is False:
			continue

		primary_unit = participant_df.PrimaryUnit[0]
		current_position = 'nurse' if participant_df.currentposition[0] == 1 or participant_df.currentposition[0] == 2 else 'non-nurse'
		shift = participant_df.Shift[0]
		age = participant_df.age[0]
		language = 'english' if participant_df.language[0] == 1 else 'non-english'
		supervise = 'supervise' if participant_df.supervise[0] == 1 else 'non-supervise'

		participant_corr_df = pd.DataFrame(index=[participant_df.index[0]])

		participant_corr_df['current_position'] = current_position
		participant_corr_df['shift'] = shift
		participant_corr_df['age'] = age
		participant_corr_df['language'] = language
		participant_corr_df['primary_unit'] = primary_unit
		participant_corr_df['supervise'] = supervise
		participant_corr_df['ope_igtb'] = participant_df.ope_igtb[0]

		save_prefix = data_config.audio_sensor_dict['final_save_prefix']

		if data_config.audio_sensor_dict['overlap'] == 'False':
			corr_file_name = os.path.join(data_cluster_path, participant_id, save_prefix + '_offset_false_corr.csv.gz')
			sent_file_name = os.path.join(data_cluster_path, participant_id, save_prefix + '_offset_false_sent.csv.gz')
		else:
			corr_file_name = os.path.join(data_cluster_path, participant_id, save_prefix + '_corr.csv.gz')
			sent_file_name = os.path.join(data_cluster_path, participant_id, save_prefix + '_sent.csv.gz')

		if os.path.exists(corr_file_name) is True:
			topic_corr_df = pd.read_csv(corr_file_name, index_col=0)
			sent_topics_df = pd.read_csv(sent_file_name, index_col=0)
			# topic_corr_df = topic_corr_df.dropna(axis=0)

			index_list = [str(i) for i in range(int(data_config.audio_sensor_dict['topic_num']))]
			col_list = [col for col in topic_corr_df.columns if col not in index_list]
			topic_corr_df = topic_corr_df.loc[index_list, col_list]

			# sort_df = topic_corr_df.abs().unstack().sort_values(kind="quicksort")
			sort_df = topic_corr_df.unstack().sort_values(kind="quicksort")
			sort_df = sort_df.dropna()

			for i, index_tuple in enumerate(list(sort_df.index)[-10:][::-1]):
				participant_corr_df['rank_' + str(i)] = str(index_tuple[0]) + '/' + str(index_tuple[1]) + ':' + str(round(topic_corr_df.loc[str(index_tuple[1]), str(index_tuple[0])], 3))

			for col in col_list:
				# participant_corr_df[col] = int(len(np.where(np.array(topic_corr_df.loc[:, col]) > 0.2)[0]))
				# participant_corr_df[col] += int(len(np.where(np.array(topic_corr_df.loc[:, col]) < -0.2)[0]))
				threshold = 0.15
				cond1 = int(len(np.where(np.array(topic_corr_df.loc[:, col]) > threshold)[0])) > 0
				cond2 = int(len(np.where(np.array(topic_corr_df.loc[:, col]) < -threshold)[0])) > 0
				participant_corr_df[col] = 1 if cond1 or cond2 else 0

			final_corr_df = final_corr_df.append(participant_corr_df)
			'''
			for index in index_list:
				for col in col_list:
					participant_corr_df[index + '/' + col] = topic_corr_df.loc[index, col]
			'''
	nurse_df = final_corr_df.loc[final_corr_df['current_position'] == 'nurse']
	print()


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
