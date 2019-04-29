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
	
	# if cluster utterance
	if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
		cluster_name = 'utterance_cluster'
	elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
		cluster_name = 'minute_cluster'
	# process audio feature for cluster
	elif data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
		cluster_name = 'raw_audio_cluster'
	else:
		cluster_name = 'raw_audio_cluster'
	
	data_cluster_path = data_config.audio_sensor_dict['clustering_path']

	for idx, participant_id in enumerate(top_participant_id_list):

		print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
		
		lda_components = data_config.audio_sensor_dict['lda_num']
		cluster_file = data_config.audio_sensor_dict['lda_clustering_path']
		# Read other sensor data, the aim is to detect whether people workes during a day
		if os.path.exists(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_file)) is False:
			continue

		data_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_file), index_col=0)
		data_df = data_df.sort_index()
		
		if len(data_df) < 720:
			continue

		owl_in_one_df = pd.read_csv(os.path.join(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id + '.csv.gz'), index_col=0)

		time_diff = pd.to_datetime(list(data_df.index)[1:]) - pd.to_datetime(list(data_df.index)[:-1])
		time_diff = list(time_diff.total_seconds())

		change_point_start_list = [0]
		change_point_end_list = list(np.where(np.array(time_diff) > 3600 * 6)[0])

		[change_point_start_list.append(change_point_end + 1) for change_point_end in change_point_end_list]
		change_point_end_list.append(len(data_df.index) - 1)

		time_start_end_list = []
		for i, change_point_end in enumerate(change_point_end_list):
			time_start_end_list.append([list(data_df.index)[change_point_start_list[i]], list(data_df.index)[change_point_end]])

		bow_list, time_list = [], []
		word_list = []
		
		topic_df = pd.DataFrame()
		for time_start_end in time_start_end_list:
			start_time = (pd.to_datetime(time_start_end[0]).replace(minute=0, second=0, microsecond=0)).strftime(load_data_basic.date_time_format)[:-3]
			end_time = ((pd.to_datetime(time_start_end[1]) + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)).strftime(load_data_basic.date_time_format)[:-3]
			
			cluster_offset = data_config.audio_sensor_dict['cluster_offset']
			time_offest = int((pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds() / (60 * int(cluster_offset)))

			for offset in range(time_offest):
				tmp_start = (pd.to_datetime(start_time) + timedelta(minutes=int(cluster_offset) * offset)).strftime(load_data_basic.date_time_format)[:-3]
				tmp_end = (pd.to_datetime(start_time) + timedelta(minutes=int(cluster_offset) * offset + 2 * int(cluster_offset))).strftime(load_data_basic.date_time_format)[:-3]

				tmp_data_df = data_df[tmp_start:tmp_end]
				tmp_owl_in_one_df = owl_in_one_df[tmp_start:tmp_end]
				
				row_df = pd.DataFrame(index=[tmp_start])
				row_df['other'] = 0
				
				sum = 0
				for col in list(tmp_owl_in_one_df.columns):
					if col == 'other_floor' or col == 'unknown' or col == 'floor2':
						row_df['other'] += np.sum(np.array(tmp_owl_in_one_df[col])) / len(tmp_owl_in_one_df)
					else:
						row_df[col] = np.sum(np.array(tmp_owl_in_one_df[col])) / len(tmp_owl_in_one_df)
					sum += np.sum(np.array(tmp_owl_in_one_df[col])) / len(tmp_owl_in_one_df)
				
				if sum > 0.1 and len(tmp_owl_in_one_df) > 10 and len(list(tmp_data_df.cluster)) > 0:
					topic_df = topic_df.append(row_df)
					
					segment_list = []
					[segment_list.append(str(word)) for word in list(tmp_data_df.cluster)]
					# [segment_list.append(str(999)) for i in range(20-len(list(tmp_data_df.cluster)))]
					word_list.append(segment_list)

		word_dictionary = Dictionary(word_list)
		word_corpus = [word_dictionary.doc2bow(text) for text in word_list]
		
		if len(word_corpus) == 0:
			continue

		if data_config.audio_sensor_dict['topic_method'] == 'lda':
			model = LdaModel(corpus=word_corpus, id2word=word_dictionary,
						     num_topics=int(data_config.audio_sensor_dict['topic_num']),
							 update_every=1, passes=1)
		else:
			model = HdpModel(word_corpus, word_dictionary, T=int(data_config.audio_sensor_dict['topic_num']))
		
		topic_final_df = pd.DataFrame()
		
		for index, topic_row_series in topic_df.iterrows():
			start_time = index
			end_time = (pd.to_datetime(index) + timedelta(minutes=10)).strftime(load_data_basic.date_time_format)[:-3]
			tmp_data_df = data_df[start_time:end_time]
			
			sent = word_dictionary.doc2bow([str(word) for word in list(tmp_data_df.cluster)])
			topics = model[sent]
			
			tmp_owl_in_one_df = owl_in_one_df[start_time:end_time]
			row_df = pd.DataFrame(index=[start_time])
			row_df['other'] = 0
			
			sum = 0
			for col in list(tmp_owl_in_one_df.columns):
				if col == 'other_floor' or col == 'unknown' or col == 'floor2':
					row_df['other'] += np.sum(np.array(tmp_owl_in_one_df[col])) / len(tmp_owl_in_one_df)
				else:
					row_df[col] = np.sum(np.array(tmp_owl_in_one_df[col])) / len(tmp_owl_in_one_df)
				sum += np.sum(np.array(tmp_owl_in_one_df[col])) / len(tmp_owl_in_one_df)
			
			top_list = []
			for topic in topics:
				row_df[str(topic[0])] = topic[1]
				top_list.append(topic[1])
			topic_sum = np.nansum(top_list)
			if topic_sum > 0.5:
				topic_final_df = topic_final_df.append(row_df)
		
		topic_final_df = topic_final_df.fillna(0)
		# topic_corr_df = topic_final_df.corr(method='spearman')
		topic_corr_df = topic_final_df.corr()
		
		save_prefix = data_config.audio_sensor_dict['final_save_prefix']
		topic_corr_df.to_csv(os.path.join(data_cluster_path, participant_id, save_prefix + '_corr.csv.gz'), compression='gzip')
		
		topic_weight_df = pd.DataFrame()
		for index, topic_tuple_list in model.show_topics(formatted=False):
			row_df = pd.DataFrame(index=[str(index)])
			for word_tuple in topic_tuple_list:
				row_df[str(word_tuple[0])] = word_tuple[1]
			topic_weight_df = topic_weight_df.append(row_df)
		
		topic_weight_df.to_csv(os.path.join(data_cluster_path, participant_id, save_prefix + '_topic_weight.csv.gz'), compression='gzip')
		

if __name__ == '__main__':

	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
