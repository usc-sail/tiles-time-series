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
import matplotlib.pyplot as plt

from datetime import datetime

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
	data_tp_path = data_config.audio_sensor_dict['tp_path']
	
	# process cluster name
	if data_config.audio_sensor_dict['cluster_data'] == 'utterance':
		cluster_name = 'utterance_cluster'
	elif data_config.audio_sensor_dict['cluster_data'] == 'minute':
		cluster_name = 'minute_cluster'
	elif data_config.audio_sensor_dict['cluster_data'] == 'snippet':
		cluster_name = 'snippet_cluster'
	elif data_config.audio_sensor_dict['cluster_data'] == 'raw_audio':
		cluster_name = 'raw_audio_cluster'
	else:
		cluster_name = 'snippet_cluster'
	
	overlap = data_config.audio_sensor_dict['overlap']
	
	# 'pcm_fftMag_spectralCentroid_sma_cluster'
	process_col_list = ['F0_sma_cluster', 'duration_cluster', # 'pcm_loudness_sma_cluster',
						'pcm_fftMag_spectralCentroid_sma_cluster', 'pcm_fftMag_spectralEntropy_sma_cluster',
						'audspecRasta_lengthL1norm_sma_cluster']
	# process_col_list = ['F0final_sma_cluster', 'duration_cluster', 'spectral_cluster']
	# pcm_loudness_sma_cluster 'audspecRasta_lengthL1norm_sma_cluster' 'audspec_lengthL1norm_sma_cluster'
	
	for idx, participant_id in enumerate(top_participant_id_list[:3]):
		
		if participant_id not in list(igtb_df.ParticipantID):
			continue
			
		participant_df = igtb_df.loc[igtb_df['ParticipantID'] == participant_id]
		if len(participant_df.Shift) != 1:
			continue
			
		# 5c7c51c5-8bd6-4997-92c1-ce4c5eda45e6
		shift = participant_df.Shift[0]
		
		# Initialize start parameters
		cluster_offset = data_config.audio_sensor_dict['cluster_offset']
		time_offest = (12 * 3600 - 60 * int(overlap)) / (60 * int(cluster_offset))
		if shift == 'Day shift':
			work_start_time, work_end_time = 7, 19
		else:
			work_start_time, work_end_time = 19, 7
		
		start = datetime(2018, 1, 1, hour=work_start_time, minute=0, second=0, microsecond=0)
		
		time_index_list = [(start + timedelta(minutes=i*int(cluster_offset))).strftime(load_data_basic.date_time_format)[:-3] for i in range(int(time_offest))]
			
		print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
		
		# Read other sensor data, the aim is to detect whether people workes during a day
		if os.path.exists(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_name + '_subspace.csv.gz')) is False:
			continue

		# Read audio data and owl-in-one data
		audio_data_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, cluster_name + '_subspace.csv.gz'), index_col=0)
		audio_data_df = audio_data_df.sort_index()

		owl_in_one_df = pd.read_csv(os.path.join(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id + '.csv.gz'), index_col=0)
		
		# Read time difference
		time_diff = pd.to_datetime(list(audio_data_df.index)[1:]) - pd.to_datetime(list(audio_data_df.index)[:-1])
		time_diff = list(time_diff.total_seconds())

		change_point_start_list = [0]
		change_point_end_list = list(np.where(np.array(time_diff) > 3600 * 6)[0])

		[change_point_start_list.append(change_point_end + 1) for change_point_end in change_point_end_list]
		change_point_end_list.append(len(audio_data_df.index) - 1)

		time_start_end_list = []
		for i, change_point_end in enumerate(change_point_end_list):
			time_start_end_list.append([list(audio_data_df.index)[change_point_start_list[i]], list(audio_data_df.index)[change_point_end]])
		
		# Init data list
		time_word_dict = {}
		for time_index in time_index_list:
			time_word_dict[time_index] = []
		
		owl_in_one_col_list = []
		for col in list(owl_in_one_df.columns):
			if col == 'other_floor' or col == 'unknown' or col == 'floor2':
				owl_in_one_col_list.append('other')
			else:
				owl_in_one_col_list.append(col)
		owl_in_one_col_list = list(set(owl_in_one_col_list))
		
		location_df = pd.DataFrame(np.zeros([len(time_index_list), len(owl_in_one_col_list)]),
								   index=time_index_list, columns=owl_in_one_col_list)

		if len(audio_data_df) < 1500 or len(time_start_end_list) < 10:
			continue
		# Iterate
		for time_start_end in time_start_end_list[:]:
			
			# Extract start and end time of a shift
			start_time, end_time = pd.to_datetime(time_start_end[0]), pd.to_datetime(time_start_end[1])
			if shift == 'Day shift':
				start_time = start_time.replace(hour=work_start_time, minute=0, second=0, microsecond=0).strftime(load_data_basic.date_time_format)[:-3]
			else:
				start_time = (start_time - timedelta(hours=12)).replace(hour=work_start_time, minute=0, second=0, microsecond=0).strftime(load_data_basic.date_time_format)[:-3]
			
			end_time = (pd.to_datetime(start_time) + timedelta(hours=12) - timedelta(minutes=int(overlap))).strftime(load_data_basic.date_time_format)[:-3]
			
			owl_in_one_day_df = owl_in_one_df[start_time:end_time]
			if len(owl_in_one_day_df) == 0:
				continue
			if len(owl_in_one_day_df.loc[owl_in_one_day_df['unknown'] == 1]) / len(owl_in_one_day_df) > 0.5:
				continue
			
			for offset in range(int(time_offest)):
				tmp_start = (pd.to_datetime(start_time) + timedelta(minutes=int(cluster_offset) * offset)).strftime(load_data_basic.date_time_format)[:-3]

				if overlap == 'False':
					minute_offset = int(cluster_offset)
				elif overlap == 'True':
					minute_offset = 2 * int(cluster_offset)
				else:
					minute_offset = int(overlap)
				tmp_end = (pd.to_datetime(start_time) + timedelta(minutes=int(cluster_offset)*offset+minute_offset)).strftime(load_data_basic.date_time_format)[:-3]
				
				tmp_data_df = audio_data_df[tmp_start:tmp_end]
				tmp_owl_in_one_df = owl_in_one_df[tmp_start:tmp_end]
				
				time_index = time_index_list[offset]
				if len(tmp_owl_in_one_df) > minute_offset / 2:
					for col in list(tmp_owl_in_one_df.columns):
						if np.nansum(np.array(tmp_owl_in_one_df[col])) > 0:
							if col == 'other_floor' or col == 'unknown' or col == 'floor2':
								location_df.loc[time_index, 'other'] += np.nansum(np.array(tmp_owl_in_one_df[col]))
							else:
								location_df.loc[time_index, col] += np.nansum(np.array(tmp_owl_in_one_df[col]))
				if len(tmp_data_df) > 2:
					for tmp_data_index, row_series in tmp_data_df[process_col_list].iterrows():
						word = ''
						for row_series_index, value in row_series.iteritems():
							word += chr(97+value)
						
						time_word_dict[time_index].append(word)
	
		word_list = []
		for time_index in time_index_list:
			word_list.append(time_word_dict[time_index])
		word_dictionary = Dictionary(word_list)
		word_corpus = [word_dictionary.doc2bow(text) for text in word_list]
		
		if len(word_corpus) == 0:
			continue

		if data_config.audio_sensor_dict['topic_method'] == 'lda':
			model = LdaModel(corpus=word_corpus, id2word=word_dictionary,
						     num_topics=int(data_config.audio_sensor_dict['topic_num']), update_every=1, passes=1)
		else:
			# model = HdpModel(word_corpus, word_dictionary, T=int(data_config.audio_sensor_dict['topic_num']))
			model = HdpModel(word_corpus, word_dictionary, T=int(data_config.audio_sensor_dict['topic_num']))
		
		# Get main topic in each document
		topic_list = []
		for time_index in time_index_list:
			if len(time_word_dict[time_index]) > 0:
				model_list = model[word_dictionary.doc2bow(time_word_dict[time_index])]
				# row = model_list[0] if model.per_word_topics else model_list
				row = model_list
				for topic in model_list:
					if topic[1] > 0:
						location_df.loc[time_index, str(topic[0])] = topic[1]
						topic_list.append(str(topic[0]))
				
				row = sorted(row, key=lambda x: (x[1]), reverse=True)
				# Get the Dominant topic, Perc Contribution and Keywords for each document
				for j, (topic_num, prop_topic) in enumerate(row):
					if j == 0:  # => dominant topic
						wp = model.show_topic(topic_num)
						topic_keywords = ", ".join([word for word, prop in wp])
						location_df.loc[time_index, 'key_word'] = topic_keywords

		# Topic
		topic_list = list(set(topic_list))
		# Fill na
		location_df = location_df.fillna(0)

		'''
			from sklearn.cluster import AffinityPropagation
			clustering_topics = AffinityPropagation().fit(np.array(location_df.loc[:, topic_list])).predict(np.array(location_df.loc[:, topic_list]))
			location_df.loc[:, 'topic_cluster'] = clustering_topics
	
			from scipy.spatial.distance import cdist
	
			dist = cdist(np.array(location_df.loc[:, topic_list]), np.array(location_df.loc[:, topic_list]), metric='euclidean')
			K = np.exp(dist)
	
			fig, ax = plt.subplots(figsize=(20, 20))
			cax = ax.matshow(K, interpolation='nearest')
			ax.grid(True)
			fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
			plt.show()
		'''

		topic_weight_df = pd.DataFrame()

		for topic in topic_list:
			row_df = pd.DataFrame(index=[str(topic)])
			topic_tuple_list = model.show_topic(int(topic))
			for word_tuple in topic_tuple_list:
				row_df[str(word_tuple[0])] = word_tuple[1]
			topic_weight_df = topic_weight_df.append(row_df)
		'''
		for index, topic_tuple_list in model.show_topics(formatted=False):
			row_df = pd.DataFrame(index=[str(index)])
			for word_tuple in topic_tuple_list:
				row_df[str(word_tuple[0])] = word_tuple[1]
			topic_weight_df = topic_weight_df.append(row_df)
		'''

		topic_method = data_config.audio_sensor_dict['topic_method']
		topic_num = str(data_config.audio_sensor_dict['topic_num'])
		save_prefix = topic_method + '_' + topic_num + '_overlap_' + str(overlap) + '_' + str(cluster_offset)

		if os.path.exists(os.path.join(data_tp_path, participant_id)) is False:
			os.mkdir(os.path.join(data_tp_path, participant_id))

		from gensim.test.utils import datapath
		# temp_file = datapath("model")
		model.save(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_model'))
		topic_weight_df.to_csv(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_weight.csv.gz'), compression='gzip')
		location_df.to_csv(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_and_location.csv.gz'), compression='gzip')
		

if __name__ == '__main__':

	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
