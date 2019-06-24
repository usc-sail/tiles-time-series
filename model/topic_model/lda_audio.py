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
import seaborn as sns
import statsmodels.api as sm
from scipy.spatial.distance import cdist

from datetime import datetime
import operator

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser


def plot_tp(data_config, participant_id, tp_weight_df, location_and_tp_df, enable_filter=False):
	# Read topic and location data
	topic_method = data_config.audio_sensor_dict['topic_method']
	topic_num = str(data_config.audio_sensor_dict['topic_num'])
	overlap = data_config.audio_sensor_dict['overlap']
	cluster_offset = data_config.audio_sensor_dict['cluster_offset']
	save_prefix = topic_method + '_' + topic_num + '_overlap_' + str(overlap) + '_' + str(cluster_offset)
	data_tp_path = data_config.audio_sensor_dict['tp_path']
	
	# Read relevant data
	tp_list = [str(i) for i in list(tp_weight_df.index)]
	loc_list = [col for col in location_and_tp_df.columns if col not in tp_list and 'key' not in col]
	location_df = location_and_tp_df.loc[:, loc_list]
	topic_df = location_and_tp_df.loc[:, tp_list]
	
	if len(tp_list) > 10:
		top_tp_list = [tp_list[i] for i in np.argsort(np.mean(np.array(topic_df.dropna()), axis=0))[-10:][::-1]]
	else:
		top_tp_list = tp_list
		
	location_norm_array = np.array(location_df) / np.sum(np.array(location_df), axis=1).reshape([len(location_df), 1])
	
	fig, ax = plt.subplots(nrows=1, figsize=(16, 10))
	corr_df = pd.DataFrame(index=[str(i) for i in range(len(top_tp_list))], columns=loc_list)
	corr_df.loc[:, :] = np.array(location_and_tp_df.drop(columns=['key_word']).corr().loc[top_tp_list, loc_list])
	sns.heatmap(corr_df, annot=True, linewidths=.5, ax=ax)
	plt.savefig(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_corr.png'))
	plt.close()
	
	location_and_tp_df.loc[:, loc_list] = location_norm_array
	
	fig1, ax1 = plt.subplots(nrows=2, figsize=(16, 10))
	fig2, ax2 = plt.subplots(nrows=2, figsize=(16, 10))
	
	# filter_array = plt_array.copy()
	for i in range(location_norm_array.shape[1]):
		# plt_array = location_norm_array[:, i]
		plt_array = location_and_tp_df.loc[:, loc_list[i]].fillna(0)
		plt_array = np.array(plt_array)
		
		cycle, tmp_filter_array = sm.tsa.filters.hpfilter(plt_array, 100)
		location_and_tp_df.loc[:, loc_list[i]] = tmp_filter_array
		
		# ax1[0].plot(list(pd.to_datetime(location_df.index)), plt_array, label=loc_list[i])
		ax1[0].plot(list(pd.to_datetime(location_df.index)), tmp_filter_array, label=loc_list[i])
		
		if enable_filter:
			cycle, filter_array = sm.tsa.filters.hpfilter(plt_array, 10)
			ax2[0].plot(list(pd.to_datetime(location_df.index)), filter_array, label=loc_list[i])
	
	plt_array = np.array(location_and_tp_df.loc[:, loc_list].fillna(0)) # .reshape([len(location_and_tp_df), len(loc_list)])
	# ax1[0].stackplot(list(pd.to_datetime(location_df.index)), plt_array)
	# ax1[0].stackplot(list(pd.to_datetime(location_df.index)), plt_array.T, labels=loc_list)
		
	ax1[0].legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=14)
	ax1[0].set_xlim([pd.to_datetime(topic_df.index[0]), pd.to_datetime(topic_df.index[-1])])
	
	ax1[0].legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=14)
	ax1[0].set_xlim([pd.to_datetime(topic_df.index[0]), pd.to_datetime(topic_df.index[-1])])
	
	topic_df = topic_df.fillna(0)
	# for i in range(topic_df.shape[1]):
	
	for i in range(len(top_tp_list)):
		# plt_array = np.array(topic_df)[:, i]
		plt_array = np.array(topic_df.loc[:, top_tp_list[i]])
		
		cycle, tmp_filter_array = sm.tsa.filters.hpfilter(plt_array, 10)
		topic_df.loc[:, top_tp_list[i]] = tmp_filter_array
		
		# ax1[1].plot(list(pd.to_datetime(topic_df.index)), plt_array, label='topic: ' + str(top_tp_list[i]))
		ax1[1].plot(list(pd.to_datetime(topic_df.index)), tmp_filter_array, label='topic: ' + str(top_tp_list[i]))
		
		if enable_filter:
			cycle, filter_array = sm.tsa.filters.hpfilter(plt_array, 10)
			ax2[1].plot(list(pd.to_datetime(topic_df.index)), filter_array, label='topic: ' + str(top_tp_list[i]))
	
	plt_array = np.array(topic_df.loc[:, top_tp_list])
	# ax1[1].stackplot(list(pd.to_datetime(location_df.index)), plt_array.T, labels=top_tp_list)
		
	ax1[1].legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=14)
	ax1[1].set_xlim([pd.to_datetime(topic_df.index[0]), pd.to_datetime(topic_df.index[-1])])
	fig1.savefig(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_and_location.png'))
	plt.close(fig=fig1)
	# data_perc = data.divide(data.sum(axis=1), axis=0)
	
	if enable_filter:
		fig2.savefig(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_and_location_filter.png'))
		plt.close(fig=fig2)
	
	X = np.array(location_and_tp_df.loc[:, tp_list])
	dist = cdist(X, X, metric='jensenshannon')
	K = np.exp(-dist)
	
	from sklearn.cluster import AffinityPropagation
	# clustering_topics = AffinityPropagation().fit(np.array(topic_df.loc[:, top_tp_list])).predict(np.array(topic_df.loc[:, top_tp_list]))
	# clustering_topics = AffinityPropagation(affinity='precomputed').fit(K).predict(K)
	
	plt.imshow(K)
	plt.show()


def get_nmf_topics(model, n_top_words, num_topics, vectorizer):
	# the word ids obtained need to be reverse-mapped to the words so we can print the topic names.
	feat_names = vectorizer.get_feature_names()
	
	word_dict = {}
	for i in range(num_topics):
		# for each topic, obtain the largest values, and add the words they map to into the dictionary.
		words_ids = model.components_[i].argsort()[:-20 - 1:-1]
		words = [feat_names[key] for key in words_ids]
		word_dict[str(i)] = words
	
	return pd.DataFrame(word_dict)


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
	process_col_list = ['F0_sma_cluster',
						'pcm_loudness_sma_cluster',
						'duration_cluster',
						# 'logHNR_sma_cluster',
						# 'audspecRasta_lengthL1norm_sma_cluster',
						# 'pcm_fftMag_spectralCentroid_sma_cluster',
						# 'pcm_fftMag_spectralEntropy_sma_cluster'
						]
						# 'audspecRasta_lengthL1norm_sma_cluster']
	'''
	process_col_list = ['F0_sma_cluster',
						'pcm_loudness_sma_cluster',
						'duration_cluster',
						'logHNR_sma_cluster',
						# 'audspecRasta_lengthL1norm_sma_cluster',
						# 'pcm_fftMag_spectralCentroid_sma_cluster',
						# 'pcm_fftMag_spectralEntropy_sma_cluster'
						]
	'''
	
	# process_col_list = ['F0final_sma_cluster', 'duration_cluster', 'spectral_cluster']
	# pcm_loudness_sma_cluster 'audspecRasta_lengthL1norm_sma_cluster' 'audspec_lengthL1norm_sma_cluster'
	
	for idx, participant_id in enumerate(top_participant_id_list[:5]):
		
		if participant_id not in list(igtb_df.ParticipantID):
			continue
			
		participant_df = igtb_df.loc[igtb_df['ParticipantID'] == participant_id]
		if len(participant_df.Shift) != 1:
			continue
			
		# 5c7c51c5-8bd6-4997-92c1-ce4c5eda45e6
		shift = participant_df.Shift[0]
		
		# Initialize start parameters
		cluster_offset = data_config.audio_sensor_dict['cluster_offset']
		# time_offest = (12 * 3600 - 60 * int(overlap)) / (60 * int(cluster_offset))
		time_offest = (10 * 3600) / (60 * int(cluster_offset))
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

		if len(audio_data_df) < 1500 or len(time_start_end_list) < 15:
			continue
			
		select_array = np.random.choice(len(time_start_end_list), 15)
		# Iterate
		# for select_index in select_array:
		
		word_count_dict = {}
		for time_start_end in time_start_end_list[:]:
			# time_start_end = time_start_end_list[select_index]
			
			# Extract start and end time of a shift
			start_time, end_time = pd.to_datetime(time_start_end[0]), pd.to_datetime(time_start_end[1])
			if shift == 'Day shift':
				start_time = start_time.replace(hour=work_start_time, minute=0, second=0, microsecond=0).strftime(load_data_basic.date_time_format)[:-3]
			else:
				start_time = (start_time - timedelta(hours=12)).replace(hour=work_start_time, minute=0, second=0, microsecond=0).strftime(load_data_basic.date_time_format)[:-3]
			
			# end_time = (pd.to_datetime(start_time) + timedelta(hours=10) - timedelta(minutes=int(overlap))).strftime(load_data_basic.date_time_format)[:-3]
			end_time = (pd.to_datetime(start_time) + timedelta(hours=10)).strftime(load_data_basic.date_time_format)[:-3]
		
			owl_in_one_day_df = owl_in_one_df[start_time:end_time]
			if len(owl_in_one_day_df) == 0:
				continue
			if len(owl_in_one_day_df.loc[owl_in_one_day_df['unknown'] == 1]) / len(owl_in_one_day_df) > 0.5:
				continue
				
			day_audio_df = audio_data_df[start_time:end_time]
			for tmp_data_index, row_series in day_audio_df[process_col_list].iterrows():
				word = ''
				for row_series_index, value in row_series.iteritems():
					word += chr(97 + value)
				if word not in list(word_count_dict.keys()):
					word_count_dict[word] = 1
				else:
					word_count_dict[word] += 1
			
			for offset in range(int(time_offest)):
				
				if overlap == 'False':
					minute_offset = int(cluster_offset)
				elif overlap == 'True':
					minute_offset = 2 * int(cluster_offset)
				else:
					minute_offset = int(overlap)
				
				tmp_start = (pd.to_datetime(start_time) + timedelta(minutes=int(cluster_offset)*offset-int(minute_offset/2))).strftime(load_data_basic.date_time_format)[:-3]
				tmp_end = (pd.to_datetime(start_time) + timedelta(minutes=int(cluster_offset)*offset+int(minute_offset/2))).strftime(load_data_basic.date_time_format)[:-3]
				
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
		
		sorted_word_count = sorted(word_count_dict.items(), key=operator.itemgetter(1))[::-1]
		valid_word_list = []
		
		'''
		for i in range(len(sorted_word_count)):
			word = sorted_word_count[i][0]
			# if 50 < sorted_word_count[i][1] < 500:
			if 10 < sorted_word_count[i][1]:
				valid_word_list.append(word)
		'''
		for i in range(10):
			word = sorted_word_count[i][0]
			valid_word_list.append(word)
		
		valid_point_list = list(np.where(np.nansum(np.array(location_df), axis=1) > (int(data_config.audio_sensor_dict['overlap']) * 3))[0])
		word_list = []
		for valid_point in valid_point_list:
			time_index = time_index_list[int(valid_point)]
			# for time_index in time_index_list:
			valid_word_at_time_list = [word for word in time_word_dict[time_index] if word in valid_word_list]
			word_list.append(valid_word_at_time_list)
			time_word_dict[time_index] = valid_word_at_time_list
		word_dictionary = Dictionary(word_list)
		word_corpus = [word_dictionary.doc2bow(text) for text in word_list]
		
		train_sentences = [' '.join(text) for text in word_list]
		vectorizer = CountVectorizer(analyzer='word', max_features=5000)
		x_counts = vectorizer.fit_transform(train_sentences)
		transformer = TfidfTransformer(smooth_idf=False)
		x_tfidf = transformer.fit_transform(x_counts)
		xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
		
		vectorizer = TfidfVectorizer(min_df=20, max_df=1000)
		xtfidf_norm = vectorizer.fit_transform(train_sentences)
		
		if len(word_corpus) == 0:
			continue

		if data_config.audio_sensor_dict['topic_method'] == 'lda':
			model = LdaModel(corpus=word_corpus, id2word=word_dictionary,
						     num_topics=int(data_config.audio_sensor_dict['topic_num']), update_every=1, passes=1)
		elif data_config.audio_sensor_dict['topic_method'] == 'nmf':
			model = NMF(n_components=int(data_config.audio_sensor_dict['topic_num']), init='nndsvd')
			# fit the model
			model.fit(xtfidf_norm)
			y = model.fit_transform(xtfidf_norm)
			
		else:
			# model = HdpModel(word_corpus, word_dictionary, T=int(data_config.audio_sensor_dict['topic_num']))
			model = HdpModel(word_corpus, word_dictionary, T=int(data_config.audio_sensor_dict['topic_num']), alpha=0.1, gamma=0.1)
		
		# Get main topic in each document
		topic_list = []
		# for time_index in time_index_list:
		if data_config.audio_sensor_dict['topic_method'] == 'nmf':
			key_word_df = get_nmf_topics(model, 3, int(data_config.audio_sensor_dict['topic_num']), vectorizer)
			for valid_point in valid_point_list:
				time_index = time_index_list[int(valid_point)]
				
				model_list = model.transform(vectorizer.transform([' '.join(time_word_dict[time_index])]))
				# row = model_list[0] if model.per_word_topics else model_list
				for topic, weight in enumerate(list(model_list)[0]):
					location_df.loc[time_index, str(topic)] = weight
				
				topic_list.append(str(np.argmax(model_list[0])))
				
				topic_keywords = ", ".join(list(key_word_df[str(np.argmax(model_list[0]))]))
				location_df.loc[time_index, 'key_word'] = topic_keywords
		else:
			for valid_point in valid_point_list:
				time_index = time_index_list[int(valid_point)]
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
		valie_time_index_list = [time_index_list[point] for point in valid_point_list]
		valid_rows_df = location_df.loc[valie_time_index_list, :].fillna(0)
		location_df.loc[valie_time_index_list, list(valid_rows_df.columns)] = valid_rows_df.loc[valie_time_index_list, list(valid_rows_df.columns)]
		# location_df.loc[:, ] = location_df.loc[time_index_list[valid_point_list], :]

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
		
		topic_method = data_config.audio_sensor_dict['topic_method']
		topic_num = str(data_config.audio_sensor_dict['topic_num'])
		save_prefix = topic_method + '_' + topic_num + '_overlap_' + str(overlap) + '_' + str(cluster_offset)
		
		if data_config.audio_sensor_dict['topic_method'] != 'nmf':
			topic_weight_df = pd.DataFrame()

			for topic in topic_list:
				row_df = pd.DataFrame(index=[str(topic)])
				topic_tuple_list = model.show_topic(int(topic))
				for word_tuple in topic_tuple_list:
					row_df[str(word_tuple[0])] = word_tuple[1]
				topic_weight_df = topic_weight_df.append(row_df)
			topic_weight_df.to_csv(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_weight.csv.gz'), compression='gzip')
		
		else:
			topic_weight_df = key_word_df.transpose()
		'''
		for index, topic_tuple_list in model.show_topics(formatted=False):
			row_df = pd.DataFrame(index=[str(index)])
			for word_tuple in topic_tuple_list:
				row_df[str(word_tuple[0])] = word_tuple[1]
			topic_weight_df = topic_weight_df.append(row_df)
		'''
		
		if os.path.exists(os.path.join(data_tp_path, participant_id)) is False:
			os.mkdir(os.path.join(data_tp_path, participant_id))
		
		if data_config.audio_sensor_dict['topic_method'] != 'nmf':
			model.save(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_model'))
		location_df.to_csv(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_and_location.csv.gz'), compression='gzip')
		
		plot_tp(data_config, participant_id, topic_weight_df, location_df, enable_filter=False)


if __name__ == '__main__':
	import bayesian_hmm
	
	# create emission sequences
	base_sequence = list(range(5)) + list(range(5, 0, -1))
	sequences = [base_sequence * 20 for _ in range(50)]
	
	# initialise object with overestimate of true number of latent states
	hmm = bayesian_hmm.HDPHMM(sequences, sticky=False)
	hmm.initialise(k=20)
	
	# estimate parameters, making use of multithreading functionality
	results = hmm.mcmc(n=500, burn_in=100)
	
	# print final probability estimates (expect 10 latent states)
	hmm.print_probabilities()

	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
