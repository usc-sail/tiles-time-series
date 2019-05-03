"""
Cluster the audio data
"""
from __future__ import print_function

from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel, LdaModel
from gensim.corpora import Dictionary
import scipy.signal

import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import matplotlib.pyplot as plt

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

		# Read basic information
		participant_df = igtb_df.loc[igtb_df['ParticipantID'] == participant_id]
		# primary_unit = participant_df.PrimaryUnit[0]
		# current_position = 'nurse' if participant_df.currentposition[0] == 1 or participant_df.currentposition[0] == 2 else 'non-nurse'
		# shift = participant_df.Shift[0]
		# age = participant_df.age[0]
		# language = 'english' if participant_df.language[0] == 1 else 'non-english'
		# supervise = 'supervise' if participant_df.supervise[0] == 1 else 'non-supervise'

		'''
		participant_corr_df = pd.DataFrame(index=[participant_df.index[0]])

		participant_corr_df['current_position'] = current_position
		participant_corr_df['shift'] = shift
		participant_corr_df['age'] = age
		participant_corr_df['language'] = language
		participant_corr_df['primary_unit'] = primary_unit
		participant_corr_df['supervise'] = supervise
		participant_corr_df['ope_igtb'] = participant_df.ope_igtb[0]
		'''
		# Read topic and location data
		topic_method = data_config.audio_sensor_dict['topic_method']
		topic_num = str(data_config.audio_sensor_dict['topic_num'])
		overlap = data_config.audio_sensor_dict['overlap']
		cluster_offset = data_config.audio_sensor_dict['cluster_offset']
		save_prefix = topic_method + '_' + topic_num + '_overlap_' + str(overlap) + '_' + str(cluster_offset)

		data_tp_path = data_config.audio_sensor_dict['tp_path']
		tp_weight_file_name = os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_weight.csv.gz')
		location_and_tp_name = os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_and_location.csv.gz')

		if os.path.exists(tp_weight_file_name) is True:
			tp_weight_df = pd.read_csv(tp_weight_file_name, index_col=0)
			location_and_tp_df = pd.read_csv(location_and_tp_name, index_col=0)

			tp_list = [str(i) for i in list(tp_weight_df.index)]
			loc_list = [col for col in location_and_tp_df.columns if col not in tp_list and 'key' not in col]
			location_df = location_and_tp_df.loc[:, loc_list]
			topic_df = location_and_tp_df.loc[:, tp_list]

			location_norm_array = np.array(location_df) / np.sum(np.array(location_df), axis=1).reshape([len(location_df), 1])

			corr_df = pd.DataFrame(index=[str(i) for i in range(len(tp_list))], columns=loc_list)
			corr_df.loc[:, :] = np.array(location_and_tp_df.drop(columns=['key_word']).corr().loc[tp_list, loc_list])
			sns.heatmap(corr_df, annot=True, linewidths=.5)
			plt.savefig(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_corr.png'))
			# plt.imshow(location_and_tp_df.drop(columns=['key_word']).corr().loc[tp_list, loc_list], cmap='hot', interpolation='nearest')
			plt.close()
			# plt.show()
			location_and_tp_df.loc[:, loc_list] = location_norm_array

			# location_norm_array = (location_norm_array - np.nanmin(location_norm_array, axis=0)) / (np.nanmax(location_norm_array, axis=0) - np.nanmin(location_norm_array, axis=0))

			fig, ax = plt.subplots(nrows=2, figsize=(16, 10))

			from sklearn.cluster import AffinityPropagation
			# clustering_topics = AffinityPropagation().fit(location_norm_array).predict(location_norm_array)
			# dpgmm = mixture.BayesianGaussianMixture(n_components=20, max_iter=500, covariance_type='full').fit(location_norm_array)
			# cluster_id = dpgmm.predict(location_norm_array)

			for i in range(location_norm_array.shape[1]):
				plt_array = location_norm_array[:, i]
				# plt_array = scipy.signal.savgol_filter(plt_array, 5, 3)
				# ax[0].plot(list(pd.to_datetime(location_df.index)), location_norm_array[:, i], label=loc_list[i])
				ax[0].plot(list(pd.to_datetime(location_df.index)), plt_array, label=loc_list[i])
			ax[0].legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=14)

			for i in range(topic_df.shape[1]):
				# ax[1].plot(list(pd.to_datetime(location_df.index)), np.array(topic_df)[:, i], label=tp_list[i])
				plt_array = np.array(topic_df)[:, i]
				# plt_array = scipy.signal.savgol_filter(plt_array, 5, 3)
				# ax[1].plot(list(pd.to_datetime(location_df.index)), np.array(topic_df)[:, i], label='topic: ' + str(i))
				ax[1].plot(list(pd.to_datetime(location_df.index)), plt_array, label='topic: ' + str(i))
			ax[1].legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=14)

			# ax[2].plot(list(pd.to_datetime(location_df.index)), list(cluster_id))

			plt.savefig(os.path.join(data_tp_path, participant_id, save_prefix + '_offset_subspace_topic_and_location.png'))
			plt.close()
	print()

if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
