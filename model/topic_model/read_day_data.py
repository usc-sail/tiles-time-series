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
		
		topic_method = data_config.audio_sensor_dict['topic_method']
		topic_num = str(data_config.audio_sensor_dict['topic_num'])
		overlap = data_config.audio_sensor_dict['overlap']
		cluster_offset = data_config.audio_sensor_dict['cluster_offset']
		
		save_prefix = topic_method + '_' + topic_num + '_overlap_' + str(overlap) + '_' + str(cluster_offset)
		
		tp_weight_name = os.path.join(data_cluster_path, participant_id, save_prefix + '_offset_subspace_topic_weight.csv.gz')
		tp_location_name = os.path.join(data_cluster_path, participant_id, save_prefix + '_offset_subspace_topic_and_location.csv.gz')
		
		if os.path.exists(tp_location_name) is True:
			tp_location_df = pd.read_csv(tp_location_name, index_col=0)
			
			index_list = [str(i) for i in range(int(data_config.audio_sensor_dict['topic_num']))]
			col_list = [col for col in tp_location_df.columns if col not in index_list and 'key' not in col]
			location_df = tp_location_df.loc[:, col_list]
			
			location_norm_df = np.array(location_df) / np.array(location_df.sum(axis=1)).reshape([len(location_df), 1])
			location_norm_df = ((np.array(location_df) - np.nanmin(location_df, axis=0))) / np.nanmax(location_df, axis=0)
			location_norm_df = pd.DataFrame(location_norm_df, index=list(location_df.index), columns=list(location_df.columns))
			
			tp_location_df.loc[list(location_df.index), list(location_df.columns)] = location_norm_df
			tp_location_corr_df = tp_location_df.drop(columns=['key_word']).corr()
			
			plt.pcolor(tp_location_df.drop(columns=['key_word']).transpose())
			plt.yticks(np.arange(0.5, len(tp_location_df.drop(columns=['key_word']).transpose().index), 1), tp_location_df.drop(columns=['key_word']).transpose().index)
			plt.xticks(np.arange(0.5, len(tp_location_df.drop(columns=['key_word']).transpose().columns), 1), tp_location_df.drop(columns=['key_word']).transpose().columns)
			plt.show()
			print()


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)
