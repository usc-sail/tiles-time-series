"""
Top level classes for the preprocess model.
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own pyspark path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'preprocess')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import segmentation
import load_sensor_data, load_data_path, load_data_basic
import numpy as np
import pandas as pd
import argparse

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

import warnings
warnings.filterwarnings("ignore")


def main(tiles_data_path, config_path, experiement):
	###########################################################
	# 1. Create Config
	###########################################################
	process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))

	data_config = config.Config()
	data_config.readConfigFile(config_path, experiement)

	# Load preprocess folder
	# Load all data path according to config file
	load_data_path.load_all_available_path(data_config, process_data_path,
										   preprocess_data_identifier='preprocess',
										   segmentation_data_identifier='segmentation',
										   filter_data_identifier='filter_data',
										   clustering_data_identifier='clustering')

	# Load Fitbit summary folder
	fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')

	###########################################################
	# 2. Get participant id list
	###########################################################
	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# read shift type
		shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
		nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]

		if nurse != 1:
			continue

		###########################################################
		# 3. Create segmentation class
		###########################################################
		ggs_segmentation = segmentation.Segmentation(data_config=data_config, participant_id=participant_id)

		###########################################################
		# 4.1 Read days at work data
		###########################################################
		days_at_work_df = load_sensor_data.read_preprocessed_days_at_work(data_config.days_at_work_path, participant_id)

		if days_at_work_df is None:
			continue

		if len(days_at_work_df) < 10:
			continue

		###########################################################
		# 4.2 Read Fitbit data that is associated with these days
		###########################################################
		# fitbit_workday_dict = load_sensor_data.read_preprocessed_fitbit_on_workdays(data_config, participant_id, days_at_work_df, shift)
		fitbit_workday_dict = load_sensor_data.read_preprocessed_fitbit_during_work(data_config, participant_id, days_at_work_df, shift)

		if fitbit_workday_dict is None:
			continue

		###########################################################
		# 5. Segmentation
		###########################################################
		fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
		fitbit_summary_df = fitbit_data_dict['summary']

		# ggs_segmentation.segment_workday_data(fitbit_workday_dict)
		ggs_segmentation.segment_shift_data(fitbit_workday_dict)

		del ggs_segmentation


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
	parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
	parser.add_argument("--experiement", required=False, help="Experiement name")

	args = parser.parse_args()

	tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
	experiement = 'dpmm' if args.config is None else args.config

	main(tiles_data_path, config_path, experiement)