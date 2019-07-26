"""
Filter the data
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import pandas as pd
import numpy as np
from datetime import timedelta
import pickle


def main(tiles_data_path, config_path, experiment):

	# Create Config
	process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))

	data_config = config.Config()
	data_config.readConfigFile(config_path, experiment)

	ubicomp_data_config = config.Config()
	ubicomp_data_config.readUbicompConfigFile(config_path)

	# Load all data path according to config file
	load_data_path.load_all_available_path(data_config, process_data_path,
										   preprocess_data_identifier='preprocess',
										   segmentation_data_identifier='segmentation',
										   filter_data_identifier='filter_data',
										   clustering_data_identifier='clustering')

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')
	mgt_df = load_data_basic.read_MGT(tiles_data_path)

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	# Location list
	loc_list = ['ns', 'pat', 'med', 'lounge', 'unknown', 'floor2', 'other_floor']

	# setting
	process_offset = int(ubicomp_data_config.omsignal_sensor_dict['offset'])
	process_window = int(ubicomp_data_config.omsignal_sensor_dict['window'])

	if os.path.exists(os.path.join('data')) is False:
		os.mkdir(os.path.join('data'))

	if os.path.exists(os.path.join('data', 'window_' + str(process_window) + '_offset_' + str(process_offset))) is False:
		os.mkdir(os.path.join('data', 'window_' + str(process_window) + '_offset_' + str(process_offset)))

	for idx, participant_id in enumerate(top_participant_id_list[:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Read id
		uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
		participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
		current_position = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition)[0]

		if current_position != 1:
			continue

		loc_dict = {}
		for loc in loc_list:
			loc_dict[loc] = {}

		# Read other sensor data, the aim is to detect whether people workes during a day
		if os.path.exists(os.path.join(data_config.omsignal_sensor_dict['filter_path'], participant_id)) is False:
			continue
		file_list = os.listdir(os.path.join(data_config.omsignal_sensor_dict['filter_path'], participant_id))

		for file_name in file_list:
			om_signal_df = pd.read_csv(os.path.join(data_config.omsignal_sensor_dict['filter_path'], participant_id, file_name), index_col=0)
			owl_in_one_df = pd.read_csv(os.path.join(data_config.owl_in_one_sensor_dict['filter_path'], participant_id, file_name), index_col=0)

			tmp_df = om_signal_df.loc[om_signal_df['HeartRate'] == 0]
			om_signal_df.loc[list(tmp_df.index), 'HeartRate'] = np.nan
			owl_in_one_index_df = pd.DataFrame(index=list(owl_in_one_df.index), columns=['index'])

			for col in list(owl_in_one_df.columns):
				loc_idx = loc_list.index(col)
				tmp_data_df = owl_in_one_df.loc[owl_in_one_df[col] == 1]
				owl_in_one_index_df.loc[list(tmp_data_df.index), 'index'] = loc_idx

			data_array = np.array(owl_in_one_index_df)
			offset_array = data_array[1:] - data_array[:-1]
			change_point_array = np.where(offset_array != 0)[0]

			start_array, end_array = np.zeros(len(change_point_array) + 1), np.zeros(len(change_point_array) + 1)
			start_array[1:], end_array[:-1] = change_point_array, change_point_array
			end_array[-1] = len(owl_in_one_df) - 1

			for i in range(len(start_array)):
				start, end = int(start_array[i]), int(end_array[i])
				length = end - start
				loc = loc_list[int(data_array[i])]

				if length < 5:
					continue

				start_str, end_str = list(owl_in_one_df.index)[start], list(owl_in_one_df.index)[end]
				tmp_om_df = om_signal_df[start_str:end_str]

				if len(tmp_om_df) < 60 * 4:
					continue

				tmp_start, tmp_end = pd.to_datetime(tmp_om_df.index[0]), pd.to_datetime(tmp_om_df.index[-1])
				time_range = int((tmp_end - tmp_start).seconds / process_offset) - 1

				heart_rate_df = tmp_om_df['HeartRate']
				intensity_df = tmp_om_df['Intensity']

				tmp_df = pd.DataFrame()
				for tmp_idx in range(time_range):

					tmp_frame_start_str = (tmp_start + timedelta(seconds=tmp_idx*process_offset)).strftime(load_data_basic.date_time_format)[:-3]
					tmp_frame_end_str = (tmp_start + timedelta(seconds=tmp_idx*process_offset+process_window)).strftime(load_data_basic.date_time_format)[:-3]

					row_df = pd.DataFrame(index=[tmp_frame_start_str])
					row_df['HeartRate'] = np.nanmean(np.array(heart_rate_df[tmp_frame_start_str:tmp_frame_end_str]))
					if len(intensity_df[tmp_frame_start_str:tmp_frame_end_str]) > 0:
						tmp_mean = np.nanmean(np.power(np.array(intensity_df[tmp_frame_start_str:tmp_frame_end_str]), 2))
						row_df['Intensity'] = np.sqrt(tmp_mean)
					else:
						row_df['Intensity'] = np.nan

					tmp_df = tmp_df.append(row_df)

				loc_dict[loc][start_str] = {}
				loc_dict[loc][start_str]['heart'] = np.array(tmp_df['HeartRate'])
				loc_dict[loc][start_str]['intensity'] = np.array(tmp_df['Intensity'])

		output = open(os.path.join('data', 'window_' + str(process_window) + '_offset_' + str(process_offset), participant_id + '.pkl'), 'wb')
		pickle.dump(loc_dict, output)


if __name__ == '__main__':
	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)