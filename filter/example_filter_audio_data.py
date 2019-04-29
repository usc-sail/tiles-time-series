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

from filter import Filter


def main(tiles_data_path, config_path, experiment):
	# Create Config
	process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))

	data_config = config.Config()
	data_config.readConfigFile(config_path, experiment)

	# Load all data path according to config file
	load_data_path.load_all_available_path(data_config, process_data_path,
										   preprocess_data_identifier='preprocess', segmentation_data_identifier='segmentation',
										   filter_data_identifier='filter_data', clustering_data_identifier='clustering')

	# Read ground truth data
	igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
	igtb_df = igtb_df.drop_duplicates(keep='first')
	mgt_df = load_data_basic.read_MGT(tiles_data_path)

	# Get participant id list, k=None, save all participant data
	top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
	top_participant_id_list = list(top_participant_id_df.index)
	top_participant_id_list.sort()

	for idx, participant_id in enumerate(top_participant_id_list[180:]):

		print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

		# Create filter class
		filter_class = Filter(data_config=data_config, participant_id=participant_id)

		# If we have save the filter data before
		if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['filter_path'], participant_id, 'filter_dict.csv.gz')) is True:
			print('%s has been filtered before' % participant_id)
			continue

		# Read id
		uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
		participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]

		# Read other sensor data, the aim is to detect whether people workes during a day
		owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
		raw_audio_df = load_sensor_data.read_raw_audio(tiles_data_path, participant_id)
		omsignal_data_df = load_sensor_data.read_preprocessed_omsignal(data_config.omsignal_sensor_dict['preprocess_path'], participant_id)
		
		# If we don't have fitbit data, no need to process it
		if raw_audio_df is None:
			print('%s has no audio data' % participant_id)
			continue

		filter_class.filter_data(raw_audio_df=raw_audio_df, mgt_df=participant_mgt,
								 owl_in_one_df=owl_in_one_df, omsignal_df=omsignal_data_df)

		del filter_class


if __name__ == '__main__':

	# Read args
	args = parser.parse_args()

	# If arg not specified, use default value
	tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
	experiment = 'dpmm' if args.experiment is None else args.experiment

	main(tiles_data_path, config_path, experiment)