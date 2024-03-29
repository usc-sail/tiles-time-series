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
import load_sensor_data, load_data_path, load_data_basic
import parser


def main(tiles_data_path, config_path, experiment):
    
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess_data',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    # Get participant id list, k=10, read 10 participants with most data in fitbit
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, k=10, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    # option 1:
    top_participant_data_list = load_sensor_data.load_all_filter_data(data_config.fitbit_sensor_dict['filter_path'], top_participant_id_list, filter_logic=None, threshold_dict=None)
    # option 2:
    '''
    top_participant_data_list = []
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # Read per participant data
        participant_data_dict = load_sensor_data.load_filter_data(data_config.fitbit_sensor_dict['filter_path'], participant_id, filter_logic=None, threshold_dict=None)

        # Append data to the final list
        if participant_data_dict is not None: top_participant_data_list.append(participant_data_dict)
    '''
    print('Successfully load all participant filter data')


if __name__ == '__main__':
    
    # Read args
    args = parser.parse_args()
    
    # If args are not specified, use default value
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'baseline' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)