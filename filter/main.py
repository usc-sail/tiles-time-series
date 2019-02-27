"""
Filter the data
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'preprocess')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic
import pandas as pd
import argparse

from filter import Filter

def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess_data', segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data', clustering_data_identifier='clustering')
    
    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')

    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    # Get participant id list
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, experiment, 'participant_id.csv.gz'), tiles_data_path, k=150, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # Create filter class
        filter_class = Filter(data_config=data_config, participant_id=participant_id)
        
        # Read all data
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']

        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]

        omsignal_data_df = load_sensor_data.read_preprocessed_omsignal(data_config.omsignal_sensor_dict['preprocess_path'], participant_id)
        owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, participant_id)

        if fitbit_df is None:
            continue

        filter_class.filter_data(fitbit_df, fitbit_summary_df=fitbit_summary_df, omsignal_df=omsignal_data_df, mgt_df=participant_mgt, owl_in_one_df=owl_in_one_df)
        
        del filter_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'baseline' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiment)