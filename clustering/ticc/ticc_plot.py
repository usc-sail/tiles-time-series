#!/usr/bin/env python3

import os
import sys
from configparser import ConfigParser
import argparse

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'plot')))

import config
import segmentation
import load_sensor_data, load_data_basic, load_data_path
import plot
import pandas as pd


def main(tiles_data_path, cluster_config_path, experiement):
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(cluster_config_path, experiement)
    
    # Load preprocess folder
    load_data_path.load_preprocess_path(data_config, process_data_path, data_name='preprocess_data')
    
    # Load segmentation folder
    load_data_path.load_segmentation_path(data_config, process_data_path, data_name='segmentation')
    
    # Load clustering folder
    load_data_path.load_clustering_path(data_config, process_data_path, data_name='clustering')
    
    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')
    
    ###########################################################
    # Read ground truth data
    ###########################################################
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    top_participant_id_df = pd.read_csv(os.path.join(process_data_path, experiement, 'participant_id.csv.gz'), index_col=0, compression='gzip')
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        ###########################################################
        # Option: Read summary data, mgt, omsignal
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']
        
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
        
        owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
        realizd_df = load_sensor_data.read_preprocessed_realizd(data_config.realizd_sensor_dict['preprocess_path'], participant_id)
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, participant_id)
        
        ###########################################################
        # 4. Read clustering and segmentation data
        ###########################################################
        if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['clustering_path'], participant_id + '.csv.gz')) is False:
            continue
        
        clustering_df = pd.read_csv(os.path.join(data_config.fitbit_sensor_dict['clustering_path'], participant_id + '.csv.gz'), index_col=0)

        ###########################################################
        # 5. Plot
        ###########################################################
        cluster_plot = plot.Plot(data_config=data_config, primary_unit=primary_unit)
        
        cluster_plot.plot_ticc(participant_id, fitbit_df=fitbit_df, cluster_df=clustering_df)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiement", required=False, help="Experiement name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.config is None else args.config
    
    main(tiles_data_path, config_path, experiment)
