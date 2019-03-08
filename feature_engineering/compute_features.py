#!/usr/bin/env python3

import os
import sys
import pdb
import glob
import argparse
import pandas as pd
from configparser import ConfigParser

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))

import config
import load_data_path
import feature_engineering

def ComputeFeatures(tiles_data_path, experiment):
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))
    
    data_config = config.Config()
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file'))
    data_config.readConfigFile(config_path, experiment)

    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path, filter_data=True,
    preprocess_data_identifier='preprocess',
    segmentation_data_identifier='segmentation',
    filter_data_identifier='filter_data',
    clustering_data_identifier='clustering')

    ###########################################################
    # 2. Get participant id list
    ###########################################################
    top_participant_id_df = pd.read_csv(os.path.join(process_data_path, experiment, 'participant_id.csv.gz'), index_col=0, compression='gzip')
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # If clustering data exist
        clustering_pid_path = os.path.join(data_config.fitbit_sensor_dict['clustering_path'], participant_id)
        if not os.path.exists(clustering_pid_path):
            continue

        ###########################################################
        # 3. Read data and segmentation data
        ###########################################################
        file_dfs = []
        clustering_files = glob.glob(os.path.join(clustering_pid_path, '*.csv.gz'))
        for clustering_file in clustering_files:
            file_dfs.append(pd.read_csv(clustering_file))

        pid_df = pd.concat(file_dfs, axis=0)
        pid_df.columns = ['Timestamp', 'cluster_id']
        pid_df = pid_df.sort_values('Timestamp', axis=0)

        # Reformat the dataframe
        cur_cluster = None
        cluster_list = []
        for row in pid_df.iterrows():
            row = row[1]
            if cur_cluster is None:
                cur_cluster = row['cluster_id']
                start_time = row['Timestamp']
                last_timestamp = row['Timestamp']
            elif row['cluster_id'] != cur_cluster:
                cluster_list.append((start_time, last_timestamp, cur_cluster))
                cur_cluster = row['cluster_id']
                start_time = row['Timestamp']
                last_timestamp = row['Timestamp']
            else:
                last_timestamp = row['Timestamp']
                continue
        cluster_list.append((start_time, last_timestamp, cur_cluster))

        zip_cluster_list = list(zip(*cluster_list))
        cluster_summary_df = pd.DataFrame.from_dict({'start': zip_cluster_list[0], 'end': zip_cluster_list[1], 'cluster_id': zip_cluster_list[2]})
        feature_list = data_config.feature_engineering_dict['features'].split(',')
        feature_df = feature_engineering.CreateFeatures(cluster_summary_df, feature_list)
        feature_df = pd.concat((cluster_summary_df['start'], feature_df), axis=1)
        feature_df.rename(columns={'start':'Timestamp'}, inplace=True)

        # Save the feature data
        file_name = participant_id+'.csv'
        #feature_df.to_csv(os.path.join(data_config.feature_engineering_dict['feature_path'], file_name), header=True, index=False)
        out_folder = '../data/features/'
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        feature_df.to_csv(os.path.join(out_folder, file_name), header=True, index=False)

    return
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    experiment = 'baseline' if args.experiment is None else args.experiment

    ComputeFeatures(tiles_data_path, experiment)
