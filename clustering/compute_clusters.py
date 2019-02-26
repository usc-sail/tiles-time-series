#!/usr/bin/env python3

import os
import sys
import pdb
import argparse
import pandas as pd
from configparser import ConfigParser

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'plot')))

import config
import segmentation
import load_sensor_data, load_data_basic, load_data_path
import summarize_sequence
import clustering
import plot

def ComputeClusters(tiles_data_path, experiment):
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))
    
    data_config = config.Config()
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file'))
    data_config.readConfigFile(config_path, experiment)
    
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
    top_participant_id_df = pd.read_csv(os.path.join(process_data_path, experiment, 'participant_id.csv.gz'), index_col=0, compression='gzip')
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # If segmentation data exist
        if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['segmentation_path'], participant_id + '.csv.gz')) is False:
            continue

        # Read basic data for each participant
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
        
        # Read fitbit summary data
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']

        # Read preprocessed sensor data
        omsignal_data_df = load_sensor_data.read_preprocessed_omsignal(data_config.omsignal_sensor_dict['preprocess_path'], participant_id)
        owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
        realizd_df = load_sensor_data.read_preprocessed_realizd(data_config.realizd_sensor_dict['preprocess_path'], participant_id)
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, participant_id)
        audio_df = load_sensor_data.read_preprocessed_audio(data_config.audio_sensor_dict['preprocess_path'], participant_id)

        # Read segmentation data
        if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['segmentation_path'], participant_id + '.csv.gz')) is False:
            continue
        
        ###########################################################
        # 3. Read data and segmentation data
        ###########################################################
        segmentation_df = load_sensor_data.load_segmentation_data(data_config.fitbit_sensor_dict['segmentation_path'], participant_id)
        times_list = segmentation_df['time']
        start_times = times_list.tolist()[0:-1]
        end_times = times_list.tolist()[1:]
        segments = pd.DataFrame(data={'start': start_times, 'end': end_times}, index=range(len(times_list)-1))

        # Get cluster configuration from the config file
        sequence = fitbit_df
        segment_summaries = summarize_sequence.SummarizeSequenceSegments(sequence, segments, method=data_config.fitbit_sensor_dict['segmentation_method'])
        clusters = clustering.ClusterSegments(segment_summaries, method=data_config.fitbit_sensor_dict['cluster_method'], num_clusters=int(data_config.fitbit_sensor_dict['num_cluster']))

        # Save the cluster data
        file_name = participant_id+'.csv'
        clustering_df = pd.concat((segments, clusters), axis=1)
        clustering_df = clustering_df[['start', 'end', 'cluster_id']] # Reorder columns
        clustering_df.to_csv(os.path.join(data_config.fitbit_sensor_dict['clustering_path'], file_name), header=True, index=False)

        ###########################################################
        # 4. Plot
        ###########################################################
        if data_config.enable_plot:
            plot_df = clustering_df
            plot_df.loc[:, 'index'] = plot_df.loc[:, 'start']
            plot_df = plot_df.set_index('index')

            cluster_plot = plot.Plot(data_config=data_config, primary_unit=primary_unit)

            cluster_plot.plot_cluster(participant_id, fitbit_df=fitbit_df, fitbit_summary_df=fitbit_summary_df,
                                      mgt_df=participant_mgt, segmentation_df=segmentation_df,
                                      omsignal_data_df=omsignal_data_df, realizd_df=realizd_df,
                                      owl_in_one_df=owl_in_one_df, audio_df=audio_df, cluster_df=plot_df)
    return
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    experiment = 'baseline' if args.experiment is None else args.experiment

    ComputeClusters(tiles_data_path, experiment)
