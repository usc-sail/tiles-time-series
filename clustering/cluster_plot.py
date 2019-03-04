#!/usr/bin/env python3

import os
import sys
from configparser import ConfigParser
import argparse

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
import plot
import pandas as pd

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


def main(tiles_data_path, cluster_config_path, experiement):
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')), experiement)
    
    load_data_path.load_all_available_path(data_config, process_data_path)

    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')

    ###########################################################
    # Read ground truth data
    ###########################################################
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    survey_df = load_data_basic.read_app_survey(tiles_data_path)
    survey_df = survey_df.loc[survey_df['survey_type'] == 'psych_flex']
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    # Get participant id list, k=10, read 10 participants with most data in fitbit
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, k=150, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        ###########################################################
        # 3. Create segmentation class
        ###########################################################
        ggs_segmentation = segmentation.Segmentation(data_config=data_config, participant_id=participant_id)

        ###########################################################
        # Option: Read summary data, mgt, omsignal
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']

        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        participant_app_survey = survey_df.loc[survey_df['participant_id'] == participant_id]
        
        if len(participant_app_survey) == 0:
            continue

        omsignal_data_df = load_sensor_data.read_preprocessed_omsignal(data_config.omsignal_sensor_dict['preprocess_path'], participant_id)
        owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
        realizd_df = load_sensor_data.read_preprocessed_realizd(data_config.realizd_sensor_dict['preprocess_path'], participant_id)
        audio_df = load_sensor_data.read_preprocessed_audio(data_config.audio_sensor_dict['preprocess_path'], participant_id)
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, participant_id)
        
        ###########################################################
        # 4. Read clustering and segmentation data, skip clustering
        ###########################################################
        if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['segmentation_path'], participant_id + '.csv.gz')) is False:
            continue
        segmentation_df = load_sensor_data.load_segmentation_data(data_config.fitbit_sensor_dict['segmentation_path'], participant_id)
        '''
        # if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['clustering_path'], participant_id + '.csv')) is False:
        #    continue
        # clustering_df = load_sensor_data.load_clustering_data(data_config.fitbit_sensor_dict['clustering_path'], participant_id)
        '''
        
        ###########################################################
        # 5. Plot
        ###########################################################
        cluster_plot = plot.Plot(data_config=data_config, primary_unit=primary_unit)

        cluster_plot.plot_app_survey(participant_id, fitbit_df=fitbit_df, fitbit_summary_df=fitbit_summary_df, audio_df=audio_df,
                                     app_survey_df=participant_app_survey, segmentation_df=segmentation_df, omsignal_data_df=omsignal_data_df,
                                     realizd_df=realizd_df, owl_in_one_df=owl_in_one_df)
        
        del ggs_segmentation


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiement", required=False, help="Experiement name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiement = 'baseline' if args.config is None else args.config

    main(tiles_data_path, config_path, experiement)
