#!/usr/bin/env python3

import os
import sys
import pandas as pd
import argparse

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
from preprocess import Preprocess
import load_data_path, load_data_basic, load_sensor_data


def main(tiles_data_path, config_path, experiment):
    ###########################################################
    # 0. Read configs
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load preprocess folder
    load_data_path.load_preprocess_path(data_config, process_data_path, data_name='preprocess_data')
    
    if os.path.exists(os.join(process_data_path, 'tiles-phase1-wav123-processed', '4_extracted_features/jelly_audio_feats_fixed')) is False:
        load_sensor_data.download_data(os.join(process_data_path, 'tiles-phase1-wav123-processed', '4_extracted_features/jelly_audio_feats_fixed'), s3.Bucket(processed_bucket_str), simulated_data=False, prefix='4_extracted_features/jelly_audio_feats_fixed')

    
    ###########################################################
    # 1. Read all participant
    ###########################################################
    participant_id_list = load_data_basic.return_participant(tiles_data_path)
    
    ###########################################################
    # 2. Iterate all participant
    ###########################################################
    for i, participant_id in enumerate(participant_id_list[:]):
        
        print('Complete process for %s: %.2f' % (participant_id, 100 * i / len(participant_id_list)))
        
        ###########################################################
        # Read audio file path
        ###########################################################
        audio_file = participant_id + '.csv.gz'
        audio_file_abs_path = os.path.join(tiles_data_path, '4_extracted_features/jelly_audio_feats_fixed/', audio_file)
        
        ###########################################################
        # Read audio data
        ###########################################################
        try:
            audio_data_df = pd.read_csv(audio_file_abs_path, index_col=0)
            audio_data_df = audio_data_df.sort_index()
        except FileNotFoundError:
            continue
        
        ###########################################################
        # 2.0 Iterate all fitbit files
        ###########################################################
        if len(audio_data_df) > 0:
            ###########################################################
            # 2.1 Init fitbit preprocess
            ###########################################################
            audio_preprocess = Preprocess(data_config=data_config, participant_id=participant_id)
            
            ###########################################################
            # 2.2 Preprocess audio data array
            ###########################################################
            audio_preprocess.preprocess_audio(audio_data_df)
            
            del audio_preprocess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'baseline' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)

