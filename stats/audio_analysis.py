#!/usr/bin/env python3

import os
import sys
import argparse

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'plot')))

import config
import segmentation
import load_sensor_data, load_data_basic, load_data_path
import plot
import pandas as pd
import numpy as np


def main(tiles_data_path, config_path, experiment):
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    load_data_path.load_all_available_path(data_config, process_data_path)
    
    ###########################################################
    # Read ground truth data
    ###########################################################
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    
    participant_stats_all_df = pd.read_csv(os.path.join('audio_stats', 'participant.csv.gz'), index_col=0)
    
    nurse_df = participant_stats_all_df.loc[participant_stats_all_df['position'] == 1 & ~participant_stats_all_df['primary_unit'].str.contains('lysis')]
    day_shift_df = nurse_df.loc[nurse_df['shift'] == 'Day shift']
    night_shift_df = nurse_df.loc[nurse_df['shift'] == 'Night shift']
    
    cond_dialysis = participant_stats_all_df['primary_unit'].str.contains('Dialysis')
    cond_lab = participant_stats_all_df['primary_unit'].str.contains('Lab')
    cond_phlebotomist = participant_stats_all_df['primary_unit'].str.contains('Phlebotomist')
    
    other_df = participant_stats_all_df.loc[cond_dialysis | cond_lab | cond_phlebotomist]

    dialysis_df = participant_stats_all_df.loc[cond_dialysis]
    lab_df = participant_stats_all_df.loc[cond_lab]
    phlebotomist = participant_stats_all_df.loc[cond_phlebotomist]
    
    ICU_df = nurse_df.loc[nurse_df['primary_unit'].str.contains('ICU') | nurse_df['primary_unit'].str.contains('5 North')]
    non_ICU_df = nurse_df.loc[~(nurse_df['primary_unit'].str.contains('ICU') | nurse_df['primary_unit'].str.contains('5 North'))]

    day_ICU_df = day_shift_df.loc[day_shift_df['primary_unit'].str.contains('ICU') | day_shift_df['primary_unit'].str.contains('5 North')]
    day_non_ICU_df = day_shift_df.loc[~(day_shift_df['primary_unit'].str.contains('ICU') | day_shift_df['primary_unit'].str.contains('5 North'))]

    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles_path", required=False, help="Path to the root folder containing TILES data")
    parser.add_argument("--config", required=False, help="Path to a config file specifying how to perform the clustering")
    parser.add_argument("--experiment", required=False, help="Experiment name")
    args = parser.parse_args()
    
    tiles_data_path = '../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'baseline' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)




