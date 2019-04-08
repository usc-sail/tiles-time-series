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


def longest_speak(speak_point):
    diff = speak_point[1:] - speak_point[:-1]
    change_point = np.where(diff > 1)[0]
    
    non_speak_time = diff[change_point]
    
    if speak_point[0] < 100:
        non_speak_time = np.append(non_speak_time, speak_point[0])

    change_point = np.append(change_point, len(speak_point) - 1)
    speak_time = [speak_point[change_point[0]] - speak_point[0] + 1]
    for index, value in enumerate(change_point):
        if index != len(change_point) - 1:
            speak_time.append(speak_point[change_point[index+1]] - speak_point[value+1] + 1)
            
    if len(np.where(non_speak_time > 200)[0]) != 0:
        print()
    
    return speak_time, non_speak_time

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
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    # Get participant id list, k=10, read 10 participants with most data in audio
    audio_top_participant_path = os.path.join(process_data_path, 'audio_participant_id.csv.gz')
    top_participant_id_df = load_data_basic.return_top_k_participant_from_stream(audio_top_participant_path, tiles_data_path, k=150, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    
    participant_stats_all_df = pd.DataFrame()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
       
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
        position = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
        # ICU = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]

        participant_stats_df = pd.DataFrame(index=[participant_id])
        
        # Get owl in one data
        owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
        
        audio_df = load_sensor_data.read_preprocessed_audio(data_config.audio_sensor_dict['preprocess_path'], participant_id)
        audio_df = audio_df * 60
        
        if owl_in_one_df is None:
            continue
        
        ###########################################################
        # 4. Extract regions of shift
        ###########################################################
        diff = pd.to_datetime(list(owl_in_one_df.index)[1:]) - pd.to_datetime(list(owl_in_one_df.index)[:-1])
        diff = list(diff.total_seconds())
        
        change_point = list((np.where(np.array(diff) > 3600))[0])
        change_point.append(len(owl_in_one_df)-1)
        
        final_stats = pd.DataFrame()

        # start_date_time = pd.to_datetime(list(owl_in_one_df.index)[0])
        # end_date_time = pd.to_datetime(list(owl_in_one_df.index)[change_point[0]])
        
        # Greater than 6 hours
        # if 3600 * 6 < (end_date_time - start_date_time).total_seconds() < 3600 * 10:
        if 360 < change_point[0] < 800:
            
            row_df = pd.DataFrame(index=[list(owl_in_one_df.index)[0]])
            audio_row = audio_df[list(owl_in_one_df.index)[0]:list(owl_in_one_df.index)[change_point[0]]]

            if len(np.where(np.array(audio_row) > 3)[0]) < 15:
                continue

            row_df['start'] = list(owl_in_one_df.index)[0]
            row_df['end'] = list(owl_in_one_df.index)[change_point[0]]
            row_df['duration'] = len(audio_row)
            
            row_df['greater_than_0s'] = len(np.where(np.array(audio_row) > 0)[0])
            row_df['greater_than_1s'] = len(np.where(np.array(audio_row) > 1)[0])
            row_df['greater_than_2s'] = len(np.where(np.array(audio_row) > 2)[0])
            row_df['greater_than_3s'] = len(np.where(np.array(audio_row) > 3)[0])
            row_df['greater_than_5s'] = len(np.where(np.array(audio_row) > 5)[0])
            row_df['greater_than_6s'] = len(np.where(np.array(audio_row) > 6)[0])

            row_df['greater_than_1s_portion'] = len(np.where(np.array(audio_row) > 1)[0]) / len(audio_row)
            row_df['greater_than_2s_portion'] = len(np.where(np.array(audio_row) > 2)[0]) / len(audio_row)
            row_df['greater_than_3s_portion'] = len(np.where(np.array(audio_row) > 3)[0]) / len(audio_row)
            row_df['greater_than_5s_portion'] = len(np.where(np.array(audio_row) > 5)[0]) / len(audio_row)
            row_df['greater_than_6s_portion'] = len(np.where(np.array(audio_row) > 6)[0]) / len(audio_row)
            
            speak_time, non_speak_time = longest_speak(np.where(np.array(audio_row) > 3)[0])
            row_df['speak_time_mean'] = np.mean(speak_time)
            row_df['speak_time_std'] = np.std(speak_time)
            row_df['speak_time_median'] = np.median(speak_time)
            row_df['speak_time_min'] = np.min(speak_time)
            row_df['speak_time_max'] = np.max(speak_time)

            row_df['non_speak_time_mean'] = np.mean(non_speak_time)
            row_df['non_speak_time_std'] = np.std(non_speak_time)
            row_df['non_speak_time_median'] = np.median(non_speak_time)
            row_df['non_speak_time_min'] = np.min(non_speak_time)
            row_df['non_speak_time_max'] = np.max(non_speak_time)
            
            if np.max(non_speak_time) > 180:
                continue

            final_stats = final_stats.append(row_df)

        participant_speak_time, participant_non_speak_time = [], []
            
        for index, value in enumerate(change_point):
            if index == len(change_point) - 1:
                continue
            
            start_date_time = pd.to_datetime(list(owl_in_one_df.index)[value+1])
            end_date_time = pd.to_datetime(list(owl_in_one_df.index)[change_point[index+1]])
            # if (end_date_time - start_date_time).total_seconds() < 3600 * 6 or (end_date_time - start_date_time).total_seconds() > 3600 * 10:
            #    continue
            
            if change_point[index+1] - value - 1 > 800 or change_point[index+1] - value - 1 < 360:
                continue

            audio_row = audio_df[list(owl_in_one_df.index)[value+1]:list(owl_in_one_df.index)[change_point[index+1]]]
            if len(np.where(np.array(audio_row) > 3)[0]) < 15:
                continue

            row_df = pd.DataFrame(index=[list(owl_in_one_df.index)[value + 1]])
            row_df['start'] = list(owl_in_one_df.index)[value + 1]
            row_df['end'] = list(owl_in_one_df.index)[change_point[index + 1]]

            row_df['duration'] = len(audio_row)
            row_df['greater_than_0s'] = len(np.where(np.array(audio_row) > 0)[0])
            row_df['greater_than_1s'] = len(np.where(np.array(audio_row) > 1)[0])
            row_df['greater_than_2s'] = len(np.where(np.array(audio_row) > 2)[0])
            row_df['greater_than_3s'] = len(np.where(np.array(audio_row) > 3)[0])
            row_df['greater_than_5s'] = len(np.where(np.array(audio_row) > 5)[0])
            row_df['greater_than_6s'] = len(np.where(np.array(audio_row) > 6)[0])
            
            row_df['greater_than_1s_portion'] = len(np.where(np.array(audio_row) > 1)[0]) / len(audio_row)
            row_df['greater_than_2s_portion'] = len(np.where(np.array(audio_row) > 2)[0]) / len(audio_row)
            row_df['greater_than_3s_portion'] = len(np.where(np.array(audio_row) > 3)[0]) / len(audio_row)
            row_df['greater_than_5s_portion'] = len(np.where(np.array(audio_row) > 5)[0]) / len(audio_row)
            row_df['greater_than_6s_portion'] = len(np.where(np.array(audio_row) > 6)[0]) / len(audio_row)
            
            speak_time, non_speak_time = longest_speak(np.where(np.array(audio_row) > 3)[0])
            
            row_df['speak_time_mean'] = np.mean(speak_time)
            row_df['speak_time_std'] = np.std(speak_time)
            row_df['speak_time_median'] = np.median(speak_time)
            row_df['speak_time_min'] = np.min(speak_time)
            row_df['speak_time_max'] = np.max(speak_time)

            row_df['non_speak_time_mean'] = np.mean(non_speak_time)
            row_df['non_speak_time_std'] = np.std(non_speak_time)
            row_df['non_speak_time_median'] = np.median(non_speak_time)
            row_df['non_speak_time_min'] = np.min(non_speak_time)
            row_df['non_speak_time_max'] = np.max(non_speak_time)
            
            if np.max(non_speak_time) > 180:
                continue
            
            [participant_speak_time.append(i) for i in speak_time]
            [participant_non_speak_time.append(i) for i in non_speak_time]

            final_stats = final_stats.append(row_df)

        if final_stats.shape[0] > 5:
            participant_stats_df['greater_than_1s_mean'] = np.nanmean(np.array(final_stats.greater_than_1s))
            participant_stats_df['greater_than_1s_portion_mean'] = np.nanmean(np.array(final_stats.greater_than_1s_portion))
            participant_stats_df['greater_than_1s_std'] = np.nanstd(np.array(final_stats.greater_than_1s))
            
            participant_stats_df['greater_than_2s_mean'] = np.nanmean(np.array(final_stats.greater_than_2s))
            participant_stats_df['greater_than_2s_portion_mean'] = np.nanmean(np.array(final_stats.greater_than_2s_portion))
            participant_stats_df['greater_than_2s_std'] = np.nanstd(np.array(final_stats.greater_than_2s))

            participant_stats_df['greater_than_3s_mean'] = np.nanmean(np.array(final_stats.greater_than_3s))
            participant_stats_df['greater_than_3s_portion_mean'] = np.nanmean(np.array(final_stats.greater_than_3s_portion))
            participant_stats_df['greater_than_3s_std'] = np.nanstd(np.array(final_stats.greater_than_3s))

            participant_stats_df['greater_than_5s_mean'] = np.nanmean(np.array(final_stats.greater_than_5s))
            participant_stats_df['greater_than_5s_portion_mean'] = np.nanmean(np.array(final_stats.greater_than_5s_portion))
            participant_stats_df['greater_than_5s_std'] = np.nanstd(np.array(final_stats.greater_than_5s))

            participant_stats_df['greater_than_6s_mean'] = np.nanmean(np.array(final_stats.greater_than_6s))
            participant_stats_df['greater_than_6s_portion_mean'] = np.nanmean(np.array(final_stats.greater_than_6s_portion))
            participant_stats_df['greater_than_6s_std'] = np.nanstd(np.array(final_stats.greater_than_6s))
    
            participant_stats_df['speak_time_mean'] = np.mean(participant_speak_time)
            participant_stats_df['speak_time_std'] = np.std(participant_speak_time)
            participant_stats_df['speak_time_median'] = np.median(participant_speak_time)
            participant_stats_df['speak_time_min'] = np.min(participant_speak_time)
            participant_stats_df['speak_time_max'] = np.max(participant_speak_time)
    
            participant_stats_df['non_speak_time_mean'] = np.mean(participant_non_speak_time)
            participant_stats_df['non_speak_time_std'] = np.std(participant_non_speak_time)
            participant_stats_df['non_speak_time_median'] = np.median(participant_non_speak_time)
            participant_stats_df['non_speak_time_min'] = np.min(participant_non_speak_time)
            participant_stats_df['non_speak_time_max'] = np.max(participant_non_speak_time)
            
            participant_stats_df['primary_unit'] = primary_unit
            participant_stats_df['shift'] = shift
            participant_stats_df['position'] = position
    
            participant_stats_all_df = participant_stats_all_df.append(participant_stats_df)
        
        if os.path.exists('audio_stats') is False: os.mkdir('audio_stats')
        
        if final_stats.shape[0] > 5:
            final_stats.to_csv(os.path.join('audio_stats', participant_id + '.csv.gz'))

    participant_stats_all_df.to_csv(os.path.join('audio_stats', 'participant.csv.gz'))
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




