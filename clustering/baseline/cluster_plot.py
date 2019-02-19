#!/usr/bin/env python3

import os
import sys

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'plot')))


import config
import segmentation
import load_sensor_data
import load_data_basic
import plot
import pandas as pd

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

# Define data type
default_signal = {'MinPeakDistance': 100, 'MinPeakHeight': 0.04, 'raw_cols': ['Cadence', 'HeartRate', 'Intensity', 'Steps', 'BreathingDepth', 'BreathingRate']}
om_signal = {'MinPeakDistance': 100, 'MinPeakHeight': 0.04, 'raw_cols': ['HeartRatePPG', 'StepCount']}

segmentation_hype = {'method': 'ma', 'offset': 60, 'overlap': 0, 'segmentation': 'ggs', 'segmentation_lamb': 10e0, 'sub_segmentation_lamb': None, 'preprocess_cols': ['HeartRatePPG', 'StepCount'], 'imputation': 'iterative'}
preprocess_hype = {'method': 'ma', 'offset': 60, 'overlap': 0, 'preprocess_cols': ['HeartRatePPG', 'StepCount'], 'imputation': 'iterative'}
omsignal_hype = {'method': 'ma', 'offset': 60, 'overlap': 0, 'imputation': None, 'preprocess_cols': ['Cadence', 'HeartRate', 'Intensity', 'Steps', 'BreathingDepth', 'BreathingRate']}
owl_in_one_hype = {'method': 'ma', 'offset': 60, 'overlap': 0, 'imputation': None}


def return_participant(main_folder):
    ###########################################################
    # 2. Read all fitbit file
    ###########################################################
    fitbit_folder = os.path.join(main_folder, '3_preprocessed_data/fitbit/')
    fitbit_file_list = os.listdir(fitbit_folder)
    fitbit_file_dict_list = {}
    
    for fitbit_file in fitbit_file_list:
        
        if '.DS' in fitbit_file:
            continue
        
        participant_id = fitbit_file.split('_')[0]
        if participant_id not in list(fitbit_file_dict_list.keys()):
            fitbit_file_dict_list[participant_id] = {}
    return list(fitbit_file_dict_list.keys())


def main(main_folder):
    ###########################################################
    # 1. Create Config
    ###########################################################
    process_data_folder = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    fitbit_config = config.Config(data_type='preprocess_data', sensor='fitbit', read_folder=process_data_folder,
                                  return_full_feature=False, process_hyper=preprocess_hype, signal_hyper=default_signal)

    omsignal_config = config.Config(data_type='preprocess_data', sensor='om_signal', read_folder=process_data_folder,
                                    return_full_feature=False, process_hyper=omsignal_hype, signal_hyper=om_signal)
    
    realizd_config = config.Config(data_type='preprocess_data', sensor='realizd', read_folder=process_data_folder,
                                   return_full_feature=False, process_hyper=preprocess_hype, signal_hyper=default_signal)

    owl_in_one_config = config.Config(data_type='preprocess_data', sensor='owl_in_one', read_folder=process_data_folder,
                                      return_full_feature=False, process_hyper=owl_in_one_hype, signal_hyper=default_signal)
    
    ggs_config = config.Config(data_type='segmentation', sensor='fitbit', read_folder=process_data_folder,
                               return_full_feature=False, process_hyper=segmentation_hype, signal_hyper=default_signal)
    
    fitbit_summary_config = config.Config(data_type='3_preprocessed_data', sensor='fitbit', read_folder=main_folder,
                                          return_full_feature=False, process_hyper=preprocess_hype, signal_hyper=default_signal)
    
    ###########################################################
    # 2. Get participant id list
    ###########################################################
    participant_id_list = return_participant(main_folder)
    participant_id_list.sort()
    
    top_participant_id_df = pd.read_csv(os.path.join(ggs_config.process_folder, 'participant_id.csv.gz'), index_col=0, compression='gzip')
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    igtb_df = load_data_basic.read_AllBasic(main_folder)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    mgt_df = load_data_basic.read_MGT(main_folder)
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(participant_id_list)))
        ###########################################################
        # 3. Create segmentation class
        ###########################################################
        ggs_segmentation = segmentation.Segmentation(read_config=fitbit_config, save_config=ggs_config,
                                                     realizd_config=realizd_config, participant_id=participant_id)
        
        ###########################################################
        # 4. Read segmentation data
        ###########################################################
        # ggs_segmentation.read_preprocess_data(participant_id)
        ggs_segmentation.read_preprocess_data_all()
        
        ###########################################################
        # Option: Read summary data, mgt, omsignal
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_config, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']

        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]

        omsignal_data_df = load_sensor_data.read_processed_omsignal(omsignal_config, participant_id)

        owl_in_one_df = load_sensor_data.read_processed_owl_in_one(owl_in_one_config, participant_id)
        
        ###########################################################
        # 5. Plot
        ###########################################################
        cluster_plot = plot.Plot(ggs_config=ggs_config, primary_unit=primary_unit)

        cluster_plot.plot_clusetr(participant_id, fitbit_df=ggs_segmentation.fitbit_df, fitbit_summary_df=fitbit_summary_df,
                                  mgt_df=participant_mgt, omsignal_data_df=omsignal_data_df,
                                  realizd_df=ggs_segmentation.realizd_df, owl_in_one_df=owl_in_one_df, cluster_df=None)
        
        del ggs_segmentation


if __name__ == '__main__':
    # Main Data folder
    main_folder = '../../../data/keck_wave_all/'
    
    main(main_folder)
