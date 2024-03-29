import config
import os
import segmentation
import load_sensor_data
import load_data_basic
import plot
import pandas as pd

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

default_signal = {'MinPeakDistance': 100, 'MinPeakHeight': 0.04,
                  'raw_cols': ['Cadence', 'HeartRate', 'Intensity', 'Steps', 'BreathingDepth', 'BreathingRate']}

segmentation_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                     'segmentation': 'ggs', 'segmentation_lamb': 10e0, 'sub_segmentation_lamb': None,
                     'preprocess_cols': ['HeartRatePPG', 'StepCount'], 'imputation': 'iterative'}

preprocess_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                   'preprocess_cols': ['HeartRatePPG', 'StepCount'], 'imputation': 'iterative'}

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
    fitbit_config = config.Config(data_type='preprocess_data', sensor='fitbit',
                                  read_folder=os.path.abspath(os.path.join(os.pardir, '../..')),
                                  return_full_feature=False, process_hyper=preprocess_hype,
                                  signal_hyper=default_signal)
    
    owl_in_one_config = config.Config(data_type='preprocess_data', sensor='owl_in_one',
                                      read_folder=os.path.abspath(os.path.join(os.pardir, '../..')),
                                      return_full_feature=False, process_hyper=owl_in_one_hype,
                                      signal_hyper=default_signal)
    
    ggs_config = config.Config(data_type='segmentation_by_sleep', sensor='fitbit',
                               read_folder=os.path.abspath(os.path.join(os.pardir, '../..')),
                               return_full_feature=False, process_hyper=segmentation_hype,
                               signal_hyper=default_signal)

    fitbit_summary_config = config.Config(data_type='3_preprocessed_data', sensor='fitbit',
                                          read_folder=main_folder, return_full_feature=False,
                                          process_hyper=preprocess_hype, signal_hyper=default_signal)
    
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
    
    for idx, participant_id in enumerate(top_participant_id_list[80:]):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(participant_id_list)))
        ###########################################################
        # 3. Create segmentation class
        ###########################################################
        ggs_segmentation = segmentation.Segmentation(read_config=fitbit_config, save_config=ggs_config, participant_id=participant_id)
        
        ###########################################################
        # 4.1 Read segmentation data
        ###########################################################
        save_folder = os.path.join(ggs_segmentation.save_config.process_folder)
        if os.path.exists(os.path.join(save_folder, participant_id + '.csv.gz')) is False:
            continue
        
        # if os.path.exists(os.path.join(save_folder, participant_id + '_fitbit.csv.gz')) is True:
        #    continue

        ###########################################################
        # 4.2 Read owl_data
        ###########################################################
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        current_position = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
        
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
        
        owl_in_one_df = load_sensor_data.read_processed_owl_in_one(owl_in_one_config, participant_id)

        ###########################################################
        # 4.2 Read fitbit summary data
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_config, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']

        ###########################################################
        # 4.3 Read fitbit data
        ###########################################################
        fitbit_df = load_sensor_data.read_processed_fitbit_with_pad(fitbit_config, participant_id)
        
        ###########################################################
        # 5. Plot
        ###########################################################
        ggs_segmentation.read_owl_in_one_in_segment(fitbit_summary_df=fitbit_summary_df, fitbit_df=fitbit_df,
                                                    owl_in_one_df=owl_in_one_df, current_position=current_position)
        
        del ggs_segmentation


if __name__ == '__main__':
    # Main Data folder
    main_folder = '../../../../data/keck_wave_all/'
    
    main(main_folder)
