"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import numpy as np
import pandas as pd
import pickle
import preprocess


def nurse_sleep(sleep_dict, psqi_raw_igtb, uid):
    sleep_summary_df = sleep_dict['summary']
    sleep_summary_dict = {'main_sleep_before_workday': pd.DataFrame(), 'main_sleep_after_workday': pd.DataFrame(),
                          'main_sleep_offday': pd.DataFrame(), 'main_sleep_workday': pd.DataFrame(),
                          'nap_offday': pd.DataFrame()}
    
    if len(sleep_summary_df) > 15:
        
        for i in range(len(sleep_summary_df)):
            current_df = sleep_summary_df.iloc[i, :]
            if i != 0:
                last_df = sleep_summary_df.iloc[i-1, :]
        
        sleep_before_workday_df = sleep_summary_df.loc[(sleep_summary_df['sleep_before_work'] == 1) & (sleep_summary_df['sleep_after_work'] == 0)]
        if len(sleep_before_workday_df) > 0:
            tmp_main_sleep_df = sleep_before_workday_df.loc[sleep_before_workday_df['duration'] >= 2]
            sleep_summary_dict['main_sleep_before_workday'] = tmp_main_sleep_df

        sleep_workday_df = sleep_summary_df.loc[(sleep_summary_df['sleep_after_work'] == 1) & (sleep_summary_df['sleep_before_work'] == 1)]
        sleep_workday_df = sleep_workday_df.append(sleep_before_workday_df)
        if len(sleep_workday_df) > 0:
            tmp_main_sleep_df = sleep_workday_df.loc[sleep_workday_df['duration'] >= 2]
            sleep_summary_dict['main_sleep_workday'] = tmp_main_sleep_df
        
        sleep_after_workday_df = sleep_summary_df.loc[(sleep_summary_df['sleep_after_work'] == 1) & (sleep_summary_df['sleep_before_work'] == 0)]
        if len(sleep_after_workday_df) > 0:
            tmp_main_sleep_df = sleep_after_workday_df.loc[sleep_after_workday_df['duration'] >= 2]
            sleep_summary_dict['main_sleep_before_workday'] = tmp_main_sleep_df
        
        sleep_offday_df = sleep_summary_df.loc[(sleep_summary_df['sleep_after_work'] == 0) & (sleep_summary_df['sleep_before_work'] == 0)]
        sleep_offday_df = sleep_offday_df.append(sleep_after_workday_df)
        if len(sleep_offday_df) > 0:
            tmp_main_sleep_df = sleep_offday_df.loc[sleep_offday_df['duration'] >= 2]
            sleep_summary_dict['main_sleep_offday'] = tmp_main_sleep_df
            
            tmp_main_sleep_df = sleep_offday_df.loc[sleep_offday_df['duration'] < 2]
            sleep_summary_dict['nap_offday'] = tmp_main_sleep_df
            
        # for col in list(sleep_summary_dict.keys()):
        for col in ['main_sleep_offday', 'main_sleep_workday', 'nap_offday']:
            for data_col in ['duration', 'SleepEfficiency']:
                data_df = sleep_summary_dict[col]
                if len(data_df) > 0:
                    psqi_raw_igtb.loc[uid, col + '_' + data_col + '_mean'] = np.nanmean(data_df[data_col])
                    psqi_raw_igtb.loc[uid, col + '_' + data_col + '_std'] = np.nanstd(data_df[data_col])
                    # psqi_raw_igtb.loc[uid, col + '_' + data_col + '_median'] = np.nanmedian(data_df[data_col])
                    
                time_list = []
                for time in list(pd.to_datetime(data_df['SleepBeginTimestamp'])):
                    if time.hour >= 12:
                        time_list.append(time.hour-24)
                    else:
                        time_list.append(time.hour)
                psqi_raw_igtb.loc[uid, col + '_start_time_std'] = np.nanstd(time_list)
                

def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')

    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
    psqi_raw_igtb = load_data_basic.read_PSQI_Raw(tiles_data_path)

    # mgt_df = load_data_basic.read_MGT(tiles_data_path)

    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    if os.path.exists(os.path.join(os.path.dirname(__file__), 'sleep_raw.csv.gz')) is True:
        psqi_raw_igtb = pd.read_csv(os.path.join(os.path.dirname(__file__), 'sleep_raw.csv.gz'), index_col=0)
    else:
        for idx, participant_id in enumerate(top_participant_id_list):
            print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
            
            sleep_path = os.path.join(data_config.sleep_path, participant_id + '.pkl')
            if os.path.exists(sleep_path) is False:
                continue
            
            pkl_file = open(os.path.join(data_config.sleep_path, participant_id + '.pkl'), 'rb')
            sleep_dict = pickle.load(pkl_file)
            
            uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
            nurse_sleep(sleep_dict, psqi_raw_igtb, uid)
    
            nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
            primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            job_str = 'nurse' if nurse == 1 and 'Dialysis' not in primary_unit else 'non_nurse'
            shift_str = 'day' if shift == 'Day shift' else 'night'
    
            psqi_raw_igtb.loc[uid, 'job'] = job_str
            psqi_raw_igtb.loc[uid, 'shift'] = shift_str
    
            for col in igtb_cols:
                psqi_raw_igtb.loc[uid, col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]
        
        psqi_raw_igtb.to_csv(os.path.join(os.path.dirname(__file__), 'sleep_raw.csv.gz'), compression='gzip')

    nurse_sleep_df = psqi_raw_igtb.loc[psqi_raw_igtb['job'] == 'nurse']
    day_nurse_sleep_df = psqi_raw_igtb.loc[psqi_raw_igtb['shift'] == 'day']
    night_nurse_sleep_df = psqi_raw_igtb.loc[psqi_raw_igtb['shift'] == 'night']

    nurse_sleep_df = nurse_sleep_df.drop(columns=['job', 'shift'])
    day_nurse_sleep_df = day_nurse_sleep_df.drop(columns=['job', 'shift'])
    night_nurse_sleep_df = night_nurse_sleep_df.drop(columns=['job', 'shift'])

    nurse_sleep_corr_df = nurse_sleep_df.corr(method='spearman')
    day_nurse_sleep_corr_df = day_nurse_sleep_df.corr(method='spearman')
    night_nurse_sleep_corr_df = night_nurse_sleep_df.corr(method='spearman')
    
    print()


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)