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
from scipy import stats
from datetime import timedelta
import collections


def plot_sensor_data(data_df, chi_data_config, participant_id, file_name):
    """ Plot sensor data based on the stream type

    Params:
    ax - axis to plot
    data_df - whole data dataframe for one stream

    Returns:
    """
    ###########################################################
    # Plot
    ###########################################################
    f, ax = plt.subplots(3, figsize=(16, 6))
    color_list = ['green', 'red', 'blue']
    
    for i in range(3):
        ax[i].plot(np.arange(len(data_df)), np.array(data_df)[:, i], label=list(data_df.columns)[i], color=color_list[i])
        ax[i].legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True, fontsize=14)
    if os.path.exists(os.path.join(chi_data_config.save_path, participant_id)) is False:
        os.mkdir(os.path.join(chi_data_config.save_path, participant_id))
    plt.savefig(os.path.join(chi_data_config.save_path, participant_id, file_name + '.png'))
    plt.close()
    

def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)

    chi_data_config = config.Config()
    chi_data_config.readChiConfigFile(config_path)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    load_data_path.load_chi_preprocess_path(chi_data_config, process_data_path)
    
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

    num_point_per_day = chi_data_config.num_point_per_day
    offset = 1440 / num_point_per_day
    window = chi_data_config.window
    
    for idx, participant_id in enumerate(top_participant_id_list[:]):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
        job_str = 'nurse' if nurse == 1 else 'non_nurse'
        shift_str = 'day' if shift == 'Day shift' else 'night'

        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        
        if os.path.exists(os.path.join(chi_data_config.save_path, participant_id + '.pkl')) is False:
            continue
        pkl_file = open(os.path.join(chi_data_config.save_path, participant_id + '.pkl'), 'rb')
        participant_id_shift_dict = pickle.load(pkl_file)
        
        if job_str == 'nurse':
            plot_sensor_data(participant_id_shift_dict['days_off_work_daily_data'], chi_data_config, participant_id, file_name='days_off_work_daily_data')
            plot_sensor_data(participant_id_shift_dict['days_at_work_daily_data'], chi_data_config, participant_id, file_name='days_at_work_daily_data')
            plot_sensor_data(participant_id_shift_dict['daily_data'], chi_data_config, participant_id, file_name='daily_data')

        
if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)