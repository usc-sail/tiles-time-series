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


icu_list = ['4 South', '5 North', '5 South ICU', '5 West', '7 West', '7 East', '7 South', '8 West']


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
    igtb_raw = load_data_basic.read_IGTB_Raw(tiles_data_path)
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    if os.path.exists(os.path.join('daily_realizd_fitbit_mr.csv.gz')) is True:
        final_df = pd.read_csv(os.path.join('daily_realizd_fitbit_mr.csv.gz'), index_col=0)
        nurse_df = final_df.loc[final_df['job'] == 'nurse']

        print(len(nurse_df))

        print('Number of participant who take nurseyear survey %d' % (len(nurse_df.dropna(subset=['employer_duration']))))
        print('Average nurse year of participant %.3f std: %.3f' % (np.mean(nurse_df['employer_duration'].dropna()), np.std(nurse_df['employer_duration'].dropna())))
        print('Nurse year range of participants %d - %d' % (np.min(nurse_df['employer_duration']), np.max(nurse_df['employer_duration'])))

        print('\n')

        print('Number of participant who take age survey %d' % (len(nurse_df.dropna(subset=['age']))))
        print('Average age of participant %.3f std: %.3f' % (np.mean(nurse_df['age'].dropna()), np.std(nurse_df['age'])))
        print('Age range of participants %d - %d' % (np.min(nurse_df['age']), np.max(nurse_df['age'])))

        print('\n')

        # 20 - 29
        print('Number of participants in range between 20 - 29: %d, percentage: %.3f' % (len(nurse_df[(nurse_df['age'] >= 20) & (nurse_df['age'] < 30)]),
        len(nurse_df[(nurse_df['age'] >= 20) & (nurse_df['age'] < 30)]) / len(nurse_df.dropna(subset=['age'])) * 100))
        # 30 - 39
        print('Number of participants in range between 30 - 39: %d, percentage: %.3f' % (len(nurse_df[(nurse_df['age'] >= 30) & (nurse_df['age'] < 40)]),
        len(nurse_df[(nurse_df['age'] >= 30) & (nurse_df['age'] < 40)]) / len(nurse_df.dropna(subset=['age'])) * 100))

        # Above 40
        print('Number of participants in range above 40: %d, %.3f\n' % (len(nurse_df[(nurse_df['age'] >= 40) & (nurse_df['age'] < 80)]),
        len(nurse_df[(nurse_df['age'] >= 40) & (nurse_df['age'] < 80)]) / len(nurse_df.dropna(subset=['age']))))

        # Nurse specific
        print('Number of nurses who take Shift survey %d' % (len(nurse_df.dropna(subset=['shift']))))
        print('Number of day shift nurse %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['shift'] == 'day']), len(nurse_df.loc[nurse_df['shift'] == 'day']) / len(nurse_df) * 100))
        print('Number of night shift nurse %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['shift'] == 'night']),
        len(nurse_df.loc[nurse_df['shift'] == 'night']) / len(nurse_df) * 100))

        print('\n')
        print('Number of nurses who take PrimaryUnit survey %d' % (len(nurse_df.dropna(subset=['icu']))))
        print('Number of ICU nurse %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['icu'] == 'icu']), len(nurse_df.loc[nurse_df['icu'] == 'icu']) / len(nurse_df) * 100))
        print('Number of Non-ICU shift nurse %d, percentage: %.3f\n' % (len(nurse_df.loc[nurse_df['icu'] == 'non_icu']), len(nurse_df.loc[nurse_df['icu'] == 'non_icu']) / len(nurse_df) * 100))

        print('Number of participant who take gender survey %d' % (len(nurse_df.dropna(subset=['gender']))))
        print('Number of male participant %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['gender'] == 'Male']), len(nurse_df.loc[nurse_df['gender'] == 'Male']) / len(nurse_df.dropna(subset=['gender'])) * 100))
        print('Number of female participant %d, percentage: %.3f\n' % (len(nurse_df.loc[nurse_df['gender'] == 'Female']), len(nurse_df.loc[nurse_df['gender'] == 'Female']) / len(nurse_df.dropna(subset=['gender'])) * 100))

        '''
        print('Number of participant who take education survey %d' % (len(nurse_df.dropna(subset=['education']))))
        print('Number of participant who attend college %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['education'] > 2]),
        len(nurse_df.loc[nurse_df['education'] > 2]) / len(nurse_df.dropna(subset=['education'])) * 100))

        print('Number of participant who attend gradute school %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['education'] > 4]),
        len(nurse_df.loc[nurse_df['education'] > 4]) / len(nurse_df.dropna(subset=['education'])) * 100))

        print('Number of participant who attend gradute school %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['education'] > 6]),
        len(nurse_df.loc[nurse_df['education'] > 6]) / len(nurse_df.dropna(subset=['education'])) * 100))
        '''
        
        # Supervise
        print('Number of participant who take supervise survey %d' % (len(nurse_df.dropna(subset=['supervise']))))
        print('Number of participant who supervise others %d, percentage: %.3f\n' % (len(nurse_df.loc[nurse_df['supervise'] == 'Supervise']),
        len(nurse_df.loc[nurse_df['supervise'] == 'Supervise']) / len(nurse_df.dropna(subset=['supervise'])) * 100))

        # Child
        print('Number of participant who take have children survey %d' % (len(nurse_df.dropna(subset=['children']))))
        print('Number of participant who have children %d, percentage: %.3f' % (len(nurse_df.loc[nurse_df['children'] > 0]), len(nurse_df.loc[nurse_df['children'] > 0]) / len(nurse_df.dropna(subset=['children'])) * 100))


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)