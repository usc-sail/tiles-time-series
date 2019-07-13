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
from scipy import stats
from scipy.stats import skew


def print_psqi_igtb(day_df, night_df, cols):
    
    print('\multicolumn{1}{l}{\\textbf{Total PSQI score}} &')
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df['psqi_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df['psqi_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df['psqi_igtb'].dropna())))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df['psqi_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df['psqi_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df['psqi_igtb'].dropna())))

    stat, p = stats.ks_2samp(day_df['psqi_igtb'].dropna(), night_df['psqi_igtb'].dropna())

    if p < 0.001:
        print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
    elif p < 0.05:
        print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
    else:
        print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
    print('\n')
    
    for col in cols:
        
        if 'dysfunction' in col:
            print_col = 'Daytime dysfunction'
        elif 'med' in col:
            print_col = 'Sleep medication'
        elif 'sub' in col:
            print_col = 'Subjective sleep quality'
        elif 'tency' in col:
            print_col = 'Sleep latency'
        elif 'fficiency' in col:
            print_col = 'Sleep efficiency'
        elif 'disturbance' in col:
            print_col = 'Sleep disturbance'
        else:
            print_col = 'Sleep duration'
        
        print('\multicolumn{1}{l}{\hspace{0.5cm}' + print_col + '} &')
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col])))

        stat, p = stats.ks_2samp(day_df[col].dropna(), night_df[col].dropna())
        
        if p < 0.001:
            print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
        elif p < 0.05:
            print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
        else:
            print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
        print('\n')


def print_personality_igtb(day_df, night_df, cols):
    
    print('\multicolumn{1}{l}{\\textbf{Personality}} & & & & & & \\rule{0pt}{3ex} \\\\')
    print('\n')
    
    for col in cols:
        
        if 'con_igtb' in col:
            print_col = 'Conscientiousness'
        elif 'ext_igtb' in col:
            print_col = 'Extraversion'
        elif 'agr_igtb' in col:
            print_col = 'Agreeableness'
        elif 'ope_igtb' in col:
            print_col = 'Openness to Experience'
        else:
            print_col = 'Neuroticism'
        
        print('\multicolumn{1}{l}{\hspace{0.5cm}' + print_col + '} &')
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col])))
        
        stat, p = stats.ks_2samp(day_df[col].dropna(), night_df[col].dropna())
        
        if p < 0.001:
            print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
        elif p < 0.05:
            print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
        else:
            print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
        
        print()
        

def print_affect_igtb(day_df, night_df, cols):
    print('\multicolumn{1}{l}{\\textbf{Affect}} & & & & & & \\rule{0pt}{3ex} \\\\')
    print('\n')
    
    for col in cols:
        
        if 'neg_af_igtb' in col:
            print_col = 'Negative Affect'
        else:
            print_col = 'Positive Affect'
        
        print('\multicolumn{1}{l}{\hspace{0.5cm}' + print_col + '} &')
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col])))
        
        stat, p = stats.ks_2samp(day_df[col].dropna(), night_df[col].dropna())
        
        if p < 0.001:
            print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
        elif p < 0.05:
            print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
        else:
            print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
        
        print()


def print_anxiety_igtb(day_df, night_df):
    print('\multicolumn{1}{l}{\\textbf{Anxiety}} &')
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df['stai_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df['stai_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df['stai_igtb'].dropna())))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df['stai_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df['stai_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df['stai_igtb'].dropna())))
    
    stat, p = stats.ks_2samp(day_df['stai_igtb'].dropna(), night_df['stai_igtb'].dropna())
    
    if p < 0.001:
        print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
    elif p < 0.05:
        print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
    else:
        print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
    print()


def print_audit_igtb(day_df, night_df):
    print('\multicolumn{1}{l}{\\textbf{AUDIT score}} &')
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df['audit_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df['audit_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df['audit_igtb'].dropna())))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df['audit_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df['audit_igtb'])))
    print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df['audit_igtb'].dropna())))
    
    stat, p = stats.ks_2samp(day_df['audit_igtb'].dropna(), night_df['audit_igtb'].dropna())
    
    if p < 0.001:
        print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
    elif p < 0.05:
        print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
    else:
        print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
    print()


def print_cognition_igtb(day_df, night_df, cols):
    print('\multicolumn{1}{l}{\\textbf{Cognition}} & & & & & & \\rule{0pt}{3ex} \\\\')
    print('\n')
    
    for col in cols:
        
        if 'shipley_abs_igtb' in col:
            print_col = 'Shipley Abstraction'
        else:
            print_col = 'Shipley Vocabulary'
        
        print('\multicolumn{1}{l}{\hspace{0.5cm}' + print_col + '} &')
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col])))
        
        stat, p = stats.ks_2samp(day_df[col].dropna(), night_df[col].dropna())
        
        if p < 0.001:
            print('\multicolumn{1}{c}{\mathbf{<0.001}} \\rule{0pt}{3ex} \\\\')
        elif p < 0.05:
            print('\multicolumn{1}{c}{$\mathbf{%.3f}$} \\rule{0pt}{3ex} \\\\' % (p))
        else:
            print('\multicolumn{1}{c}{$%.3f$} \\rule{0pt}{3ex} \\\\' % (p))
        
        print()
        

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
    
    if os.path.exists(os.path.join(data_config.phone_usage_path, 'daily_summary.csv.gz')) is True:
        all_df = pd.read_csv(os.path.join(data_config.phone_usage_path, 'daily_summary.csv.gz'), index_col=0)
        
        for participant_id in list(all_df.index):
            nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
            primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            job_str = 'nurse' if nurse == 1 else 'non_nurse'
            shift_str = 'day' if shift == 'Day shift' else 'night'
            uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
            
            all_df.loc[participant_id, 'job'] = job_str
            all_df.loc[participant_id, 'shift'] = shift_str
            
            for col in list(igtb_df.columns):
                all_df.loc[participant_id, col] = igtb_df.loc[uid, col]
                
            for col in list(psqi_raw_igtb.columns):
                all_df.loc[participant_id, col] = psqi_raw_igtb.loc[uid, col]

        # shift_pre-study
        all_df = all_df.loc[all_df['job'] == 'nurse']
        day_nurse_df = all_df.loc[all_df['Shift'] == 'Day shift']
        night_nurse_df = all_df.loc[all_df['Shift'] == 'Night shift']

        big5_col = ['neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb']
        for col in big5_col:
            print(col)
            print('Number of valid participant: day: %i; night: %i\n' % (len(day_nurse_df), len(night_nurse_df)))
        
            # Print
            print('Total: mean = %.2f, std = %.2f, range is %.3f - %.3f' % (np.mean(all_df[col]), np.std(all_df[col]), np.min(all_df[col]), np.max(all_df[col])))
            print('Day shift: mean = %.2f, std = %.2f' % (np.mean(day_nurse_df[col]), np.std(day_nurse_df[col])))
            print('Day shift: range is %.3f - %.3f' % (np.min(day_nurse_df[col]), np.max(day_nurse_df[col])))
            print('Day shift: skew = %.3f' % (skew(day_nurse_df[col])))

            print('Night shift: mean = %.2f, std = %.2f' % (np.mean(night_nurse_df[col]), np.std(night_nurse_df[col])))
            print('Night shift: range is %.3f - %.3f' % (np.min(night_nurse_df[col]), np.max(night_nurse_df[col])))
            print('Night shift: skew = %.3f' % (skew(night_nurse_df[col])))
            # K-S test
            stat, p = stats.ks_2samp(day_nurse_df[col].dropna(), night_nurse_df[col].dropna())
            print('K-S test for %s' % col)
            print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))

        affect_col = ['stai_igtb', 'pos_af_igtb', 'neg_af_igtb']
        for col in affect_col:
            print(col)
            print('Number of valid participant: day: %i; night: %i\n' % (len(day_nurse_df), len(night_nurse_df)))

            print('Total: mean = %.2f, std = %.2f, range is %.3f - %.3f' % (np.mean(all_df[col]), np.std(all_df[col]), np.min(all_df[col]), np.max(all_df[col])))
            print('Day shift: mean = %.2f, std = %.2f' % (np.mean(day_nurse_df[col]), np.std(day_nurse_df[col])))
            print('Day shift: range is %.3f - %.3f' % (np.min(day_nurse_df[col]), np.max(day_nurse_df[col])))
            print('Day shift: skew = %.3f' % (skew(day_nurse_df[col])))

            print('Night shift: mean = %.2f, std = %.2f' % (np.mean(night_nurse_df[col]), np.std(night_nurse_df[col])))
            print('Night shift: range is %.3f - %.3f' % (np.min(night_nurse_df[col]), np.max(night_nurse_df[col])))
            print('Night shift: skew = %.3f' % (skew(night_nurse_df[col])))
            # K-S test
            stat, p = stats.ks_2samp(day_nurse_df[col].dropna(), night_nurse_df[col].dropna())
            print('K-S test for %s' % col)
            print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))

        psqi_col = ['audit_igtb', 'psqi_igtb']
        psqi_col = psqi_col + list(psqi_raw_igtb.columns)
        for col in psqi_col:
            print(col)
            print('Number of valid participant: day: %i; night: %i\n' % (len(day_nurse_df), len(night_nurse_df)))
    
            # Print
            print('Total: mean = %.2f, std = %.2f, range is %.3f - %.3f' % (np.mean(all_df[col]), np.std(all_df[col]), np.min(all_df[col]), np.max(all_df[col])))
            print('Day shift: mean = %.2f, std = %.2f' % (np.mean(day_nurse_df[col]), np.std(day_nurse_df[col])))
            print('Day shift: range is %.3f - %.3f' % (np.min(day_nurse_df[col]), np.max(day_nurse_df[col])))
            print('Day shift: skew = %.3f' % (skew(day_nurse_df[col])))

            print('Night shift: mean = %.2f, std = %.2f' % (np.mean(night_nurse_df[col]), np.std(night_nurse_df[col])))
            print('Night shift: range is %.3f - %.3f' % (np.min(night_nurse_df[col]), np.max(night_nurse_df[col])))
            print('Night shift: skew = %.3f' % (skew(night_nurse_df[col])))
            
            if col == 'audit_igtb':
                print('Day above 8: %d' % (len(day_nurse_df.loc[day_nurse_df[col] >= 8])))
                print('Night above 8: %d' % (len(night_nurse_df.loc[night_nurse_df[col] >= 8])))

            # K-S test
            stat, p = stats.ks_2samp(day_nurse_df[col].dropna(), night_nurse_df[col].dropna())
            print('K-S test for %s' % col)
            print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))

    # print_psqi_igtb(day_nurse_df, night_nurse_df, list(psqi_raw_igtb.columns))
    # print_personality_igtb(day_nurse_df, night_nurse_df, big5_col)
    # print_affect_igtb(day_nurse_df, night_nurse_df, ['pos_af_igtb', 'neg_af_igtb'])
    # print_anxiety_igtb(day_nurse_df, night_nurse_df)
    # print_audit_igtb(day_nurse_df, night_nurse_df)
    # print_cognition_igtb(day_nurse_df, night_nurse_df, ['shipley_abs_igtb', 'shipley_voc_igtb'])
    

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)