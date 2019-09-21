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

compare_cols = ['itp_igtb', 'irb_igtb', 'ocb_igtb', 'stai_igtb', 'pos_af_igtb', 'neg_af_igtb', 'Pain', 'energy_fatigue']


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


def print_overall_comparison(first_df):
    print('\n')
    print('\multicolumn{1}{l}{Overall} &')
    for i, col in enumerate(compare_cols):
        mean = np.mean(first_df[col].dropna())
        std = np.std(first_df[col].dropna())
        if mean > 10:
            mean = np.round(mean, 1)
            std = np.round(std, 1)
        else:
            mean = np.round(mean, 2)
            std = np.round(std, 2)
        
        if i == len(compare_cols) - 1:
            print('\multicolumn{1}{|c}{$%s$($%s$)} \\rule{0pt}{2.25ex} \\\\' % (str(mean), str(std)))
        else:
            print('\multicolumn{1}{|c}{$%s$($%s$)} &' % (str(mean), str(std)))
    

def print_comparison(first_df, second_df, cond_list):
    
    print('\n')
    print('\multicolumn{1}{l}{\hspace{0.5cm}' + cond_list[0] + '} &')
    for i, col in enumerate(compare_cols):
        mean = np.mean(first_df[col].dropna())
        std = np.std(first_df[col].dropna())
        if mean > 10:
            mean = np.round(mean, 1)
            std = np.round(std, 1)
        else:
            mean = np.round(mean, 2)
            std = np.round(std, 2)
        
        if i == len(compare_cols) - 1:
            print('\multicolumn{1}{|c}{$%s$($%s$)} \\rule{0pt}{2.25ex} \\\\' % (str(mean), str(std)))
        else:
            print('\multicolumn{1}{|c}{$%s$($%s$)} &' % (str(mean), str(std)))
        # print('\multicolumn{1}{|c}{$%s$} &' % (str(mean)))
        # print('\multicolumn{1}{|c}{$%s$} &' % (str(std)))
        
    print()
    print('\multicolumn{1}{l}{\hspace{0.5cm}' + cond_list[1] + '} &')
    for i, col in enumerate(compare_cols):
        mean = np.mean(second_df[col].dropna())
        std = np.std(second_df[col].dropna())
        if mean > 10:
            mean = np.round(mean, 1)
            std = np.round(std, 1)
        else:
            mean = np.round(mean, 2)
            std = np.round(std, 2)

        if i == len(compare_cols) - 1:
            print('\multicolumn{1}{|c}{$%s$($%s$)} \\rule{0pt}{2.25ex} \\\\' % (str(mean), str(std)))
        else:
            print('\multicolumn{1}{|c}{$%s$($%s$)} &' % (str(mean), str(std)))
        # print('\multicolumn{1}{|c}{$%s$} &' % (str(std)))

    print()

    print('\multicolumn{1}{l}{\hspace{0.5cm}' + 'p-value' + '} &')
    for i, col in enumerate(compare_cols):
        stat, p = stats.ks_2samp(first_df[col].dropna(), second_df[col].dropna())

        if i == len(compare_cols) - 1:
            if p < 0.001:
                print('\multicolumn{1}{|c}{\mathbf{<0.001}} \\rule{0pt}{2.25ex} \\\\ \\hline')
            elif p < 0.05:
                print('\multicolumn{1}{|c}{$\mathbf{%.3f}$} \\rule{0pt}{2.25ex} \\\\ \\hline' % (p))
            else:
                print('\multicolumn{1}{|c}{$%.3f$} \\rule{0pt}{2.25ex} \\\\ \\hline' % (p))
        else:
            if p < 0.001:
                print('\multicolumn{1}{|c}{\mathbf{<0.001}} &')
            elif p < 0.05:
                print('\multicolumn{1}{|c}{$\mathbf{%.3f}$} &' % (p))
            else:
                print('\multicolumn{1}{|c}{$%.3f$} &' % (p))
            
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

    # compare_method_list = ['shift', 'language', 'supervise', 'gender', 'icu', 'job', 'children', 'age', 'employer_duration']
    compare_method_list = ['supervise', 'icu', 'children', 'age']

    
    if os.path.exists(os.path.join('daily_realizd_fitbit_mr.csv.gz')) is True:
        final_df = pd.read_csv(os.path.join('daily_realizd_fitbit_mr.csv.gz'), index_col=0)
        nurse_df = final_df.loc[final_df['job'] == 'nurse']

        print_overall_comparison(nurse_df)

        print(len(nurse_df))
        
        for compare_method in compare_method_list:
            if compare_method == 'supervise':
                first_data_df = nurse_df.loc[nurse_df['supervise'] == 'Supervise']
                second_data_df = nurse_df.loc[nurse_df['supervise'] == 'Non-Supervise']
                cond_list = ['Nurse Manager', 'Non-nurse Manager']
            elif compare_method == 'children':
                first_data_df = nurse_df.loc[final_df['children'] > 0]
                second_data_df = nurse_df.loc[final_df['children'] == 0]
                cond_list = ['Have Child', 'Don\'t Have Child']
            elif compare_method == 'age':
                first_data_df = nurse_df.loc[nurse_df['age'] > np.nanmedian(nurse_df['age'])]
                second_data_df = nurse_df.loc[nurse_df['age'] <= np.nanmedian(nurse_df['age'])]
                cond_list = ['Above Median Age', 'Below Median Age']
            else:
                first_data_df = nurse_df.loc[nurse_df['icu'] == 'icu']
                second_data_df = nurse_df.loc[nurse_df['icu'] == 'non_icu']
                cond_list = ['ICU Unit', 'Non-ICU Unit']

            print_comparison(first_data_df, second_data_df, cond_list=cond_list)


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)