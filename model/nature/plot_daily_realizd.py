"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import spearmanr

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

import seaborn as sns
sns.set(style="whitegrid")

row_cols = ['Total Usage Time', 'Mean Session Length', 'Mean Inter-session Time',
            'Session Frequency', 'Session Frequency (<1min)', 'Session Frequency (>1min)']

col_cols = ['Shipley Abs.', 'Shipley Voc.',
            'Anxiety', 'Pos. Affect', 'Neg. Affect',
            'Neuroticism', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Openness', 'Total PSQI']


def plot_corr(data_df, ax):
    coef, p = spearmanr(np.array(data_df))
    coef_df = pd.DataFrame(coef, index=list(data_df.columns), columns=list(data_df.columns))
    p_value_df = pd.DataFrame(p, index=list(data_df.columns), columns=list(data_df.columns))
    coef_df = coef_df.loc[row_cols, col_cols]
    p_value_df = p_value_df.loc[row_cols, col_cols]
    sns.heatmap(coef_df, ax=ax, vmin=-0.6, vmax=0.6, cbar=False, cmap='bwr', annot_kws={"weight": "bold", "size": 15})
    
    for y in range(p_value_df.shape[0]):
        for x in range(p_value_df.shape[1]):
            color = 'black' if np.abs(np.array(coef_df)[y, x]) < 0.3 else 'white'
            p_text = '*' if np.array(p_value_df)[y, x] < 0.05 else ''
            p_text = p_text + '*' if np.array(p_value_df)[y, x] < 0.01 else p_text
            coef_text = str(np.array(coef_df)[y, x])[:5] if np.array(coef_df)[y, x] < 0 else str(np.array(coef_df)[y, x])[:4]
            
            ax.text(x + 0.5, y + 0.5, coef_text + p_text, fontsize=14, fontweight='bold',
                    horizontalalignment='center', verticalalignment='center', color=color)


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
            
            all_df.loc[participant_id, 'job'] = job_str
            all_df.loc[participant_id, 'shift'] = shift_str
        
        nurse_df = all_df.loc[all_df['job'] == 'nurse']
        day_nurse_df = nurse_df.loc[nurse_df['shift'] == 'day']
        night_nurse_df = nurse_df.loc[nurse_df['shift'] == 'night']
        
        final_cols = []
        for col in list(nurse_df.columns):
            if 'work' not in col and 'off' not in col:
                final_cols.append(col)

        ana_igtb_cols = ['shipley_abs_igtb', 'shipley_voc_igtb', 'stai_igtb', 'pos_af_igtb', 'neg_af_igtb',
                         'neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb', 'psqi_igtb']

        ana_igtb_dict = {'shipley_abs_igtb': 'Shipley Abs.', 'shipley_voc_igtb': 'Shipley Voc.',
                         'stai_igtb': 'Anxiety', 'psqi_igtb': 'Total PSQI',
                         'pos_af_igtb': 'Pos. Affect', 'neg_af_igtb': 'Neg. Affect',
                         'neu_igtb': 'Neuroticism', 'con_igtb': 'Conscientiousness',
                         'ext_igtb': 'Extraversion', 'agr_igtb': 'Agreeableness', 'ope_igtb': 'Openness'}
        
        plot_df = pd.DataFrame()
        for i in range(len(nurse_df)):
            participant_df = nurse_df.iloc[i, :]
            participant_id = list(nurse_df.index)[i]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            row_df = pd.DataFrame(index=[participant_id])
            row_df['Total Usage Time'] = participant_df['total_time_mean']
            row_df['Mean Session Length'] = participant_df['mean_time_mean']
            row_df['Session Frequency'] = participant_df['frequency_mean']
            row_df['Mean Inter-session Time'] = participant_df['mean_inter_mean']
            row_df['Session Frequency (<1min)'] = participant_df['less_than_1min_mean']
            row_df['Session Frequency (>1min)'] = participant_df['above_1min_mean']
            row_df['Data Type'] = 'Combined'
            row_df['Shift Type'] = shift

            for col in ana_igtb_cols:
                row_df[ana_igtb_dict[col]] = participant_df[col]

            plot_df = plot_df.append(row_df)
            
            row_df = pd.DataFrame(index=[participant_id + 'work'])
            row_df['Total Usage Time'] = participant_df['work_total_time_mean']
            row_df['Mean Session Length'] = participant_df['work_mean_time_mean']
            row_df['Session Frequency'] = participant_df['work_frequency_mean']
            row_df['Mean Inter-session Time'] = participant_df['work_mean_inter_mean']
            row_df['Session Frequency (<1min)'] = participant_df['work_less_than_1min_mean']
            row_df['Session Frequency (>1min)'] = participant_df['work_above_1min_mean']
            row_df['Data Type'] = 'Workday'
            row_df['Shift Type'] = shift
            
            for col in ana_igtb_cols:
                row_df[ana_igtb_dict[col]] = participant_df[col]
            
            plot_df = plot_df.append(row_df)
            
            row_df = pd.DataFrame(index=[participant_id + 'off'])
            row_df['Total Usage Time'] = participant_df['off_total_time_mean']
            row_df['Mean Session Length'] = participant_df['off_mean_time_mean']
            row_df['Session Frequency'] = participant_df['off_frequency_mean']
            row_df['Mean Inter-session Time'] = participant_df['off_mean_inter_mean']
            row_df['Session Frequency (<1min)'] = participant_df['off_less_than_1min_mean']
            row_df['Session Frequency (>1min)'] = participant_df['off_above_1min_mean']
            row_df['Data Type'] = 'Off-day'
            row_df['Shift Type'] = shift

            for col in ana_igtb_cols:
                row_df[ana_igtb_dict[col]] = participant_df[col]

            plot_df = plot_df.append(row_df)
        
        day_df = plot_df.loc[plot_df['Shift Type'] == 'Day shift']
        night_df = plot_df.loc[plot_df['Shift Type'] == 'Night shift']

        work_day_df = day_df.loc[day_df['Data Type'] == 'Workday']
        work_night_df = night_df.loc[night_df['Data Type'] == 'Workday']
        off_day_df = day_df.loc[day_df['Data Type'] == 'Off-day']
        off_night_df = night_df.loc[night_df['Data Type'] == 'Off-day']

        fig = plt.figure(figsize=(20, 8))
        axes = fig.subplots(nrows=2, ncols=2)
        # cbar_ax = fig.add_axes([1.015,0.13, 0.015, 0.8])

        plot_corr(work_day_df, axes[0][0])
        plot_corr(off_day_df, axes[0][1])
        plot_corr(work_night_df, axes[1][0])
        plot_corr(off_night_df, axes[1][1])
        
        for i in range(2):
            axes[i][1].set_yticklabels('')
            axes[i][0].set_yticklabels(row_cols, fontdict={'fontweight': 'bold', 'fontsize': 16})
            for j in range(2):
                axes[i][j].tick_params(axis="x", labelsize=1)
                axes[0][j].set_xticklabels('')
                axes[1][j].set_xticklabels(col_cols, fontdict={'fontweight': 'bold', 'fontsize': 16})

        axes[0][0].set_title('Day shift (Workday)', fontweight='bold', fontsize=16)
        axes[0][1].set_title('Day shift (Non-Workday)', fontweight='bold', fontsize=16)
        
        axes[1][0].set_title('Night shift (Workday)', fontweight='bold', fontsize=16)
        axes[1][1].set_title('Night shift (Non-Workday)', fontweight='bold', fontsize=16)

        if os.path.exists(os.path.join('plot')) is False:
            os.mkdir(os.path.join('plot'))
        if os.path.exists(os.path.join('plot', 'daily')) is False:
            os.mkdir(os.path.join('plot', 'daily'))

        plt.tight_layout()

        cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        cb = plt.colorbar(axes[0][0].get_children()[0], cax=cax, **kw)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(16)

        # plt.colorbar(cbar_ax)
        plt.savefig(os.path.join('plot', 'daily', 'corr'), dpi=300)
        plt.close()
        

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)