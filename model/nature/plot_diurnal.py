"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import seaborn as sns

sns.set(style="whitegrid")

total_row_cols = ['Total Usage Time(night)', 'Total Usage Time(morning)', 'Total Usage Time(afternoon)', 'Total Usage Time(evening)']
mean_row_cols = ['Mean Session Time(night)', 'Mean Session Time(morning)', 'Mean Session Time(afternoon)', 'Mean Session Time(evening)']
session_row_cols = ['Session Frequency(night)', 'Session Frequency(morning)', 'Session Frequency(afternoon)', 'Session Frequency(evening)']

col_cols = ['Shipley Abs.', 'Shipley Voc.',
            'Anxiety', 'Pos. Affect', 'Neg. Affect',
            'Neuroticism', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Openness', 'Total PSQI']


def plot_corr(data_df, ax, row_cols, col_cols):
    heat_fontsize = 19
    
    coef, p = spearmanr(np.array(data_df))
    coef_df = pd.DataFrame(coef, index=list(data_df.columns), columns=list(data_df.columns))
    p_value_df = pd.DataFrame(p, index=list(data_df.columns), columns=list(data_df.columns))
    coef_df = coef_df.loc[row_cols, col_cols]
    p_value_df = p_value_df.loc[row_cols, col_cols]
    sns.heatmap(coef_df, ax=ax, vmin=-0.6, vmax=0.6, cbar=False, cmap='bwr')
    
    for y in range(p_value_df.shape[0]):
        for x in range(p_value_df.shape[1]):
            color = 'black' if np.abs(np.array(coef_df)[y, x]) < 0.3 else 'white'
            p_text = '*' if np.array(p_value_df)[y, x] < 0.05 else ''
            p_text = p_text + '*' if np.array(p_value_df)[y, x] < 0.01 else p_text
            coef_text = str(np.array(coef_df)[y, x])[:5] if np.array(coef_df)[y, x] < 0 else str(np.array(coef_df)[y, x])[:4]
            
            ax.text(x + 0.5, y + 0.5, coef_text + p_text, fontsize=heat_fontsize, fontweight='bold',
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

    ana_igtb_cols = ['shipley_abs_igtb', 'shipley_voc_igtb', 'stai_igtb', 'pos_af_igtb', 'neg_af_igtb',
                     'neu_igtb', 'con_igtb', 'ext_igtb', 'agr_igtb', 'ope_igtb', 'psqi_igtb']

    ana_igtb_dict = {'shipley_abs_igtb': 'Shipley Abs.', 'shipley_voc_igtb': 'Shipley Voc.',
                     'stai_igtb': 'Anxiety', 'pos_af_igtb': 'Pos. Affect', 'neg_af_igtb': 'Neg. Affect',
                     'neu_igtb': 'Neuroticism', 'con_igtb': 'Conscientiousness', 'psqi_igtb': 'Total PSQI',
                     'ext_igtb': 'Extraversion', 'agr_igtb': 'Agreeableness', 'ope_igtb': 'Openness'}

    psqi_igtb_dict = {'psqi_igtb': 'Total PSQI',
                      'psqi_subject_quality': 'Subjective quality',
                      'psqi_sleep_latency': 'Sleep latency',
                      'psqi_sleep_duration': 'Sleep duration',
                      'psqi_sleep_efficiency': 'Sleep efficiency',
                      'psqi_sleep_disturbance': 'Sleep Disturbance',
                      'psqi_sleep_medication': 'Sleep medication',
                      'psqi_day_dysfunction': 'Day dysfunction'}
    
    if os.path.exists(os.path.join(data_config.phone_usage_path, 'dianual_summary.csv.gz')) is True:
        all_df = pd.read_csv(os.path.join(data_config.phone_usage_path, 'dianual_summary.csv.gz'), index_col=0)
        
        for participant_id in list(all_df.index):
            nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
            primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            job_str = 'nurse' if nurse == 1 else 'non_nurse'
            shift_str = 'day' if shift == 'Day shift' else 'night'
            
            all_df.loc[participant_id, 'job'] = job_str
            all_df.loc[participant_id, 'shift'] = shift_str
        
        nurse_df = all_df.loc[all_df['job'] == 'nurse']
        corr_df = pd.DataFrame()
        
        for i in range(len(nurse_df)):
            participant_df = nurse_df.iloc[i, :]
            participant_id = list(nurse_df.index)[i]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]

            row_df = pd.DataFrame(index=[participant_id + 'work'])
            for i, diurnal_type in enumerate(['night', 'morning', 'afternoon', 'evening']):
                if diurnal_type == 'night':
                    diurnal_str = 'Night'
                elif diurnal_type == 'morning':
                    diurnal_str = 'Morning'
                elif diurnal_type == 'afternoon':
                    diurnal_str = 'Afternoon'
                else:
                    diurnal_str = 'Evening'
                row_df['Total Usage Time' + '(' + diurnal_type + ')'] = participant_df['work_' + diurnal_type + '_total_time_mean']
                row_df['Session Frequency' + '(' + diurnal_type + ')'] = participant_df['work_' + diurnal_type + '_frequency_mean']
                row_df['Mean Session Time' + '(' + diurnal_type + ')'] = participant_df['work_' + diurnal_type + '_mean_time_mean']

            for col in ana_igtb_cols:
                row_df[ana_igtb_dict[col]] = participant_df[col]
            
            row_df['Data Type'] = 'Workday'
            row_df['Shift Type'] = shift
            corr_df = corr_df.append(row_df)

            row_df = pd.DataFrame(index=[participant_id + 'work'])
            for i, diurnal_type in enumerate(['night', 'morning', 'afternoon', 'evening']):
                if diurnal_type == 'night':
                    diurnal_str = 'Night'
                elif diurnal_type == 'morning':
                    diurnal_str = 'Morning'
                elif diurnal_type == 'afternoon':
                    diurnal_str = 'Afternoon'
                else:
                    diurnal_str = 'Evening'
                row_df['Total Usage Time' + '(' + diurnal_type + ')'] = participant_df['off_' + diurnal_type + '_total_time_mean']
                row_df['Session Frequency' + '(' + diurnal_type + ')'] = participant_df['off_' + diurnal_type + '_frequency_mean']
                row_df['Mean Session Time' + '(' + diurnal_type + ')'] = participant_df['off_' + diurnal_type + '_mean_time_mean']

            for col in ana_igtb_cols:
                row_df[ana_igtb_dict[col]] = participant_df[col]
            
            row_df['Data Type'] = 'Off-day'
            row_df['Shift Type'] = shift
            corr_df = corr_df.append(row_df)
                
        day_corr_df = corr_df.loc[corr_df['Shift Type'] == 'Day shift']
        night_corr_df = corr_df.loc[corr_df['Shift Type'] == 'Night shift']

        work_day_corr_df = day_corr_df.loc[day_corr_df['Data Type'] == 'Workday']
        work_night_corr_df = night_corr_df.loc[night_corr_df['Data Type'] == 'Workday']
        off_day_corr_df = day_corr_df.loc[day_corr_df['Data Type'] == 'Off-day']
        off_night_corr_df = night_corr_df.loc[night_corr_df['Data Type'] == 'Off-day']
        
        fig = plt.figure(figsize=(24, 16))
        axes = fig.subplots(nrows=6, ncols=2)

        plot_corr(work_day_corr_df, axes[0][0], total_row_cols, col_cols)
        plot_corr(work_night_corr_df, axes[0][1], total_row_cols, col_cols)
        plot_corr(off_day_corr_df, axes[1][0], total_row_cols, col_cols)
        plot_corr(off_night_corr_df, axes[1][1], total_row_cols, col_cols)

        plot_corr(work_day_corr_df, axes[2][0], mean_row_cols, col_cols)
        plot_corr(work_night_corr_df, axes[2][1], mean_row_cols, col_cols)
        plot_corr(off_day_corr_df, axes[3][0], mean_row_cols, col_cols)
        plot_corr(off_night_corr_df, axes[3][1], mean_row_cols, col_cols)

        plot_corr(work_day_corr_df, axes[4][0], session_row_cols, col_cols)
        plot_corr(work_night_corr_df, axes[4][1], session_row_cols, col_cols)
        plot_corr(off_day_corr_df, axes[5][0], session_row_cols, col_cols)
        plot_corr(off_night_corr_df, axes[5][1], session_row_cols, col_cols)

        fontsize = 22
        axes[0][0].set_title('Day shift (Workday, Total Usage Time)', fontweight='bold', fontsize=fontsize)
        axes[0][1].set_title('Night shift (Workday, Total Usage Time)', fontweight='bold', fontsize=fontsize)
        axes[1][0].set_title('Day shift (Non-Workday Total Usage Time)', fontweight='bold', fontsize=fontsize)
        axes[1][1].set_title('Night shift (Non-Workday, Total Usage Time)', fontweight='bold', fontsize=fontsize)

        axes[2][0].set_title('Day shift (Workday, Mean Session Length)', fontweight='bold', fontsize=fontsize)
        axes[2][1].set_title('Night shift (Workday, Mean Session Length)', fontweight='bold', fontsize=fontsize)
        axes[3][0].set_title('Day shift (Non-Workday, Mean Session Length)', fontweight='bold', fontsize=fontsize)
        axes[3][1].set_title('Night shift (Non-Workday, Mean Session Length)', fontweight='bold', fontsize=fontsize)
        
        axes[4][0].set_title('Day shift (Workday, Session Frequency)', fontweight='bold', fontsize=fontsize)
        axes[4][1].set_title('Night shift (Workday, Session Frequency)', fontweight='bold', fontsize=fontsize)
        axes[5][0].set_title('Day shift (Non-Workday, Session Frequency)', fontweight='bold', fontsize=fontsize)
        axes[5][1].set_title('Night shift (Non-Workday, Session Frequency)', fontweight='bold', fontsize=fontsize)

        for i in range(6):
            axes[i][0].set_yticklabels(['Night', 'Morning', 'Afternoon', 'Evening'], fontdict={'fontweight': 'bold', 'fontsize': fontsize})
            axes[i][1].set_yticklabels('')
            for j in range(2):
                axes[i][j].tick_params(axis="y", labelsize=fontsize)
                axes[i][j].yaxis.set_tick_params(size=1)
                axes[i][j].xaxis.set_tick_params(size=1)
                axes[i][j].set_xticklabels('')

        axes[5][0].set_xticklabels(col_cols, fontdict={'fontweight': 'bold', 'fontsize': fontsize})
        axes[5][1].set_xticklabels(col_cols, fontdict={'fontweight': 'bold', 'fontsize': fontsize})
            
        plt.tight_layout()

        cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        cb = plt.colorbar(axes[0][0].get_children()[0], cax=cax, **kw)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(fontsize)
        
        plt.savefig(os.path.join('plot', 'daily', 'diurnal_corr'), dpi=300)
        plt.close()


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)