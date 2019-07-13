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
from scipy.stats import skew
import matplotlib as mpl
from scipy.stats import spearmanr
from datetime import timedelta

import seaborn as sns
sns.set(style="whitegrid")

row_cols = ['Total Usage Time', 'Mean Session Length',
            'Session Frequency', 'Session Frequency (<1min)', 'Session Frequency (>1min)']

col_cols = ['Shipley Abs.', 'Shipley Voc.',
            'Anxiety', 'Pos. Affect', 'Neg. Affect',
            'Neuroticism', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Openness']

psqi_col_cols = ['Total PSQI', 'Subjective quality', 'Sleep latency',
                 'Sleep duration', 'Sleep efficiency', 'Sleep Disturbance',
                 'Sleep medication', 'Day dysfunction']


def plot_corr(data_df, ax, row_cols, col_cols, heat_fontsize=15):
    
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
            

def process_sleep_realizd(data_config, data_df, sleep_df, igtb_df, participant_id):

    process_df = pd.DataFrame()
    
    if len(data_df) < 700 or len(sleep_df) < 10:
        return None
    
    for i in range(len(sleep_df)):
        
        start_str = sleep_df.iloc[i, :]['SleepBeginTimestamp']
        end_str = sleep_df.iloc[i, :]['SleepEndTimestamp']
        
        if sleep_df.iloc[i, :]['duration'] < 4:
            continue
        
        # date_start_str = (pd.to_datetime(start_str)).strftime(load_data_basic.date_time_format)[:-3]
        # date_end_str = (pd.to_datetime(end_str)).strftime(load_data_basic.date_time_format)[:-3]
        row_df = pd.DataFrame(index=[start_str])
        raw_df = data_df[start_str:end_str]
        
        if len(raw_df) > 0:
            row_df['frequency'] = len(raw_df)
            row_df['total_time'] = np.nansum(np.array(raw_df))
            row_df['mean_time'] = np.nanmean(np.array(raw_df))
            '''
            row_df['less_than_1min'] = len(np.where(np.array(raw_df) <= 60)[0])
            row_df['above_1min'] = len(np.where((np.array(raw_df) > 60))[0])
            '''
            row_df['less_than_1min'] = len(np.where(np.array(raw_df) <= 60)[0]) / len(raw_df)
            row_df['above_1min'] = len(np.where((np.array(raw_df) > 60))[0]) / len(raw_df)
            # row_df['1min_5min'] = len(np.where((np.array(raw_df) > 60) & (np.array(raw_df) <= 300))[0])
            # row_df['above_5min'] = len(np.where(np.array(raw_df) >= 300)[0])
        else:
            row_df['frequency'] = 0
            row_df['total_time'] = 0
            row_df['mean_time'] = np.nan
            '''
            row_df['less_than_1min'] = 0
            row_df['above_1min'] = 0
            '''
            row_df['less_than_1min'] = np.nan
            row_df['above_1min'] = np.nan
            
        start_str = (pd.to_datetime(start_str) - timedelta(minutes=30)).strftime(load_data_basic.date_time_format)[:-3]
        end_str = sleep_df.iloc[i, :]['SleepBeginTimestamp']

        raw_df = data_df[start_str:end_str]
        if len(raw_df) > 0:
            row_df['1hour_prior_frequency'] = len(raw_df)
            row_df['1hour_prior_total_time'] = np.nansum(np.array(raw_df))
            row_df['1hour_prior_mean_time'] = np.nanmean(np.array(raw_df))
            row_df['1hour_prior_less_than_1min'] = len(np.where(np.array(raw_df) <= 60)[0]) / len(raw_df)
            row_df['1hour_prior_above_1min'] = len(np.where(np.array(raw_df) > 60)[0]) / len(raw_df)
        else:
            row_df['1hour_prior_frequency'] = 0
            row_df['1hour_prior_total_time'] = 0
            row_df['1hour_prior_mean_time'] = np.nan
            row_df['1hour_prior_above_1min'] = np.nan
            row_df['1hour_prior_less_than_1min'] = np.nan

        start_str = sleep_df.iloc[i, :]['SleepEndTimestamp']
        end_str = (pd.to_datetime(sleep_df.iloc[i, :]['SleepEndTimestamp']) + timedelta(minutes=30)).strftime(load_data_basic.date_time_format)[:-3]
        
        raw_df = data_df[start_str:end_str]
        if len(raw_df) > 0:
            row_df['1hour_after_frequency'] = len(raw_df)
            row_df['1hour_after_total_time'] = np.nansum(np.array(raw_df))
            row_df['1hour_after_mean_time'] = np.nanmean(np.array(raw_df))
            row_df['1hour_after_above_1min'] = len(np.where(np.array(raw_df) > 60)[0])
            row_df['1hour_after_less_than_1min'] = len(np.where(np.array(raw_df) <= 60)[0])
        else:
            row_df['1hour_after_frequency'] = 0
            row_df['1hour_after_total_time'] = 0
            row_df['1hour_after_mean_time'] = np.nan
            row_df['1hour_after_above_1min'] = 0
            row_df['1hour_after_less_than_1min'] = 0
        
        process_df = process_df.append(row_df)
    
    process_df.to_csv(os.path.join(data_config.phone_usage_path, participant_id + '_sleep.csv.gz'), compression='gzip')
    participant_df = pd.DataFrame(index=[participant_id])
    
    for col in list(process_df.columns):
        participant_df[col + '_mean'] = np.nanmean(np.array(process_df[col]))
        
    igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
    for col in igtb_cols:
        participant_df[col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]
    
    return participant_df


def print_stats(day_df, night_df):
    
    for col in list(day_df.columns):
        
        if 'igtb' in col or 'shift' in col or 'job' in col:
            continue
            
        if 'prior' not in col:
        # if 'prior' in col or 'after' in col:
            continue
        
        if 'total_time_mean' in col:
            text_col = 'Total Usage Time'
        elif 'mean_time_mean' in col:
            text_col = 'Mean Session Time'
        elif 'frequency_mean' in col:
            text_col = 'Session Frequency'
        elif 'mean_inter_mean' in col:
            text_col = 'Mean Inter-session Time'
        elif 'less_than_1min_mean' in col:
            text_col = 'Number of Sessions (<1min)'
        else:
            text_col = 'Number of Sessions (>1min)'
    
        print('\multicolumn{1}{l}{\\textbf{' + text_col + '}} &')
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(day_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(day_df[col].dropna())))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.mean(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (np.std(night_df[col])))
        print('\multicolumn{1}{c}{$%.2f$} &' % (skew(night_df[col].dropna())))
        
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
    psqi_raw_igtb = load_data_basic.read_PSQI_Raw(tiles_data_path)
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    if os.path.exists(os.path.join(data_config.phone_usage_path, 'sleep_summary.csv.gz')) is True:
        all_df = pd.read_csv(os.path.join(data_config.phone_usage_path, 'sleep_summary.csv.gz'), index_col=0)
        
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

        plot_df = pd.DataFrame()
        one_hour_plot_df = pd.DataFrame()
        
        for i in range(len(nurse_df)):
            participant_df = nurse_df.iloc[i, :]
            participant_id = list(nurse_df.index)[i]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            uid = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index[0]
            
            row_df = pd.DataFrame(index=[participant_id])
            onehour_row_df = pd.DataFrame(index=[participant_id])
            
            row_df['Total Usage Time'] = participant_df['total_time_mean']
            row_df['Mean Session Length'] = participant_df['mean_time_mean']
            row_df['Session Frequency'] = participant_df['frequency_mean']
            row_df['Session Frequency (<1min)'] = participant_df['less_than_1min_mean']
            row_df['Session Frequency (>1min)'] = participant_df['above_1min_mean']
            
            onehour_row_df['Total Usage Time'] = participant_df['1hour_prior_total_time_mean']
            onehour_row_df['Mean Session Length'] = participant_df['1hour_prior_mean_time_mean']
            onehour_row_df['Session Frequency'] = participant_df['1hour_prior_frequency_mean']
            onehour_row_df['Session Frequency (<1min)'] = participant_df['1hour_prior_less_than_1min_mean']
            onehour_row_df['Session Frequency (>1min)'] = participant_df['1hour_prior_above_1min_mean']
            
            row_df['Shift Type'] = shift
            onehour_row_df['Shift Type'] = shift
            
            for col in ana_igtb_cols:
                row_df[ana_igtb_dict[col]] = participant_df[col]
                onehour_row_df[ana_igtb_dict[col]] = participant_df[col]
                
            for col in list(psqi_raw_igtb.columns):
                row_df[psqi_igtb_dict[col]] = psqi_raw_igtb.loc[uid, col]
                onehour_row_df[psqi_igtb_dict[col]] = psqi_raw_igtb.loc[uid, col]

            plot_df = plot_df.append(row_df)
            one_hour_plot_df = one_hour_plot_df.append(onehour_row_df)

        day_plot_df = plot_df.loc[plot_df['Shift Type'] == 'Day shift']
        night_plot_df = plot_df.loc[plot_df['Shift Type'] == 'Night shift']

        day_one_hour_plot_df = one_hour_plot_df.loc[one_hour_plot_df['Shift Type'] == 'Day shift']
        night_one_hour_plot_df = one_hour_plot_df.loc[one_hour_plot_df['Shift Type'] == 'Night shift']
        
        fig = plt.figure(figsize=(18, 12))
        axes = fig.subplots(nrows=4, ncols=1)

        heat_fontsize = 16

        '''
        plot_corr(day_one_hour_plot_df, axes[0][0], row_cols, col_cols, heat_fontsize=heat_fontsize)
        plot_corr(night_one_hour_plot_df, axes[1][0], row_cols, col_cols, heat_fontsize=heat_fontsize)
        plot_corr(day_one_hour_plot_df, axes[0][1], row_cols, psqi_col_cols, heat_fontsize=heat_fontsize)
        plot_corr(night_one_hour_plot_df, axes[1][1], row_cols, psqi_col_cols, heat_fontsize=heat_fontsize)
        
        plot_corr(day_plot_df, axes[2][0], row_cols, col_cols, heat_fontsize=heat_fontsize)
        plot_corr(night_plot_df, axes[3][0], row_cols, col_cols, heat_fontsize=heat_fontsize)
        plot_corr(day_plot_df, axes[2][1], row_cols, psqi_col_cols, heat_fontsize=heat_fontsize)
        plot_corr(night_plot_df, axes[3][1], row_cols, psqi_col_cols, heat_fontsize=heat_fontsize)
        '''

        plot_corr(day_one_hour_plot_df, axes[0], row_cols, col_cols+psqi_col_cols, heat_fontsize=heat_fontsize)
        plot_corr(night_one_hour_plot_df, axes[1], row_cols, col_cols+psqi_col_cols, heat_fontsize=heat_fontsize)
        plot_corr(day_plot_df, axes[2], row_cols, col_cols+psqi_col_cols, heat_fontsize=heat_fontsize)
        plot_corr(night_plot_df, axes[3], row_cols, col_cols+psqi_col_cols, heat_fontsize=heat_fontsize)
        
        axes[0].set_xticklabels('')
        axes[1].set_xticklabels('')
        axes[2].set_xticklabels('')
        
        axes[0].set_title('Day shift (Before-bedtime usage)', fontweight='bold', fontsize=heat_fontsize+2)
        axes[1].set_title('Night shift (Before-bedtime usage)', fontweight='bold', fontsize=heat_fontsize+2)
        axes[2].set_title('Day shift (Bedtime usage)', fontweight='bold', fontsize=heat_fontsize+2)
        axes[3].set_title('Night shift (Bedtime usage)', fontweight='bold', fontsize=heat_fontsize+2)

        axes[0].set_yticklabels(row_cols, fontdict={'fontweight': 'bold', 'fontsize': heat_fontsize})
        axes[1].set_yticklabels(row_cols, fontdict={'fontweight': 'bold', 'fontsize': heat_fontsize})
        axes[2].set_yticklabels(row_cols, fontdict={'fontweight': 'bold', 'fontsize': heat_fontsize})
        axes[3].set_yticklabels(row_cols, fontdict={'fontweight': 'bold', 'fontsize': heat_fontsize})

        axes[3].set_xticklabels(['Shipley Abs.', 'Shipley Voc.',
                                 'Anxiety', 'Pos. Affect', 'Neg. Affect',
                                 'Neuroticism', 'Conscientiousness', 'Extraversion',
                                 'Agreeableness', 'Openness',
                                 'Total PSQI', 'Subjective quality', 'Sleep latency',
                                 'Sleep duration', 'Sleep efficiency', 'Sleep Disturbance',
                                 'Sleep medication', 'Day dysfunction'], rotation=90, ha='center',
                                 fontdict={'fontweight': 'bold', 'fontsize': heat_fontsize})

        axes[0].tick_params(axis="y", labelsize=heat_fontsize)
        axes[1].tick_params(axis="y", labelsize=heat_fontsize)
        axes[2].tick_params(axis="y", labelsize=heat_fontsize)
        axes[3].tick_params(axis="y", labelsize=heat_fontsize)
        axes[3].tick_params(axis="x", labelsize=heat_fontsize)
        
        plt.tight_layout()

        cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
        cb = plt.colorbar(axes[0].get_children()[0], cax=cax, **kw)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(heat_fontsize)
            
        plt.savefig(os.path.join('plot', 'daily', 'sleep_corr'), dpi=300)
        plt.close()

        for col in list(nurse_df.columns):
            
            if col is not 'shift' or col is not 'job':
                
                # K-S test
                stat, p = stats.ks_2samp(day_nurse_df[col].dropna(), night_nurse_df[col].dropna())
                print('K-S test for %s' % col)
                print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))
        
        print_stats(day_nurse_df, night_nurse_df)
        
    all_df = pd.DataFrame()
    for idx, participant_id in enumerate(top_participant_id_list[:]):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        if os.path.exists(os.path.join(data_config.sleep_path, participant_id + '.pkl')) is False:
            continue

        pkl_file = open(os.path.join(data_config.sleep_path, participant_id + '.pkl'), 'rb')
        participant_sleep_dict = pickle.load(pkl_file)

        realizd_raw_df = load_sensor_data.read_realizd(os.path.join(tiles_data_path, '2_raw_csv_data/realizd/'), participant_id)
        if realizd_raw_df is None:
            continue

        participant_df = process_sleep_realizd(data_config, realizd_raw_df, participant_sleep_dict['summary'], igtb_df, participant_id)
        
        if participant_df is not None:
            all_df = all_df.append(participant_df)
            all_df = all_df.loc[:, list(participant_df.columns)]
        
    all_df.to_csv(os.path.join(data_config.phone_usage_path, 'sleep_summary.csv.gz'), compression='gzip')


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)