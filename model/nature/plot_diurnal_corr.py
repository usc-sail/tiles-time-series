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
import pickle
import preprocess
from scipy import stats
from datetime import timedelta
import collections

import seaborn as sns

sns.set(style="whitegrid")


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
        # day_df = nurse_df.loc[nurse_df['shift'] == 'day']
        # night_df = nurse_df.loc[nurse_df['shift'] == 'night']
        
        corr_df = pd.DataFrame()
        plot_df = pd.DataFrame()
        
        for i in range(len(nurse_df)):
            participant_df = nurse_df.iloc[i, :]
            participant_id = list(nurse_df.index)[i]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]

            for i, diurnal_type in enumerate(['night', 'morning', 'afternoon', 'evening']):
                row_df = pd.DataFrame(index=[participant_id + 'work'])
                row_df['Total Usage Time'] = participant_df['work_' + diurnal_type + '_total_time_mean']
                row_df['Mean Session Length'] = participant_df['work_' + diurnal_type + '_mean_time_mean']
                row_df['Session Frequency'] = participant_df['work_' + diurnal_type + '_frequency_mean']
                row_df['Mean Inter-session Time']  = participant_df['work_' + diurnal_type + '_mean_inter_mean']
                row_df['Session Frequency (<1min)'] = participant_df['work_' + diurnal_type + '_less_than_1min_mean']
                row_df['Session Frequency (>1min)'] = participant_df['work_' + diurnal_type + '_above_1min_mean']
                row_df['Data Type'] = 'Workday'
                row_df['Shift Type'] = shift
                row_df['time'] = i
            
                plot_df = plot_df.append(row_df)
            
                row_df = pd.DataFrame(index=[participant_id + 'off'])
                row_df['Total Usage Time'] = participant_df['off_' + diurnal_type + '_total_time_mean']
                row_df['Mean Session Length'] = participant_df['off_' + diurnal_type + '_mean_time_mean']
                row_df['Session Frequency'] = participant_df['off_' + diurnal_type + '_frequency_mean']
                row_df['Mean Inter-session Time'] = participant_df['off_' + diurnal_type + '_mean_inter_mean']
                row_df['Session Frequency (<1min)'] = participant_df['off_' + diurnal_type + '_less_than_1min_mean']
                row_df['Session Frequency (>1min)'] = participant_df['off_' + diurnal_type + '_above_1min_mean']
                row_df['Data Type'] = 'Off-day'
                row_df['Shift Type'] = shift
                row_df['time'] = i
            
                plot_df = plot_df.append(row_df)
        
        
        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["font.size"] = 16
        fig = plt.figure(figsize=(18, 6))
        axes = fig.subplots(nrows=2, ncols=3)
        
        day_df = plot_df.loc[plot_df['Shift Type'] == 'Day shift']
        night_df = plot_df.loc[plot_df['Shift Type'] == 'Night shift']

        work_df = plot_df.loc[plot_df['Data Type'] == 'Workday']
        off_df = plot_df.loc[plot_df['Data Type'] == 'Off-day']
        
        work_day_df = day_df.loc[day_df['Data Type'] == 'Workday']
        work_night_df = night_df.loc[night_df['Data Type'] == 'Workday']
        off_day_df = day_df.loc[day_df['Data Type'] == 'Off-day']
        off_night_df = night_df.loc[night_df['Data Type'] == 'Off-day']
        
        row_cols = ['Total Usage Time', 'Mean Session Length', 'Mean Inter-session Time',
                    'Session Frequency', 'Session Frequency (<1min)', 'Session Frequency (>1min)']

        plt.rcParams["font.weight"] = "bold"
        plt.rcParams["font.size"] = 16
        fig = plt.figure(figsize=(18, 6))
        axes = fig.subplots(nrows=2, ncols=3)

        data_df = off_df
        
        day_data_df = data_df.loc[data_df['Shift Type'] == 'Day shift']
        night_data_df = data_df.loc[data_df['Shift Type'] == 'Night shift']
        for col in row_cols:
            time_col_list = ['Night', 'Morning', 'Afternoon', 'Evening']
            for i in range(len(time_col_list)):
                day_tmp_df = day_data_df.loc[day_data_df['time'] == i]
                night_tmp_df = night_data_df.loc[night_data_df['time'] == i]

                print('\n\n')
                print('K-S test for %s workday' % (time_col_list[i] + '_' + col))
                stat, p = stats.ks_2samp(day_tmp_df[col].dropna(), night_tmp_df[col].dropna())

                print('Statistics = %.3f, p = %.3f' % (stat, p))
                print('Day shift work: mean = %.2f, std = %.2f' % (np.mean(day_tmp_df[col].dropna()), np.std(day_tmp_df[col].dropna())))
                print('Night shift off: mean = %.2f, std = %.2f' % (np.mean(night_tmp_df[col].dropna()), np.std(night_tmp_df[col].dropna())))

        ax = sns.lineplot(x="time", y='Total Usage Time', dashes=False, marker="o", hue="Shift Type", data=data_df, palette="seismic", ax=axes[0][0])
        ax = sns.lineplot(x="time", y='Mean Session Length', dashes=False, marker="o", hue="Shift Type", data=data_df, palette="seismic", ax=axes[0][1])
        ax = sns.lineplot(x="time", y='Mean Inter-session Time', dashes=False, marker="o", hue="Shift Type", data=data_df, palette="seismic", ax=axes[0][2])

        ax = sns.lineplot(x="time", y='Session Frequency', dashes=False, marker="o", hue="Shift Type", data=data_df, palette="seismic", ax=axes[1][0])
        ax = sns.lineplot(x="time", y='Session Frequency (<1min)', dashes=False, marker="o", hue="Shift Type", data=data_df, palette="seismic", ax=axes[1][1])
        ax = sns.lineplot(x="time", y='Session Frequency (>1min)', dashes=False, marker="o", hue="Shift Type", data=data_df, palette="seismic", ax=axes[1][2])

        # axes[0][0].grid(False, axis='x')
        
        for i in range(2):
            for j in range(3):
                axes[i][j].set_xlim([-0.25, 3.25])
                axes[i][j].set_xlabel('')
                axes[i][j].set_xticks([0, 1, 2, 3])

                time_col_list = ['Night', 'Morning', 'Afternoon', 'Evening']
                data_col = row_cols[int(i * 2 + j)]
                plot_time_col_list = time_col_list
                for time in range(len(time_col_list)):
                    day_tmp_df = day_data_df.loc[day_data_df['time'] == time]
                    night_tmp_df = night_data_df.loc[night_data_df['time'] == time]

                    stat, p = stats.ks_2samp(day_tmp_df[data_col].dropna(), night_tmp_df[data_col].dropna())

                    cohens_d = (np.nanmean(day_tmp_df[data_col]) - np.nanmean(night_tmp_df[data_col]))
                    cohens_d = cohens_d / np.sqrt((np.nanstd(day_tmp_df[data_col]) ** 2 + np.nanstd(night_tmp_df[data_col] ** 2)) / 2)
                    
                    if cohens_d < 0:
                        cohens_d = str(cohens_d)[:5]
                    else:
                        cohens_d = str(cohens_d)[:4]
                    
                    if p < 0.01:
                        plot_time_col_list[time] = plot_time_col_list[time] + '\n(p<0.01, \nd=' + cohens_d + ')'
                    else:
                        plot_time_col_list[time] = plot_time_col_list[time] + '\n(p=' + str(p)[:4] + ', \nd=' + cohens_d + ')'

                axes[i][j].set_xticklabels(plot_time_col_list, fontdict={'fontweight': 'bold', 'fontsize': 14})
                axes[i][j].yaxis.set_tick_params(size=1)
                # axes[0][j].set_xticklabels([])
                # axes[i][j].grid(False, axis='x')
                axes[i][j].grid(linestyle='--')
                axes[i][j].grid(False, axis='y')
                
                handles, labels = axes[i][j].get_legend_handles_labels()
                axes[i][j].legend(handles=handles[1:], labels=labels[1:], prop={'size': 10.5})
                axes[i][j].lines[0].set_linestyle("--")
                axes[i][j].lines[1].set_linestyle("--")
                axes[i][j].tick_params(axis="y", labelsize=14)
        
        axes[0][0].set_ylim([0, 10000])
        axes[0][1].set_ylim([0, 1500])
        axes[0][2].set_ylim([0, 2500])
        axes[1][0].set_ylim([0, 50])
        axes[1][1].set_ylim([0, 30])
        axes[1][2].set_ylim([0, 30])
        
        '''
        axes[0][0].set_ylim([1000, 10000])
        axes[0][1].set_ylim([0, 1200])
        axes[0][2].set_ylim([600, 2400])
        axes[1][0].set_ylim([0, 40])
        axes[1][1].set_ylim([0, 25])
        axes[1][2].set_ylim([0, 25])
        '''
        axes[0][0].set_title('Total Usage Time', fontweight='bold', fontsize=15)
        axes[0][1].set_title('Mean Session Length', fontweight='bold', fontsize=15)
        axes[0][2].set_title('Mean Inter-session Time', fontweight='bold', fontsize=15)
        axes[1][0].set_title('Session Frequency', fontweight='bold', fontsize=15)
        axes[1][1].set_title('Session Frequency (<1min)', fontweight='bold', fontsize=15)
        axes[1][2].set_title('Session Frequency (>1min)', fontweight='bold', fontsize=15)

        axes[0][0].set_ylabel('Seconds', fontweight='bold', fontsize=14)
        axes[0][1].set_ylabel('Seconds', fontweight='bold', fontsize=14)
        axes[0][2].set_ylabel('Seconds', fontweight='bold', fontsize=14)
        axes[1][0].set_ylabel('Num. of Sessions', fontweight='bold', fontsize=14)
        axes[1][1].set_ylabel('Num. of Sessions', fontweight='bold', fontsize=14)
        axes[1][2].set_ylabel('Num. of Sessions', fontweight='bold', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join('plot', 'daily', 'offday_diurnal'), dpi=300)
        plt.close()


if __name__ == '__main__':
    '''
    import ssl
    
    ssl._create_default_https_context = ssl._create_unverified_context
    fmri = sns.load_dataset("fmri")
    '''
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)