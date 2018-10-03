"""
This is script is modified based on Karel Mundnich's script: days_at_work.py
Script is modified by Tiantian Feng
"""

import os, errno
import glob
import argparse
import numpy as np
import pandas as pd
import sys
from dateutil import rrule
from datetime import datetime, timedelta

from matplotlib.dates import MONTHLY, DateFormatter, rrulewrapper, RRuleLocator
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.dates
from pylab import *

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'util'))
from load_data_basic import getParticipantIDJobShift, getParticipantInfo
from load_data_basic import getParticipantStartTime, getParticipantEndTime

start_data_collection_date = datetime.datetime(year=2018, month=2, day=20)
end_data_collection_date = datetime.datetime(year=2018, month=6, day=10)


def getDataFrame(file):
    # Read and prepare owl data per participant
    data = pd.read_csv(file, index_col=0)
    data.index = pd.to_datetime(data.index)
    
    return data


def plot_timeline(ax, timeline, color, offset, expected_work=False, alpha=0.75, stream='sleep', wave_number=1):
    
    if expected_work is False:
        start_str = 'start_recording_time'
        end_str = 'end_recording_time'
    else:
        start_str = 'expected_start_work_time'
        end_str = 'expected_end_work_time'
        
    is_initialize_label = False

    if wave_number == 1:
        start_data_collection_date = datetime.datetime(year=2018, month=3, day=5)
        end_data_collection_date = datetime.datetime(year=2018, month=5, day=15)
    elif wave_number == 2:
        start_data_collection_date = datetime.datetime(year=2018, month=4, day=5)
        end_data_collection_date = datetime.datetime(year=2018, month=6, day=10)
    else:
        start_data_collection_date = datetime.datetime(year=2018, month=5, day=1)
        end_data_collection_date = datetime.datetime(year=2018, month=6, day=10)

    timeline_plot = pd.DataFrame()
    if len(timeline) > 0:
        
        for index, row in timeline.iterrows():
            
            diff = (pd.to_datetime(row[end_str]).replace(hour=0, minute=0, second=0, microsecond=0) - pd.to_datetime(row[start_str]).replace(hour=0, minute=0, second=0, microsecond=0)).days
            
            if stream == 'sleep':
                if row['is_sleep_before_work'] == 1 or row['is_sleep_after_work'] == 1:
                    # plot_color = 'y'
                    alpha = 1
                elif row['is_sleep_adaption'] == 1:
                    alpha = 1
                else:
                    alpha = 0.75
            
            if diff > 0 or diff < -20:
                
                end_recording_time = row[end_str]
                frame = row
                frame[end_str] = pd.to_datetime(row[start_str]).replace(hour=23, minute=59, second=59, microsecond=59).strftime(date_time_format)[:-3]
                timeline_plot = timeline_plot.append(frame)
                
                start_time = pd.to_datetime(row[start_str]).hour + pd.to_datetime(row[start_str]).minute / 60
                end_time = 23.99
                
                delta_time = pd.to_datetime(row[start_str]) - start_data_collection_date

                if is_initialize_label is False and stream != '':
                    ax.barh(delta_time.days * 3 + offset, end_time - start_time, left=start_time,
                            height=0.5, align='center', color=color, alpha=alpha, label=stream)
                    is_initialize_label = True
                else:
                    ax.barh(delta_time.days * 3 + offset, end_time - start_time, left=start_time,
                            height=0.5, align='center', color=color, alpha=alpha)
                    
                frame[start_str] = pd.to_datetime(end_recording_time).replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
                frame[end_str] = end_recording_time
                timeline_plot = timeline_plot.append(frame)
                
                start_time = 0
                end_time = pd.to_datetime(end_recording_time).hour + pd.to_datetime(end_recording_time).minute / 60
                
                if is_initialize_label is False and stream != '':
                    if stream == 'fitbit':
                        for i in range(diff-1):
                            ax.barh((delta_time.days + i + 1) * 3 + offset, 23.99, left=start_time,
                                    height=0.5, align='center', color=color, alpha=alpha)

                        ax.barh((delta_time.days + diff) * 3 + offset, end_time - start_time, left=start_time,
                                height=0.5, align='center', color=color, alpha=alpha, label=stream)
                    else:
                        ax.barh((delta_time.days + 1) * 3 + offset, end_time - start_time, left=start_time,
                                height=0.5, align='center', color=color, alpha=alpha, label=stream)
                    is_initialize_label = True
                else:
                    if stream == 'fitbit':
                        for i in range(diff-1):
                            ax.barh((delta_time.days + i + 1) * 3 + offset, 23.99, left=start_time,
                                    height=0.5, align='center', color=color, alpha=alpha)

                        ax.barh((delta_time.days + diff) * 3 + offset, end_time - start_time, left=start_time,
                                height=0.5, align='center', color=color, alpha=alpha)
                    else:
                        ax.barh((delta_time.days + 1) * 3 + offset, end_time - start_time, left=start_time,
                                height=0.5, align='center', color=color, alpha=alpha)
            else:
                timeline_plot = timeline_plot.append(row)
                
                start_time = pd.to_datetime(row[start_str]).hour + pd.to_datetime(row[start_str]).minute / 60
                end_time = pd.to_datetime(row[end_str]).hour + pd.to_datetime(row[end_str]).minute / 60
                delta_time = pd.to_datetime(row[start_str]) - start_data_collection_date
                
                if is_initialize_label is False and stream != '':
                    ax.barh(delta_time.days * 3 + offset, end_time - start_time, left=start_time,
                            height=0.5, align='center', color=color, alpha=alpha, label=stream)
                    is_initialize_label = True
                else:
                    ax.barh(delta_time.days * 3 + offset, end_time - start_time, left=start_time,
                            height=0.5, align='center', color=color, alpha=alpha)
                    
            # ax.grid(color='g', linestyle=':', which='major')
        if stream == 'fitbit':
            timeline_plot = timeline_plot[timeline.columns]


def main(main_data_directory, recording_timeline_directory, individual_timeline_directory, plot_timeline_directory):
    
    # job_shift = getParticipantIDJobShift(main_data_directory)
    participant_info = getParticipantInfo(main_data_directory)
    participant_info = participant_info.set_index('MitreID')
    
    colors = ['b', 'g', 'r', 'y', 'black', 'purple', 'lime']
    
    shift_type = ['day', 'night', 'unknown']
    
    for user_id in participant_info.index:
    
        participant_id = participant_info.loc[user_id]['ParticipantID']
        shift = 1 if participant_info.loc[user_id]['Shift'] == 'Day shift' else 2
        wave_number = participant_info.loc[user_id]['Wave']
        
        if wave_number == 1:
            start_data_collection_date = datetime.datetime(year=2018, month=3, day=5)
        elif wave_number == 2:
            start_data_collection_date = datetime.datetime(year=2018, month=4, day=5)
        else:
            start_data_collection_date = datetime.datetime(year=2018, month=5, day=1)

        # 10 weeks data collection
        data_collection_days = 70

        data_collection_date = []
        for i in range(data_collection_days):
            data_collection_date.append((start_data_collection_date + timedelta(days=i)).strftime(date_only_date_time_format))
    
        print('Start Processing (Individual timeline): ' + participant_id)
        
        if wave_number != 3 and os.path.exists(os.path.join(individual_timeline_directory, shift_type[shift-1], participant_id + '.csv')) is True:
            individual_timeline = getDataFrame(os.path.join(individual_timeline_directory, shift_type[shift-1], participant_id + '.csv'))
            
            sleep_timeline = individual_timeline.loc[individual_timeline['type'] == 'sleep']
            sleep_timeline.index = pd.to_datetime(sleep_timeline['start_recording_time'])

            owl_in_one_timeline = individual_timeline.loc[individual_timeline['type'] == 'owl_in_one']
            owl_in_one_timeline.index = pd.to_datetime(owl_in_one_timeline['start_recording_time'])

            om_signal_timeline = individual_timeline.loc[individual_timeline['type'] == 'omsignal']
            om_signal_timeline.index = pd.to_datetime(om_signal_timeline['start_recording_time'])

            ground_truth_timeline = individual_timeline.loc[individual_timeline['type'] == 'ground_truth']
            ground_truth_timeline.index = pd.to_datetime(ground_truth_timeline['start_recording_time'])

            fitbit_timeline = individual_timeline.loc[individual_timeline['type'] == 'fitbit']
            fitbit_timeline.index = pd.to_datetime(fitbit_timeline['start_recording_time'])

            realizd_timeline = individual_timeline.loc[individual_timeline['type'] == 'realizd']
            realizd_timeline.index = pd.to_datetime(realizd_timeline['start_recording_time'])

            if len(sleep_timeline) > 0 or len(owl_in_one_timeline) > 0 or len(ground_truth_timeline) > 0:
                fig = plt.figure(figsize=(15, 12))
                ax = fig.add_subplot(111)
                ax.set_title('Timeline for participant: ' + participant_id)
                ax.axis('tight')

                ax.set_ylim(ymin=-0.5, ymax=data_collection_days * 3 - 0.5)
                ax.set_xlim(xmin=0, xmax=24)

            if len(sleep_timeline) > 0:
                plot_timeline(ax, sleep_timeline, colors[0], 0, alpha=0.75, stream='sleep', wave_number=wave_number)

            if len(owl_in_one_timeline) > 0:
                plot_timeline(ax, owl_in_one_timeline, colors[1], 0, expected_work=True, alpha=1, stream='expected_work', wave_number=wave_number)
                plot_timeline(ax, owl_in_one_timeline, colors[6], 1, alpha=0.6, stream='owl_in_one', wave_number=wave_number)
                
            if len(ground_truth_timeline) > 0:
                plot_timeline(ax, ground_truth_timeline, colors[1], 0, expected_work=True, alpha=1, stream='', wave_number=wave_number)
                plot_timeline(ax, ground_truth_timeline, colors[2], 1, alpha=0.6, stream='survey', wave_number=wave_number)
                
            if len(om_signal_timeline) > 0:
                plot_timeline(ax, om_signal_timeline, colors[1], 0, expected_work=True, alpha=1, stream='', wave_number=wave_number)
                plot_timeline(ax, om_signal_timeline, colors[5], 1, alpha=0.6, stream='om_signal', wave_number=wave_number)
                
            if len(fitbit_timeline) > 0:
                plot_timeline(ax, fitbit_timeline, colors[3], 2, alpha=1, stream='fitbit', wave_number=wave_number)

            if len(realizd_timeline) > 0:
                plot_timeline(ax, realizd_timeline, colors[4], 0, alpha=1, stream='RealizD', wave_number=wave_number)

            if len(sleep_timeline) > 0 or len(owl_in_one_timeline) > 0:
                for i in np.arange(-0.5, len(data_collection_date) * 3 + 1, 3):
                    ax.axhline(y=i, ls='--', color='black', alpha=0.5)
                
                legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                yticks(range(1, len(data_collection_date) * 3 + 1, 3), data_collection_date)
                if os.path.exists(os.path.join(plot_timeline_directory, shift_type[shift-1])) is False:
                    os.mkdir(os.path.join(plot_timeline_directory, shift_type[shift - 1]))
                plt.savefig(os.path.join(plot_timeline_directory, shift_type[shift-1], participant_id + '.png'),
                            bbox_extra_artists=(legend,),
                            bbox_inches='tight')
                plt.close()

                
if __name__ == "__main__":
    
    DEBUG = 1
    
    if DEBUG == 0:
        """
            Parse the args:
            1. main_data_directory: directory to store keck data
            2. output_directory: main output directory

        """
        parser = argparse.ArgumentParser(description='Create a dataframe of worked days.')
        parser.add_argument('-i', '--main_data_directory', type=str, required=True,
                            help='Directory for data.')
        parser.add_argument('-o', '--output_directory', type=str, required=True,
                            help='Directory for output.')
        
        # stream_types = ['omsignal', 'owl_in_one', 'ground_truth']
        
        args = parser.parse_args()
        
        """
            days_at_work_directory = '../output/days_at_work'
            main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
            recording_timeline_directory = '../output/recording_timeline'
        """
        
        main_data_directory = os.path.join(os.path.expanduser(os.path.normpath(args.main_data_directory)), 'keck_wave1/2_preprocessed_data')
        days_at_work_directory = os.path.join(os.path.expanduser(os.path.normpath(args.output_directory)), 'days_at_work')
        recording_timeline_directory = os.path.join(os.path.expanduser(os.path.normpath(args.output_directory)), 'recording_timeline')
        individual_timeline_directory = os.path.join(os.path.expanduser(os.path.normpath(args.output_directory)), 'individual_timeline')
        plot_timeline_directory = '../plot'
        
        print('main_data_directory: ' + main_data_directory)
        print('days_at_work_directory: ' + days_at_work_directory)
        print('recording_timeline_directory: ' + recording_timeline_directory)
    
    else:
        days_at_work_directory = '../output/days_at_work'
        # main_data_directory = '../../data/keck_wave1/2_preprocessed_data'
        main_data_directory = '../../data'
        recording_timeline_directory = '../output/recording_timeline'
        individual_timeline_directory = '../output/individual_timeline'
        plot_timeline_directory = '../plot'
        
    if os.path.exists(plot_timeline_directory) is False: os.mkdir(plot_timeline_directory)
    if os.path.exists(os.path.join(plot_timeline_directory, 'output')) is False: os.mkdir(os.path.join(plot_timeline_directory, 'output'))
        
    main(main_data_directory, recording_timeline_directory, individual_timeline_directory, plot_timeline_directory)