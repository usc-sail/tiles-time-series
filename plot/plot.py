#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import timedelta
from matplotlib.patches import Patch
from hmmlearn import hmm

###########################################################
# Add package path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
import util

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

color_dict = {'lounge': 'green', 'ns': 'gold', 'pat': 'grey', 'other_floor': 'violet', 'med': 'blue', 'floor2': 'cyan'}

cluster_color_list = ['red', 'blue', 'green', 'cyan', 'grey', 'yellow', 'purple', 'brown', 'black', 'gold']


class Plot(object):
    
    def __init__(self, data_config=None, participant_id_list=None, primary_unit=None, shift=None):
        """
        Initialization method
        """
        ###########################################################
        # Assert if these parameters are not parsed
        ###########################################################
        self.data_config = data_config
        self.primary_unit = primary_unit
        self.shift = shift
        
        self.participant_id_list = participant_id_list
    
    def plot_segmentation(self, participant_id, fitbit_df=None, realizd_df=None, owl_in_one_df=None,
                          fitbit_summary_df=None, mgt_df=None, omsignal_data_df=None):
        ###########################################################
        # If folder not exist
        ###########################################################
        save_folder = os.path.join(self.data_config.fitbit_sensor_dict['segmentation_path'])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        ###########################################################
        # Read data
        ###########################################################
        fitbit_df = fitbit_df.sort_index()
        realizd_data_df = None if realizd_df is None else realizd_df
        
        if os.path.exists(os.path.join(save_folder, participant_id + '.csv.gz')) is False:
            return
        bps_df = pd.read_csv(os.path.join(save_folder, participant_id + '.csv.gz'), index_col=0)
        
        interval = int((pd.to_datetime(fitbit_df.index[1]) - pd.to_datetime(fitbit_df.index[0])).total_seconds() / 60)
        
        ###########################################################
        # Define plot range
        ###########################################################
        start_str = fitbit_df.index[0]
        end_str = fitbit_df.index[-1]
        day_offset = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days
        
        for day in range(day_offset):
            ###########################################################
            # Define start and end string
            ###########################################################
            day_start_str = (pd.to_datetime(start_str) + timedelta(days=day)).strftime(date_time_format)[:-3]
            day_end_str = (pd.to_datetime(start_str) + timedelta(days=day +1)).strftime(date_time_format)[:-3]
            
            day_str = (pd.to_datetime(day_start_str)).strftime(date_only_date_time_format)
            day_before_str = (pd.to_datetime(day_start_str) - timedelta(days=1)).strftime(date_only_date_time_format)
            day_after_str = (pd.to_datetime(day_start_str) + timedelta(days=1)).strftime(date_only_date_time_format)
            
            ###########################################################
            # Read data in the range
            ###########################################################
            day_data_df = fitbit_df[day_start_str:day_end_str]
            day_data_array = np.array(day_data_df)
            day_time_array = pd.to_datetime(day_data_df.index)
            day_data_array[:, 1] = day_data_array[:, 1] / interval
            day_data_array[np.where(day_data_array[:, 1] < 0)[0], :] = -25
            
            ###########################################################
            # Read plot data_df
            ###########################################################
            plt_df = bps_df[day_start_str:day_end_str]
            xposition = pd.to_datetime(list(plt_df.index))
            
            ###########################################################
            # Plot
            ###########################################################
            f, ax = plt.subplots(3, figsize=(18, 8))
            
            ax[0].plot(day_time_array, np.array(day_data_array)[:, 0], label='heart rate')
            ax[0].plot(day_time_array, np.array(day_data_array)[:, 1], label='step count')
            
            ###########################################################
            # Plot omsignal data
            ###########################################################
            if omsignal_data_df is not None:
                day_omsignal_data_df = omsignal_data_df[day_start_str:day_end_str]
                day_omsignal_time_array = pd.to_datetime(day_omsignal_data_df.index)
                if len(day_omsignal_time_array) > 0:
                    ax[1].plot(day_omsignal_time_array, np.array(day_omsignal_data_df)[:, 0], label='om heart rate', color='purple')
                    ax[1].plot(day_omsignal_time_array, np.array(day_omsignal_data_df)[:, 1], label='om step count', color='brown')

            ###########################################################
            # Plot realizd data
            ###########################################################
            if realizd_data_df is not None:
                day_realizd_data_df = realizd_data_df[day_start_str:day_end_str]
                day_realizd_time_array = pd.to_datetime(day_realizd_data_df.index)

                if len(day_realizd_data_df) > 0:
                    ax[2].plot(day_realizd_time_array, np.array(day_realizd_data_df.SecondsOnPhone),
                               label='phone usage', color='green')

            ###########################################################
            # Plot owl_in_one data
            ###########################################################
            if owl_in_one_df is not None:
                day_owl_in_one_df = owl_in_one_df[day_before_str:day_end_str]
                day_owl_in_one_time_array = pd.to_datetime(day_owl_in_one_df.index)
                if len(day_owl_in_one_time_array) > 0:
                    
                    unit_in_time = np.where(np.array(day_owl_in_one_df) > 0)[1]
                    unit_diff_in_time = unit_in_time[1:] - unit_in_time[:-1]

                    unit_change_time_array = np.where(unit_diff_in_time != 0)[0]
                    unit_change_time_array = np.append(unit_change_time_array, len(day_owl_in_one_df) - 2)
                    
                    unit_time_df = pd.DataFrame()
                    start_time = day_owl_in_one_df.index[0]
                    unit_type = day_owl_in_one_df.columns[unit_in_time[0]]
                    
                    for unit_change_time in unit_change_time_array:
                        
                        end_time = day_owl_in_one_df.index[unit_change_time+1]
                        unit_row_df = pd.DataFrame(index=[start_time])
                        unit_row_df['start'] = start_time
                        unit_row_df['end'] = end_time
                        unit_row_df['unit'] = unit_type
                        unit_time_df = unit_time_df.append(unit_row_df)
                        
                        if unit_type == 'med' or unit_type == 'lounge' or unit_type == 'ns' \
                                or unit_type == 'pat' or unit_type == 'other_floor' or unit_type == 'floor2':
                            self.plot_owl_in_one_span(ax, unit_type, start_time, end_time, day_str, day_after_str)

                        start_time = day_owl_in_one_df.index[unit_change_time+1]
                        unit_type = day_owl_in_one_df.columns[unit_in_time[unit_change_time+1]]

            for xc in xposition:
                ax[0].axvline(x=xc, color='k', linestyle='--')
                ax[1].axvline(x=xc, color='k', linestyle='--')
                ax[2].axvline(x=xc, color='k', linestyle='--')
            
            ax[0].set_xlim([day_time_array[0], day_time_array[-1]])
            ax[1].set_xlim([day_time_array[0], day_time_array[-1]])
            ax[2].set_xlim([day_time_array[0], day_time_array[-1]])
            ax[2].set_ylim([-10, 100])
            ax[2].set_xlabel(self.primary_unit)

            
            ax[0].legend()
            ax[1].legend()
            # ax[2].legend()
            # ax[2].legend(['lounge'], loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True)

            legend_elements = [Patch(facecolor=color_dict[loc], label=loc, alpha=0.3) for loc in list(color_dict.keys())]
            ax[2].legend(handles=legend_elements)

            ###########################################################
            # Plot fitbit summary
            ###########################################################
            if fitbit_summary_df is not None:
                day_fitbit_summary = fitbit_summary_df[day_before_str:day_after_str]
                
                if len(day_fitbit_summary) > 0:
                    for index, row in day_fitbit_summary.iterrows():
                        
                        self.plot_sleep_span(ax, row.Sleep1BeginTimestamp, row.Sleep1EndTimestamp, day_str, day_after_str)
                        self.plot_sleep_span(ax, row.Sleep2BeginTimestamp, row.Sleep2EndTimestamp, day_str, day_after_str)
                        self.plot_sleep_span(ax, row.Sleep3BeginTimestamp, row.Sleep3EndTimestamp, day_str, day_after_str)
            
            ###########################################################
            # Plot mgt data
            ###########################################################
            if mgt_df is not None:
                day_mgt_df = mgt_df[day_str:day_after_str]
                if len(day_mgt_df) > 0:
                    for index, row in day_mgt_df.iterrows():
                        ax[0].axvline(x=pd.to_datetime(index), color='red', linestyle='--')
                        ax[1].axvline(x=pd.to_datetime(index), color='red', linestyle='--')
                        ax[2].axvline(x=pd.to_datetime(index), color='red', linestyle='--')
                        
                        plt_str = 'loc:' + str(day_mgt_df.location_mgt.values[0]) + ',' + \
                                  'pos:' + str(day_mgt_df.pos_af_mgt.values[0]) + ',' + \
                                  'neg:' + str(day_mgt_df.neg_af_mgt.values[0]) + ', \n' + \
                                  'stress:' + str(day_mgt_df.stress_mgt.values[0]) + ',' + \
                                  'anx:' + str(day_mgt_df.anxiety_mgt.values[0])

                        if pd.to_datetime(index).hour > 22:
                            ax[0].text(pd.to_datetime(index) - timedelta(hours=1), np.nanmax(np.array(day_data_array)[:, 0]) + 15, plt_str)
                        else:
                            ax[0].text(pd.to_datetime(index), np.nanmax(np.array(day_data_array)[:, 0]) + 15, plt_str)
            
            # plt.tight_layout()
            
            plt.savefig(os.path.join(save_folder, participant_id, day_str + '.png'), dpi=300)
            plt.close()

    def plot_cluster(self, participant_id, fitbit_df=None, realizd_df=None, owl_in_one_df=None, segmentation_df=None,
                     fitbit_summary_df=None, mgt_df=None, omsignal_data_df=None, audio_df=None, cluster_df=None, save_folder=None):
            
        ###########################################################
        # If folder not exist
        ###########################################################
        if save_folder is None:
            save_folder = os.path.join(self.data_config.fitbit_sensor_dict['clustering_path'], participant_id)
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
    
        ###########################################################
        # Read data
        ###########################################################
        fitbit_df = fitbit_df.sort_index()
        interval = int((pd.to_datetime(fitbit_df.index[1]) - pd.to_datetime(fitbit_df.index[0])).total_seconds() / 60)
    
        ###########################################################
        # Define plot range
        ###########################################################
        start_str = fitbit_df.index[0]
        end_str = fitbit_df.index[-1]
        day_offset = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days
    
        for day in range(day_offset):
            ###########################################################
            # Define start and end string
            ###########################################################
            day_start_str = (pd.to_datetime(start_str) + timedelta(days=day)).strftime(date_time_format)[:-3]
            day_end_str = (pd.to_datetime(start_str) + timedelta(days=day + 1)).strftime(date_time_format)[:-3]
        
            day_str = (pd.to_datetime(day_start_str)).strftime(date_only_date_time_format)
            day_before_str = (pd.to_datetime(day_start_str) - timedelta(days=1)).strftime(date_only_date_time_format)
            day_after_str = (pd.to_datetime(day_start_str) + timedelta(days=1)).strftime(date_only_date_time_format)
        
            ###########################################################
            # Read fitbit data in the range
            ###########################################################
            fitbit_window_df = fitbit_df[day_start_str:day_end_str]
            fitbit_window_array = np.array(fitbit_window_df)
            fitbit_window_time_array = pd.to_datetime(fitbit_window_df.index)

            ###########################################################
            # If we have over 10% of missing data, continue
            ###########################################################
            if len(fitbit_window_df) < 60 * 4:
                continue

            if len(np.where(fitbit_window_array[:, 1] < 0)[0]) / len(fitbit_window_time_array) > 0.1:
                continue
            fitbit_window_array[np.where(fitbit_window_array[:, 1] < 0)[0], :] = -25

            ###########################################################
            # Plot
            ###########################################################
            f, ax = plt.subplots(4, figsize=(18, 8))

            ###########################################################
            # Plot Fitbit data
            ###########################################################
            self.plot_sensor_data(ax[0], fitbit_window_df, day_start_str, day_end_str, stream='fitbit')

            ###########################################################
            # Plot omsignal data
            ###########################################################
            self.plot_sensor_data(ax[1], omsignal_data_df, day_start_str, day_end_str, stream='om_signal')

            ###########################################################
            # Plot realizd data
            ###########################################################
            self.plot_sensor_data(ax[2], realizd_df, day_start_str, day_end_str, stream='realizd')

            ###########################################################
            # Plot owl_in_one data
            ###########################################################
            self.plot_sensor_data(ax, owl_in_one_df, day_start_str, day_end_str, stream='owl_in_one')
            ax[3].set_xlabel(self.primary_unit + '(' + self.shift + ')')

            ###########################################################
            # Plot audio data
            ###########################################################
            self.plot_sensor_data(ax[3], audio_df, day_start_str, day_end_str, stream='audio')

            ###########################################################
            # Plot sleep data
            ###########################################################
            self.plot_sensor_data(ax, fitbit_summary_df, day_start_str, day_end_str, stream='sleep')

            ###########################################################
            # Plot segment data data
            ###########################################################
            self.plot_segment_data(ax, segmentation_df, day_start_str, day_end_str)
            
            ###########################################################
            # Plot mgt data
            ###########################################################
            self.plot_mgt_data(ax, mgt_df, day_start_str, day_after_str, np.nanmax(np.array(fitbit_window_array)[:, 0]) + 15)
            
            ###########################################################
            # Plot cluster data
            ###########################################################
            self.plot_cluster_data(ax, cluster_df, day_before_str, day_after_str, day_str)

            for i in range(4):
                ax[i].set_xlim([pd.to_datetime(day_start_str), pd.to_datetime(day_end_str)])

            # ax[0].set_title(participant_id)
            
            plt.savefig(os.path.join(save_folder, day_str + '.png'), dpi=300)
            plt.close()
            
    def plot_cluster_data(self, ax, cluster_df, start_str, end_str, day_str):
    
        if cluster_df is not None:
            day_cluster_df = cluster_df[start_str:end_str]
            if len(day_cluster_df) > 0:
                for index, row_cluster in day_cluster_df.iterrows():
                    for i in range(4):
                        ymin, ymax = ax[i].get_ylim()
                        self.plot_cluster_span(ax[i], row_cluster.start, row_cluster.end, row_cluster.cluster_id, day_str, end_str)
                        ax[i].set_ylim([ymin, ymax])

    def plot_segment_data(self, ax, segmentation_df, start_str, end_str):
        if segmentation_df is not None:
            plt_df = segmentation_df[start_str:end_str]
            for xc in pd.to_datetime(list(plt_df.index)):
                for i in range(4):
                    ax[i].axvline(x=xc, color='k', linestyle='--')
                    
    def plot_mgt_data(self, ax, mgt_df, start_str, end_str, max):
        
        if mgt_df is not None:
            day_mgt_df = mgt_df[start_str:end_str]
            if len(day_mgt_df) == 0:
                return
            for index, row in day_mgt_df.iterrows():
                for i in range(4):
                    ax[i].axvline(x=pd.to_datetime(index), color='red', linestyle='--')
            
                plt_str = 'loc:' + str(day_mgt_df.location_mgt.values[0]) + ',' + 'pos:' + str(day_mgt_df.pos_af_mgt.values[0]) + ',' + \
                          'neg:' + str(day_mgt_df.neg_af_mgt.values[0]) + ', \n' + 'stress:' + str(day_mgt_df.stress_mgt.values[0]) + ',' + \
                          'anx:' + str(day_mgt_df.anxiety_mgt.values[0])
            
                if pd.to_datetime(index).hour > 22:
                    ax[0].text(pd.to_datetime(index) - timedelta(hours=1), max, plt_str)
                else:
                    ax[0].text(pd.to_datetime(index), max, plt_str)

    def plot_app_survey(self, participant_id, fitbit_df=None, realizd_df=None, owl_in_one_df=None, segmentation_df=None,
                        fitbit_summary_df=None, app_survey_df=None, omsignal_data_df=None, audio_df=None, cluster_df=None):
    
        ###########################################################
        # If folder not exist
        ###########################################################
        save_folder = os.path.join(self.data_config.fitbit_sensor_dict['plot_path'], self.primary_unit + '_' + self.shift + '_' + participant_id)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
    
        ###########################################################
        # Read data
        ###########################################################
        fitbit_df = fitbit_df.sort_index()
        
        if len(app_survey_df) < 15:
            return

        if owl_in_one_df is None:
            return
        
        for index, day_app_survey in app_survey_df.iterrows():
            ###########################################################
            # Define plot range
            ###########################################################
            # 'Question 12'
            utc_end_str = pd.to_datetime(day_app_survey.completed_ts_utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            end_str = util.GetLocalTimestamp(utc_end_str)
            start_str = (pd.to_datetime(end_str) - timedelta(hours=4)).strftime(date_time_format)[:-3]

            ###########################################################
            # Read data in the range
            ###########################################################
            fitbit_window_df = fitbit_df[start_str:end_str]
            fitbit_window_array = np.array(fitbit_window_df)
            fitbit_window_time_array = pd.to_datetime(fitbit_window_df.index)
            # If we have over 10% of missing data
            if len(fitbit_window_df) < 60 * 4:
                continue
            
            if len(owl_in_one_df[start_str:end_str]) < 60 * 3:
                continue
            
            if len(np.where(fitbit_window_array[:, 1] < 0)[0]) / len(fitbit_window_time_array) > 0.1:
                continue
            fitbit_window_array[np.where(fitbit_window_array[:, 1] < 0)[0], :] = -25
            
            ###########################################################
            # Plot
            ###########################################################
            f, ax = plt.subplots(4, figsize=(18, 8))

            ###########################################################
            # Plot Fitbit data
            ###########################################################
            self.plot_sensor_data(ax[0], fitbit_window_df, start_str, end_str, stream='fitbit')
            
            ###########################################################
            # Plot omsignal data
            ###########################################################
            self.plot_sensor_data(ax[1], omsignal_data_df, start_str, end_str, stream='om_signal')
            
            ###########################################################
            # Plot realizd data
            ###########################################################
            self.plot_sensor_data(ax[2], realizd_df, start_str, end_str, stream='realizd')
            
            ###########################################################
            # Plot owl_in_one data
            ###########################################################
            self.plot_sensor_data(ax, owl_in_one_df, start_str, end_str, stream='owl_in_one')
            ax[3].set_xlabel(self.primary_unit + '(' + self.shift + ')')

            ###########################################################
            # Plot audio data
            ###########################################################
            self.plot_sensor_data(ax[3], audio_df, start_str, end_str, stream='audio')
            
            ###########################################################
            # Plot owl_in_one data
            ###########################################################
            self.plot_sensor_data(ax, fitbit_summary_df, start_str, end_str, stream='sleep')

            ###########################################################
            # Plot segment data data
            ###########################################################
            plt_df = segmentation_df[start_str:end_str]
            for xc in pd.to_datetime(list(plt_df.index)):
                for i in range(4):
                    ax[i].axvline(x=xc, color='k', linestyle='--')
            
            question_list = ['Question 3', 'Question 4', 'Question 5', 'Question 6', 'Question 7',
                             'Question 8', 'Question 9', 'Question 10', 'Question 11', 'Question 12',
                             'Question 13', 'Question 14', 'Question 15']
            score = np.nanmean(np.array(day_app_survey[question_list], dtype=float).reshape([-1]))
            for i in range(4):
                ax[i].axvline(x=pd.to_datetime(end_str) - timedelta(hours=4), color='red', linestyle='--')
                # ax[i].axvline(x=pd.to_datetime(end_str) - timedelta(hours=4), color='blue', linestyle='--')
                ax[i].set_xlim([pd.to_datetime(start_str), pd.to_datetime(end_str)])
                
            ax[0].set_title('PF score:' + str(score))
            plt.savefig(os.path.join(save_folder, start_str + '.png'), dpi=300)
            plt.close()

    def plot_sensor_data(self, ax, data_df, start_str, end_str, stream='realizd'):
        """ Plot sensor data based on the stream type

        Params:
        ax - axis to plot
        data_df - whole data dataframe for one stream
        start_str - start time in str
        end_str - end time in str
        stream - stream type

        Returns:
        """
        if data_df is not None:
    
            sleep_start_str = (pd.to_datetime(start_str) - timedelta(days=1)).strftime(date_time_format)[:-3]
            sleep_end_str = (pd.to_datetime(end_str) + timedelta(days=1)).strftime(date_time_format)[:-3]
    
            if stream == 'sleep':
                data_window_df = data_df[sleep_start_str:sleep_end_str]
            else:
                data_window_df = data_df[start_str:end_str]
            data_window_time_array = pd.to_datetime(data_window_df.index)
            
            # If there is no data, just return
            if len(data_window_df) == 0:
                return
            
            if stream == 'realizd':
                # Plot realizd
                ax.plot(data_window_time_array, np.array(data_window_df.SecondsOnPhone), label='phone usage', color='green')
            elif stream == 'om_signal':
                # Plot om_signal
                ax.plot(data_window_time_array, np.array(data_window_df)[:, 0], label='om heart rate', color='purple')
                ax.plot(data_window_time_array, np.array(data_window_df)[:, 1], label='om step count', color='brown')
                ax.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
            elif stream == 'audio':
                # Plot audio
                ax.plot(data_window_time_array, np.array(data_window_df.foreground), label='foreground', color='blue')
                ax.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
            elif stream == 'owl_in_one':
                # Plot owl_in_one
                unit_in_time = np.where(np.array(data_window_df) > 0)[1]
                unit_diff_in_time = unit_in_time[1:] - unit_in_time[:-1]
    
                unit_change_time_array = np.where(unit_diff_in_time != 0)[0]
                unit_change_time_array = np.append(unit_change_time_array, len(data_window_df) - 2)
    
                unit_time_df = pd.DataFrame()
                start_time, unit_type = data_window_df.index[0], data_window_df.columns[unit_in_time[0]]
                
                for unit_change_time in unit_change_time_array:
    
                    end_time = data_window_df.index[unit_change_time + 1]
                    unit_row_df = pd.DataFrame(index=[start_time])
                    unit_row_df['start'], unit_row_df['end'], unit_row_df['unit'] = start_time, end_time, unit_type
                    unit_time_df = unit_time_df.append(unit_row_df)
                    
                    owl_in_one_start_str = (pd.to_datetime(start_str) - timedelta(hours=12)).strftime(date_time_format)[:-3]
                    owl_in_one_end_str = (pd.to_datetime(end_str) + timedelta(hours=12)).strftime(date_time_format)[:-3]
                    if unit_type != 'unknown':
                        self.plot_owl_in_one_span(ax, unit_type, start_time, end_time, owl_in_one_start_str, owl_in_one_end_str)
    
                    start_time = data_window_df.index[unit_change_time + 1]
                    unit_type = data_window_df.columns[unit_in_time[unit_change_time + 1]]

                legend_elements = [Patch(facecolor=color_dict[loc], label=loc, alpha=0.3) for loc in list(color_dict.keys())]
                ax[2].legend(handles=legend_elements, bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
            elif stream == 'sleep':
                for index, row in data_window_df.iterrows():
                    self.plot_sleep_span(ax, row.Sleep1BeginTimestamp, row.Sleep1EndTimestamp, start_str, end_str)
                    self.plot_sleep_span(ax, row.Sleep2BeginTimestamp, row.Sleep2EndTimestamp, start_str, end_str)
                    self.plot_sleep_span(ax, row.Sleep3BeginTimestamp, row.Sleep3EndTimestamp, start_str, end_str)
            else:
                # Plot fitbit
                interval = int((pd.to_datetime(data_df.index[1]) - pd.to_datetime(data_df.index[0])).total_seconds() / 60)
                data_window_array = np.array(data_window_df)
                data_window_array[:, 1] = data_window_array[:, 1] / int(interval)
                data_window_array[np.where(data_window_array[:, 1] < 0)[0], :] = -25
                ax.plot(data_window_time_array, data_window_array[:, 0], label='heart rate')
                ax.plot(data_window_time_array, data_window_array[:, 1], label='step count')
                ax.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)

    def plot_owl_in_one_span(self, ax, room_type, begin_str, end_str, day_str, day_after_str):
        """ Plot owl_in_one span

        Params:
        ax - axis to plot
        room_type - room type
        begin_str - start time in str
        end_str - end time in str
        day_str -  start time in str
        day_after_str - end time in str

        Returns:
        """
        color = color_dict[room_type]
        
        condition1 = pd.to_datetime(day_str) < pd.to_datetime(begin_str) < pd.to_datetime(day_after_str)
        condition2 = pd.to_datetime(day_str) < pd.to_datetime(end_str) < pd.to_datetime(day_after_str)
        condition3 = (pd.to_datetime(end_str) - pd.to_datetime(begin_str)).total_seconds() > 60 * 0
        condition4 = (pd.to_datetime(end_str) - pd.to_datetime(begin_str)).total_seconds() < 60 * 60 * 4
        
        if condition3 and condition4 is False:
            return
        
        if condition1 is True and condition2 is True:
            
            ax[0].axvspan(pd.to_datetime(begin_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            ax[1].axvspan(pd.to_datetime(begin_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            ax[2].axvspan(pd.to_datetime(begin_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0, label=room_type)
            if len(ax) > 3:
                ax[3].axvspan(pd.to_datetime(begin_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
        
        elif condition1 is True:
            ax[0].axvline(x=pd.to_datetime(begin_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(begin_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(begin_str), color='blue', linestyle='--')
            if len(ax) > 3:
                ax[3].axvline(x=pd.to_datetime(begin_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(begin_str), pd.to_datetime(day_after_str), alpha=0.3, color=color, lw=0)
            ax[1].axvspan(pd.to_datetime(begin_str), pd.to_datetime(day_after_str), alpha=0.3, color=color, lw=0)
            ax[2].axvspan(pd.to_datetime(begin_str), pd.to_datetime(day_after_str), alpha=0.3, color=color, label=room_type, lw=0)
            if len(ax) > 3:
                ax[3].axvspan(pd.to_datetime(begin_str), pd.to_datetime(day_after_str), alpha=0.3, color=color, lw=0)
    
        elif condition2 is True:
            ax[0].axvline(x=pd.to_datetime(end_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(end_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(end_str), color='blue', linestyle='--')
            if len(ax) > 3:
                ax[3].axvline(x=pd.to_datetime(end_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(day_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            ax[1].axvspan(pd.to_datetime(day_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            ax[2].axvspan(pd.to_datetime(day_str), pd.to_datetime(end_str), alpha=0.3, color=color, label=room_type, lw=0)
            if len(ax) > 3:
                ax[3].axvspan(pd.to_datetime(day_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            
    def plot_cluster_span(self, ax, cluster_begin_str, cluster_end_str, cluster_id, day_str, day_after_str):
        condition1 = pd.to_datetime(day_str) < pd.to_datetime(cluster_begin_str) < pd.to_datetime(day_after_str)
        condition2 = pd.to_datetime(day_str) < pd.to_datetime(cluster_end_str) < pd.to_datetime(day_after_str)
        ymin, ymax = ax.get_ylim()
        
        if condition1 is True and condition2 is True:
            ax.hlines(y=ymax, color=cluster_color_list[cluster_id], linewidth=20,
                      xmin=pd.to_datetime(cluster_begin_str), xmax=pd.to_datetime(cluster_end_str))

        elif condition1 is True:
            ax.hlines(y=ymax, color=cluster_color_list[cluster_id], linewidth=20,
                      xmin=pd.to_datetime(cluster_begin_str), xmax=pd.to_datetime(day_after_str))
            
        elif condition2 is True:
            ax.hlines(y=ymax, color=cluster_color_list[cluster_id], linewidth=20,
                      xmin=pd.to_datetime(day_str), xmax=pd.to_datetime(cluster_end_str))
            
    def plot_sleep_span(self, ax, sleep_begin_str, sleep_end_str, day_str, day_after_str):
    
        condition1 = pd.to_datetime(day_str) < pd.to_datetime(sleep_begin_str) < pd.to_datetime(day_after_str)
        condition2 = pd.to_datetime(day_str) < pd.to_datetime(sleep_end_str) < pd.to_datetime(day_after_str)
    
        if condition1 is True and condition2 is True:
            ax[0].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[3].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
        
            ax[0].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[3].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[1].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[2].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[3].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
    
        elif condition1 is True:
            ax[0].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[3].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(day_after_str), alpha=0.1, color='red')
            ax[1].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(day_after_str), alpha=0.1, color='red')
            ax[2].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(day_after_str), alpha=0.1, color='red')
            ax[3].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(day_after_str), alpha=0.1, color='red')
    
        elif condition2 is True:
            ax[0].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[3].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(day_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[1].axvspan(pd.to_datetime(day_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[2].axvspan(pd.to_datetime(day_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[3].axvspan(pd.to_datetime(day_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')

    def plot_ticc(self, participant_dict, fitbit_df=None, cluster_df=None):
    
        ###########################################################
        # If folder not exist
        ###########################################################
        save_folder = os.path.join(self.data_config.fitbit_sensor_dict['clustering_path'], participant_dict['participant_id'])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        ###########################################################
        # Read data
        ###########################################################
        fitbit_df = fitbit_df.sort_index()
        
        ###########################################################
        # Define plot range
        ###########################################################
        start_str = fitbit_df.index[0]
        end_str = fitbit_df.index[-1]
        day_offset = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days
        day_plot_list = []

        for day in range(day_offset):
            ###########################################################
            # Define start and end string
            ###########################################################
            day_start_str = (pd.to_datetime(start_str) + timedelta(days=day)).strftime(date_time_format)[:-3]
            day_end_str = (pd.to_datetime(start_str) + timedelta(days=day + 1)).strftime(date_time_format)[:-3]
            
            if cluster_df is not None:
                day_cluster_df = cluster_df[day_start_str:day_end_str]
                if len(day_cluster_df) > 0:
                    heart_rate_array = np.array(day_cluster_df.HeartRatePPG)
                    invalid_heart_len = len(np.where(heart_rate_array < 0)[0])
                    
                    if invalid_heart_len / len(heart_rate_array) < 0.15:
                        day_plot_list.append(day_start_str)

        if len(day_plot_list) < 5:
            return
        
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111)
        ax.set_title('Cluster for participant: ' + participant_dict['participant_id'] + '\n'
                     + '(' + 'uid: ' + participant_dict['uid'] + ', shift:' + participant_dict['shift']
                     + '_' + participant_dict['primary_unit'] + ')')
        ax.axis('tight')

        ax.set_ylim(ymin=-0.5, ymax=len(day_plot_list) + 0.5)
        
        for i in range(len(day_plot_list)):
    
            ###########################################################
            # Define start and end string
            ###########################################################
            day_start_str = (pd.to_datetime(day_plot_list[i])).strftime(date_time_format)[:-3]
            day_end_str = (pd.to_datetime(day_plot_list[i]) + timedelta(days=1)).strftime(date_time_format)[:-3]

            ###########################################################
            # Plot cluster data
            ###########################################################
            if cluster_df is not None:
                day_cluster_df = cluster_df[day_start_str:day_end_str]
                if len(day_cluster_df) > 0:
                    day_cluster_diff = np.array(day_cluster_df.cluster[1:]) - np.array(day_cluster_df.cluster[:-1])
                    change_point_array = np.where(day_cluster_diff != 0)[0]
                    cluster_array = np.array(day_cluster_df.cluster)[change_point_array]
                    cluster_array = np.array(cluster_array).reshape([1, len(cluster_array)])

                    final_plot_array = np.zeros([len(change_point_array) + 1, 3])

                    final_plot_array[1:len(change_point_array)+1, 0] = np.array(change_point_array).reshape([1, len(change_point_array)])
                    final_plot_array[:len(change_point_array), 1] = np.array(change_point_array).reshape([1, len(change_point_array)])
                    final_plot_array[len(change_point_array), 1] = len(day_cluster_df)
                    final_plot_array[:len(change_point_array), 2] = cluster_array
                    final_plot_array[len(change_point_array), 2] = np.array(day_cluster_df.cluster)[-1]
                    
                    for plot_array in final_plot_array:
                        if plot_array[2] >= 0:
                            ax.barh(i + 0.5, (plot_array[1] - plot_array[0]) / 60, left=plot_array[0] / 60, height=0.8,
                                    align='center', color=cluster_color_list[int(plot_array[2])])
                ax.set_xlim(xmin=0, xmax=24)
            plt.savefig(os.path.join(save_folder, participant_dict['participant_id'] + '.png'), dpi=300)
        plt.close()

    def plot_ticc_hmm(self, participant_dict, fitbit_df=None, cluster_df=None):
    
        ###########################################################
        # If folder not exist
        ###########################################################
        save_folder = os.path.join(self.data_config.fitbit_sensor_dict['clustering_path'], participant_dict['participant_id'])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        
        '''
        category_sequence = np.array([list(np.array(cluster_df.cluster_correct).astype(int))]).T

        category_sequence[np.where(category_sequence == 100)[0]] = 1
        model = hmm.MultinomialHMM(n_components=4, n_iter=300)
        model.fit(category_sequence)
        # model.decode(category_sequence, algorithm="viterbi")
        Z2 = model.predict(category_sequence)
        '''
        
        if len(cluster_df) > 0:
            cluster_diff = np.array(cluster_df.cluster_correct[1:]) - np.array(cluster_df.cluster_correct[:-1])
            change_point_array = np.where(cluster_diff != 0)[0]
            cluster_array = np.array(cluster_df.cluster_correct)[change_point_array]
            cluster_array = np.array(cluster_array).reshape([-1, 1])
            cluster_array = np.append(cluster_array, np.array(cluster_df.cluster_correct)[-1])
            
            invalid_point = np.where(cluster_array == 100)[0]
            cluster_array[invalid_point] = 1
            
            hmm_training = []
            hmm_len = []
            for i in range(int(len(cluster_array) / 300)):
                
                tmp_seq = []
                for x in np.array(cluster_array[i*300:(i+1)*300]):
                    tmp_seq.append([int(x)])
                # category_sequence = np.array([list(np.array(cluster_array[i*100:(i+1)*100]).astype(int))])
                
                if len(hmm_training) == 0:
                    hmm_training = tmp_seq
                else:
                    hmm_training = np.concatenate([hmm_training, tmp_seq])
                hmm_len.append(len(tmp_seq))

            change_point_index_list = [0]
            for change_point_index in list(cluster_df.index[change_point_array]):
                change_point_index_list.append(change_point_index)
            change_point_index_list.append(cluster_df.index[-1])
            
            hmm_df = pd.DataFrame(index=change_point_index_list[:-1])
            hmm_df['start'] = change_point_index_list[:-1]
            hmm_df['end'] = change_point_index_list[1:]

            category_sequence = np.array([list(np.array(cluster_array).astype(int))]).T

            model = hmm.MultinomialHMM(n_components=6, n_iter=3000)
            model.fit(hmm_training, hmm_len)
            decode_array = model.decode(hmm_training, hmm_len, algorithm="viterbi")
            Z2 = model.predict(hmm_training)

            hmm_df.loc[change_point_index_list[:-1], 'hidden_state'] = Z2
            
        ###########################################################
        # Read data
        ###########################################################
        fitbit_df = fitbit_df.sort_index()
    
        ###########################################################
        # Define plot range
        ###########################################################
        start_str = fitbit_df.index[0]
        end_str = fitbit_df.index[-1]
        day_offset = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).days
        day_plot_list = []
    
        for day in range(day_offset):
            ###########################################################
            # Define start and end string
            ###########################################################
            day_start_str = (pd.to_datetime(start_str) + timedelta(days=day)).strftime(date_time_format)[:-3]
            day_end_str = (pd.to_datetime(start_str) + timedelta(days=day + 1)).strftime(date_time_format)[:-3]
        
            if cluster_df is not None:
                day_cluster_df = cluster_df[day_start_str:day_end_str]
                if len(day_cluster_df) > 0:
                    heart_rate_array = np.array(day_cluster_df.HeartRatePPG)
                    invalid_heart_len = len(np.where(heart_rate_array < 0)[0])
                
                    if invalid_heart_len / len(heart_rate_array) < 0.15:
                        day_plot_list.append(day_start_str)
    
        if len(day_plot_list) < 5:
            return
    
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111)
        ax.set_title('Cluster for participant: ' + participant_dict['participant_id'] + '\n'
                     + '(' + 'uid: ' + participant_dict['uid'] + ', shift:' + participant_dict['shift']
                     + '_' + participant_dict['primary_unit'] + ')')
        ax.axis('tight')
    
        ax.set_ylim(ymin=-0.5, ymax=len(day_plot_list) + 0.5)
    
        for i in range(len(day_plot_list)):
        
            ###########################################################
            # Define start and end string
            ###########################################################
            day_start_str = (pd.to_datetime(day_plot_list[i])).strftime(date_time_format)[:-3]
            day_end_str = (pd.to_datetime(day_plot_list[i]) + timedelta(days=1)).strftime(date_time_format)[:-3]
        
            ###########################################################
            # Plot cluster data
            ###########################################################
            if cluster_df is not None:
                day_cluster_df = cluster_df[day_start_str:day_end_str]
                if len(day_cluster_df) > 0:
                    day_cluster_diff = np.array(day_cluster_df.hidden_state[1:]) - np.array(day_cluster_df.hidden_state[:-1])
                    change_point_array = np.where(day_cluster_diff != 0)[0]
                    cluster_array = np.array(day_cluster_df.hidden_state)[change_point_array]
                    cluster_array = np.array(cluster_array).reshape([1, len(cluster_array)])
                
                    final_plot_array = np.zeros([len(change_point_array) + 1, 4])
                
                    final_plot_array[1:len(change_point_array) + 1, 0] = np.array(change_point_array).reshape([1, len(change_point_array)])
                    final_plot_array[:len(change_point_array), 1] = np.array(change_point_array).reshape([1, len(change_point_array)])
                    final_plot_array[len(change_point_array), 1] = len(day_cluster_df)
                    final_plot_array[:len(change_point_array), 3] = cluster_array
                    final_plot_array[len(change_point_array), 3] = np.array(day_cluster_df.hidden_state)[-1]
                
                    for plot_array in final_plot_array:
                        if plot_array[3] >= 0:
                            ax.barh(i + 0.5, (plot_array[1] - plot_array[0]) / 60, left=plot_array[0] / 60, height=0.8, align='center', color=cluster_color_list[int(plot_array[3])])
            ax.set_xlim(xmin=0, xmax=24)
            plt.savefig(os.path.join(save_folder, participant_dict['participant_id'] + '_hmm.png'), dpi=300)
        plt.close()
