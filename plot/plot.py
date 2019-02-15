import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta
from matplotlib.patches import Patch

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

color_dict = {'lounge': 'green', 'ns': 'gold', 'pat': 'grey',
              'other_floor': 'violet', 'med': 'blue', 'floor2': 'cyan'}

class Plot(object):
    
    def __init__(self, fitbit_config=None, ggs_config=None, participant_id_list=None, realizd_config=None,
                 primary_unit=None):
        """
        Initialization method
        """
        ###########################################################
        # Assert if these parameters are not parsed
        ###########################################################
        self.fitbit_config = fitbit_config
        self.ggs_config = ggs_config
        self.realizd_config = realizd_config
        self.primary_unit = primary_unit
        
        self.participant_id_list = participant_id_list
    
    def plot_segmentation(self, participant_id, fitbit_df=None, realizd_df=None, owl_in_one_df=None,
                          fitbit_summary_df=None, mgt_df=None, omsignal_data_df=None):
        ###########################################################
        # If folder not exist
        ###########################################################
        save_folder = os.path.join(self.ggs_config.process_folder)
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
            
    def plot_owl_in_one_span(self, ax, room_type, begin_str, end_str, day_str, day_after_str):
        
        color = color_dict[room_type]
        
        condition1 = pd.to_datetime(day_str) < pd.to_datetime(begin_str) < pd.to_datetime(day_after_str)
        condition2 = pd.to_datetime(day_str) < pd.to_datetime(end_str) < pd.to_datetime(day_after_str)
        condition3 = (pd.to_datetime(end_str) - pd.to_datetime(begin_str)).total_seconds() > 60 * 0
        condition4 = (pd.to_datetime(end_str) - pd.to_datetime(begin_str)).total_seconds() < 60 * 60 * 2
        
        if condition3 and condition4 is False:
            return
        
        if condition1 is True and condition2 is True:
            
            ax[0].axvspan(pd.to_datetime(begin_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            ax[1].axvspan(pd.to_datetime(begin_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            ax[2].axvspan(pd.to_datetime(begin_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0, label=room_type)
        
        elif condition1 is True:
            ax[0].axvline(x=pd.to_datetime(begin_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(begin_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(begin_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(begin_str), pd.to_datetime(day_after_str), alpha=0.3, color=color, lw=0)
            ax[1].axvspan(pd.to_datetime(begin_str), pd.to_datetime(day_after_str), alpha=0.3, color=color, lw=0)
            ax[2].axvspan(pd.to_datetime(begin_str), pd.to_datetime(day_after_str), alpha=0.3, color=color, label=room_type, lw=0)
    
        elif condition2 is True:
            ax[0].axvline(x=pd.to_datetime(end_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(end_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(end_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(day_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            ax[1].axvspan(pd.to_datetime(day_str), pd.to_datetime(end_str), alpha=0.3, color=color, lw=0)
            ax[2].axvspan(pd.to_datetime(day_str), pd.to_datetime(end_str), alpha=0.3, color=color, label=room_type, lw=0)

    def plot_sleep_span(self, ax, sleep_begin_str, sleep_end_str, day_str, day_after_str):
    
        condition1 = pd.to_datetime(day_str) < pd.to_datetime(sleep_begin_str) < pd.to_datetime(day_after_str)
        condition2 = pd.to_datetime(day_str) < pd.to_datetime(sleep_end_str) < pd.to_datetime(day_after_str)
    
        if condition1 is True and condition2 is True:
            ax[0].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
        
            ax[0].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[1].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[2].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
    
        elif condition1 is True:
            ax[0].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(sleep_begin_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(day_after_str), alpha=0.1, color='red')
            ax[1].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(day_after_str), alpha=0.1, color='red')
            ax[2].axvspan(pd.to_datetime(sleep_begin_str), pd.to_datetime(day_after_str), alpha=0.1, color='red')
    
        elif condition2 is True:
            ax[0].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[1].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
            ax[2].axvline(x=pd.to_datetime(sleep_end_str), color='blue', linestyle='--')
        
            ax[0].axvspan(pd.to_datetime(day_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[1].axvspan(pd.to_datetime(day_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')
            ax[2].axvspan(pd.to_datetime(day_str), pd.to_datetime(sleep_end_str), alpha=0.1, color='red')