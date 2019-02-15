"""
Top level classes for the preprocess model.
"""
from __future__ import print_function

import os
import sys

###########################################################
# Change to your own pyspark path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'preprocess')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'plot')))


from om_signal.helper import *
from fitbit.helper import *
from realizd.helper import *
import pandas as pd

from GGS.ggs import *
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Segmentation']


class Segmentation(object):
    
    def __init__(self, read_config=None, save_config=None, participant_id=None, realizd_config=None):
        """
        Initialization method
        """
        ###########################################################
        # Assert if these parameters are not parsed
        ###########################################################
        assert read_config is not None and save_config is not None and participant_id is not None
        
        self.read_config = read_config
        self.save_config = save_config
        self.realizd_config = realizd_config
        
        self.participant_id = participant_id

        self.processed_data_dict_array = {}
    
    def segment_data(self, participant_id):
    
        ###########################################################
        # If folder not exist
        ###########################################################
        save_participant_folder = os.path.join(self.save_config.process_folder)
        if not os.path.exists(save_participant_folder):
            os.mkdir(save_participant_folder)

        ###########################################################
        # Read data
        ###########################################################
        self.processed_data_dict_array[participant_id]['data'] = self.processed_data_dict_array[participant_id]['data'].sort_index()
        data = np.array(self.processed_data_dict_array[participant_id]['data']).astype(float)

        ###########################################################
        # Normalizing
        ###########################################################
        mean = self.processed_data_dict_array[participant_id]['mean']
        std = self.processed_data_dict_array[participant_id]['std']
        norm_data = np.divide(data - mean, std)

        ###########################################################
        # Convert to an n-by-T matrix
        ###########################################################
        norm_data = norm_data.T

        ###########################################################
        # Find breakpoints with lambda = segmentation_lamb
        ###########################################################
        if not os.path.exists(os.path.join(save_participant_folder, participant_id + '.csv.gz')):
        
            bps, objectives = GGS(norm_data, Kmax=int(data.shape[0] / 5), lamb=self.save_config.segmentation_lamb)
            bps_df = pd.DataFrame(self.processed_data_dict_array[participant_id]['data'].index[bps[:-1]], columns=['time'],
                                  index=self.processed_data_dict_array[participant_id]['data'].index[bps[:-1]])
            bps_df['objectives'] = objectives
            bps_df.to_csv(os.path.join(save_participant_folder, participant_id + '.csv.gz'), compression='gzip')
    
            ###########################################################
            # Define plot range
            ###########################################################
            start_str = self.processed_data_dict_array[participant_id]['data'].index[0]
            end_str = self.processed_data_dict_array[participant_id]['data'].index[720]
            plt_df = bps_df[start_str:end_str]
            xposition = bps[:len(plt_df)]
            
            ###########################################################
            # Plot
            ###########################################################
            plt.figure(figsize=(18, 4))
            plt.plot(np.array(data[:720, :]))
            for xc in xposition:
                plt.axvline(x=xc, color='k', linestyle='--')
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(save_participant_folder, participant_id + '.png'), dpi=300)
            plt.close()
            return True
        else:
            return False
        
    def read_preprocess_data(self, participant_id):
        """
        Read preprocessed data
        """
        ###########################################################
        # If folder not exist
        ###########################################################
        save_participant_folder = os.path.join(self.save_config.process_folder, participant_id)
        if not os.path.exists(save_participant_folder):
            os.mkdir(save_participant_folder)
        
        read_participant_folder = os.path.join(self.read_config.process_folder, participant_id)
        if not os.path.exists(read_participant_folder):
            return

        ###########################################################
        # List files and remove 'DS' file in mac system
        ###########################################################
        data_file_array = os.listdir(read_participant_folder)
            
        for data_file in data_file_array:
            if 'DS' in data_file: data_file_array.remove(data_file)
    
        if len(data_file_array) > 0:
            ###########################################################
            # Create dict for participant
            ###########################################################
            self.processed_data_dict_array[participant_id] = {}
            self.processed_data_dict_array[participant_id]['data'] = pd.DataFrame()
            
            for data_file in data_file_array:
                ###########################################################
                # Read data and append
                ###########################################################
                csv_path = os.path.join(read_participant_folder, data_file)
                data_df = pd.read_csv(csv_path, index_col=0)

                ###########################################################
                # Pad data
                ###########################################################
                end_str = data_df.index[-1]
                interval = int(self.read_config.offset / 60)
                time_arr = [(pd.to_datetime(end_str) + timedelta(minutes=i*interval)).strftime(date_time_format)[:-3] for i in range(10)]
                
                pad_df = pd.DataFrame(np.zeros([len(time_arr), len(data_df.columns)]), index=time_arr, columns=list(data_df.columns))
                temp = np.random.normal(size=(2, 2))
                temp2 = np.dot(temp, temp.T)
                for j in range(len(pad_df)):
                    data_pad_tmp = np.random.multivariate_normal(np.zeros(2) - 10, temp2)
                    pad_df.loc[time_arr[j], :] = data_pad_tmp

                ###########################################################
                # Append data
                ###########################################################
                self.processed_data_dict_array[participant_id]['data'] = self.processed_data_dict_array[participant_id]['data'].append(data_df)
                self.processed_data_dict_array[participant_id]['data'] = self.processed_data_dict_array[participant_id]['data'].append(pad_df)

                self.processed_data_dict_array[participant_id]['raw'] = self.processed_data_dict_array[participant_id]['data'].append(data_df)
                
            self.processed_data_dict_array[participant_id]['mean'] = np.nanmean(self.processed_data_dict_array[participant_id]['raw'], axis=0)
            self.processed_data_dict_array[participant_id]['std'] = np.nanstd(self.processed_data_dict_array[participant_id]['raw'], axis=0)

    def read_preprocess_data_all(self):
        """
        Read preprocessed data
        """
        ###########################################################
        # If folder not exist
        ###########################################################
        participant_id = self.participant_id
        
        save_participant_folder = os.path.join(self.save_config.process_folder, participant_id)
        if not os.path.exists(save_participant_folder):
            os.mkdir(save_participant_folder)
    
        read_participant_folder = os.path.join(self.read_config.process_folder, participant_id)
        if not os.path.exists(read_participant_folder):
            return
    
        ###########################################################
        # List files and remove 'DS' file in mac system
        ###########################################################
        data_file_array = os.listdir(read_participant_folder)
    
        for data_file in data_file_array:
            if 'DS' in data_file: data_file_array.remove(data_file)

        self.fitbit_df = None
    
        if len(data_file_array) > 0:
            ###########################################################
            # Create dict for participant
            ###########################################################
            self.processed_data_dict_array[participant_id] = {}
            self.processed_data_dict_array[participant_id]['data'] = pd.DataFrame()
            self.processed_data_dict_array[participant_id]['raw'] = pd.DataFrame()

            ###########################################################
            # Read realizd data if exist
            ###########################################################
            if self.realizd_config is not None:
                realizd_participant_folder = os.path.join(self.realizd_config.process_folder, participant_id)
    
                self.realizd_df = None
                if os.path.exists(os.path.join(realizd_participant_folder, participant_id + '.csv.gz')) is True:
                    self.realizd_df = pd.read_csv(os.path.join(realizd_participant_folder, participant_id + '.csv.gz'), index_col=0)
        
            for data_file in data_file_array:
                ###########################################################
                # Read data and append
                ###########################################################
                csv_path = os.path.join(read_participant_folder, data_file)
                data_df = pd.read_csv(csv_path, index_col=0)
            
                ###########################################################
                # Append data
                ###########################################################
                self.processed_data_dict_array[participant_id]['raw'] = self.processed_data_dict_array[participant_id]['raw'].append(data_df)
            
            ###########################################################
            # Assign data
            ###########################################################
            interval = int(self.read_config.offset / 60)
            final_df = self.processed_data_dict_array[participant_id]['raw'].sort_index()
            start_str = pd.to_datetime(final_df.index[0]).replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
            end_str = (pd.to_datetime(final_df.index[-1]) + timedelta(days=1)).replace(hour=0, minute=0, second=0,microsecond=0).strftime(date_time_format)[:-3]

            time_length = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).total_seconds()
            point_length = int(time_length / self.read_config.offset) + 1
            time_arr = [(pd.to_datetime(start_str) + timedelta(minutes=i*interval)).strftime(date_time_format)[:-3] for i in range(0, point_length)]
            
            final_df_all = pd.DataFrame(index=time_arr, columns=final_df.columns)
            final_df_all.loc[final_df.index, :] = final_df

            ###########################################################
            # Assign pad
            ###########################################################
            pad_time_arr = list(set(time_arr) - set(final_df.dropna().index))
            pad_df = pd.DataFrame(np.zeros([len(pad_time_arr), len(final_df_all.columns)]), index=pad_time_arr, columns=list(final_df_all.columns))
            temp = np.random.normal(size=(2, 2))
            temp2 = np.dot(temp, temp.T)
            for j in range(len(pad_df)):
                data_pad_tmp = np.random.multivariate_normal(np.zeros(2) - 50, temp2)
                pad_df.loc[pad_time_arr[j], :] = data_pad_tmp
            final_df_all.loc[pad_df.index, :] = pad_df

            self.processed_data_dict_array[participant_id]['data'] = final_df_all
            self.fitbit_df = final_df_all

            self.processed_data_dict_array[participant_id]['mean'] = np.nanmean(self.processed_data_dict_array[participant_id]['raw'], axis=0)
            self.processed_data_dict_array[participant_id]['std'] = np.nanstd(self.processed_data_dict_array[participant_id]['raw'], axis=0)

    def read_data(self, participant_id):
        """
        Read preprocessed data
        """
        ###########################################################
        # If folder not exist
        ###########################################################
        save_participant_folder = os.path.join(self.save_config.process_folder, participant_id)
        if not os.path.exists(save_participant_folder):
            os.mkdir(save_participant_folder)
    
        read_participant_folder = os.path.join(self.read_config.process_folder, participant_id)
        if not os.path.exists(read_participant_folder):
            return
    
        ###########################################################
        # List files and remove 'DS' file in mac system
        ###########################################################
        data_file_array = os.listdir(read_participant_folder)
    
        for data_file in data_file_array:
            if 'DS' in data_file: data_file_array.remove(data_file)
    
        self.fitbit_df = None
    
        if len(data_file_array) > 0:
            ###########################################################
            # Create dict for participant
            ###########################################################
            self.processed_data_dict_array[participant_id] = {}
            self.processed_data_dict_array[participant_id]['data'] = pd.DataFrame()
            self.processed_data_dict_array[participant_id]['raw'] = pd.DataFrame()
            
            for data_file in data_file_array:
                ###########################################################
                # Read data and append
                ###########################################################
                csv_path = os.path.join(read_participant_folder, data_file)
                data_df = pd.read_csv(csv_path, index_col=0)
            
                ###########################################################
                # Append data
                ###########################################################
                self.processed_data_dict_array[participant_id]['raw'] = self.processed_data_dict_array[participant_id][
                    'raw'].append(data_df)
        
            ###########################################################
            # Assign data
            ###########################################################
            interval = int(self.read_config.offset / 60)
            final_df = self.processed_data_dict_array[participant_id]['raw'].sort_index()
            self.fitbit_df = final_df
     
    def segment_data_by_sleep(self, fitbit_summary_df=None):
    
        participant_id = self.participant_id
    
        ###########################################################
        # If folder not exist
        ###########################################################
        save_participant_folder = os.path.join(self.save_config.process_folder)
        if not os.path.exists(save_participant_folder):
            os.mkdir(save_participant_folder)
    
        ###########################################################
        # Read data
        ###########################################################
        data_df = self.processed_data_dict_array[participant_id]['data'].sort_index()
        data = np.array(data_df).astype(float)
        
        fitbit_summary_df = fitbit_summary_df
        sleep_df = pd.DataFrame()

        ###########################################################
        # Parse sleep df
        ###########################################################
        if fitbit_summary_df is not None:
            if len(fitbit_summary_df) > 0:
                for index, row in fitbit_summary_df.iterrows():
                    return_df = self.add_sleep_data_frame(row.Sleep1BeginTimestamp, row.Sleep1EndTimestamp)
                    if return_df is not None: sleep_df = sleep_df.append(return_df)
                    return_df = self.add_sleep_data_frame(row.Sleep2BeginTimestamp, row.Sleep2EndTimestamp)
                    if return_df is not None: sleep_df = sleep_df.append(return_df)
                    return_df = self.add_sleep_data_frame(row.Sleep3BeginTimestamp, row.Sleep3EndTimestamp)
                    if return_df is not None: sleep_df = sleep_df.append(return_df)
        sleep_df = sleep_df.sort_index()
        
        if len(sleep_df) < 5:
            return False
                    
        ###########################################################
        # Normalizing
        ###########################################################
        mean = self.processed_data_dict_array[participant_id]['mean']
        std = self.processed_data_dict_array[participant_id]['std']
        
        ###########################################################
        # Find breakpoints with lambda = segmentation_lamb
        ###########################################################
        if not os.path.exists(os.path.join(save_participant_folder, participant_id + '.csv.gz')):
            
            last_sleep_end = data_df.index[0]
            bps_df, final_bps_df = pd.DataFrame(), pd.DataFrame()

            ###########################################################
            # First seg
            ###########################################################
            for index, row in sleep_df.iterrows():
                start_str = last_sleep_end
                end_str = row.start
                last_sleep_end = row.end
                
                row_data_df = data_df[start_str:end_str]
                row_data = np.array(row_data_df).astype(float)

                ###########################################################
                # Normalizing
                ###########################################################
                norm_data = np.divide(row_data - mean, std)

                ###########################################################
                # Convert to an n-by-T matrix
                ###########################################################
                norm_data = norm_data.T
                
                if len(row_data_df) > 20:
                    bps, objectives = GGS(norm_data, Kmax=int(norm_data.shape[1] / 2),
                                          lamb=self.save_config.segmentation_lamb)
                    bps[-1] = norm_data.shape[1] - 1

                    bps_final = []
                    last_index = 0
                    for bps_index, bps_row in enumerate(bps):
                        if bps_index == 0 or bps_index == len(bps) - 1:
                            bps_final.append(bps[bps_index])
                        else:
                            if bps[bps_index + 1] - bps[bps_index] > 5 and bps[bps_index] - bps[last_index] > 5:
                                bps_final.append(bps[bps_index])
                                last_index = bps_index

                    bps_start_str = [list(row_data_df.index)[i] for i in bps_final[:-1]]
                    bps_end_str = [list(row_data_df.index)[i] for i in bps_final[1:]]
                    bps_str = [list(row_data_df.index)[i] for i in bps_final]

                    row_bps_df = pd.DataFrame(bps_start_str, columns=['start'], index=bps_start_str)
                    row_bps_df['end'] = bps_end_str

                    bps_df = bps_df.append(pd.DataFrame(bps_str, columns=['time'], index=bps_str))

                    ###########################################################
                    # Sub seg
                    ###########################################################
                    if len(row_bps_df) > 0 and self.save_config.sub_segmentation_lamb is not None:
                        for row_bps_index, bps_row_series in row_bps_df.iterrows():
                            if (pd.to_datetime(bps_row_series.end) - pd.to_datetime(bps_row_series.start)).total_seconds() > 30 * 60:
                                bps_row_data_df = data_df[bps_row_series.start:bps_row_series.end]
                                bps_row_data = np.array(bps_row_data_df).astype(float)
    
                                ###########################################################
                                # Normalizing
                                ###########################################################
                                norm_data = np.divide(bps_row_data - mean, std)
    
                                ###########################################################
                                # Convert to an n-by-T matrix
                                ###########################################################
                                norm_data = norm_data.T

                                bps_row, bps_objectives = GGS(norm_data, Kmax=int(norm_data.shape[1] / 2), lamb=self.save_config.sub_segmentation_lamb)
                                bps_row[-1] = norm_data.shape[1] - 1

                                bps_row_final = []
                                last_index = 0
                                for bps_index, bps in enumerate(bps_row):
                                    if bps_index == 0 or bps_index == len(bps_row) - 1:
                                        bps_row_final.append(bps_row[bps_index])
                                    else:
                                        if bps_row[bps_index+1] - bps_row[bps_index] > 5 and bps_row[bps_index] - bps_row[last_index] > 5:
                                            bps_row_final.append(bps_row[bps_index])
                                            last_index = bps_index
                                
                                bps_row_str = [list(bps_row_data_df.index)[i] for i in bps_row_final[:-1]]
                                
                                final_row_bps_df = pd.DataFrame(bps_row_str, columns=['time'], index=bps_row_str)
                                final_bps_df = final_bps_df.append(final_row_bps_df)
                            else:
                                bps_tmp_df = pd.DataFrame(row_bps_index, index=[row_bps_index], columns=['time'])
                                final_bps_df = final_bps_df.append(bps_tmp_df)
                                
                        ###########################################################
                        # Last data
                        ###########################################################
                        row_bps_index = row_bps_df.loc[row_bps_df.index[-1], 'end']
                        bps_tmp_df = pd.DataFrame(row_bps_index, index=[row_bps_index], columns=['time'])
                        final_bps_df = final_bps_df.append(bps_tmp_df)

            if self.save_config.sub_segmentation_lamb is not None:
                final_bps_df.to_csv(os.path.join(save_participant_folder, participant_id + '.csv.gz'), compression='gzip')
            else:
                bps_df.to_csv(os.path.join(save_participant_folder, participant_id + '.csv.gz'), compression='gzip')
        
            return True
        else:
            return False

    def segment_data_by_sleep_and_inactive(self, participant_id, fitbit_summary_df=None):
    
        ###########################################################
        # If folder not exist
        ###########################################################
        save_participant_folder = os.path.join(self.save_config.process_folder)
        if not os.path.exists(save_participant_folder):
            os.mkdir(save_participant_folder)
    
        ###########################################################
        # Read data
        ###########################################################
        data_df = self.processed_data_dict_array[participant_id]['data'].sort_index()
        data = np.array(data_df).astype(float)
    
        fitbit_summary_df = fitbit_summary_df
        sleep_df = pd.DataFrame()
    
        ###########################################################
        # Parse sleep df
        ###########################################################
        if fitbit_summary_df is not None:
            if len(fitbit_summary_df) > 0:
                for index, row in fitbit_summary_df.iterrows():
                    return_df = self.add_sleep_data_frame(row.Sleep1BeginTimestamp, row.Sleep1EndTimestamp)
                    if return_df is not None: sleep_df = sleep_df.append(return_df)
                    return_df = self.add_sleep_data_frame(row.Sleep2BeginTimestamp, row.Sleep2EndTimestamp)
                    if return_df is not None: sleep_df = sleep_df.append(return_df)
                    return_df = self.add_sleep_data_frame(row.Sleep3BeginTimestamp, row.Sleep3EndTimestamp)
                    if return_df is not None: sleep_df = sleep_df.append(return_df)
        sleep_df = sleep_df.sort_index()
    
        if len(sleep_df) < 5:
            return False
    
        ###########################################################
        # Normalizing
        ###########################################################
        mean = self.processed_data_dict_array[participant_id]['mean']
        std = self.processed_data_dict_array[participant_id]['std']
    
        ###########################################################
        # Find breakpoints with lambda = segmentation_lamb
        ###########################################################
        if not os.path.exists(os.path.join(save_participant_folder, participant_id + '.csv.gz')):
        
            last_sleep_end = data_df.index[0]
            bps_df, final_bps_df = pd.DataFrame(), pd.DataFrame()
        
            ###########################################################
            # First seg
            ###########################################################
            for index, row in sleep_df.iterrows():
                start_str = last_sleep_end
                end_str = row.start
                last_sleep_end = row.end
            
                row_data_df = data_df[start_str:end_str]
                row_data = np.array(row_data_df).astype(float)
            
                ###########################################################
                # Normalizing
                ###########################################################
                norm_data = np.divide(row_data - mean, std)
            
                ###########################################################
                # Convert to an n-by-T matrix
                ###########################################################
                norm_data = norm_data.T
            
                if len(row_data_df) > 20:
                    bps, objectives = GGS(norm_data, Kmax=int(norm_data.shape[1] / 2),
                                          lamb=self.save_config.segmentation_lamb)
                    bps[-1] = norm_data.shape[1] - 1
                
                    bps_final = []
                    last_index = 0
                    for bps_index, bps_row in enumerate(bps):
                        if bps_index == 0 or bps_index == len(bps) - 1:
                            bps_final.append(bps[bps_index])
                        else:
                            if bps[bps_index + 1] - bps[bps_index] > 5 and bps[bps_index] - bps[last_index] > 5:
                                bps_final.append(bps[bps_index])
                                last_index = bps_index
                
                    bps_start_str = [list(row_data_df.index)[i] for i in bps_final[:-1]]
                    bps_end_str = [list(row_data_df.index)[i] for i in bps_final[1:]]
                
                    row_bps_df = pd.DataFrame(bps_start_str, columns=['start'], index=bps_start_str)
                    row_bps_df['end'] = bps_end_str
                
                    bps_df = bps_df.append(row_bps_df)
                
                    ###########################################################
                    # Sub seg
                    ###########################################################
                    if len(row_bps_df) > 0:
                        for row_bps_index, bps_row_series in row_bps_df.iterrows():
                            if (pd.to_datetime(bps_row_series.end) - pd.to_datetime(
                                    bps_row_series.start)).total_seconds() > 30 * 60:
                                bps_row_data_df = data_df[bps_row_series.start:bps_row_series.end]
                                bps_row_data = np.array(bps_row_data_df).astype(float)
                            
                                ###########################################################
                                # Normalizing
                                ###########################################################
                                norm_data = np.divide(bps_row_data - mean, std)
                            
                                ###########################################################
                                # Convert to an n-by-T matrix
                                ###########################################################
                                norm_data = norm_data.T
                            
                                bps_row, bps_objectives = GGS(norm_data, Kmax=int(norm_data.shape[1] / 2),
                                                              lamb=self.save_config.sub_segmentation_lamb)
                                bps_row[-1] = norm_data.shape[1] - 1
                            
                                bps_row_final = []
                                last_index = 0
                                for bps_index, bps in enumerate(bps_row):
                                    if bps_index == 0 or bps_index == len(bps_row) - 1:
                                        bps_row_final.append(bps_row[bps_index])
                                    else:
                                        if bps_row[bps_index + 1] - bps_row[bps_index] > 5 and bps_row[bps_index] - \
                                                bps_row[last_index] > 5:
                                            bps_row_final.append(bps_row[bps_index])
                                            last_index = bps_index
                            
                                bps_row_str = [list(bps_row_data_df.index)[i] for i in bps_row_final[:-1]]
                            
                                final_row_bps_df = pd.DataFrame(bps_row_str, columns=['time'], index=bps_row_str)
                                final_bps_df = final_bps_df.append(final_row_bps_df)
                            else:
                                bps_tmp_df = pd.DataFrame(row_bps_index, index=[row_bps_index], columns=['time'])
                                final_bps_df = final_bps_df.append(bps_tmp_df)
                    
                        ###########################################################
                        # Last data
                        ###########################################################
                        row_bps_index = row_bps_df.loc[row_bps_df.index[-1], 'end']
                        bps_tmp_df = pd.DataFrame(row_bps_index, index=[row_bps_index], columns=['time'])
                        final_bps_df = final_bps_df.append(bps_tmp_df)
        
            final_bps_df.to_csv(os.path.join(save_participant_folder, participant_id + '.csv.gz'), compression='gzip')
        
            return True
        else:
            return False
        
    def add_sleep_data_frame(self, sleep_begin_str, sleep_end_str):
        
        if pd.to_datetime(sleep_begin_str).year > 0:
            
            index = pd.to_datetime(sleep_begin_str).strftime(date_time_format)[:-3]
            start = pd.to_datetime(sleep_begin_str).strftime(date_time_format)[:-3]
            end = pd.to_datetime(sleep_end_str).strftime(date_time_format)[:-3]

            return_df = pd.DataFrame(index=[index], columns=['start', 'end'])
            return_df.loc[index, 'start'] = start
            return_df.loc[index, 'end'] = end
            return return_df
            
        else:
            return None

    def read_owl_in_one_in_segment(self, fitbit_summary_df=None, fitbit_df=None, owl_in_one_df=None, current_position=1):
        
        ###########################################################
        # Read participant id
        ###########################################################
        participant_id = self.participant_id

        ###########################################################
        # Read segmentation file
        ###########################################################
        save_folder = os.path.join(self.save_config.process_folder)
        if os.path.exists(os.path.join(save_folder, participant_id + '.csv.gz')) is False:
            return
        
        bps_df = pd.read_csv(os.path.join(save_folder, participant_id + '.csv.gz'), index_col=0)
        sensor_df = pd.DataFrame()
        
        if fitbit_df is not None and owl_in_one_df is not None:
    
            ###########################################################
            # Room type list
            ###########################################################
            room_type_list = list(owl_in_one_df.columns)
    
            ###########################################################
            # Complete bps data frame
            ###########################################################
            bps_start_time_str = list(bps_df.index)[:-1]
            bps_end_time_str = list(bps_df.index)[1:]

            bps_full_df = pd.DataFrame(bps_start_time_str, index=[bps_start_time_str], columns=['start'])
            bps_full_df['end'] = bps_end_time_str

            ###########################################################
            # Compute start and end
            ###########################################################
            if len(owl_in_one_df) == 0:
                return
            owl_in_one_diff = (pd.to_datetime(owl_in_one_df.index[1:]) - pd.to_datetime(owl_in_one_df.index[:-1])).total_seconds()
            owl_in_one_diff_change = np.where(np.array(owl_in_one_diff) > 60 * 60 * 6)[0]

            owl_in_one_diff_change_time_start = [list(owl_in_one_df.index)[0]]
            [owl_in_one_diff_change_time_start.append(list(owl_in_one_df.index)[i + 1]) for i in owl_in_one_diff_change]
            
            owl_in_one_diff_change_time_end = [list(owl_in_one_df.index)[i] for i in owl_in_one_diff_change]
            owl_in_one_diff_change_time_end.append(list(owl_in_one_df.index)[-1])
            
            owl_in_one_recording_df = pd.DataFrame(owl_in_one_diff_change_time_start, index=owl_in_one_diff_change_time_start, columns=['start'])
            owl_in_one_recording_df['end'] = owl_in_one_diff_change_time_end

            for owl_in_one_recording_index, owl_in_one_recording_row in owl_in_one_recording_df.iterrows():
    
                owl_in_one_recording_df.loc[owl_in_one_recording_index, 'hr_mean'] = np.nan
                owl_in_one_recording_df.loc[owl_in_one_recording_index, 'step_mean'] = np.nan
                owl_in_one_recording_df.loc[owl_in_one_recording_index, 'hr_std'] = np.nan
                owl_in_one_recording_df.loc[owl_in_one_recording_index, 'step_std'] = np.nan
    
                if (pd.to_datetime(owl_in_one_recording_row.end) - pd.to_datetime(owl_in_one_recording_row.start)).total_seconds() > 60 * 60 * 4:
                    fitbit_recording_df = fitbit_df[owl_in_one_recording_row.start:owl_in_one_recording_row.end]
                    fitbit_recording_valid_index = np.where(np.array(fitbit_recording_df)[:, 1] >= 0)
                    fitbit_recording_array = np.array(fitbit_recording_df)[fitbit_recording_valid_index[0]]
                    
                    if len(fitbit_recording_array) > 0:
                        owl_in_one_recording_df.loc[owl_in_one_recording_index, 'hr_mean'] = np.mean(fitbit_recording_array, axis=0)[0]
                        owl_in_one_recording_df.loc[owl_in_one_recording_index, 'step_mean'] = np.mean(fitbit_recording_array, axis=0)[1]
                        owl_in_one_recording_df.loc[owl_in_one_recording_index, 'hr_std'] = np.std(fitbit_recording_array[:, 0])
                        owl_in_one_recording_df.loc[owl_in_one_recording_index, 'step_std'] = np.std(fitbit_recording_array[:, 1])
                
            owl_in_one_recording_df = owl_in_one_recording_df.dropna()
            if len(owl_in_one_recording_df) == 0:
                return

            ###########################################################
            # Iterate time segments
            ###########################################################
            for index, bps_row in bps_full_df.iterrows():
                
                owl_in_one_row_df = owl_in_one_df[bps_row.start:bps_row.end]
                fitbit_row_df = fitbit_df[bps_row.start:bps_row.end]
                
                if (pd.to_datetime(bps_row.end) - pd.to_datetime(bps_row.start)).total_seconds() > 60 * 60 * 3:
                    continue
                
                if len(owl_in_one_row_df) > 0:
    
                    for owl_in_one_recording_index, owl_in_one_recording_row in owl_in_one_recording_df.iterrows():
                        
                        cond1 = (pd.to_datetime(owl_in_one_recording_row.end) - pd.to_datetime(bps_row.start)).total_seconds() >= 0
                        cond2 = (pd.to_datetime(bps_row.start) - pd.to_datetime(owl_in_one_recording_row.start)).total_seconds() >= 0
                    
                        if cond1 and cond2:
                            owl_index_start = owl_in_one_recording_row.start
                            owl_index_end = owl_in_one_recording_row.end
                    
                            ###########################################################
                            # Sensor data
                            ###########################################################
                            sensor_row_df = pd.DataFrame(index=[bps_row.start])
                            
                            hr_array = np.array(fitbit_row_df.HeartRatePPG.values)
                            step_array = np.array(fitbit_row_df.StepCount.values)
        
                            sensor_row_df['start'] = bps_row.start
                            sensor_row_df['end'] = bps_row.end
                            sensor_row_df['norm_hr_mean'] = owl_in_one_recording_df.loc[owl_index_start, 'hr_mean']
                            sensor_row_df['norm_hr_std'] = owl_in_one_recording_df.loc[owl_index_start, 'hr_std']
                            sensor_row_df['norm_step_mean'] = owl_in_one_recording_df.loc[owl_index_start, 'step_mean']
                            sensor_row_df['norm_step_std'] = owl_in_one_recording_df.loc[owl_index_start, 'step_std']
                            
                            sensor_row_df['hr_std'] = np.std(hr_array)
                            sensor_row_df['hr_ave'] = np.mean(hr_array)
                            sensor_row_df['hr_max'] = np.max(hr_array)
                            sensor_row_df['hr_min'] = np.min(hr_array)
                            sensor_row_df['hr_range'] = np.max(hr_array) - np.min(hr_array)
        
                            sensor_row_df['step_std'] = np.std(step_array)
                            sensor_row_df['step_ave'] = np.mean(step_array)
                            sensor_row_df['step_max'] = np.max(step_array)
                            sensor_row_df['step_min'] = np.min(step_array)
                            sensor_row_df['step_range'] = np.max(step_array) - np.min(step_array)

                            sensor_row_df['recording_start'] = owl_index_start
                            sensor_row_df['recording_end'] = owl_index_end
                            
                            for room_type in room_type_list:
                                sensor_row_df[room_type] = np.nansum(owl_in_one_row_df.loc[:, room_type])
                            
                            sensor_row_df['duration'] = (pd.to_datetime(bps_row.end) - pd.to_datetime(bps_row.start)).total_seconds() / 60 + 1
                            sensor_row_df['start_hour'] = pd.to_datetime(bps_row.start).hour + pd.to_datetime(bps_row.start).minute / 60
                            sensor_row_df['end_hour'] = pd.to_datetime(bps_row.end).hour + pd.to_datetime(bps_row.end).minute / 60

                            ###########################################################
                            # Work hours
                            ###########################################################
                            if current_position == 1 or current_position == 2:
                                if pd.to_datetime(owl_in_one_recording_row.start).hour <= 4:
                                    start_work_time = (pd.to_datetime(owl_in_one_recording_row.start) - timedelta(days=1)).replace(hour=19, minute=0, second=0)
                                elif 17 < pd.to_datetime(owl_in_one_recording_row.start).hour < 24:
                                    start_work_time = (pd.to_datetime(owl_in_one_recording_row.start)).replace(hour=19, minute=0, second=0)
                                else:
                                    start_work_time = (pd.to_datetime(owl_in_one_recording_row.start)).replace(hour=7, minute=0, second=0)

                                sensor_row_df['relative_start_hour'] = (pd.to_datetime(bps_row.start) - pd.to_datetime(start_work_time)).total_seconds() / 3600
                            else:
                                sensor_row_df['relative_start_hour'] = (pd.to_datetime(bps_row.start) - pd.to_datetime(owl_in_one_recording_row.start)).total_seconds() / 3600

                            ###########################################################
                            # Append data
                            ###########################################################
                            sensor_df = sensor_df.append(sensor_row_df)

        sensor_df.to_csv(os.path.join(save_folder, participant_id + '_owl_in_one.csv.gz'), compression='gzip')