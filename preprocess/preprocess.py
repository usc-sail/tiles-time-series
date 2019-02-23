"""
Top level classes for the preprocess model.
"""
from __future__ import print_function

import copy
from om_signal.helper import *
from fitbit.helper import *
from realizd.helper import *
from owl_in_one.helper import *
import pandas as pd

__all__ = ['Preprocess']

class Preprocess(object):
    """
    Preprocess script for all signal
    """
    
    def __init__(self, data_config=None, participant_id=None):
        """
        Initialization method
        """
        
        ###########################################################
        # Assert if these parameters are not parsed
        ###########################################################
        assert data_config is not None
        self.data_config = data_config
        
        ###########################################################
        # Initialize data array within class
        ###########################################################
        self.sliced_data_array = []
        self.processed_sliced_data_array = []
        
        self.participant_id = participant_id
        
    def process_fitbit(self, ppg_df, step_df, valid_slice_in_min=60):
        """
        Slice the data based on shift
        """
        print('---------------------------------------------------------------------')
        print('Function: process_fitbit')
        print('---------------------------------------------------------------------')
        
        ###########################################################
        # Get start and end time of a chunk
        ###########################################################
        start_time_array, end_time_array = fitbit_sliced_data_start_end_array(ppg_df, threshold=timedelta(seconds=60*15))

        preprocess_data_all_df = pd.DataFrame()
        
        ###########################################################
        # Slice the data
        ###########################################################
        for i in range(len(start_time_array)):
            
            print('Complete process for %s: %.2f' % (self.participant_id, 100 * i / len(start_time_array)))

            start_time, end_time = start_time_array[i], end_time_array[i]
            tmp_ppg_data_df = ppg_df[start_time:end_time]
            tmp_step_data_df = step_df[start_time:end_time]
            
            if (pd.to_datetime(end_time) - pd.to_datetime(start_time)).seconds > 60 * valid_slice_in_min:
                ###########################################################
                # Process sliced data
                ###########################################################
                preprocess_data_df = fitbit_process_sliced_data(tmp_ppg_data_df, tmp_step_data_df,
                                                                participant=self.participant_id,
                                                                data_config=self.data_config)

                if len(preprocess_data_df) > 0:
                    preprocess_data_all_df = preprocess_data_df if len(preprocess_data_all_df) == 0 else preprocess_data_all_df.append(preprocess_data_df)
                    if os.path.exists(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id)) is False:
                        os.mkdir(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id))
                    
                    preprocess_data_df.to_csv(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id, start_time + '.csv.gz'), compression='gzip')
                    preprocess_data_all_df.to_csv(os.path.join(self.data_config.fitbit_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz'), compression='gzip')
                    
    def preprocess_realizd(self, data_df):
        """
        Process realizd data based on shift
        """
        print('---------------------------------------------------------------------')
        print('Function: process_realizd')
        print('---------------------------------------------------------------------')
        self.preprocess_data_all_df = pd.DataFrame()
    
        ###########################################################
        # Get start and end time of a shift
        ###########################################################
        if len(data_df) > 300:
            self.preprocess_data_all_df = realizd_process_data(data_df, offset=self.data_config.realizd_sensor_dict['offset'])
            self.preprocess_data_all_df.to_csv(os.path.join(self.data_config.realizd_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz'), compression='gzip')

    def preprocess_owl_in_one(self, data_df):
        """
        Process owl_in_one data based on shift
        """
        print('---------------------------------------------------------------------')
        print('Function: process_owl_in_one')
        print('---------------------------------------------------------------------')
        self.preprocess_data_all_df = pd.DataFrame()
    
        ###########################################################
        # Get start and end time of a shift
        ###########################################################
        if len(data_df) > 300:
            self.preprocess_data_all_df = process_owl_in_one_data(data_df, offset=self.data_config.owl_in_one_sensor_dict['offset'])
            self.preprocess_data_all_df.to_csv(os.path.join(self.data_config.owl_in_one_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz'), compression='gzip')
            
    def preprocess_audio(self, data_df):
        """
        Process audio data
        """
        print('---------------------------------------------------------------------')
        print('Function: preprocess_audio')
        print('---------------------------------------------------------------------')
        

    def slice_raw_data(self, method=None, valid_slice_in_min=180, data_df=None):
        """
        Slice the data based on block (chunk)
        """

        # Assert if method parameter is not parsed
        assert method is not None

        print('---------------------------------------------------------------------')
        print('Function: slice_raw_data')
        print('---------------------------------------------------------------------')

        if method == 'block':
    
            ###########################################################
            # 1. om_signal type
            ###########################################################
            if self.signal_type == 'om_signal':
        
                ###########################################################
                # Get start and end time of a shift
                ###########################################################
                start_time_array, end_time_array = om_signal_sliced_data_start_end_array(self.raw_data_df, threshold=timedelta(seconds=60 * 2))
        
                ###########################################################
                # Slice the data
                ###########################################################
                for i in range(len(start_time_array)):
            
                    start_time, end_time = start_time_array[i], end_time_array[i]
            
                    shift_data_df = self.raw_data_df[start_time:end_time]
            
                    # At least 3 hours of data for a valid shift
                    if len(shift_data_df) / 60 > valid_slice_in_min:
                        self.sliced_data_array.append(shift_data_df)

        return self.sliced_data_array

    def preprocess_slice_raw_data(self, check_saved=False):
        
        ###########################################################
        # 1. om_signal type
        ###########################################################
        if self.signal_type == 'om_signal':
            
            for index, sliced_data_df in enumerate(self.sliced_data_array):
                print('---------------------------------------------------------------------')
                print('Function: preprocess_slice_raw_data')
                print('Process data with start index: %s' % (sliced_data_df.index[0]))
                print('Process data process %.2f' % (index / len(self.sliced_data_array) * 100))
                print('---------------------------------------------------------------------')
                
                preprocess_data_df = om_signal_process_sliced_data(sliced_data_df, self.process_hyper,
                                                                   check_saved=check_saved,
                                                                   check_folder=self.participant_folder)
                
                if preprocess_data_df is not None:
                    self.processed_sliced_data_array.append(preprocess_data_df)
    
    def preprocess_slice_raw_data_full_feature(self, check_saved=False):
        
        ###########################################################
        # 1. om_signal type
        ###########################################################
        if self.signal_type == 'om_signal':
            
            for index, sliced_data_df in enumerate(self.sliced_data_array):
                print('---------------------------------------------------------------------')
                print('Function: preprocess_slice_raw_data')
                print('Process data with start index: %s' % (sliced_data_df.index[0]))
                print('Process data process %.2f' % (index / len(self.sliced_data_array) * 100))
                print('---------------------------------------------------------------------')

                ###########################################################
                # 2. If we have weird data, skip
                ###########################################################
                if len(np.unique(np.array(sliced_data_df)[:, 0])) < 5:
                    continue
                
                preprocess_data_df = om_signal_process_sliced_data_full_feature(sliced_data_df, self.process_hyper,
                                                                                check_saved=check_saved,
                                                                                check_folder=self.participant_folder)
                
                if preprocess_data_df is not None:
                    self.processed_sliced_data_array.append(preprocess_data_df)
    
    def save_preprocess_slice_raw_data(self):
        
        print('---------------------------------------------------------------------')
        print('Function: save_preprocess_slice_raw_data')
        print('---------------------------------------------------------------------')
        
        ###########################################################
        # Iterate data and save
        ###########################################################
        if len(self.processed_sliced_data_array) > 0:
            for i in range(len(self.processed_sliced_data_array)):
                processed_sliced_data_df = self.processed_sliced_data_array[i]
                csv_path = os.path.join(self.participant_folder, processed_sliced_data_df.index[0] + '.csv')
                processed_sliced_data_df.to_csv(csv_path)