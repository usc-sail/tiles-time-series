"""
Top level classes for the preprocess model.
"""
from __future__ import print_function

import copy
import pdb
from om_signal.helper import *
from fitbit.helper import *
from realizd.helper import *
from owl_in_one.helper import *
import pandas as pd

date_time_format = '%Y-%m-%dT%H:%M:%S.%f'


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
        
        if os.path.exists(os.path.join(self.data_config.realizd_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz')) is True:
            return
    
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
        sample_rate = 0.01 # The sampling rate of the audio data in seconds (ignoring missing data)
        offset = int(self.data_config.audio_sensor_dict['offset'])

        out_file_path = os.path.join(self.data_config.audio_sensor_dict['preprocess_path'], self.participant_id + '.csv.gz')
        if os.path.exists(out_file_path): # Skip files that have already been written
            return

        data_df.index = pd.to_datetime(data_df.index)
        unix_times = pd.DatetimeIndex([data_df.index[0], data_df.index[data_df.shape[0]-1]]).astype(np.int64)
        start_unix_time_minute_clamp = int(unix_times[0] - (unix_times[0]% int(offset*1e9)))
        end_unix_time_minute_clamp = int(unix_times[1] - (unix_times[1] % int(offset*1e9)))
        time_index = pd.to_datetime(range(start_unix_time_minute_clamp, end_unix_time_minute_clamp+int(offset*1e9), int(offset*1e9))).strftime(date_time_format)[:-3]
        foreground_data = np.zeros(len(time_index))

        idx = 0
        last_time_index = 0
        while idx < len(time_index):
            if (idx%(len(time_index)/20)) == 0:
                print("Percent complete: %f"%(float(idx)/len(time_index)))
            cur_time_index = np.searchsorted(data_df.index, time_index[idx])
            num_foreground_samples = max(cur_time_index-last_time_index-1, 0)
            percentage_foreground = (num_foreground_samples*sample_rate)/float(offset)
            foreground_data[idx] = percentage_foreground
            last_time_index = cur_time_index
            idx += 1

        self.preprocess_data_all_df = pd.DataFrame(data={'foreground': foreground_data}, index=time_index)
        self.preprocess_data_all_df.to_csv(out_file_path, index=True, index_label='Timestamp', compression='gzip')


    def slice_raw_data(self, method=None, valid_slice_in_min=180, data_df=None):
        """
        Slice the data based on block (chunk)
        """

        # Assert if method parameter is not parsed
        assert method is not None

        print('---------------------------------------------------------------------')
        print('Function: slice_raw_data')
        print('---------------------------------------------------------------------')

        ###########################################################
        # Get start and end time of a shift
        ###########################################################
        start_time_array, end_time_array = om_signal_sliced_data_start_end_array(data_df, threshold=timedelta(seconds=60 * 2))

        ###########################################################
        # Slice the data
        ###########################################################
        for i in range(len(start_time_array)):
    
            start_time, end_time = start_time_array[i], end_time_array[i]
    
            shift_data_df = data_df[start_time:end_time]
    
            # At least 3 hours of data for a valid shift
            if len(shift_data_df) / 60 > valid_slice_in_min:
                self.sliced_data_array.append(shift_data_df)

        return self.sliced_data_array

    
    def preprocess_slice_raw_data_full_feature(self, check_saved=False):
        
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
            
            preprocess_data_df = om_signal_process_sliced_data_full_feature(sliced_data_df, self.data_config, self.participant_id)
            
            if preprocess_data_df is not None:
                self.processed_sliced_data_array.append(preprocess_data_df)
