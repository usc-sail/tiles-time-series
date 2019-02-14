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
    _process_hype = {'method': 'ma', 'offset': 30, 'overlap': 0}
    
    _default_om_signal = {'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence',
                                       'HeartRate', 'Intensity', 'Steps'],
                          'MinPeakDistance': 100, 'MinPeakHeight': 0.04}
    
    def __init__(self, data_df=None, save_main_folder=None, participant_id=None, enable_impute=False,
                 process_hyper=_process_hype, signal_hyper=_default_om_signal, return_full_feature=False,
                 read_config=None, save_config=None,
                 imputation_method='knn'):
        """
        Initialization method
        """
        
        ###########################################################
        # Assert if these parameters are not parsed
        ###########################################################
        assert read_config is not None and save_config is not None
        self.signal_type = read_config.sensor
        self.read_config = read_config
        self.save_config = save_config
        
        self.imputation_method = imputation_method

        ###########################################################
        # 1. Update hyper paramters for signal and preprocess method
        ###########################################################
        self.process_hyper = copy.deepcopy(self._process_hype)
        self.process_hyper.update(process_hyper)

        ###########################################################
        # Initialize data array within class
        ###########################################################
        self.sliced_data_array = []
        self.processed_sliced_data_array = []
        
        ###########################################################
        # If is om_signal type
        ###########################################################
        if self.signal_type == 'om_signal':
            ###########################################################
            # Update hyper paramters for signal and preprocess method
            ###########################################################
            self.signal_hypers = copy.deepcopy(self._default_om_signal)
            self.signal_hypers.update(signal_hyper)

        ###########################################################
        # 2. save data folder
        ###########################################################
        self.save_main_folder = self.save_config.main_folder
        self.save_process_folder = self.save_config.process_folder
        
        self.participant_id = participant_id
        
        ###########################################################
        # 2. Create Sub folder
        ###########################################################
        self.return_full_feature = return_full_feature
        self.enable_impute = enable_impute
        self.update_folder_name()
        
    def update_folder_name(self):
        ###########################################################
        # Create folders for preprocess data
        ###########################################################
        if os.path.exists(self.save_main_folder) is False:
            os.mkdir(self.save_main_folder)

        signal_type_folder = os.path.join(self.save_main_folder, self.signal_type)
        if os.path.exists(signal_type_folder) is False:
            os.mkdir(signal_type_folder)
    
        ###########################################################
        # 2.2 Create Sub folder
        ###########################################################
        if self.return_full_feature == True:
            signal_type_folder = os.path.join(signal_type_folder, 'feat')
        elif self.enable_impute == True:
            signal_type_folder = os.path.join(signal_type_folder, 'impute_'+ self.imputation_method)
        else:
            signal_type_folder = os.path.join(signal_type_folder, 'original')
        
        if os.path.exists(signal_type_folder) is False:
            os.mkdir(signal_type_folder)

        self.participant_folder = os.path.join(self.save_process_folder, self.participant_id)
        if os.path.exists(self.participant_folder) is False:
            os.mkdir(self.participant_folder)
        
    def read_preprocess_slice_raw_data(self):
        """
        Read preprocessed slice raw data
        """
        self.processed_sliced_data_array = []
        
        ###########################################################
        # Get all data file array
        ###########################################################
        data_file_array = os.listdir(self.participant_folder)
        for data_file in data_file_array:
            if 'DS' in data_file: data_file_array.remove(data_file)
        
        if len(data_file_array) > 0:
            for data_file in data_file_array:
                ###########################################################
                # Read data and append
                ###########################################################
                csv_path = os.path.join(self.participant_folder, data_file)
                data_df = pd.read_csv(csv_path, index_col=0)
                
                data_dict = {}
                data_dict['file_name'] = data_file
                data_dict['data'] = data_df
                
                self.processed_sliced_data_array.append(data_dict)
    
    def read_all_data(self):
        """
		Read preprocessed slice raw data
		"""
        self.data_df_all = pd.DataFrame()
        
        ###########################################################
        # Get all data file array
        ###########################################################
        data_file_array = os.listdir(self.participant_folder)
        for data_file in data_file_array:
            if 'DS' in data_file: data_file_array.remove(data_file)
        
        if len(data_file_array) > 0:
            for data_file in data_file_array:
                ###########################################################
                # Read data and append
                ###########################################################
                csv_path = os.path.join(self.participant_folder, data_file)
                data_df = pd.read_csv(csv_path, index_col=0)
                if data_df.shape[0] > 300:
                    self.data_df_all = self.data_df_all.append(data_df)
    
    def process_fitbit(self, ppg_df, step_df, method=None, valid_slice_in_min=60):
        """
        Slice the data based on shift
        """
        # Assert if method parameter is not parsed
        assert method is not None
        
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
                if self.enable_impute == True:
                    preprocess_data_df = fitbit_process_sliced_data(tmp_ppg_data_df, tmp_step_data_df, self.process_hyper,
                                                                    check_folder=self.participant_folder,
                                                                    imputation_method=self.imputation_method)
                else:
                    preprocess_data_df = fitbit_process_sliced_data(tmp_ppg_data_df, tmp_step_data_df, self.process_hyper,
                                                                    check_folder=self.participant_folder,
                                                                    imputation_method=None)

                if len(preprocess_data_df) > 0:
                    preprocess_data_all_df = preprocess_data_df if len(preprocess_data_all_df) == 0 else preprocess_data_all_df.append(preprocess_data_df)
                    
                    preprocess_data_df.to_csv(os.path.join(self.participant_folder, start_time + '.csv.gz'), compression='gzip')
                    preprocess_data_all_df.to_csv(os.path.join(self.save_process_folder, self.participant_id + '.csv.gz'), compression='gzip')
                    
    def process_realizd(self, data_df, offset=60):
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
            self.preprocess_data_all_df = realizd_process_data(data_df, offset=offset)

    def process_owl_in_one(self, data_df, offset=60):
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
            self.preprocess_data_all_df = process_owl_in_one_data(data_df, self.participant_id, self.save_process_folder, offset=offset)
            
        
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