from datetime import timedelta

import os
import numpy as np
import pandas as pd
import scipy.signal

from fancyimpute import (
	IterativeImputer,
    KNN
)

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


def fitbit_sliced_data_start_end_array(raw_data_df, threshold=timedelta(seconds=1)):
    ###########################################################
    # Trick: cal time offset of consecutive rows
    ###########################################################
    raw_data_time_array = pd.to_datetime(raw_data_df.index)
    
    # Get consecutive time offset
    raw_data_time_offset_array = raw_data_time_array[1:] - raw_data_time_array[:-1]
    offset_mask_true_index = np.where((raw_data_time_offset_array > threshold))[0]
    
    ###########################################################
    # Get start and end time index
    ###########################################################
    start_time_array, end_time_array = [raw_data_df.index[0]], []
    
    for i, true_idx, in enumerate(offset_mask_true_index):
        end_time_array.append(raw_data_df.index[true_idx])
        start_time_array.append(raw_data_df.index[true_idx + 1])
    
    end_time_array.append(raw_data_df.index[-1])
    
    return start_time_array, end_time_array

def fitbit_process_sliced_data(ppg_data_df, step_data_df, process_hyper,
                               check_saved=False, check_folder=None, imputation_method=None):
    
    ###########################################################
    # Initialization
    ###########################################################
    method = process_hyper['method']
    offset = process_hyper['offset']
    overlap = process_hyper['overlap']
    preprocess_cols = process_hyper['preprocess_cols']

    threshold = 2
    interval = int(offset / 60)
    
    # Returned data
    preprocess_data_df = pd.DataFrame()

    ###########################################################
    # Process function, only for heart rate, cadence, intensity
    ###########################################################
    process_func = np.nanmean
    if method == 'ma':
        process_func = np.nanmean

    ###########################################################
    # Start iterate data with given parameters
    ###########################################################
    replace_minute = int(pd.to_datetime(ppg_data_df.index[0]).minute * 60 / offset) * interval
    start_str = pd.to_datetime(ppg_data_df.index[0]).replace(minute=replace_minute, second=0, microsecond=0).strftime(date_time_format)[:-3]

    replace_minute = int(pd.to_datetime(ppg_data_df.index[-1]).minute * 60 / offset) * interval
    end_str = (pd.to_datetime(ppg_data_df.index[-1]).replace(minute=replace_minute, second=0, microsecond=0) + timedelta(minutes=interval)).strftime(date_time_format)[:-3]

    time_length = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).total_seconds()
    start_off_dt = pd.to_datetime(start_str)

    for i in range(int(time_length / offset)):
        
        ###########################################################
        # For normal data, calculate each time step time range
        ###########################################################
        start_off_str = (start_off_dt + timedelta(seconds=offset*i-1)).strftime(date_time_format)[:-3]
        end_off_str = (start_off_dt + timedelta(seconds=offset*(i+1)+overlap-1)).strftime(date_time_format)[:-3]
        
        save_str = (start_off_dt + timedelta(seconds=offset*i)).strftime(date_time_format)[:-3]

        ###########################################################
        # Initialize save data array
        ###########################################################
        process_row_df = pd.DataFrame(index=[save_str])

        ###########################################################
        # Filter data in the time range
        ###########################################################
        ppg_data_row_df = ppg_data_df[start_off_str:end_off_str]
        step_data_row_df = step_data_df[start_off_str:end_off_str]

        ###########################################################
        # Process the data
        ###########################################################
        process_row_df['HeartRatePPG'] = process_func(ppg_data_row_df) if len(ppg_data_row_df) > threshold else np.nan
        process_row_df['StepCount'] = np.nansum(step_data_row_df) if len(step_data_row_df) > 0 else np.nan
        preprocess_data_df = preprocess_data_df.append(process_row_df)
        
        ###########################################################
        # If check saved or not
        ###########################################################
        if check_saved is True and check_folder is not None:
            if len(preprocess_data_df) > 0 and os.path.join(check_folder, preprocess_data_df.index[0] + '.csv') is True:
                return None

    ###########################################################
    # If we add imputation or not
    ###########################################################
    if imputation_method is not None:
        
        len_seq = len(preprocess_data_df)
        iteration = int(len_seq / 30)

        if imputation_method == 'knn':
            model = KNN(k=5)
        else:
            model = IterativeImputer()
        
        last_un_imputed_idx = -1
        for iter in range(iteration):
            data_iter_df = preprocess_data_df[iter*30:(iter+1)*30+30]
            if len(data_iter_df.dropna()) > 10 and len(data_iter_df.dropna()) / len(data_iter_df) > 0.75:
                impute_array = model.fit_transform(np.array(data_iter_df))
                    
                preprocess_data_df.loc[data_iter_df.index, 'HeartRatePPG'] = impute_array[:, 0]
                preprocess_data_df.loc[data_iter_df.index, 'StepCount'] = impute_array[:, 1]
            
            else:
                filter_df = preprocess_data_df[(last_un_imputed_idx + 1) * 30:iter * 30]
                if len(filter_df.dropna()) == len(filter_df) and len(filter_df.dropna()) > 20:
                    filter_array = np.array(filter_df)[:, 0]
                    filter_array = scipy.signal.savgol_filter(filter_array, 5, 3)
        
                    preprocess_data_df.loc[filter_df.index, 'HeartRatePPG'] = filter_array
                    last_un_imputed_idx = iter
        
        if len(preprocess_data_df[(last_un_imputed_idx + 1) * 30:]) > 60:
            filter_df = preprocess_data_df[(last_un_imputed_idx + 1) * 30:]
            filter_array = np.array(filter_df)[:, 0]
            filter_array = scipy.signal.savgol_filter(filter_array, 5, 3)
    
            preprocess_data_df.loc[filter_df.index, 'HeartRatePPG'] = filter_array

    return preprocess_data_df