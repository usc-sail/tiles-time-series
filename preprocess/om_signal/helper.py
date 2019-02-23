from datetime import timedelta

import os
import numpy as np
import pandas as pd
import peakutils
from hrvanalysis import get_time_domain_features

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


def om_signal_sliced_data_start_end_array(raw_data_df, threshold=timedelta(seconds=1)):
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


def om_signal_process_sliced_data(sliced_data_df, process_hyper, check_saved=False, check_folder=None):
    
    ###########################################################
    # Initialization
    ###########################################################
    method = process_hyper['method']
    offset = process_hyper['offset']
    overlap = process_hyper['overlap']
    preprocess_cols = process_hyper['preprocess_cols']

    threshold = offset / 3
    
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
    start_str = pd.to_datetime(sliced_data_df.index[0]).replace(minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
    time_length = (pd.to_datetime(sliced_data_df.index[-1]) - pd.to_datetime(start_str)).total_seconds()
    
    start_off_dt = pd.to_datetime(start_str)

    for i in range(int(time_length / offset)):
        
        ###########################################################
        # For normal data
        ###########################################################
        start_off_str = (start_off_dt + timedelta(seconds=offset*i)).strftime(date_time_format)[:-3]
        end_off_str = (start_off_dt + timedelta(seconds=offset*(i+1)+overlap)).strftime(date_time_format)[:-3]
        
        ###########################################################
        # For breath data
        ###########################################################
        start_off_breath_str = (start_off_dt + timedelta(seconds=offset * i - 2)).strftime(date_time_format)[:-3]
        end_off_breath_str = (start_off_dt + timedelta(seconds=offset * (i + 1) + overlap + 2)).strftime(date_time_format)[:-3]

        sliced_data_row_df = sliced_data_df[start_off_str:end_off_str]
        sliced_breath_data_row_df = sliced_data_df[start_off_breath_str:end_off_breath_str]
        
        # Row data to append
        process_row_df = pd.DataFrame(index=[end_off_str])
        
        # 'BreathingDepth', 'BreathingRate', 'Cadence', 'HeartRate', 'Intensity', 'Steps'
        if len(sliced_data_row_df) > 0:
            
            for preprocess_col in preprocess_cols:
                if 'RR' not in preprocess_col:
                    process_row_df[preprocess_col] = calculate_proc_func_for_series(sliced_data_row_df,
                                                                                    sliced_breath_data_row_df,
                                                                                    preprocess_col, process_func,
                                                                                    threshold)
                else:
                    rr_results = calculate_rr_std_avg_for_series(sliced_data_row_df, threshold)
                
                    if rr_results is not None:
                        for key in rr_results.keys():
                            process_row_df['rr_' + key] = rr_results[key]
            
            preprocess_data_df = preprocess_data_df.append(process_row_df)

        # If check saved or not
        if check_saved is True and check_folder is not None:
            if len(preprocess_data_df) > 0 and os.path.join(check_folder, preprocess_data_df.index[0] + '.csv') is True:
                return None
            
    return preprocess_data_df


def om_signal_process_sliced_data_full_feature(sliced_data_df, data_config, participant_id, check_saved=True):
    ###########################################################
    # Initialization
    ###########################################################
    offset = data_config.omsignal_sensor_dict['offset']
    overlap = data_config.omsignal_sensor_dict['overlap']
    preprocess_cols = data_config.omsignal_sensor_dict['preprocess_cols']
    
    threshold = offset / 3
    
    # Returned data
    preprocess_data_df = pd.DataFrame()
    
    ###########################################################
    # Start iterate data with given parameters
    ###########################################################
    start_str = pd.to_datetime(sliced_data_df.index[0]).replace(minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
    time_length = (pd.to_datetime(sliced_data_df.index[-1]) - pd.to_datetime(start_str)).total_seconds()
    
    start_off_dt = pd.to_datetime(start_str)
    
    for i in range(int(time_length / offset)):
        
        ###########################################################
        # For normal data
        ###########################################################
        start_off_str = (start_off_dt + timedelta(seconds=offset * i)).strftime(date_time_format)[:-3]
        end_off_str = (start_off_dt + timedelta(seconds=offset * (i + 1) + overlap)).strftime(date_time_format)[:-3]
        
        ###########################################################
        # For breath data
        ###########################################################
        start_off_breath_str = (start_off_dt + timedelta(seconds=offset * i - 2)).strftime(date_time_format)[:-3]
        end_off_breath_str = (start_off_dt + timedelta(seconds=offset * (i + 1) + overlap + 2)).strftime(date_time_format)[:-3]
        
        sliced_data_row_df = sliced_data_df[start_off_str:end_off_str]
        sliced_breath_data_row_df = sliced_data_df[start_off_breath_str:end_off_breath_str]
        
        # Row data to append
        process_row_df = pd.DataFrame(index=[end_off_str])
        
        # 'BreathingDepth', 'BreathingRate', 'Cadence', 'HeartRate', 'Intensity', 'Steps'
        if len(sliced_data_row_df) > 0:
            
            for preprocess_col in preprocess_cols:
                if 'RR' not in preprocess_col:
                    process_row_dict = calculate_proc_func_for_series_full(sliced_data_row_df,
                                                                           sliced_breath_data_row_df,
                                                                           preprocess_col, threshold)

                    for key in process_row_dict.keys():
                        process_row_df[key] = process_row_dict[key]
                else:
                    rr_results = calculate_rr_std_avg_for_series(sliced_data_row_df, threshold)
                    
                    if rr_results is not None:
                        for key in rr_results.keys():
                            process_row_df['rr_' + key] = rr_results[key]
            
            preprocess_data_df = preprocess_data_df.append(process_row_df)
        
        # If check saved or not
        if check_saved is True:
            if len(preprocess_data_df) > 0 and os.path.join(data_config.omsignal_sensor_dict['preprocess_path'], participant_id, preprocess_data_df.index[0] + '.csv') is True:
                return None
    
    return preprocess_data_df


def calculate_proc_func_for_series(data_df, breath_data_df, col, process_func, threshold):
    
    ###########################################################
    # Process the valid data only
    ###########################################################
    if 'Breathing' in col:
        if len(np.array(breath_data_df[col])[breath_data_df[col].nonzero()]) > 0 and len(breath_data_df) > threshold:
            peak_indexes = peakutils.indexes(np.array(breath_data_df[col]), thres=3/max(breath_data_df[col]), min_dist=6)
            if len(peak_indexes) > 0:
                col_data = np.array(breath_data_df[col])[peak_indexes]
                return process_func(col_data)
            else:
                if np.nanmax(np.array(breath_data_df[col])) > 0:
                    return np.nanmax(np.array(breath_data_df[col]))
                else:
                    return np.nan
        else:
            if np.nanmax(np.array(breath_data_df[col])) > 0:
                return np.nanmax(np.array(breath_data_df[col]))
            else:
                return np.nan
    else:
        col_data = np.array(data_df[col])[data_df[col].nonzero()]

        if 'Cadence' in col or 'Steps' in col:
            return np.nansum(col_data)
        elif 'Intensity' in col:
            return np.nanmean(col_data)
        else:
            if len(col_data) > threshold:
                return process_func(col_data)
            else:
                return np.nan


def calculate_proc_func_for_series_full(data_df, breath_data_df, col, threshold):
    ###########################################################
    # Process the valid data only
    ###########################################################
    process_func_list = {'std': np.nanstd, 'mean': np.nanmean, 'max': np.nanmax, 'min': np.nanmin, 'sum': np.nansum}
    return_dict = {}

    if 'Breathing' in col:
        if len(np.array(breath_data_df[col])[breath_data_df[col].nonzero()]) > 0 and len(breath_data_df) > threshold:
            peak_indexes = peakutils.indexes(np.array(breath_data_df[col]), thres=3 / max(breath_data_df[col]), min_dist=6)
            
            if len(peak_indexes) > 1:
                col_data = np.array(breath_data_df[col])[peak_indexes]
                return_dict[col + '_max'] = process_func_list['max'](col_data)
                return_dict[col + '_min'] = process_func_list['min'](col_data)
                return_dict[col + '_mean'] = process_func_list['mean'](col_data)
                return return_dict
            else:
                return_dict[col + '_max'] = np.nan
                return_dict[col + '_min'] = np.nan
                return_dict[col + '_mean'] = np.nan
                return return_dict
        else:
            return_dict[col + '_max'] = np.nan
            return_dict[col + '_min'] = np.nan
            return_dict[col + '_mean'] = np.nan
            return return_dict
    else:
        
        if 'Cadence' in col or 'Steps' in col:
            col_data = np.array(data_df[col])
            return_dict[col + '_max'] = process_func_list['max'](col_data)
            return_dict[col + '_mean'] = process_func_list['mean'](col_data)
            return_dict[col + '_std'] = process_func_list['std'](col_data)
            return_dict[col + '_sum'] = process_func_list['sum'](col_data)
            return return_dict
        
        elif 'Intensity' in col:
            col_data = np.array(data_df[col])[data_df[col].nonzero()]
            if len(col_data) > threshold:
                return_dict[col + '_max'] = process_func_list['max'](col_data)
                return_dict[col + '_min'] = process_func_list['min'](col_data)
                return_dict[col + '_mean'] = process_func_list['mean'](col_data)
                return_dict[col + '_std'] = process_func_list['std'](col_data)
            else:
                return_dict[col + '_max'] = np.nan
                return_dict[col + '_min'] = np.nan
                return_dict[col + '_mean'] = np.nan
                return_dict[col + '_std'] = np.nan
            return return_dict
        
        else:
            col_data = np.array(data_df[col])[data_df[col].nonzero()]
            if len(col_data) > threshold:
                return_dict[col + '_max'] = process_func_list['max'](col_data)
                return_dict[col + '_min'] = process_func_list['min'](col_data)
                return_dict[col + '_mean'] = process_func_list['mean'](col_data)
                return_dict[col + '_std'] = process_func_list['std'](col_data)
                return return_dict
            else:
                return_dict[col + '_max'] = np.nan
                return_dict[col + '_min'] = np.nan
                return_dict[col + '_mean'] = np.nan
                return_dict[col + '_std'] = np.nan
                return return_dict


def calculate_rr_std_avg_for_series(data_df, threshold):
    ###########################################################
    # Process the valid data only
    ###########################################################
    rr_df = data_df[['RR0', 'RR1', 'RR2', 'RR3']]
    rr_array = np.array(rr_df)[np.nonzero(np.array(rr_df))]
    
    if len(rr_array) > threshold:
        time_domain_features = get_time_domain_features(list(rr_array*4))
        return time_domain_features
    else:
        return None