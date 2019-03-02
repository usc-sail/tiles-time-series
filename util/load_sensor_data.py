import os
import pandas as pd
import datetime
import numpy as np

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

from datetime import timedelta


def read_omsignal(path, participant_id):
    # Read data and participant id first
    omsignal_file_abs_path = os.path.join(path, participant_id + '_omsignal.csv.gz')
    omsignal_df = pd.read_csv(omsignal_file_abs_path, index_col=0)
    
    omsignal_df = omsignal_df.fillna(0)
    omsignal_df = omsignal_df.drop_duplicates(keep='first')
    omsignal_df = omsignal_df.sort_index()
    
    return omsignal_df


def read_fitbit(path, participant_id):
    ###########################################################
    # 1. Read all fitbit file
    ###########################################################
    fitbit_data_dict = {}

    ppg_file_abs_path = os.path.join(path, participant_id + '_heartRate.csv.gz')
    step_file_abs_path = os.path.join(path, participant_id + '_stepCount.csv.gz')
    summary_file_abs_path = os.path.join(path, participant_id + '_dailySummary.csv.gz')

    ppg_df = pd.read_csv(ppg_file_abs_path, index_col=0)
    ppg_df = ppg_df.sort_index()

    step_df = pd.read_csv(step_file_abs_path, index_col=0)
    step_df = step_df.sort_index()

    summary_df = pd.read_csv(summary_file_abs_path, index_col=0)
    summary_df = summary_df.sort_index()

    fitbit_data_dict['ppg'] = ppg_df
    fitbit_data_dict['step'] = step_df
    fitbit_data_dict['summary'] = summary_df

    return fitbit_data_dict


def read_realizd(path, participant_id):
    ###########################################################
    # 1. Read all omsignal file
    ###########################################################
    realizd_file_abs_path = os.path.join(path, participant_id + '_realizd.csv.gz')
    realizd_all_df = pd.DataFrame()
    
    if os.path.exists(realizd_file_abs_path) is True:
        realizd_all_df = pd.read_csv(realizd_file_abs_path, index_col=0)
        realizd_all_df = realizd_all_df.sort_index()
        
    return realizd_all_df


def read_preprocessed_omsignal(path, participant_id):
    ###########################################################
    # 1. Read all omsignal file
    ###########################################################
    omsignal_folder = os.path.join(path, participant_id)
    omsignal_all_df = pd.DataFrame()

    if os.path.exists(omsignal_folder) is True:
        omsignal_file_list = os.listdir(omsignal_folder)
    
        for omsignal_file in omsignal_file_list:
            omsignal_file_abs_path = os.path.join(omsignal_folder, omsignal_file)
            
            omsignal_df = pd.read_csv(omsignal_file_abs_path, index_col=0)
            omsignal_df = omsignal_df.loc[:, ['HeartRate_mean', 'Steps_sum']]
            omsignal_all_df = omsignal_all_df.append(omsignal_df)
    
        omsignal_all_df = omsignal_all_df.sort_index()
    
    return omsignal_all_df


def read_preprocessed_fitbit_with_pad(data_config, participant_id):
    ###########################################################
    # 1. Read all fitbit file
    ###########################################################
    fitbit_folder = os.path.join(data_config.fitbit_sensor_dict['preprocess_path'], participant_id)
    if not os.path.exists(fitbit_folder):
        return None, None, None

    ###########################################################
    # List files and remove 'DS' file in mac system
    ###########################################################
    data_file_array = os.listdir(fitbit_folder)

    for data_file in data_file_array:
        if 'DS' in data_file: data_file_array.remove(data_file)

    processed_data_dict_array = {}
    
    if len(data_file_array) > 0:
        ###########################################################
        # Create dict for participant
        ###########################################################
        processed_data_dict_array['data'] = pd.DataFrame()
        processed_data_dict_array['raw'] = pd.DataFrame()
    
        for data_file in data_file_array:
            ###########################################################
            # Read data and append
            ###########################################################
            csv_path = os.path.join(fitbit_folder, data_file)
            data_df = pd.read_csv(csv_path, index_col=0)
        
            ###########################################################
            # Append data
            ###########################################################
            processed_data_dict_array['raw'] = processed_data_dict_array['raw'].append(data_df)
    
        ###########################################################
        # Assign data
        ###########################################################
        interval = int(data_config.fitbit_sensor_dict['offset'] / 60)
    
        final_df = processed_data_dict_array['raw'].sort_index()
        final_df[final_df < 0] = 0
        start_str = pd.to_datetime(final_df.index[0]).replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
        end_str = (pd.to_datetime(final_df.index[-1]) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
    
        time_length = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).total_seconds()
        point_length = int(time_length / data_config.fitbit_sensor_dict['offset']) + 1
        time_arr = [(pd.to_datetime(start_str) + timedelta(minutes=i * interval)).strftime(date_time_format)[:-3] for i in range(0, point_length)]
    
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
    
        processed_data_dict_array['data'] = final_df_all
        processed_data_dict_array['mean'] = np.nanmean(processed_data_dict_array['raw'], axis=0)
        processed_data_dict_array['std'] = np.nanstd(processed_data_dict_array['raw'], axis=0)
        
        return final_df_all, processed_data_dict_array['mean'], processed_data_dict_array['std']
    
    else:
        return None, None, None


def read_preprocessed_owl_in_one(path, participant_id):
    ###########################################################
    # 1. Read all omsignal file
    ###########################################################
    owl_in_one_file_abs_path = os.path.join(path, participant_id + '.csv.gz')
    if os.path.exists(owl_in_one_file_abs_path) is True:
        owl_in_one_all_df = pd.read_csv(owl_in_one_file_abs_path, index_col=0)
        owl_in_one_all_df = owl_in_one_all_df.sort_index()
    
        return owl_in_one_all_df
    else:
        return None


def read_preprocessed_realizd(path, participant_id):
    ###########################################################
    # 1. Read all realizd file
    ###########################################################
    realizd_file_abs_path = os.path.join(path, participant_id + '.csv.gz')
    if os.path.exists(realizd_file_abs_path) is True:
        realizd_all_df = pd.read_csv(realizd_file_abs_path, index_col=0)
        realizd_all_df = realizd_all_df.sort_index()
        
        return realizd_all_df
    else:
        return None

def read_preprocessed_audio(path, participant_id):
    ###########################################################
    # 1. Read all audio file
    ###########################################################
    audio_file_abs_path = os.path.join(path, participant_id + '.csv.gz')
    if os.path.exists(audio_file_abs_path) is True:
        audio_all_df = pd.read_csv(audio_file_abs_path, index_col=0)
        audio_all_df = audio_all_df.sort_index()
        
        return audio_all_df
    else:
        return None


def read_owl_in_one(path, participant_id):
    ###########################################################
    # 1. Read all owl-in-one file
    ###########################################################
    owl_in_one_file_abs_path = os.path.join(path, participant_id + '_bleProximity.csv.gz')
    owl_in_one_all_df = pd.DataFrame()
    
    if os.path.exists(owl_in_one_file_abs_path) is True:
        owl_in_one_all_df = pd.read_csv(owl_in_one_file_abs_path, index_col=0)
        owl_in_one_all_df = owl_in_one_all_df.sort_index()
        owl_in_one_all_df = owl_in_one_all_df.drop(columns='participantId')

        # Drop RSSI under 140
        owl_in_one_all_df = owl_in_one_all_df.loc[owl_in_one_all_df['rssi'] >= 140]
    
    return owl_in_one_all_df


def read_preprocessed_fitbit(path, participant_id):
    """
    Read preprocessed data
    """
    ###########################################################
    # If folder not exist
    ###########################################################
    read_participant_folder = os.path.join(path, participant_id)
    if not os.path.exists(read_participant_folder):
        return

    ###########################################################
    # List files and remove 'DS' file in mac system
    ###########################################################
    data_file_array = os.listdir(read_participant_folder)

    for data_file in data_file_array:
        if 'DS' in data_file: data_file_array.remove(data_file)

    fitbit_df = None

    if len(data_file_array) > 0:
        ###########################################################
        # Create dict for participant
        ###########################################################
        processed_data_dict_array = {}
        processed_data_dict_array['data'] = pd.DataFrame()
        
        for data_file in data_file_array:
            ###########################################################
            # Read data and append
            ###########################################################
            csv_path = os.path.join(read_participant_folder, data_file)
            data_df = pd.read_csv(csv_path, index_col=0)
        
            ###########################################################
            # Append data
            ###########################################################
            processed_data_dict_array['data'] = processed_data_dict_array['data'].append(data_df)
    
        ###########################################################
        # Assign data
        ###########################################################
        fitbit_df = processed_data_dict_array['data'].sort_index()
    return fitbit_df


def load_clustering_data(path, participant_id):
    clustering_df = pd.read_csv(os.path.join(path, participant_id + '.csv.gz'), index_col=0)
    clustering_df.loc[:, 'index'] = clustering_df.loc[:, 'start']
    clustering_df = clustering_df.set_index('index')
    
    return clustering_df


def load_segmentation_data(path, participant_id):
    segmentation_df = pd.read_csv(os.path.join(path, participant_id + '.csv.gz'), index_col=0)
    return segmentation_df


def load_filter_data(path, participant_id, filter_logic=None, threshold_dict=None, valid_data_rate=None):
    """ Load filter data

    Params:
    data_config - config setting
    participant_id - participant id
    filter_logic - how to filter the data
        None, no filter, return all data
        'work', return work days only
        'off_work', return non-work days only
    threshold_dict - extract data with only reasonable length:
        'min': minimum length of accepted recording for a day
        'max': maximum length of accepted recording for a day
        threshold_dict = {'min': 16, 'max': 32}
    valid_data_len - num of valid data present
    
    Returns:
    return_dict - contains dictionary of filter data
    keys:
        participant_id, data, filter_dict, filter_data_list

    """
    filter_dict_path_exist_cond = os.path.exists(os.path.join(path, participant_id, 'filter_dict.csv.gz')) == False
    data_path_exist_cond = os.path.exists(os.path.join(path, participant_id, participant_id + '.csv.gz')) == False
    
    if filter_dict_path_exist_cond or data_path_exist_cond:
        return None
    
    # Read filter dict df
    filter_dict_df = pd.read_csv(os.path.join(path, participant_id, 'filter_dict.csv.gz'), index_col=0)

    # Read whole data df
    data_df = pd.read_csv(os.path.join(path, participant_id, participant_id + '.csv.gz'), index_col=0)
    
    # Define return dict list
    return_dict = {}
    return_dict['participant_id'] = participant_id
    return_dict['data'] = data_df
    return_dict['filter_dict'] = filter_dict_df
    return_dict['filter_data_list'] = []

    # Add global statistics
    for col in list(data_df.columns):
        return_dict[col + '_mean'], return_dict[col + '_std'] = np.nan, np.nan

    # If we have enough amount of data, get statistics
    if len(np.where(data_df.StepCount >= 0)[0]) > 3 * 60:
    
        # Calculate stats on valid data
        mean = np.nanmean(data_df[data_df >= 0].dropna(), axis=0)
        std = np.std(data_df[data_df >= 0].dropna(), axis=0)
    
        # Save mean and std for each stream
        for i, col in enumerate(list(data_df.columns)):
            return_dict[col + '_mean'] = mean[i]
            return_dict[col + '_std'] = std[i]
    
    if len(filter_dict_df) > 0:
        for index, row_filter_dict_series in filter_dict_df.iterrows():
            
            # If we only select reasonable recordings
            cond_recording_duration1, cond_recording_duration2, cond_valid_data = False, False, False
            if threshold_dict is not None:
                cond_recording_duration1 = (pd.to_datetime(row_filter_dict_series.end) - pd.to_datetime(row_filter_dict_series.start)).total_seconds() < threshold_dict['min'] * 3600
                cond_recording_duration2 = (pd.to_datetime(row_filter_dict_series.end) - pd.to_datetime(row_filter_dict_series.start)).total_seconds() > threshold_dict['max'] * 3600
            
            # Work condition
            work_cond = row_filter_dict_series.work == 1

            # Day data dict
            day_filter_data_dict = {}
            day_filter_data_dict['data'] = data_df[row_filter_dict_series.start:row_filter_dict_series.end]

            if valid_data_rate is not None:
                cond_valid_data = (row_filter_dict_series.valid_length / len(day_filter_data_dict['data'])) < valid_data_rate

            if cond_recording_duration1 or cond_recording_duration2 or cond_valid_data:
                continue
            
            for tmp_index in list(row_filter_dict_series.index):
                day_filter_data_dict[tmp_index] = row_filter_dict_series[tmp_index]

            # If we want to get work days data only
            if filter_logic == 'work' and work_cond:
                return_dict['filter_data_list'].append(day_filter_data_dict)
            # If we want to get off_work days data only
            elif filter_logic == 'off_work' and not work_cond:
                return_dict['filter_data_list'].append(day_filter_data_dict)
            # Get everything, non-filter
            elif filter_logic is None:
                return_dict['filter_data_list'].append(day_filter_data_dict)
    
    if len(return_dict['filter_data_list']) == 0:
        return None
    else:
        return return_dict


def load_all_filter_data(path, participant_id_list, filter_logic=None, threshold_dict=None):
    """ Load filter data

    Params:
    data_config - config setting
    participant_id_list - participant id list
    filter_logic - how to filter the data
        None, no filter, return all data
        'work', return work days only
        'off_work', return non-work days only
    threshold_dict - extract data with only reasonable length:
        'min': minimum length of accepted recording for a day
        'max': maximum length of accepted recording for a day
        threshold_dict = {'min': 16, 'max': 32}

    Returns:
    participant_data_list - contains dictionary of filter data for all participants
    """

    participant_data_list = []
    for idx, participant_id in enumerate(participant_id_list):
    
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(participant_id_list)))
    
        # Read per participant data
        participant_data_dict = load_filter_data(path, participant_id, filter_logic=filter_logic, threshold_dict=threshold_dict)
    
        # Append data to the final list
        if participant_data_dict is not None: participant_data_list.append(participant_data_dict)

    print('Successfully load all participant filter data')
    
    return participant_data_list


def load_filter_clustering(path, participant_id):
    
    if os.path.exists(os.path.join(path, participant_id)) is False:
        return None
    
    return_dict = []
    file_list = os.listdir(os.path.join(path, participant_id))
    
    for file in file_list:
        data_dict = {}
        data_dict['participant_id'] = participant_id
        data_dict['data'] = pd.read_csv(os.path.join(path, participant_id, file), index_col=0)
        data_dict['start'] = list(data_dict['data'].index)[0]
        data_dict['length'] = len(list(data_dict['data'].index))

        return_dict.append(data_dict)
    
    return return_dict