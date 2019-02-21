import os
import pandas as pd
import datetime
import numpy as np

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

from datetime import timedelta


def read_fitbit(fitbit_folder, participant_id):
    ###########################################################
    # 1. Read all fitbit file
    ###########################################################
    fitbit_data_dict = {}

    ppg_file_abs_path = os.path.join(fitbit_folder, participant_id + '_heartRate.csv.gz')
    step_file_abs_path = os.path.join(fitbit_folder, participant_id + '_stepCount.csv.gz')
    summary_file_abs_path = os.path.join(fitbit_folder, participant_id + '_dailySummary.csv.gz')

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


def read_processed_omsignal(omsignal_pat, participant_id):
    ###########################################################
    # 1. Read all omsignal file
    ###########################################################
    omsignal_folder = os.path.join(omsignal_pat, participant_id)
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


def read_processed_owl_in_one(owl_in_one_path, participant_id):
    ###########################################################
    # 1. Read all omsignal file
    ###########################################################
    owl_in_one_file_abs_path = os.path.join(owl_in_one_path, participant_id + '.csv.gz')
    if os.path.exists(owl_in_one_file_abs_path) is True:
        owl_in_one_all_df = pd.read_csv(owl_in_one_file_abs_path, index_col=0)
        owl_in_one_all_df = owl_in_one_all_df.sort_index()
    
        return owl_in_one_all_df
    else:
        return None


def read_processed_realizd(realizd_path, participant_id):
    ###########################################################
    # 1. Read all omsignal file
    ###########################################################
    realizd_file_abs_path = os.path.join(realizd_path, participant_id + '.csv.gz')
    if os.path.exists(realizd_file_abs_path) is True:
        realizd_all_df = pd.read_csv(realizd_file_abs_path, index_col=0)
        realizd_all_df = realizd_all_df.sort_index()
        
        return realizd_all_df
    else:
        return None


def read_owl_in_one(owl_in_one_config, participant_id):
    ###########################################################
    # 1. Read all omsignal file
    ###########################################################
    owl_in_one_file_abs_path = os.path.join(owl_in_one_config.signal_type_folder,
                                            participant_id + '_bleProximity.csv.gz')
    owl_in_one_all_df = pd.DataFrame()
    
    if os.path.exists(owl_in_one_file_abs_path) is True:
        owl_in_one_all_df = pd.read_csv(owl_in_one_file_abs_path, index_col=0)
        owl_in_one_all_df = owl_in_one_all_df.sort_index()
        owl_in_one_all_df = owl_in_one_all_df.drop(columns='participantId')

        # Drop RSSI under 140
        owl_in_one_all_df = owl_in_one_all_df.loc[owl_in_one_all_df['rssi'] >= 140]
    
    return owl_in_one_all_df


def read_processed_fitbit(fitbit_config, participant_id):
    """
    Read preprocessed data
    """
    ###########################################################
    # If folder not exist
    ###########################################################
    read_participant_folder = os.path.join(fitbit_config.process_folder, participant_id)
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


def read_processed_fitbit_with_pad(fitbit_config, participant_id):
    """
    Read preprocessed data
    """
    ###########################################################
    # If folder not exist
    ###########################################################
    read_participant_folder = os.path.join(fitbit_config.process_folder, participant_id)
    if not os.path.exists(read_participant_folder):
        return

    ###########################################################
    # List files and remove 'DS' file in mac system
    ###########################################################
    data_file_array = os.listdir(read_participant_folder)

    for data_file in data_file_array:
        if 'DS' in data_file: data_file_array.remove(data_file)

    fitbit_df, final_df_all = None, None

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

        ###########################################################
        # If stepcount is below zero
        ###########################################################
        zero_index = list(fitbit_df.loc[fitbit_df['StepCount'] < 0].index)
        fitbit_df.loc[zero_index, 'StepCount'] = 0

        ###########################################################
        # Assign data
        ###########################################################
        interval = int(fitbit_config.offset / 60)
        start_str = pd.to_datetime(fitbit_df.index[0]).replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]
        end_str = (pd.to_datetime(fitbit_df.index[-1]) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).strftime(date_time_format)[:-3]

        time_length = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).total_seconds()
        point_length = int(time_length / fitbit_config.offset) + 1
        time_arr = [(pd.to_datetime(start_str) + timedelta(minutes=i * interval)).strftime(date_time_format)[:-3] for i in range(0, point_length)]

        final_df_all = pd.DataFrame(index=time_arr, columns=fitbit_df.columns)
        final_df_all.loc[fitbit_df.index, :] = fitbit_df

        ###########################################################
        # Assign pad
        ###########################################################
        pad_time_arr = list(set(time_arr) - set(fitbit_df.dropna().index))
        pad_df = pd.DataFrame(np.zeros([len(pad_time_arr), len(final_df_all.columns)]), index=pad_time_arr, columns=list(final_df_all.columns))
        temp = np.random.normal(size=(2, 2))
        temp2 = np.dot(temp, temp.T)
        for j in range(len(pad_df)):
            data_pad_tmp = np.random.multivariate_normal(np.zeros(2) - 50, temp2)
            pad_df.loc[pad_time_arr[j], :] = data_pad_tmp
        final_df_all.loc[pad_df.index, :] = pad_df
        final_df_all = final_df_all

    return final_df_all
