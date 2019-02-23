from datetime import timedelta

import os
import numpy as np
import pandas as pd

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


def read_unit(receiver_directory):
    """
    :param receiver_directory: string of receiver directory
    :return: string of receiver section only
    """
    receiver_section = receiver_directory.split(':')[:-1]
    
    receiver_section_str = ''
    for tmp in receiver_section:
        receiver_section_str = receiver_section_str + tmp + ':'
    receiver_section_str = receiver_section_str[:-1]
    
    return receiver_section_str


def return_primary_unit(owl_in_one_df):
    # Read where nurses worked
    receiver_unit_list = []
    for receiver_directory in owl_in_one_df['receiverDirectory']:
        receiver_unit_list.append(read_unit(receiver_directory))
    
    # Get unique receiver section
    unique_receiver_unit_dict = dict((x, receiver_unit_list.count(x)) for x in set(receiver_unit_list))
    
    primary_unit = max(unique_receiver_unit_dict, key=unique_receiver_unit_dict.get)
    
    return primary_unit


def read_shift_start_end_time(owl_in_one_df):
    """
    :param owl_in_one_df: owl-in-one pandas frame
    :return: start and end time of shift
    """
    last_time, start_time = pd.to_datetime(owl_in_one_df.index[0]), pd.to_datetime(owl_in_one_df.index[0])
    shift_start_end_time_list = []
    
    # We want to get the shift start and end time
    for index in owl_in_one_df.index:
        # Read current time
        cur_time = pd.to_datetime(index)
        
        # If last time minus curr time is less than 6 hours, then it is one shift
        time_offset = (cur_time - last_time).total_seconds()
        
        if time_offset > 3600 * 8:
            shift_start_end_time_list.append({'start': start_time.strftime(date_time_format)[:-3],
                                              'end': last_time.strftime(date_time_format)[:-3]})
            start_time = cur_time
        
        # Save current time as last time
        last_time = cur_time
    
    shift_start_end_time_list.append({'start': start_time.strftime(date_time_format)[:-3],
                                      'end': last_time.strftime(date_time_format)[:-3]})
    return shift_start_end_time_list


def init_shift_df(shift_start_end_time, cols, offset=1):
    """
    :param shift_start_end_time: shift start and end time
    :return: initialized pandas frame
    """
    start_time = pd.to_datetime(shift_start_end_time['start']).replace(second=0, microsecond=0)
    end_time = (pd.to_datetime(shift_start_end_time['end']) + timedelta(minutes=1)).replace(second=0, microsecond=0)
    
    # Total time offset
    time_offset_in_min = int(((end_time - start_time).total_seconds() / 60) / offset)
    
    # In case somebody not wearing the phone
    if 30 / offset < time_offset_in_min < 720 / offset:
        # Construct index
        index = [(start_time + timedelta(minutes=i * offset)).strftime(date_time_format)[:-3] for i in
                 range(time_offset_in_min)]
        
        # Initializing pandas frame
        shift_df = pd.DataFrame(data=np.zeros([len(index), len(cols)]), index=index, columns=cols, dtype=int)
        
        return shift_df
    
    else:
        return None


def process_owl_in_one_data(owl_in_one_df, offset=60):
    
    ###########################################################
    # Initialization
    ###########################################################
    interval = int(offset / 60)
    preprocess_data_df = pd.DataFrame()

    ###########################################################
    # Read primary unit
    ###########################################################
    primary_unit = return_primary_unit(owl_in_one_df)

    ###########################################################
    # Read shift start and end time
    ###########################################################
    shift_start_end_time_list = read_shift_start_end_time(owl_in_one_df)

    ###########################################################
    # Directory list
    ###########################################################
    receiverDirectory_list = list(set(owl_in_one_df.receiverDirectory))
    receiverDirectory_short_list = []
    for receiverDirectory in receiverDirectory_list:
        if primary_unit + ':' in receiverDirectory:
            if 'pat' in receiverDirectory:
                receiverDirectory_short_list.append('pat')
            elif 'lounge' in receiverDirectory:
                receiverDirectory_short_list.append('lounge')
            elif 'ns' in receiverDirectory:
                receiverDirectory_short_list.append('ns')
            elif 'med' in receiverDirectory:
                receiverDirectory_short_list.append('med')
            elif 'lab' in receiverDirectory:
                receiverDirectory_short_list.append('lab')
            elif 'receiving' in receiverDirectory:
                receiverDirectory_short_list.append('receiving')

    receiverDirectory_short_list.append('unknown')
    receiverDirectory_short_list.append('other_floor')
    receiverDirectory_short_list.append('floor2')
    
    receiverDirectory_short_list = list(set(receiverDirectory_short_list))

    for idx, shift_start_end_time in enumerate(shift_start_end_time_list):
        
        print('Current: %.2f%%' % ((idx + 1) / len(shift_start_end_time_list) * 100))

        ###########################################################
        # Init shift df
        ###########################################################
        shift_df = init_shift_df(shift_start_end_time, receiverDirectory_short_list)
        if shift_df is None:
            continue

        start_time = pd.to_datetime(shift_start_end_time['start'])
        end_time = pd.to_datetime(shift_start_end_time['end'])

        for index, row in owl_in_one_df.iterrows():
            if start_time <= pd.to_datetime(index) <= end_time:
                time_in_min = pd.to_datetime(index).replace(second=0, microsecond=0).strftime(date_time_format)[:-3]
                # There can't be 3 or more pins in one minute
                if primary_unit + ':' in row['receiverDirectory']:
                    if 'pat' in row['receiverDirectory']:
                        if shift_df.loc[time_in_min, 'pat'] < row['rssi']:
                            shift_df.loc[time_in_min, 'pat'] = row['rssi']
                    elif 'lounge' in row['receiverDirectory']:
                        if shift_df.loc[time_in_min, 'lounge'] < row['rssi']:
                            shift_df.loc[time_in_min, 'lounge'] = row['rssi']
                    elif 'ns' in row['receiverDirectory']:
                        if shift_df.loc[time_in_min, 'ns'] < row['rssi']:
                            shift_df.loc[time_in_min, 'ns'] = row['rssi']
                    elif 'med' in row['receiverDirectory']:
                        if shift_df.loc[time_in_min, 'med'] < row['rssi']:
                            shift_df.loc[time_in_min, 'med'] = row['rssi']
                    elif 'lab' in row['receiverDirectory']:
                        if shift_df.loc[time_in_min, 'lab'] < row['rssi']:
                            shift_df.loc[time_in_min, 'lab'] = row['rssi']
                    elif 'receiving' in row['receiverDirectory']:
                        if shift_df.loc[time_in_min, 'receiving'] < row['rssi']:
                            shift_df.loc[time_in_min, 'receiving'] = row['rssi']
                elif 'floor2' in row['receiverDirectory']:
                    if shift_df.loc[time_in_min, 'floor2'] < row['rssi']:
                        shift_df.loc[time_in_min, 'floor2'] = row['rssi']
                else:
                    if shift_df.loc[time_in_min, 'other_floor'] < row['rssi']:
                        shift_df.loc[time_in_min, 'other_floor'] = row['rssi']

        # We want to do converge here, three minutes version
        start_time = pd.to_datetime(shift_df.index[0])
        start_time = start_time.replace(minute=int(start_time.minute / interval) * interval)
        end_time = pd.to_datetime(shift_df.index[-1])
        end_time = end_time.replace(minute=int(end_time.minute / interval) * interval)

        shift_interval_min_df = init_shift_df({'start': start_time, 'end': end_time},
                                              list(set(receiverDirectory_short_list)), offset=interval)

        ###########################################################
        # Return final df
        ###########################################################
        shift_final_df = pd.DataFrame()

        if shift_interval_min_df is not None:
            for index in shift_interval_min_df.index:
                start_time = (pd.to_datetime(index) - timedelta(minutes=interval)).strftime(date_time_format)[:-3]
        
                if interval == 1:
                    end_time = (pd.to_datetime(index) + timedelta(minutes=interval)).strftime(date_time_format)[:-3]
                    end_time_index = (pd.to_datetime(index) + timedelta(minutes=interval)).strftime(date_time_format)[:-3]
                else:
                    end_time = (pd.to_datetime(index) + timedelta(minutes=interval - 1, seconds=59)).strftime(date_time_format)[:-3]
                    end_time_index = (pd.to_datetime(index) + timedelta(minutes=interval - 1)).strftime(date_time_format)[:-3]
        
                data_tmp_df = shift_df[start_time:end_time]
        
                if len(data_tmp_df) > interval:
                    tmp = pd.DataFrame(np.zeros([1, len(data_tmp_df.columns)]), index=[end_time_index], columns=data_tmp_df.columns)
                    tmp_series = data_tmp_df.max().sort_values(ascending=False)
            
                    if tmp_series.max() == 0:
                        tmp.loc[end_time_index, 'unknown'] = 1
                    else:
                        tmp.loc[end_time_index, tmp_series.index[0]] = 1
                
                        if tmp_series[0] - tmp_series[1] < 0:
                            tmp.loc[end_time_index, tmp_series.index[1]] = 1
            
                    shift_final_df = shift_final_df.append(tmp)
    
            shift_final_df = shift_final_df.loc[:, (shift_final_df != 0).any(axis=0)]
            preprocess_data_df = preprocess_data_df.append(shift_final_df)
            
    preprocess_data_df = preprocess_data_df.fillna(0)
    return preprocess_data_df

