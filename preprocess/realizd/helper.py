from datetime import timedelta

import os
import numpy as np
import pandas as pd
import peakutils
from hrvanalysis import get_time_domain_features

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


def realizd_process_data(data_df, offset=60, check_saved=False, check_folder=None):
    
    ###########################################################
    # Initialization
    ###########################################################
    # Returned data
    interval = int(offset / 60)

    ###########################################################
    # Start iterate data with given parameters
    ###########################################################
    minute_offset = int(pd.to_datetime(data_df.index[0]).minute * 60 / offset)
    start_time = pd.to_datetime(data_df.index[0]).replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute_offset*interval)
    start_str = start_time.strftime(date_time_format)[:-3]

    minute_offset = pd.to_datetime(data_df.index[-1]) - pd.to_datetime(data_df.index[-1]).replace(minute=0, second=0, microsecond=0)
    minute_offset = int((minute_offset + timedelta(seconds=np.array(data_df)[-1][0])).total_seconds() / offset) + 1
    end_time = (pd.to_datetime(data_df.index[-1])).replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute_offset*interval)
    end_str = (end_time).strftime(date_time_format)[:-3]

    time_length = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).total_seconds()
    point_length = int(time_length / offset) + 1
    time_arr = [(pd.to_datetime(start_str) + timedelta(minutes=i*interval)).strftime(date_time_format)[:-3] for i in range(0, point_length+1, 1)]

    # Returned data
    preprocess_data_df = pd.DataFrame(index=time_arr, columns=['SecondsOnPhone', 'NumberOfTime'])
    preprocess_data_df = preprocess_data_df.fillna(0)

    for index, row in data_df.iterrows():
        seconds_on_phone = row.SecondsOnPhone

        minute_offset = int(pd.to_datetime(index).minute * 60 / offset)
        row_start_time = pd.to_datetime(index).replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute_offset*interval)
        row_start_time = row_start_time.strftime(date_time_format)[:-3]

        minute_offset = int((pd.to_datetime(index) + timedelta(seconds=seconds_on_phone) - pd.to_datetime(row_start_time)).total_seconds() / offset)
        row_end_time = (pd.to_datetime(row_start_time) + timedelta(minutes=(minute_offset+1)*interval))
        row_end_time = row_end_time.strftime(date_time_format)[:-3]

        row_offset = (pd.to_datetime(row_end_time) - pd.to_datetime(row_start_time)).total_seconds()
        
        for j in range(int(row_offset / offset)):
            tmp_start_time = (pd.to_datetime(row_start_time) + timedelta(seconds=offset*j)).strftime(date_time_format)[:-3]
            tmp_end_time = (pd.to_datetime(row_start_time) + timedelta(seconds=offset*(j+1))).strftime(date_time_format)[:-3]
            
            if j == 0:
                if int(row_offset / offset) != 1:
                    tmp_seconds_on_phone = pd.to_datetime(tmp_end_time) - pd.to_datetime(index)
                else:
                    tmp_seconds_on_phone = pd.to_datetime(index) + timedelta(seconds=seconds_on_phone) - pd.to_datetime(index)
            elif j == int(row_offset / offset) - 1 and j != 0:
                tmp_seconds_on_phone = pd.to_datetime(index) + timedelta(seconds=seconds_on_phone) - pd.to_datetime(tmp_start_time)
            else:
                tmp_seconds_on_phone = pd.to_datetime(tmp_end_time) - pd.to_datetime(tmp_start_time)

            preprocess_data_df.loc[tmp_start_time, 'SecondsOnPhone'] = preprocess_data_df.loc[tmp_start_time, 'SecondsOnPhone'] + tmp_seconds_on_phone.total_seconds()
            preprocess_data_df.loc[tmp_start_time, 'NumberOfTime'] = preprocess_data_df.loc[tmp_start_time, 'NumberOfTime'] + 1
        
        # If check saved or not
        if check_saved is True and check_folder is not None:
            if len(preprocess_data_df) > 0 and os.path.join(check_folder, preprocess_data_df.index[0] + '.csv') is True:
                return None

    return preprocess_data_df

