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
    # Initiate data df
    ###########################################################
    start_time = pd.to_datetime(data_df.index[0]).replace(hour=0, minute=0, second=0, microsecond=0)
    start_str = start_time.strftime(date_time_format)[:-3]

    end_time = (pd.to_datetime(data_df.index[-1]) + timedelta(days=1)).replace(hour=23, minute=59, second=0, microsecond=0)
    end_str = (end_time).strftime(date_time_format)[:-3]

    time_length = (pd.to_datetime(end_str) - pd.to_datetime(start_str)).total_seconds()
    point_length = int(time_length / offset) + 1
    time_arr = [(pd.to_datetime(start_str) + timedelta(minutes=i*interval)).strftime(date_time_format)[:-3] for i in range(0, point_length+1, 1)]

    # fill return df with 0
    return_df = pd.DataFrame(index=time_arr, columns=['SecondsOnPhone', 'NumberOfTime'])
    return_df = return_df.fillna(0)

    # Iterate data df
    for index, row in data_df.iterrows():
        seconds_on_phone = row.SecondsOnPhone
        
        row_time_arr = [(pd.to_datetime(index) + timedelta(seconds=i)).strftime(date_time_format)[:-3] for i in range(0, int(seconds_on_phone) + 1, 1)]
        row_time_df = pd.DataFrame(index=row_time_arr, columns=['phone'])
        row_time_df['phone'] = 1

        # Row data
        row_start_time = pd.to_datetime(index).replace(second=0, microsecond=0)
        row_end_time = (pd.to_datetime(index) + timedelta(seconds=seconds_on_phone) + timedelta(minutes=1))
        row_offset = (pd.to_datetime(row_end_time) - pd.to_datetime(row_start_time)).total_seconds()

        for j in range(int(row_offset / offset)):
            tmp_start_time = (pd.to_datetime(row_start_time) + timedelta(seconds=offset*j)).strftime(date_time_format)[:-3]
            tmp_end_time = (pd.to_datetime(row_start_time) + timedelta(seconds=offset*(j+1))).strftime(date_time_format)[:-3]

            return_df.loc[tmp_start_time, 'SecondsOnPhone'] = return_df.loc[tmp_start_time, 'SecondsOnPhone'] + np.sum(np.array(row_time_df[tmp_start_time:tmp_end_time])) - 1
            return_df.loc[tmp_start_time, 'NumberOfTime'] = return_df.loc[tmp_start_time, 'NumberOfTime'] + 1
        
        # If check saved or not
        if check_saved is True and check_folder is not None:
            if len(return_df) > 0 and os.path.join(check_folder, return_df.index[0] + '.csv') is True:
                return None

    return return_df

