import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def medfilt(x, window):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert window % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (window - 1) // 2
    y = np.zeros((len(x), window), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)[:-1 * window + 1]


# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'


if __name__ == '__main__':

    # read sleep data
    sleep_routine_work_df = pd.read_csv(os.path.join('../output/sleep_routine_work', 'workday_sleep_routine.csv'))
    
    # read id mapping
    id_data_df = pd.read_csv(os.path.join('../../data', 'keck_wave1/2_preprocessed_data/ground_truth', 'IDs.csv'))
    
    # time window
    time_window = 3
    
    # Commute array
    final_df = []
    
    before_work_commute_col = ['before_work_start_commute_time', 'before_work_end_commute_time',
                               'before_work_commute_duration']
    after_work_commute_col = ['after_work_start_commute_time',
                              'after_work_end_commute_time',
                              'after_work_commute_duration']
    
    for index, sleep_routine in sleep_routine_work_df.iterrows():
        
        # om signal id, read fitbit data: heart rate and step count
        om_signal_id = id_data_df.loc[id_data_df['user_id'] == sleep_routine['user_id']]['OMuser_id'].values[0]
        user_id = sleep_routine['user_id']
        
        heart_rate_data_df = pd.read_csv(os.path.join('../../data', 'keck_wave1/2_preprocessed_data/fitbit/fitbit', om_signal_id + '_heartRate.csv'), index_col=0)
        step_count_data_df = pd.read_csv(os.path.join('../../data', 'keck_wave1/2_preprocessed_data/fitbit/fitbit', om_signal_id + '_stepCount.csv'), index_col=0)
        
        work_shift_type = sleep_routine['work_shift_type']
        
        sleep_routine = sleep_routine.to_frame().transpose()

        sleep_routine[before_work_commute_col[0]] = np.nan
        sleep_routine[before_work_commute_col[1]] = np.nan
        sleep_routine[before_work_commute_col[2]] = np.nan
        
        # a. before work sleep end and start work time, read time point
        if sleep_routine['wake_before_work_standard_work_time'].values[0] < 8:
            
            before_work_timeline = sleep_routine[
                ['sleep_before_work_SleepEndTimestamp', 'start_work_time', 'start_recording_time']]
            wake_up_time = datetime.datetime.strptime(
                    before_work_timeline['sleep_before_work_SleepEndTimestamp'].values[0],
                    date_time_format)
            start_work_time = datetime.datetime.strptime(
                    before_work_timeline['start_work_time'].values[0],
                    date_time_format)

            wake_up_time = wake_up_time - datetime.timedelta(minutes=120)
            start_work_time = start_work_time + datetime.timedelta(minutes=30)
            
            commute_time = np.nan
            
            # is find commute set to be false
            is_find_commute = False
            is_initialize_commute_end = False
            is_find_peak_before_work = False
            
            if len(heart_rate_data_df) > 0:
                
                # 1.1 Read the data between wake up and work
                before_work_heartrate_df = heart_rate_data_df.loc[
                                           wake_up_time.strftime(date_time_format):
                                           start_work_time.strftime(date_time_format)]
                before_work_step_count_df = step_count_data_df.loc[
                                            wake_up_time.strftime(date_time_format):
                                            start_work_time.strftime(date_time_format)]
                
                # 1.2 preprocessing
                if time_window > 1:
                    # before_work_step_count_ma = moving_average(np.asarray(before_work_step_count_df['StepCount']), time_window)
                    before_work_step_count_ma = medfilt(np.asarray(before_work_step_count_df['StepCount']), time_window)
                else:
                    before_work_step_count_ma = np.asarray(before_work_step_count_df['StepCount'])
                
                before_work_step_count_time = []
                before_work_heart_rate_time = []
                
                for date in before_work_step_count_df.index.values:
                    before_work_step_count_time.append(datetime.datetime.strptime(date, date_time_format))
                
                for date in before_work_heartrate_df.index.values:
                    before_work_heart_rate_time.append(datetime.datetime.strptime(date, date_time_format))
                
                # 1.3 decide timeline before work
                if (before_work_step_count_time[-1] - before_work_step_count_time[0]).total_seconds() > 60 * 45:
                    
                    # Now we will try if we can get commute data out of step count
                    last_inactive_time = before_work_step_count_time[-1]
                    end_commute_time = before_work_step_count_time[-1]
                    
                    for i in range(len(before_work_step_count_ma) - 1, -1, -1):
                        if is_find_commute is False:
                            if is_find_peak_before_work is False:
                                if before_work_step_count_ma[i] > 45:
                                    is_find_peak_before_work = True
                            else:
                                # store current time
                                current_time = before_work_step_count_time[i + time_window - 1]
                                
                                if before_work_step_count_ma[i] < 8:
                                    
                                    # if we haven't have the first inactive point
                                    if is_initialize_commute_end is False:
                                        end_commute_time = current_time
                                        last_inactive_time = current_time
                                        is_initialize_commute_end = True
                                    else:
                                        # in case we have a spike, we will ignore
                                        if (last_inactive_time - current_time).total_seconds() < 60 * 4:
                                            last_inactive_time = current_time
                                        # if we have a region which people moves a lot, it is not regular commute
                                        else:
                                            # if start and end is greater than 10 minutes
                                            if 15 * 60 < (end_commute_time - last_inactive_time).total_seconds() < 60 * 90:
                                                commute_time_in_minute = (end_commute_time - last_inactive_time).total_seconds() / 60
                                                
                                                # We have found the commute time
                                                is_find_commute = True
                                            else:
                                                end_commute_time = current_time
                                                last_inactive_time = current_time
                                    
                                
                                # if step count is larger than 30, it is for sure not in commuting
                                elif before_work_step_count_ma[i] > 30:
                                    if 15 * 60 < (end_commute_time - last_inactive_time).total_seconds() < 60 * 90:
                                        commute_time_in_minute = (end_commute_time - last_inactive_time).total_seconds() / 60
                                        # print('user_id: ' + user_id)
                                        # print('commute time: ' + str(commute_time_in_minute))
                                        # print('commute start time: ' + last_inactive_time.strftime(date_time_format))
                                        # print('commute end time: ' + end_commute_time.strftime(date_time_format))
                                        
                                        # We have found the commute time
                                        is_find_commute = True
                                    else:
                                        is_initialize_commute_end = False

                # plot Step count
                fig, ax = plt.subplots()
                # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                if time_window > 1:
                    # ax.plot(before_work_step_count_time[:(-1 * time_window + 1)], before_work_step_count_ma)
                    ax.plot(before_work_step_count_time[time_window - 1:],
                            before_work_step_count_ma)
                else:
                    ax.plot(before_work_step_count_time, before_work_step_count_ma)
                plt.axvline(x=end_commute_time, color='r', linestyle='--')
                plt.axvline(x=last_inactive_time, color='r', linestyle='--')
                plt.xticks(rotation=90)
                plt.savefig(os.path.join('../output/commute_before_work',
                                         user_id + '_StepCount_' +
                                         sleep_routine['work_date'].values[0] + '.png'))
                # plt.show()
                plt.close()

                # plot Heart rate count
                fig, ax = plt.subplots()
                ax.plot(before_work_heart_rate_time,
                        np.array(before_work_heartrate_df['HeartRate']))
                plt.xticks(rotation=90)
                plt.axvline(x=end_commute_time, color='r', linestyle='--')
                plt.axvline(x=last_inactive_time, color='r', linestyle='--')
                plt.savefig(os.path.join('../output/commute_before_work',
                                         user_id + '_HeartRate_' +
                                         sleep_routine['work_date'].values[0] + '.png'))
                # plt.show()
                plt.close()

                if is_find_commute is True:
                    
                    # Construct frame and append
                    print('Commute before work')
                    print('user_id: ' + user_id)
                    print('commute time: ' + str(commute_time_in_minute))
                    print('commute start time: ' + last_inactive_time.strftime(date_time_format))
                    print('commute end time: ' + end_commute_time.strftime(date_time_format))
                    
                    sleep_routine[before_work_commute_col[0]] = last_inactive_time.strftime(date_time_format)[:-3]
                    sleep_routine[before_work_commute_col[1]] = end_commute_time.strftime(date_time_format)[:-3]
                    sleep_routine[before_work_commute_col[2]] = commute_time_in_minute
            else:
                sleep_routine[before_work_commute_col[0]] = np.nan
                sleep_routine[before_work_commute_col[1]] = np.nan
                sleep_routine[before_work_commute_col[2]] = np.nan

        sleep_routine[after_work_commute_col[0]] = np.nan
        sleep_routine[after_work_commute_col[1]] = np.nan
        sleep_routine[after_work_commute_col[2]] = np.nan
        
        # b. after work sleep end and start work time, read time point
        if sleep_routine['sleep_after_work_standard_work_time'].values[0] < 8:
            after_work_timeline = sleep_routine[
                ['end_work_time', 'end_recording_time', 'sleep_after_work_SleepEndTimestamp']]
            work_end_time_standard = datetime.datetime.strptime(after_work_timeline['end_work_time'].values[0],
                                                                date_time_format)
            
            commute_time = np.nan
            
            # is find commute set to be false
            is_find_commute = False
            is_initialize_commute_start = False
            is_find_peak_after_work = False
            
            if len(heart_rate_data_df) > 0:
                
                # b.1 Read the data between wake up and work
                after_work_heartrate_df = heart_rate_data_df.loc[sleep_routine['end_work_time'].values[0]:sleep_routine[
                    'sleep_after_work_SleepBeginTimestamp'].values[0]]
                after_work_step_count_df = step_count_data_df.loc[sleep_routine['end_work_time'].values[0]:sleep_routine[
                    'sleep_after_work_SleepBeginTimestamp'].values[0]]
                
                # b.2 preprocessing
                if time_window > 1:
                    # before_work_step_count_ma = moving_average(np.asarray(before_work_step_count_df['StepCount']), time_window)
                    after_work_step_count_ma = medfilt(np.asarray(after_work_step_count_df['StepCount']), time_window)
                else:
                    after_work_step_count_ma = np.asarray(after_work_step_count_df['StepCount'])
                
                after_work_step_count_time = []
                after_work_heart_rate_time = []
                
                for date in after_work_step_count_df.index.values:
                    after_work_step_count_time.append(datetime.datetime.strptime(date, date_time_format))
                
                for date in after_work_heartrate_df.index.values:
                    after_work_heart_rate_time.append(datetime.datetime.strptime(date, date_time_format))
                
                # b.3 decide timeline before work
                if (after_work_step_count_time[-1] - after_work_step_count_time[0]).total_seconds() > 60 * 45:
                    
                    # Now we will try if we can get commute data out of step count
                    last_inactive_time = after_work_step_count_time[0]
                    start_commute_time = after_work_step_count_time[0]
                    
                    for i in range(0, len(after_work_step_count_ma), 1):
                        if is_find_commute is False:
                            if is_find_peak_after_work is False:
                                if after_work_step_count_ma[i] > 60:
                                    is_find_peak_after_work = True
                            else:
                                # store current time
                                current_time = after_work_step_count_time[i]
                                
                                if after_work_step_count_ma[i] < 10:
                                    
                                    # if we haven't have the first inactive point
                                    if is_initialize_commute_start is False:
                                        start_commute_time = current_time
                                        last_inactive_time = current_time
                                        is_initialize_commute_start = True
                                    else:
                                        # in case we have a spike, we will ignore
                                        if (current_time - last_inactive_time).total_seconds() < 60 * 4:
                                            last_inactive_time = current_time
                                        # if we have a region which people moves a lot, it is not regular commute
                                        else:
                                            # if start and end is greater than 10 minutes
                                            if 15 * 60 < (last_inactive_time - start_commute_time).total_seconds() < 60 * 90:
                                                commute_time_in_minute = (last_inactive_time - start_commute_time).total_seconds() / 60
                                                
                                                # We have found the commute time
                                                is_find_commute = True
                                            else:
                                                start_commute_time = current_time
                                                last_inactive_time = current_time
                                
                                # if step count is larger than 30, it is for sure not in commuting
                                elif after_work_step_count_ma[i] > 40:
                                    if 15 * 60 < (last_inactive_time - start_commute_time).total_seconds() < 60 * 90:
                                        commute_time_in_minute = (last_inactive_time - start_commute_time).total_seconds() / 60
                                        
                                        # We have found the commute time
                                        is_find_commute = True
                                    else:
                                        is_initialize_commute_start = False

                    # plot Step count
                    fig, ax = plt.subplots()
                    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                    if time_window > 1:
                        # ax.plot(before_work_step_count_time[:(-1 * time_window + 1)], before_work_step_count_ma)
                        ax.plot(after_work_step_count_time[:-time_window + 1],
                                after_work_step_count_ma)
                    else:
                        ax.plot(after_work_step_count_time, after_work_step_count_ma)
                    plt.axvline(x=start_commute_time, color='r', linestyle='--')
                    plt.axvline(x=last_inactive_time, color='r', linestyle='--')
                    plt.xticks(rotation=90)
                    plt.savefig(os.path.join('../output/commute_after_work',
                                             user_id + '_StepCount_' +
                                             sleep_routine['work_date'].values[0] + '.png'))
                    # plt.show()
                    plt.close()

                    # plot Heart rate count
                    fig, ax = plt.subplots()
                    # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                    ax.plot(after_work_heart_rate_time,
                            np.array(after_work_heartrate_df['HeartRate']))
                    plt.axvline(x=start_commute_time, color='r', linestyle='--')
                    plt.axvline(x=last_inactive_time, color='r', linestyle='--')
                    plt.xticks(rotation=90)
                    plt.savefig(os.path.join('../output/commute_after_work',
                                             user_id + '_HeartRate_' +
                                             sleep_routine['work_date'].values[0] + '.png'))
                    # plt.show()
                    plt.close()
                    
                    if is_find_commute is True:
                        
                        # Construct frame and append
                        print('Commute after work')
                        print('user_id: ' + user_id)
                        print('commute time: ' + str(commute_time_in_minute))
                        print('commute start time: ' + start_commute_time.strftime(date_time_format))
                        print('commute end time: ' + last_inactive_time.strftime(date_time_format))
                        
                        sleep_routine[after_work_commute_col[0]] = start_commute_time.strftime(date_time_format)[:-3]
                        sleep_routine[after_work_commute_col[1]] = last_inactive_time.strftime(date_time_format)[:-3]
                        sleep_routine[after_work_commute_col[2]] = commute_time_in_minute
            
            # if no Fitbit data
            else:
                sleep_routine[after_work_commute_col[0]] = np.nan
                sleep_routine[after_work_commute_col[1]] = np.nan
                sleep_routine[after_work_commute_col[2]] = np.nan
        
        # save to final array
        if len(final_df) == 0:
            final_df = sleep_routine
        else:
            final_df = final_df.append(sleep_routine)
            final_df = final_df[sleep_routine.columns]

        final_df.to_csv(os.path.join('../output/sleep_routine_work', 'workday_sleep_routine_commute.csv'), index=False)



