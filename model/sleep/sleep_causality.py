"""
Filter the data
"""
from __future__ import print_function

import os
import sys
import matplotlib.pyplot as plt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import numpy as np
import pandas as pd
import pickle
import preprocess
from statsmodels.tsa.stattools import grangercausalitytests

from fancyimpute import (
	IterativeImputer,
    SoftImpute,
    KNN
)

max_order = 5


# Initial output dataframe
def init_data_df(max_order, col_list1, col_list2, participant_id=None):
    cols = []
    for col1 in col_list1:
        for col2 in col_list2:
            cols.append(col1 + '->' + col2)
            cols.append(col2 + '->' + col1)
    
    init_nan = np.ones([len(range(2, max_order)), len(cols)])
    init_nan = init_nan.fill(np.nan)
    data_dict = {}
    
    for i in range(2, max_order):
        data_dict['order' + str(i)] = {}
        
        if participant_id != None:
            data_dict['order' + str(i)]['f_stat'] = pd.DataFrame(init_nan, index=[participant_id], columns=cols)
            data_dict['order' + str(i)]['f_p'] = pd.DataFrame(init_nan, index=[participant_id], columns=cols)
            data_dict['order' + str(i)]['chi2_stat'] = pd.DataFrame(init_nan, index=[participant_id], columns=cols)
            data_dict['order' + str(i)]['chi2_p'] = pd.DataFrame(init_nan, index=[participant_id], columns=cols)
        else:
            data_dict['order' + str(i)]['f_stat'] = pd.DataFrame()
            data_dict['order' + str(i)]['f_p'] = pd.DataFrame()
            data_dict['order' + str(i)]['chi2_stat'] = pd.DataFrame()
            data_dict['order' + str(i)]['chi2_p'] = pd.DataFrame()
    
    return data_dict


def sleep_causality(fitbit_summary_final_df, participant_id):
    sleep_df = fitbit_summary_final_df[['SleepMinutesAsleep', 'SleepMinutesInBed']]
    other_df = fitbit_summary_final_df[['Fat Burn_minutes', 'Cardio_minutes',
                                        'Out of Range_minutes', 'Peak_minutes',
                                        'RestingHeartRate', 'NumberSteps']]
    
    sleep_cols = list(sleep_df.columns)
    other_cols = list(other_df.columns)
    
    # initialize containers
    data_dict = init_data_df(max_order, other_cols, sleep_cols, participant_id)
    
    for i in range(len(sleep_cols)):
        for j in range(len(other_cols)):
            input1 = sleep_df.iloc[:, i].to_frame()
            input2 = other_df.iloc[:, j].to_frame()
            input = pd.merge(input1, input2, how='outer', left_index=True, right_index=True)

            # granger analysis
            temp = grangercausalitytests(input, max_order, verbose=False)
            col = sleep_cols[i] + '->' + other_cols[j]

            for order in range(2, max_order):
                data_dict['order' + str(order)]['f_stat'].loc[participant_id, col] = temp[order + 1][0]['ssr_ftest'][0]
                data_dict['order' + str(order)]['f_p'].loc[participant_id, col] = temp[order + 1][0]['ssr_ftest'][1]
                data_dict['order' + str(order)]['chi2_stat'].loc[participant_id, col] = temp[order + 1][0]['ssr_chi2test'][0]
                data_dict['order' + str(order)]['chi2_p'].loc[participant_id, col] = temp[order + 1][0]['ssr_chi2test'][1]
    
    for i in range(len(other_cols)):
        for j in range(len(sleep_cols)):
            input1 = other_df.iloc[:, i].to_frame()
            input2 = sleep_df.iloc[:, j].to_frame()
            input = pd.merge(input1, input2, how='outer', left_index=True, right_index=True)
            # granger analysis
            temp = grangercausalitytests(input, max_order, verbose=False)
            col = other_cols[i] + '->' + sleep_cols[j]

            for order in range(2, max_order):
                data_dict['order' + str(order)]['f_stat'].loc[participant_id, col] = temp[order + 1][0]['ssr_ftest'][0]
                data_dict['order' + str(order)]['f_p'].loc[participant_id, col] = temp[order + 1][0]['ssr_ftest'][1]
                data_dict['order' + str(order)]['chi2_stat'].loc[participant_id, col] = temp[order + 1][0]['ssr_chi2test'][0]
                data_dict['order' + str(order)]['chi2_p'].loc[participant_id, col] = temp[order + 1][0]['ssr_chi2test'][1]
    
    return data_dict
    

def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)

    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')

    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    igtb_cols = [col for col in list(igtb_df.columns) if 'igtb' in col]
    psqi_raw_igtb = load_data_basic.read_PSQI_Raw(tiles_data_path)
    
    # mgt_df = load_data_basic.read_MGT(tiles_data_path)
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    # gc df
    col1 = ['SleepMinutesAsleep', 'SleepMinutesInBed']
    col2 = ['Fat Burn_minutes', 'Cardio_minutes', 'Out of Range_minutes', 'Peak_minutes', 'RestingHeartRate', 'NumberSteps']
    gc_dict = init_data_df(max_order, col1, col2)
    
    if os.path.exists(os.path.join(os.path.dirname(__file__), 'sleep_gc.pkl')) is True:
        pkl_file = open(os.path.join(os.path.dirname(__file__), 'sleep_gc.pkl'), 'rb')
        sleep_gc_dict = pickle.load(pkl_file)
        # data_df = sleep_gc_dict['order2']['chi2_p'][['SleepMinutesInBed->NumberSteps', 'SleepMinutesAsleep->NumberSteps']]
        data_df = sleep_gc_dict['order3']['chi2_p']
        
        for idx, participant_id in enumerate(list(data_df.index)):
            print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(list(data_df.index))))
    
            nurse = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].currentposition[0]
            primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
            shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
            job_str = 'nurse' if nurse == 1 and 'Dialysis' not in primary_unit else 'non_nurse'
            shift_str = 'day' if shift == 'Day shift' else 'night'
    
            data_df.loc[participant_id, 'job'] = job_str
            data_df.loc[participant_id, 'shift'] = shift_str
            
            for col in igtb_cols:
                data_df.loc[participant_id, col] = igtb_df.loc[igtb_df['ParticipantID'] == participant_id][col][0]

        nurse_sleep_df = data_df.loc[data_df['job'] == 'nurse']
        day_nurse_sleep_df = data_df.loc[data_df['shift'] == 'day']
        night_nurse_sleep_df = data_df.loc[data_df['shift'] == 'night']

        nurse_sleep_df = nurse_sleep_df.drop(columns=['job', 'shift'])
        day_nurse_sleep_df = day_nurse_sleep_df.drop(columns=['job', 'shift'])
        night_nurse_sleep_df = night_nurse_sleep_df.drop(columns=['job', 'shift'])
        
        for col in list(day_nurse_sleep_df.columns):
            if col not in igtb_cols:
                print(col)
                print(len(np.where(np.array(day_nurse_sleep_df[col]) < 0.05)[0]) / len(day_nurse_sleep_df))

        for col in list(night_nurse_sleep_df.columns):
            if col not in igtb_cols:
                print(col)
                print(len(np.where(np.array(night_nurse_sleep_df[col]) < 0.05)[0]) / len(night_nurse_sleep_df))
        
        tmp1_df = day_nurse_sleep_df.loc[day_nurse_sleep_df['SleepMinutesInBed->NumberSteps'] < 0.05]
        tmp2_df = night_nurse_sleep_df.loc[night_nurse_sleep_df['SleepMinutesInBed->NumberSteps'] < 0.05]

        tmp3_df = day_nurse_sleep_df.loc[day_nurse_sleep_df['SleepMinutesInBed->NumberSteps'] > 0.05]
        tmp4_df = night_nurse_sleep_df.loc[night_nurse_sleep_df['SleepMinutesInBed->NumberSteps'] > 0.05]

        tmp5_df = nurse_sleep_df.loc[nurse_sleep_df['SleepMinutesInBed->NumberSteps'] < 0.05]
        tmp6_df = nurse_sleep_df.loc[nurse_sleep_df['SleepMinutesInBed->NumberSteps'] > 0.05]

        from scipy import stats
        for col in list(tmp1_df.columns):
            if col in igtb_cols:
                print(col)
                stat, p = stats.ks_2samp(tmp5_df[col].dropna(), tmp6_df[col].dropna())
                print('K-S test for %s' % col)
                print('Statistics = %.3f, p = %.3f\n\n' % (stat, p))
                
        print()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        print('read_preprocess_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        # Read all data
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        fitbit_summary_df = fitbit_data_dict['summary']
        
        '''
        cols = ['Fat Burn_caloriesOut', 'Fat Burn_minutes',
                'Cardio_caloriesOut', 'Cardio_minutes',
                'Out of Range_caloriesOut', 'Out of Range_minutes',
                'Peak_caloriesOut', 'Peak_minutes',
                'RestingHeartRate', 'SleepMinutesAsleep', 'SleepMinutesInBed', 'NumberSteps']
        '''
        
        cols = ['Fat Burn_minutes', 'Cardio_minutes', 'Out of Range_minutes', 'Peak_minutes',
                'RestingHeartRate', 'SleepMinutesAsleep', 'SleepMinutesInBed', 'NumberSteps']
        
        fitbit_summary_df = fitbit_summary_df[cols]
        
        dates_list = pd.date_range(pd.to_datetime(list(fitbit_summary_df.index)[0]), pd.to_datetime(list(fitbit_summary_df.index)[-1]))
        dates_str_list = [date.strftime(load_data_basic.date_only_date_time_format) for date in dates_list]
        
        fitbit_summary_final_df = pd.DataFrame(index=dates_str_list, columns=cols)
        
        for i in range(len(fitbit_summary_df)):
            date_str = pd.to_datetime(list(fitbit_summary_df.index)[i]).strftime(load_data_basic.date_only_date_time_format)
            
            row_df = fitbit_summary_df.iloc[i, :]
            minutes_in_total = float(row_df['Cardio_minutes']) + float(row_df['Peak_minutes']) + float(row_df['Fat Burn_minutes']) + float(row_df['Out of Range_minutes'])

            fitbit_summary_final_df.loc[date_str, 'Cardio_minutes'] = float(row_df['Cardio_minutes']) / minutes_in_total
            fitbit_summary_final_df.loc[date_str, 'Peak_minutes'] = float(row_df['Peak_minutes']) / minutes_in_total
            fitbit_summary_final_df.loc[date_str, 'Fat Burn_minutes'] = float(row_df['Fat Burn_minutes']) / minutes_in_total
            fitbit_summary_final_df.loc[date_str, 'Out of Range_minutes'] = float(row_df['Out of Range_minutes']) / minutes_in_total

            fitbit_summary_final_df.loc[date_str, 'RestingHeartRate'] = float(row_df['RestingHeartRate'])
            fitbit_summary_final_df.loc[date_str, 'NumberSteps'] = float(row_df['NumberSteps'])
            
            '''
            fitbit_summary_final_df.loc[date_str, 'Out of Range_caloriesOut'] = float(row_df['Out of Range_caloriesOut'])
            fitbit_summary_final_df.loc[date_str, 'Fat Burn_caloriesOut'] = float(row_df['Fat Burn_caloriesOut'])
            fitbit_summary_final_df.loc[date_str, 'Peak_caloriesOut'] = float(row_df['Peak_caloriesOut'])
            fitbit_summary_final_df.loc[date_str, 'Cardio_caloriesOut'] = float(row_df['Cardio_caloriesOut'])
            '''
            
            if float(row_df['SleepMinutesInBed']) < 240:
                fitbit_summary_final_df.loc[date_str, 'SleepMinutesInBed'] = np.nan
            else:
                fitbit_summary_final_df.loc[date_str, 'SleepMinutesInBed'] = float(row_df['SleepMinutesInBed'])
                
            if float(row_df['SleepMinutesAsleep']) < 240:
                fitbit_summary_final_df.loc[date_str, 'SleepMinutesAsleep'] = np.nan
            else:
                fitbit_summary_final_df.loc[date_str, 'SleepMinutesAsleep'] = float(row_df['SleepMinutesAsleep'])
        
        if len(fitbit_summary_final_df.dropna()) < 30:
            continue

        # fitbit_summary_df
        # model = KNN(k=3)
        model = IterativeImputer()
        fitbit_summary_final_array = model.fit_transform(np.array(fitbit_summary_final_df))
        fitbit_summary_final_df.loc[:, :] = fitbit_summary_final_array
        
        data_dict = sleep_causality(fitbit_summary_final_df, participant_id)

        for order in range(2, max_order):
            gc_dict['order' + str(order)]['f_stat'] = gc_dict['order' + str(order)]['f_stat'].append(data_dict['order' + str(order)]['f_stat'])
            gc_dict['order' + str(order)]['f_p'] = gc_dict['order' + str(order)]['f_p'].append(data_dict['order' + str(order)]['f_p'])
            gc_dict['order' + str(order)]['chi2_stat'] = gc_dict['order' + str(order)]['chi2_stat'].append(data_dict['order' + str(order)]['chi2_stat'])
            gc_dict['order' + str(order)]['chi2_p'] = gc_dict['order' + str(order)]['chi2_p'].append(data_dict['order' + str(order)]['chi2_p'])
    
    output = open(os.path.join(os.path.dirname(__file__), 'sleep_gc.pkl'), 'wb')
    pickle.dump(gc_dict, output)


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)