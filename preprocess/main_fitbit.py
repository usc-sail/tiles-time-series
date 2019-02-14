import os, sys

# date_time format
date_time_format = '%Y-%m-%dT%H:%M:%S.%f'
date_only_date_time_format = '%Y-%m-%d'

sys.path.append(os.path.join(os.path.curdir, '../', 'utils'))
from load_data_basic import *
from preprocess import Preprocess

raw_cols = ['HeartRatePPG', 'StepCount']


def main(main_folder, enable_impute=False):
    
    ###########################################################
    # 1. Read all fitbit file
    ###########################################################
    fitbit_folder = os.path.join(main_folder, '3_preprocessed_data/fitbit/')
    fitbit_file_list = os.listdir(fitbit_folder)
    fitbit_file_dict_list = {}
    
    for fitbit_file in fitbit_file_list:
        
        if '.DS' in fitbit_file:
            continue
        
        participant_id = fitbit_file.split('_')[0]
        if participant_id not in list(fitbit_file_dict_list.keys()):
            fitbit_file_dict_list[participant_id] = {}
        
        if 'heartRate.csv.gz' in fitbit_file:
            fitbit_file_dict_list[participant_id]['ppg'] = fitbit_file
        elif 'stepCount.csv.gz' in fitbit_file:
            fitbit_file_dict_list[participant_id]['step'] = fitbit_file

    participant_id_list = list(fitbit_file_dict_list.keys())
    participant_id_list.sort()

    ###########################################################
    # 2. Iterate all participant
    ###########################################################
    for i, participant_id in enumerate(participant_id_list[:]):
    
        print('Complete process for %s: %.2f' % (participant_id, 100 * i / len(participant_id_list)))
        
        ###########################################################
        # Read ppg and step count file path
        ###########################################################
        ppg_file = fitbit_file_dict_list[participant_id]['ppg']
        step_file = fitbit_file_dict_list[participant_id]['step']

        ppg_file_abs_path = os.path.join(fitbit_folder, ppg_file)
        step_file_abs_path = os.path.join(fitbit_folder, step_file)

        ###########################################################
        # Read ppg and step count data
        ###########################################################
        ppg_df = pd.read_csv(ppg_file_abs_path, index_col=0)
        ppg_df = ppg_df.sort_index()
        
        step_df = pd.read_csv(step_file_abs_path, index_col=0)
        step_df = step_df.sort_index()

        # step_df = step_df.drop_duplicates(keep='first')
        # ppg_df = ppg_df.drop_duplicates(keep='first')

        ###########################################################
        # 2.0 Iterate all omsignal files
        ###########################################################
        process_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                        'preprocess_cols': ['HeartRatePPG', 'StepCount']}
        
        if len(ppg_df) > 0:
            
            ###########################################################
            # 2.1 Init om_signal preprocess
            ###########################################################
            fitbit_preprocess = Preprocess(data_df=None, signal_type = 'fitbit',
                                           participant_id=participant_id, imputation_method='iterative',
                                           save_main_folder=os.path.abspath(os.path.join(os.pardir, '../preprocess_data')),
                                           process_hyper=process_hype, enable_impute=enable_impute)
        
            ###########################################################
            # 2.2 Preprocess fitbit raw data array (No slice)
            ###########################################################
            fitbit_preprocess.process_fitbit(ppg_df, step_df, method='block')
            
            ###########################################################
            # 2.3 Delete current omsignal_preprocess object
            ###########################################################
            del fitbit_preprocess
      
        
if __name__ == "__main__":
    
    # Main Data folder
    main_folder = '../../../data/keck_wave_all/'

    main(main_folder, enable_impute=True)

