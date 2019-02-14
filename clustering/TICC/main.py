import os
import pandas as pd
import numpy as np
from preprocess.main.data_loader.data_loader import ImputedDataLoader
from preprocess.main.TICC.TICC_solver import TICC

default_process_hype = {'method': 'ma', 'offset': 60, 'overlap': 0,
                        'preprocess_cols': ['Cadence', 'HeartRate', 'Intensity', 'Steps',
                                            'BreathingDepth', 'BreathingRate']}

default_om_signal = {'raw_cols': ['BreathingDepth', 'BreathingRate', 'Cadence',
                                  'HeartRate', 'Intensity', 'Steps'],
                     'MinPeakDistance': 100, 'MinPeakHeight': 0.04}


def init_process_hyperparameter(method='trmf'):
    method_dict_array = []
    
    if method == 'trmf':
        ###########################################################
        # Initiate trmf
        ###########################################################
        lag_array, K_array = [[3]], [4, 5]
        eta_array, alpha_array = [10.0], [10.0, 100.0]
        lambda_f_array, lambda_x_array, lambda_w_array = [1.], [10.], [1., 10.]
        
        for lag in lag_array:
            for K in K_array:
                for eta in eta_array:
                    for alpha in alpha_array:
                        for lambda_f in lambda_f_array:
                            for lambda_w in lambda_w_array:
                                for lambda_x in lambda_x_array:
                                    method_dict = {}
                                    method_dict['K'] = K
                                    method_dict['lag'] = lag
                                    method_dict['eta'] = eta
                                    method_dict['alpha'] = alpha
                                    method_dict['lambda_f'] = lambda_f
                                    method_dict['lambda_x'] = lambda_x
                                    method_dict['lambda_w'] = lambda_w
                                    
                                    method_dict_array.append(method_dict)
    elif method == 'brits' or method == 'brits_multitask':
        rnn_hid_size_list, drop_out_list, seq_len_list = [10, 20, 30, 40], [0.5], [10]
        
        for rnn_hid_size in rnn_hid_size_list:
            for drop_out in drop_out_list:
                for seq_len in seq_len_list:
                    method_dict = {'rnn_hid_size': rnn_hid_size, 'drop_out': drop_out, 'seq_len': seq_len}
                    method_dict_array.append(method_dict)
    
    return method_dict_array


def read_participant_array(main_folder):
    ###########################################################
    # 1.1 Read all participant
    ###########################################################
    omsignal_folder = os.path.join(main_folder, '2_raw_csv_data/omsignal/')
    omsignal_file_list = os.listdir(omsignal_folder)
    omsignal_file_list.sort()
    
    for omsignal_file in omsignal_file_list:
        if 'DS' in omsignal_file:
            omsignal_file_list.remove(omsignal_file)
    
    participant_id_array = [omsignal_file.split('_omsignal')[0] for omsignal_file in omsignal_file_list]
    participant_id_array.sort()
    
    return participant_id_array


def main(main_folder, imputed_folder):
    ###########################################################
    # 1. Read all participant
    ###########################################################
    participant_id_array = read_participant_array(main_folder)
    
    ###########################################################
    # 2. All method
    ###########################################################
    method_array = ['brits', 'trmf', 'mean', 'naive']
    
    ###########################################################
    # 3. Missing rate
    ###########################################################
    mp_array = [0.05, 0.1, 0.25, 0.5]
    
    ###########################################################
    # 4. Iterate over settings
    ###########################################################
    for mp in mp_array:
        ###########################################################
        # 4.1 Init imputed data loader
        ###########################################################
        signal_type, imputation_folder_postfix = 'om_signal', '_imputation_all'

        ticc = TICC(window_size=10, number_of_clusters=10,
                    lambda_parameter=11e-2, beta=600, maxIters=100,
                    threshold=2e-5, write_out_file=False,
                    prefix_string="output_folder/", num_proc=1)
        
        data_loader = ImputedDataLoader(signal_type=signal_type,
                                        main_folder=imputed_folder,
                                        participant_id_array=participant_id_array,
                                        mp=mp, postfix=imputation_folder_postfix, method_hyper=None,
                                        original_main_folder_postfix='_set', method='mean',
                                        process_hyper=default_process_hype,
                                        signal_hyper=default_om_signal)
        
        ###########################################################
        # 4.2 Load ground truth data and mask
        ###########################################################
        data_loader.load_ground_truth_data()

        ticc.fit(data_loader.ground_truth_data_array)
        
        error_df_all = pd.DataFrame()
        for method in method_array:
            
            ###########################################################
            # 4.3 Load imputed values
            ###########################################################
            if method == 'trmf' or method == 'brits' or method == 'brits_multitask':
                method_dict_array = init_process_hyperparameter(method=method)
                for method_dict in method_dict_array:
                    data_loader.update_method(method=method, method_hyper=method_dict, mp=mp)
                    data_loader.load_imputed_data()
                    
                    ###########################################################
                    # 4.4 Fit using TICC
                    ###########################################################
                    ticc.fit(data_loader.imputed_data_array)
                    
            else:
                data_loader.update_method(method=method, mp=mp)
                data_loader.load_imputed_data()
                
                ###########################################################
                # 4.4 Fit using TICC
                ###########################################################
                ticc.fit(data_loader.imputed_data_array)
        
        save_folder = data_loader.save_result_folder
        error_df_all = error_df_all.dropna()
        error_df_all.to_csv(os.path.join(save_folder, 'mp_' + str(mp) + '.csv'))
        print('Finished')
        
        del data_loader


def ticc_ground_truth_main(main_folder, imputed_folder, ticc_folder):
    ###########################################################
    # 1. Read all participant
    ###########################################################
    participant_id_array = read_participant_array(main_folder)

    ###########################################################
    # 2. Learn TICC
    ###########################################################
    signal_type, imputation_folder_postfix = 'om_signal', '_imputation_all'
    
    ticc = TICC(signal_type=signal_type, main_folder=ticc_folder,
                process_hyper=default_process_hype,
                signal_hyper=default_om_signal,
                window_size=5, number_of_clusters=10,
                lambda_parameter=1e-2, beta=100, maxIters=300,
                threshold=2e-5, write_out_file=False,
                prefix_string="output_folder/", num_proc=1)

    ###########################################################
    # 3. Init imputed data loader
    ###########################################################
    data_loader = ImputedDataLoader(signal_type=signal_type,
                                    main_folder=imputed_folder,
                                    participant_id_array=participant_id_array,
                                    mp=0, postfix=imputation_folder_postfix, method_hyper=None,
                                    original_main_folder_postfix='_set', method='mean',
                                    process_hyper=default_process_hype,
                                    signal_hyper=default_om_signal)

    ###########################################################
    # 4. Load ground truth data and mask
    ###########################################################
    data_loader.load_ground_truth_data()

    ###########################################################
    # 5. Load global statistics of data
    ###########################################################
    data_loader.load_data_stats(method='mean')
    data_loader.load_data_stats(method='std')

    ###########################################################
    # 6. TICC
    ###########################################################
    print('Save Folder: %s' % (ticc_folder))
    ticc.fit(data_loader.ground_truth_data_array, global_stats=data_loader.global_stat_dict)
    # ticc.fit(data_loader.ground_truth_data_array)


def ticc_imputed_main(main_folder, imputed_folder, ticc_folder, batch_norm=True):
    ###########################################################
    # 1. Read all participant
    ###########################################################
    participant_id_array = read_participant_array(main_folder)

    ###########################################################
    # 2. Learn TICC
    ###########################################################
    signal_type, imputation_folder_postfix = 'om_signal', '_imputation_all'

    ticc = TICC(signal_type=signal_type, main_folder=ticc_folder,
                process_hyper=default_process_hype,
                signal_hyper=default_om_signal,
                window_size=5, number_of_clusters=10,
                lambda_parameter=1e-2, beta=100, maxIters=300,
                threshold=2e-5, write_out_file=False,
                prefix_string="output_folder/", num_proc=1)
    ticc.load_model_parameters()

    ###########################################################
    # 3.1 All method
    ###########################################################
    method_array = ['brits', 'trmf', 'mean', 'naive']
    # method_array = ['mean']

    ###########################################################
    # 3.2 Missing rate
    ###########################################################
    mp_array = [0.05, 0.1, 0.25, 0.5]

    ###########################################################
    # 4. Iterate over settings
    ###########################################################
    for mp in mp_array:
        ###########################################################
        # 4.1 Init imputed data loader
        ###########################################################
        signal_type, imputation_folder_postfix = 'om_signal', '_imputation_all'
    
        data_loader = ImputedDataLoader(signal_type=signal_type,
                                        main_folder=imputed_folder,
                                        participant_id_array=participant_id_array,
                                        mp=mp, postfix=imputation_folder_postfix, method_hyper=None,
                                        original_main_folder_postfix='_set', method='mean',
                                        process_hyper=default_process_hype,
                                        signal_hyper=default_om_signal)
    
        ###########################################################
        # 4.2 Load ground truth data and mask
        ###########################################################
        data_loader.load_ground_truth_data()

        ###########################################################
        # 4.3 Load global statistics of data
        ###########################################################
        data_loader.load_data_stats(method='mean')
        data_loader.load_data_stats(method='std')

        for method in method_array:
            
            ###########################################################
            # 4.4 Load imputed values
            ###########################################################
            if method == 'trmf' or method == 'brits' or method == 'brits_multitask':
                method_dict_array = init_process_hyperparameter(method=method)
                for method_dict in method_dict_array:
                    ###########################################################
                    # Update method
                    ###########################################################
                    data_loader.update_method(method=method, method_hyper=method_dict, mp=mp)
                    print('mp: %s, method %s, param: %s' % (str(mp), method, data_loader.method_str))

                    ticc.update_imputation_method(method=method, method_hyper=method_dict, mp=mp)
                    
                    data_loader.load_imputed_data()
                    if batch_norm == True:
                        ticc.predict_data(data_loader.imputed_data_array)
                    else:
                        ticc.predict_data(data_loader.imputed_data_array, global_stats=data_loader.global_stat_dict)
            else:
                ###########################################################
                # Update method
                ###########################################################
                data_loader.update_method(method=method, mp=mp)
                print('mp: %s, method %s' % (str(mp), method))
                
                ticc.update_imputation_method(method=method, mp=mp)
                
                data_loader.load_imputed_data()
                if batch_norm == True:
                    ticc.predict_data(data_loader.imputed_data_array)
                else:
                    ticc.predict_data(data_loader.imputed_data_array, global_stats=data_loader.global_stat_dict)

        del data_loader


def ticc_error(main_folder, imputed_folder, ticc_folder):
    ###########################################################
    # 1. Read all participant
    ###########################################################
    participant_id_array = read_participant_array(main_folder)
    
    ###########################################################
    # 2. Learn TICC
    ###########################################################
    signal_type, imputation_folder_postfix = 'om_signal', '_imputation_all'
    
    ticc = TICC(signal_type=signal_type, main_folder=ticc_folder,
                process_hyper=default_process_hype,
                signal_hyper=default_om_signal, window_size=5, number_of_clusters=10,
                lambda_parameter=1e-2, beta=10, maxIters=300,
                threshold=2e-5, write_out_file=False,
                prefix_string="output_folder/", num_proc=1)
    ticc.load_model_parameters()
    ticc.load_ground_truth_ticc_data(participant_id_array)
    
    ###########################################################
    # 3.1 All method
    ###########################################################
    method_array = ['brits', 'trmf', 'mean', 'naive']
    # method_array = ['mean']
    # method_array = ['brits']
    
    ###########################################################
    # 3.2 Missing rate
    ###########################################################
    mp_array = [0.05, 0.1, 0.25, 0.5]
    
    ###########################################################
    # 4. Iterate over settings
    ###########################################################
    error_df_final = pd.DataFrame()
    for mp in mp_array:
        ###########################################################
        # 4.1 Iterate over method
        ###########################################################
        error_df = pd.DataFrame()
        
        for method in method_array:
            
            ###########################################################
            # 4.4 Load imputed values
            ###########################################################
            if method == 'trmf' or method == 'brits' or method == 'brits_multitask':
                method_dict_array = init_process_hyperparameter(method=method)
                for method_dict in method_dict_array:
                    ###########################################################
                    # Update method
                    ###########################################################
                    ticc.update_imputation_method(method=method, method_hyper=method_dict, mp=mp)
                    print('mp: %s, method %s, param: %s' % (str(mp), method, ticc.method_str))
                    
                    ticc.load_imputed_ticc_data(participant_id_array)
                    error_tmp_df = ticc.compute_error()
                    error_df = error_df.append(error_tmp_df)
                    
            else:
                ###########################################################
                # Update method
                ###########################################################
                ticc.update_imputation_method(method=method, mp=mp)
                print('mp: %s, method %s' % (str(mp), method))
                
                ticc.load_imputed_ticc_data(participant_id_array)
                error_tmp_df = ticc.compute_error()
                error_df = error_df.append(error_tmp_df)

        error_df = pd.DataFrame(np.array(error_df), index=[error_df.index], columns=[mp])
        error_df.to_csv(os.path.join(ticc.save_TICC_path, ticc.ticc_model_str, 'mp_' + str(mp) + '.csv'))
        error_df_final = error_df.copy() if len(error_df_final) == 0 else pd.concat([error_df_final.copy(), error_df.copy()], axis=1)
        error_df_final.to_csv(os.path.join(ticc.save_TICC_path, ticc.ticc_model_str, 'mp_all.csv'))
   
     
if __name__ == '__main__':
    # Main Data folder
    main_folder = '../../../../data/keck_wave_all/'
    
    # Imputed folder
    imputed_folder = os.path.abspath(os.path.join(os.pardir, '../../imputation_test'))
    
    # TICC folder
    batch_norm = False
    if batch_norm == True:
        ticc_folder = os.path.abspath(os.path.join(os.pardir, '../../TICC_result_batch_norm'))
    else:
        ticc_folder = os.path.abspath(os.path.join(os.pardir, '../../TICC_result'))

    ticc_imputed_main(main_folder, imputed_folder, ticc_folder, batch_norm=batch_norm)
    # ticc_ground_truth_main(main_folder, imputed_folder, ticc_folder)
    # ticc_error(main_folder, imputed_folder, ticc_folder)
