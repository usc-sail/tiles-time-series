"""
IGTB variance
"""
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'segmentation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'plot')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
import plot

min_dict = {'itp_igtb': 1, 'irb_igtb': 7, 'iod_id_igtb': 7, 'iod_od_igtb': 12, 'ocb_igtb': 20,
                'shipley_abs_igtb': 0, 'shipley_voc_igtb': 0,
                'neu_igtb': 1, 'con_igtb': 1, 'ext_igtb': 1, 'agr_igtb': 1, 'ope_igtb': 1,
                'pos_af_igtb': 10, 'neg_af_igtb': 10, 'stai_igtb': 20,
                'audit_igtb': 0, 'gats_status_igtb': 1, 'psqi_igtb': 0}

max_dict = {'itp_igtb': 5, 'irb_igtb': 49, 'iod_id_igtb': 49, 'iod_od_igtb': 84, 'ocb_igtb': 100,
            'shipley_abs_igtb': 25, 'shipley_voc_igtb': 40,
            'neu_igtb': 5, 'con_igtb': 5, 'ext_igtb': 5, 'agr_igtb': 5, 'ope_igtb': 5,
            'pos_af_igtb': 50, 'neg_af_igtb': 50, 'stai_igtb': 80,
            'audit_igtb': 40, 'gats_status_igtb': 3, 'psqi_igtb': 21}


def main(tiles_data_path, config_path, experiment):
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    igtb_cols = [col for col in igtb_df.columns if 'igtb' in col]
    igtb_df = igtb_df.dropna(subset=igtb_cols)
    
    igtb_std_df = pd.DataFrame(index=igtb_cols, columns=['igtb_std'])
    for col in igtb_cols:
        igtb_array = np.array(igtb_df[col])
        if col == 'ipaq_igtb' or col == 'gats_quantity_igtb':
            igtb_array = (igtb_array - np.min(igtb_array)) / (np.max(igtb_array) - np.min(igtb_array))
        else:
            igtb_array = (igtb_array - min_dict[col]) / (max_dict[col] - min_dict[col])
        std_igtb_array = np.std(igtb_array)
        igtb_std_df.loc[col, 'igtb_std'] = std_igtb_array

    seq_rank = np.argsort(np.array(igtb_std_df).reshape([1, -1]))[0][::-1]
    final_df = pd.DataFrame()

    for index, i in enumerate(seq_rank):
        igtb_col = igtb_std_df.index[i]
        user_igtb = igtb_df[igtb_col]
        
        # Highest
        user_rank_array = np.argsort(np.array(user_igtb).reshape([1, -1]))[0][::-1][:5]
        user_id_list = igtb_df.index[user_rank_array]
        
        for user_id in user_id_list:
            participant_id = igtb_df.loc[user_id, 'ParticipantID']
            row_df = pd.DataFrame(index=[participant_id])
            row_df['igtb_val'] = igtb_df.loc[user_id, igtb_col]
            row_df['igtb_rank'] = index + 1
            row_df['igtb_status'] = 'high'
            row_df['igtb_col'] = igtb_col
            final_df = final_df.append(row_df)

        # Lowest
        user_rank_array = np.argsort(np.array(user_igtb).reshape([1, -1]))[0][:5]
        user_id_list = igtb_df.index[user_rank_array]

        for user_id in user_id_list:
            participant_id = igtb_df.loc[user_id, 'ParticipantID']
            row_df = pd.DataFrame(index=[participant_id])
            row_df['igtb_val'] = igtb_df.loc[user_id, igtb_col]
            row_df['igtb_rank'] = index + 1
            row_df['igtb_status'] = 'low'
            row_df['igtb_col'] = igtb_col
            final_df = final_df.append(row_df)

    final_df.to_csv('igtb_rank.csv')
    print('Success!')

    ###########################################################
    # Plot code
    ###########################################################

    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))

    data_config = config.Config()
    data_config.readConfigFile(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')), experiment)

    load_data_path.load_all_available_path(data_config, process_data_path)

    # Load Fitbit summary folder
    fitbit_summary_path = load_data_path.load_fitbit_summary_path(tiles_data_path, data_name='3_preprocessed_data')

    ###########################################################
    # Read ground truth data
    ###########################################################
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    survey_df = load_data_basic.read_app_survey(tiles_data_path)
    survey_df = survey_df.loc[survey_df['survey_type'] == 'psych_flex']
    mgt_df = load_data_basic.read_MGT(tiles_data_path)

    final_df = final_df[60:100]

    for index, row_series in final_df.iterrows():
        # row_series.igtb_status
        
        if os.path.exists(os.path.join(row_series.igtb_col)) is False:
            os.mkdir(os.path.join(row_series.igtb_col))
        if os.path.exists(os.path.join(row_series.igtb_col, row_series.igtb_status)) is False:
            os.mkdir(os.path.join(row_series.igtb_col, row_series.igtb_status))
        if os.path.exists(os.path.join(row_series.igtb_col, row_series.igtb_status, index)) is False:
            os.mkdir(os.path.join(row_series.igtb_col, row_series.igtb_status, index))
        
        print('read_preprocess_data: participant: %s' % (index))
        
        ###########################################################
        # 3. Read summary data, mgt, omsignal
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, index)
        if fitbit_data_dict is None:
            continue
        
        fitbit_summary_df = fitbit_data_dict['summary']
    
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == index].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == index].PrimaryUnit[0]
        shift = igtb_df.loc[igtb_df['ParticipantID'] == index].Shift[0]
        participant_app_survey = survey_df.loc[survey_df['participant_id'] == index]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]

        omsignal_data_df = load_sensor_data.read_preprocessed_omsignal(data_config.omsignal_sensor_dict['preprocess_path'], index)
        owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], index)
        realizd_df = load_sensor_data.read_preprocessed_realizd(os.path.join(data_config.realizd_sensor_dict['preprocess_path'], index), index)
        audio_df = load_sensor_data.read_preprocessed_audio(data_config.audio_sensor_dict['preprocess_path'], index)
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, index)

        segmentation_df = None
        ###########################################################
        # 4. Plot
        ###########################################################
        cluster_plot = plot.Plot(data_config=data_config, primary_unit=primary_unit, shift=shift)
    
        cluster_plot.plot_cluster(index, fitbit_df=fitbit_df, fitbit_summary_df=fitbit_summary_df,
                                  audio_df=audio_df, mgt_df=participant_mgt,
                                  segmentation_df=segmentation_df, omsignal_data_df=omsignal_data_df,
                                  realizd_df=realizd_df, owl_in_one_df=owl_in_one_df,
                                  save_folder=os.path.join(row_series.igtb_col, row_series.igtb_status, index))
    

if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'baseline' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)