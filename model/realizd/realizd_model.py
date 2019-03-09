"""
IGTB variance
"""
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering


###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config')))
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'segmentation')))
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

demographic_col_dict = {'age': 'age', 'gender': 'gender', 'bornUS': 'bornUS', 'supervise': 'supervise',
                        'lang': 'language', 'duration': 'employer_duration'}

ocb_col_dict = {'ocb1': 'Picked up a meal for others at work',
                'ocb2': 'Took time to advise, coach, or mentor a co-worker',
                'ocb3': 'Helped a co-worker learn new skills or shared job knowledge',
                'ocb4': 'Helped new employees get oriented to the job',
                'ocb5': 'Lent a compassionate ear when someone had a work problem',
                'ocb6': 'Lent a compassionate ear when someone had a personal problem',
                'ocb7': 'Changed vacation schedule, workdays, or shifts to accommodate co-worker’s needs',
                'ocb8': 'Offered suggestions to improve how work is done',
                'ocb9': 'Offered suggestions for improving the work environment',
                'ocb10': 'Finished something for co-worker who had to leave early',
                'ocb11': 'Helped a less capable co-worker lift a heavy box or other object',
                'ocb12': 'Helped a co-worker who had too much to do',
                'ocb13': 'Volunteered for extra work assignments',
                'ocb14': 'Took phone messages for absent or busy co-worker',
                'ocb15': 'Said good things about your employer in front of others',
                'ocb16': 'Gave up meal and other breaks to complete work',
                'ocb17': 'Volunteered to help a co-worker deal with a difficult customer, vendor, or co-worker',
                'ocb18': 'Went out of the way to give co-worker encouragement or express appreciation',
                'ocb19': 'Decorated, straightened up, or otherwise beautified common work space',
                'ocb20': 'Defended a co-worker who was being ‘put-down’ or spoken ill of by other co-workers or supervisor'}


def compute_igtb_std(filter_igtb):
    # Read ground truth data
    igtb_cols = [col for col in filter_igtb.columns if 'igtb' in col]
    
    igtb_std_df = pd.DataFrame(index=igtb_cols, columns=['igtb_std'])
    for col in igtb_cols:
        igtb_array = np.array(filter_igtb[col])
        if col == 'ipaq_igtb' or col == 'gats_quantity_igtb':
            igtb_array = (igtb_array - np.min(igtb_array)) / (np.max(igtb_array) - np.min(igtb_array))
        else:
            igtb_array = (igtb_array - min_dict[col]) / (max_dict[col] - min_dict[col])
        std_igtb_array = np.nanstd(igtb_array)
        igtb_std_df.loc[col, 'igtb_std'] = std_igtb_array
        
    return_std_df = igtb_std_df.sort_values(by=['igtb_std'])
    seq_rank = np.argsort(np.array(igtb_std_df).reshape([1, -1]))[0][::-1]
    final_dict = []
    
    # Sort
    for index, i in enumerate(seq_rank):
    
        igtb_col = igtb_std_df.index[i]
        user_igtb = filter_igtb[igtb_col]
        
        igtb_dict = {}
        igtb_dict['igtb_col'] = igtb_col

        # Highest
        user_rank_array = np.argsort(np.array(user_igtb).reshape([1, -1]))[0][::-1]
        user_id_list = filter_igtb.index[user_rank_array]
        final_df = pd.DataFrame()
        
        for user_id in user_id_list:
            
            # if 'ICU' not in filter_igtb.loc[user_id, 'PrimaryUnit'] and '5 North' not in filter_igtb.loc[user_id, 'PrimaryUnit']:
            participant_id = filter_igtb.loc[user_id, 'ParticipantID']
            row_df = pd.DataFrame(index=[participant_id])
            row_df['igtb_val'] = filter_igtb.loc[user_id, igtb_col]
            row_df['igtb_col'] = igtb_col
            row_df['neu_igtb'] = filter_igtb.loc[user_id, 'neu_igtb']
            row_df['stai_igtb'] = filter_igtb.loc[user_id, 'stai_igtb']
            row_df['pos_af_igtb'] = filter_igtb.loc[user_id, 'pos_af_igtb']
            row_df['neg_af_igtb'] = filter_igtb.loc[user_id, 'neg_af_igtb']
            row_df['fatigue'] = float(filter_igtb.loc[user_id, 'fatigue'])
            row_df['Flexbility'] = float(filter_igtb.loc[user_id, 'Flexbility'])
            row_df['Inflexbility'] = float(filter_igtb.loc[user_id, 'Inflexbility'])
            row_df['cluster'] = filter_igtb.loc[user_id, 'cluster']
            row_df['duration'] = float(filter_igtb.loc[user_id, 'employer_duration'])
            row_df['age'] = float(filter_igtb.loc[user_id, 'age'])
            row_df['icu'] = 'ICU' if 'ICU' in filter_igtb.loc[user_id, 'PrimaryUnit'] or '5 North' in filter_igtb.loc[user_id, 'PrimaryUnit'] else 'Non-ICU'
            row_df['unit'] = filter_igtb.loc[user_id, 'PrimaryUnit']
            row_df['position'] = filter_igtb.loc[user_id, 'currentposition']
            row_df['language'] = filter_igtb.loc[user_id, 'language']
            row_df['shift'] = filter_igtb.loc[user_id, 'Shift']
            row_df['supervise'] = float(filter_igtb.loc[user_id, 'supervise'])
            row_df['sex'] = filter_igtb.loc[user_id, 'Sex']
            row_df['LifeSatisfaction'] = float(filter_igtb.loc[user_id, 'LifeSatisfaction'])
            row_df['igtb_rank'] = index + 1
            row_df['igtb_status'] = 'high'
            final_df = final_df.append(row_df)

        igtb_dict['data'] = final_df
        final_dict.append(igtb_dict)

    return final_dict, return_std_df


def compute_igtb_sub_std(filter_igtb):
    # Read ground truth data
    ocb_std_df = pd.DataFrame(index=list(ocb_col_dict.keys()), columns=['igtb_std'])
    for col in list(ocb_col_dict.keys()):
        igtb_array = np.array(filter_igtb[col]).astype(int)
        igtb_array = (igtb_array - 1) / 5
        std_igtb_array = np.std(igtb_array)
        ocb_std_df.loc[col, 'igtb_std'] = std_igtb_array
        ocb_std_df.loc[col, 'std'] = std_igtb_array
        ocb_std_df.loc[col, 'question'] = ocb_col_dict[col]

    seq_rank = np.argsort(np.array(ocb_std_df['std']).reshape([1, -1]))[0][::-1]
    ocb_std_df = ocb_std_df.sort_values(by=['std'])

    final_dict = []
    
    # Sort
    for index, i in enumerate(seq_rank):
        
        igtb_col = ocb_std_df.index[i]
        user_igtb = filter_igtb[igtb_col]
        
        igtb_dict = {}
        igtb_dict['igtb_col'] = ocb_col_dict[igtb_col]
        
        # Highest
        user_rank_array = np.argsort(np.array(user_igtb).reshape([1, -1]))[0][::-1]
        user_id_list = filter_igtb.index[user_rank_array]
        final_df = pd.DataFrame()
        
        for user_id in user_id_list:
            # if 'ICU' not in filter_igtb.loc[user_id, 'PrimaryUnit'] and '5 North' not in filter_igtb.loc[user_id, 'PrimaryUnit']:
            participant_id = filter_igtb.loc[user_id, 'ParticipantID']
            row_df = pd.DataFrame(index=[participant_id])
            row_df['igtb_val'] = filter_igtb.loc[user_id, igtb_col]
            row_df['ocb'] = filter_igtb.loc[user_id, 'ocb_igtb']
            row_df['igtb_col'] = igtb_col
            row_df['duration'] = float(filter_igtb.loc[user_id, 'employer_duration'])
            row_df['icu'] = 'ICU' if 'ICU' in filter_igtb.loc[user_id, 'PrimaryUnit'] or '5 North' in filter_igtb.loc[user_id, 'PrimaryUnit'] else 'Non-ICU'
            row_df['unit'] = filter_igtb.loc[user_id, 'PrimaryUnit']
            row_df['position'] = filter_igtb.loc[user_id, 'currentposition']
            row_df['shift'] = filter_igtb.loc[user_id, 'Shift']
            row_df['supervise'] = float(filter_igtb.loc[user_id, 'supervise'])
            row_df['sex'] = filter_igtb.loc[user_id, 'Sex']
            row_df['LifeSatisfaction'] = filter_igtb.loc[user_id, 'LifeSatisfaction']
            row_df['igtb_rank'] = index + 1
            row_df['igtb_status'] = 'high'
            
            final_df = final_df.append(row_df)
        
        igtb_dict['data'] = final_df
        final_dict.append(igtb_dict)
    
    return final_dict


def main(tiles_data_path, config_path, experiment):
    ###########################################################
    # 0. Read ground truth data
    ###########################################################
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    igtb_cols = [col for col in igtb_df.columns if 'igtb' in col]
    igtb_df = igtb_df.dropna(subset=igtb_cols)
    
    igtb_sub_df = load_data_basic.read_IGTB_sub(tiles_data_path)

    ###########################################################
    # 0. Additional igtb information
    ###########################################################
    for col in list(demographic_col_dict.keys()):
        igtb_df.loc[list(igtb_df.index), demographic_col_dict[col]] = igtb_sub_df.loc[list(igtb_df.index), col]
        
    for col in list(ocb_col_dict.keys()):
        igtb_df.loc[list(igtb_df.index), col] = igtb_sub_df.loc[list(igtb_df.index), col]

    igtb_df = igtb_df.dropna(subset=list(ocb_col_dict.keys()))

    
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))

    data_config = config.Config()
    data_config.readConfigFile(os.path.abspath(config_path), experiment)
    load_data_path.load_all_available_path(data_config, process_data_path)

    ###########################################################
    # 2. Read filter participant
    ###########################################################
    filter_participant_id_list = [participant_id for participant_id
                                  in os.listdir(data_config.fitbit_sensor_dict['filter_path'])
                                  if '.csv.gz' in participant_id]

    filter_participant_id_list = [participant_id.split('.csv.gz')[0] for participant_id in filter_participant_id_list]
    
    filter_igtb = igtb_df[igtb_df['ParticipantID'].isin(filter_participant_id_list)]
    filter_igtb = filter_igtb.loc[filter_igtb['currentposition'] == 1]
    filter_igtb = filter_igtb.loc[filter_igtb['Shift'] == 'Day shift']
    
    '''
    plt.figure(figsize=(10, 7))
    plt.title("Customer Dendograms")
    dend = shc.dendrogram(shc.linkage(np.array(filter_igtb[['neu_igtb', 'pos_af_igtb', 'neg_af_igtb', 'stai_igtb']]), method='ward'))
    '''
    
    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster_data = np.array(filter_igtb[['neu_igtb', 'pos_af_igtb', 'neg_af_igtb', 'stai_igtb']])
    cluster_data = (cluster_data - np.mean(cluster_data, axis=0)) / np.std(cluster_data, axis=0)
    cluster_array = cluster.fit_predict(cluster_data)
    filter_igtb.loc[:, 'cluster'] = cluster_array

    final_dict, igtb_std_df = compute_igtb_std(filter_igtb)
    # final_dict = compute_igtb_sub_std(filter_igtb)
    
    ###########################################################
    # Plot code
    ###########################################################
    
    ###########################################################
    # 1. Create Config, load data paths
    ###########################################################
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
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
    
    for index, row_series in filter_igtb.iterrows():
        
        participant_id = row_series.ParticipantID
        cluster = row_series.cluster
        
        max_pos_igtb = np.nanmean(filter_igtb.loc[filter_igtb['cluster'] == cluster].pos_af_igtb)
        
        print('read_preprocess_data: participant: %s' % (participant_id))
        
        ###########################################################
        # 3. Read summary data, mgt, omsignal
        ###########################################################
        fitbit_data_dict = load_sensor_data.read_fitbit(fitbit_summary_path, participant_id)
        if fitbit_data_dict is None:
            continue
        
        fitbit_summary_df = fitbit_data_dict['summary']
        
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        primary_unit = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].PrimaryUnit[0]
        shift = igtb_df.loc[igtb_df['ParticipantID'] == participant_id].Shift[0]
        participant_app_survey = survey_df.loc[survey_df['participant_id'] == participant_id]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]
        
        omsignal_data_df = load_sensor_data.read_preprocessed_omsignal(data_config.omsignal_sensor_dict['preprocess_path'], participant_id)
        owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
        realizd_df = load_sensor_data.read_preprocessed_realizd(os.path.join(data_config.realizd_sensor_dict['preprocess_path'], participant_id), participant_id)
        audio_df = load_sensor_data.read_preprocessed_audio(data_config.audio_sensor_dict['preprocess_path'], participant_id)
        fitbit_df, fitbit_mean, fitbit_std = load_sensor_data.read_preprocessed_fitbit_with_pad(data_config, participant_id)
        
        segmentation_df = None
        
        if fitbit_df is None or realizd_df is None:
            continue
        
        if os.path.exists(os.path.join('plot')) is False:
            os.mkdir(os.path.join('plot'))
        if os.path.exists(os.path.join('plot', str(cluster) + '_pos_' + str(max_pos_igtb))) is False:
            os.mkdir(os.path.join('plot', str(cluster) + '_pos_' + str(max_pos_igtb)))
        if os.path.exists(os.path.join('plot', str(cluster) + '_pos_' + str(max_pos_igtb), participant_id)) is False:
            os.mkdir(os.path.join('plot', str(cluster) + '_pos_' + str(max_pos_igtb), participant_id))

        ###########################################################
        # 4. Plot
        ###########################################################
        cluster_plot = plot.Plot(data_config=data_config, primary_unit=primary_unit, shift=shift)
        
        cluster_plot.plot_cluster(index, fitbit_df=fitbit_df, fitbit_summary_df=fitbit_summary_df,
                                  audio_df=audio_df, mgt_df=participant_mgt,
                                  segmentation_df=segmentation_df, omsignal_data_df=omsignal_data_df,
                                  realizd_df=realizd_df, owl_in_one_df=owl_in_one_df,
                                  save_folder=os.path.join('plot', str(cluster) + '_pos_' + str(max_pos_igtb), participant_id))


if __name__ == '__main__':
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'baseline' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)