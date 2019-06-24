"""
Cluster the audio data
"""
from __future__ import print_function

from sklearn import mixture
import os
import sys
import pandas as pd
import numpy as np
import dpmm
from collections import Counter
import dpgmm_gibbs
from vdpgmm import VDPGMM

from pybgmm.prior import NIW
from pybgmm.igmm import PCRPMM
from datetime import timedelta
from sklearn.datasets import load_iris
from dpkmeans import dpmeans
from numpy import linalg as LA

from filter import Filter

from scipy.stats import invwishart, invgamma, wishart

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

SEED = 5132290  # from random.org
np.random.seed(SEED)


def main(tiles_data_path, config_path, experiment):
    # Create Config
    process_data_path = os.path.abspath(os.path.join(os.pardir, os.pardir, 'data'))
    
    data_config = config.Config()
    data_config.readConfigFile(config_path, experiment)
    
    # Load all data path according to config file
    load_data_path.load_all_available_path(data_config, process_data_path,
                                           preprocess_data_identifier='preprocess',
                                           segmentation_data_identifier='segmentation',
                                           filter_data_identifier='filter_data',
                                           clustering_data_identifier='clustering')
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()

    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    igtb_df = igtb_df.drop_duplicates(keep='first')
    mgt_df = load_data_basic.read_MGT(tiles_data_path)
    
    data_df = pd.DataFrame()
    
    n_om = 0
    n_fitbit = 0
    n_owl_in_one = 0
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))

        # Create filter class
        filter_class = Filter(data_config=data_config, participant_id=participant_id)

        # If we have save the filter data before
        if os.path.exists(os.path.join(data_config.fitbit_sensor_dict['filter_path'], participant_id, 'filter_dict.csv.gz')) is True:
            print('%s has been filtered before' % participant_id)
            continue

        # Read id
        uid = list(igtb_df.loc[igtb_df['ParticipantID'] == participant_id].index)[0]
        participant_mgt = mgt_df.loc[mgt_df['uid'] == uid]

        # Read other sensor data, the aim is to detect whether people workes during a day
        owl_in_one_df = load_sensor_data.read_preprocessed_owl_in_one(data_config.owl_in_one_sensor_dict['preprocess_path'], participant_id)
        omsignal_data_df = load_sensor_data.read_preprocessed_omsignal(data_config.omsignal_sensor_dict['preprocess_path'], participant_id)
        fitbit_df = load_sensor_data.read_preprocessed_fitbit(data_config.fitbit_sensor_dict['preprocess_path'], participant_id)
        
        ratio, day, number_hour_per_day = filter_class.filter_data(data_df=fitbit_df, mgt_df=participant_mgt, owl_in_one_df=owl_in_one_df, omsignal_df=omsignal_data_df)
        
        participant_df = pd.DataFrame(index=[participant_id])

        if omsignal_data_df is not None:
            participant_df['om_signal'] = number_hour_per_day
        else:
            participant_df['om_signal'] = 0
        
        '''
        if omsignal_data_df is not None:
            participant_df['om_signal'] = number_hour_per_day
            n_om += 1
        else:
            participant_df['om_signal'] = 0

        if fitbit_df is not None:
            participant_df['fitbit'] = number_hour_per_day
            n_fitbit += 1
        else:
            participant_df['fitbit'] = 0

        if owl_in_one_df is not None:
            participant_df['owl_in_one'] = number_hour_per_day
            n_owl_in_one += 1
        else:
            participant_df['owl_in_one'] = 0
        '''
        data_df = data_df.append(participant_df)

    data_df.to_csv(data_config.filter_method + '.csv')
    '''
    print(data_df.columns)
    print(np.nansum(data_df, axis=0))
    
    print('om: %.3f' % (n_om / len(igtb_df)))
    print('fitbit: %.3f' % (n_fitbit / len(igtb_df)))
    print('owl_in_one: %.3f' % (n_owl_in_one / len(igtb_df)))
    
    print(len(igtb_df))
    print(n_om)
    print(n_fitbit)
    print(n_owl_in_one)
    '''

if __name__ == '__main__':
    
    '''
    data_df = pd.read_csv('fitbit.csv', index_col=0)
    
    data_df = data_df.loc[data_df['ratio'] > 0.2]
    print(np.nanmean(data_df))
    print(data_df)
    '''
    
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'config_file')) if args.config is None else args.config
    experiment = 'dpmm' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)
