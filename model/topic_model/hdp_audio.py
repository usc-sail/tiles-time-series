"""
Cluster the audio data
"""
from __future__ import print_function

from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel
from gensim.corpora import Dictionary

from sklearn import mixture
import os
import sys
import pandas as pd
import numpy as np
import random
from collections import Counter
from datetime import timedelta
import pymc3 as pm
from theano import tensor as tt

###########################################################
# Change to your own library path
###########################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'config')))

import config
import load_sensor_data, load_data_path, load_data_basic, parser


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
    
    # Read ground truth data
    igtb_df = load_data_basic.read_AllBasic(tiles_data_path)
    
    # Get participant id list, k=None, save all participant data
    top_participant_id_df = load_data_basic.return_top_k_participant(os.path.join(process_data_path, 'participant_id.csv.gz'), tiles_data_path, data_config=data_config)
    top_participant_id_list = list(top_participant_id_df.index)
    top_participant_id_list.sort()
    
    for idx, participant_id in enumerate(top_participant_id_list):
        
        print('read_filter_data: participant: %s, process: %.2f' % (participant_id, idx * 100 / len(top_participant_id_list)))
        
        # Read other sensor data, the aim is to detect whether people workes during a day
        if os.path.exists(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id)) is False:
            continue
        
        file_list = [file for file in os.listdir(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id))]

        for file in file_list:
            data_df = pd.read_csv(os.path.join(data_config.audio_sensor_dict['clustering_path'], participant_id, file), index_col=0)
            data_df = data_df.sort_index()
            
            time_diff = pd.to_datetime(list(data_df.index)[1:]) - pd.to_datetime(list(data_df.index)[:-1])
            time_diff = list(time_diff.total_seconds())

            change_point_start_list = [0]
            change_point_end_list = list(np.where(np.array(time_diff) > 3600 * 6)[0])

            [change_point_start_list.append(change_point_end + 1) for change_point_end in change_point_end_list]
            change_point_end_list.append(len(data_df.index) - 1)

            time_start_end_list = []
            for i, change_point_end in enumerate(change_point_end_list):
                time_start_end_list.append([list(data_df.index)[change_point_start_list[i]],
                                            list(data_df.index)[change_point_end]])

            bow_list, time_list = [], []
            word_list = []
            for time_start_end in time_start_end_list:
                start_time = (pd.to_datetime(time_start_end[0]).replace(minute=0, second=0, microsecond=0)).strftime(load_data_basic.date_time_format)[:-3]
                end_time = ((pd.to_datetime(time_start_end[1]) + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)).strftime(load_data_basic.date_time_format)[:-3]
                
                time_offest = int((pd.to_datetime(end_time) - pd.to_datetime(start_time)).total_seconds() / (60 * 15))
                
                for offset in range(time_offest):
                    tmp_start = (pd.to_datetime(start_time) + timedelta(minutes=15 * offset)).strftime(load_data_basic.date_time_format)[:-3]
                    tmp_end = (pd.to_datetime(start_time) + timedelta(minutes=15 * offset + 30)).strftime(load_data_basic.date_time_format)[:-3]
                
                    tmp_data_df = data_df[tmp_start:tmp_end]
                    
                    if len(tmp_data_df) > 3:
                        word_list.append([str(word) for word in list(tmp_data_df.cluster)])
                        # for i in np.unique(tmp_data_df.cluster):
                        # cluster_count_list = Counter(list(tmp_data_df.cluster))
                        # tmp_bow_list = [(cluster_key, cluster_count_list[cluster_key]) for cluster_key in cluster_count_list]
                        # bow_list.append(tmp_bow_list)
                        time_list.append([tmp_start, tmp_end])
            
            word_dictionary = Dictionary(word_list)
            word_corpus = [word_dictionary.doc2bow(text) for text in word_list]
            
            hdp = HdpModel(word_corpus, word_dictionary)
            topic_info = hdp.print_topics(num_topics=20, num_words=5)

            unseen_document = [(1, 3.), (2, 4)]
            doc_hdp = hdp[unseen_document]

            print()
        
        '''
        hdp = HdpModel(common_corpus, common_dictionary)

        unseen_document = [(1, 3.), (2, 4)]
        doc_hdp = hdp[unseen_document]
        '''
        
if __name__ == '__main__':
    def stick_breaking(beta):
        portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
        
        return beta * portion_remaining
    
    
    d0 = np.concatenate([np.random.binomial(15, .1, size=(100, 3)), np.random.binomial(15, .5, size=(100, 3))])
    d1 = np.ones(200) * 15
    
    with pm.Model() as model:
        alpha = pm.Gamma('alpha', 1., 1.)
        beta = pm.Beta('beta', 1, alpha, shape=3)
        w = pm.Deterministic('w', stick_breaking(beta))
        
        dpmm_comp_mu = pm.Normal('dpmm_comp_mu', 0., 100., shape=3)
        
        visit_rate_like = pm.Mixture(
                'visit_rate_like',
                w,
                pm.Binomial.dist(
                        p=pm.math.invlogit(dpmm_comp_mu),
                        n=d1.astype('int32')[:, None]
                ),
                observed=d0.astype('int32')[:, None]
        )
    
    with model:
        trace = pm.sample(step=pm.Metropolis())
    
    '''
    from gensim.models import CoherenceModel, HdpModel

    hdpmodel = HdpModel(corpus=corpus, id2word=id2word)
    hdptopics = hdpmodel.show_topics(formatted=False)
    '''
    
    # Read args
    args = parser.parse_args()
    
    # If arg not specified, use default value
    tiles_data_path = '../../../../../data/keck_wave_all/' if args.tiles_path is None else args.tiles_path
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir,
                                               'config_file')) if args.config is None else args.config
    experiment = 'ticc' if args.experiment is None else args.experiment
    
    main(tiles_data_path, config_path, experiment)
