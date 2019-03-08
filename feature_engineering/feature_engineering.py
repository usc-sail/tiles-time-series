#!/usr/bin/env python3 
import os
import sys
import pdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'util')))

import util

def CreateFeatures(cluster_summary_df, feature_list):
    """ Produces a set of features for each cluster

    Params:
    cluster_summary_df - pandas dataframe where each row contains cluster data for a segment
    feature_list - a list names of the desired feature to create

    Returns:
    Pandas dataframe containing the features, one feature per column and some with encoding
    """
    feature_df_list = []
    cluster_ids = cluster_summary_df['cluster_id'].values.astype(int)
    unique_clusters = np.unique(cluster_ids).astype(int).tolist()
    for feature in feature_list:
        if feature == 'duration':
            start_times = util.GetUnixTimeFromTimestamp(cluster_summary_df['start'].values)
            end_times = util.GetUnixTimeFromTimestamp(cluster_summary_df['end'].values)

            duration_mat = np.zeros((cluster_summary_df.shape[0], len(unique_clusters)))
            for i in range(len(cluster_ids)):
                cluster_id = cluster_ids[i]
                unique_cluster_index = unique_clusters.index(cluster_id)
                duration = end_times[i] - start_times[i]
                duration_mat[i, unique_cluster_index] = duration
                
            duration_df = pd.DataFrame(duration_mat)
            duration_df.columns = ['cluster_%d_duration'%(cluster_id) for cluster_id in unique_clusters]
            feature_df_list.append(duration_df)

        elif feature == 'count':
            count_mat = np.zeros((cluster_summary_df.shape[0], len(unique_clusters)))
            for i in range(len(cluster_ids)):
                cluster_id = cluster_ids[i]
                unique_cluster_index = unique_clusters.index(cluster_id)
                count_mat[i, unique_cluster_index] = 1
                
            count_df = pd.DataFrame(count_mat)
            count_df.columns = ['cluster_%d_count'%(cluster_id) for cluster_id in unique_clusters]
            feature_df_list.append(count_df)
        else:
            print("Unknown feature name. Fix me!")

    # TODO - Allow the window of time to be specified if aggregating multiple cluster data over time
    feature_df = pd.concat(feature_df_list, axis=1)
    return feature_df
