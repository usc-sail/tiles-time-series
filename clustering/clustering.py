#!/usr/bin/env python3 
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def ClusterSegments(segment_summary_df, method, num_clusters):
    """ Produces a fixed-length summary vector of the sequence for each segment.

    Params:
    segment_summary_df - pandas dataframe where each row contains summary data for a segment
    method - a string indicating which clustering method to use
    num_clusters - an integer for the number of clusters to attempt to produce

    Returns:
    Pandas dataframe containing a cluster assignment for each row of the input segment_summary_df
    """
    cluster_df = None
    if method == 'kmeans':
        model = KMeans(n_clusters=num_clusters).fit(segment_summary_df)
        cluster_df = pd.DataFrame(data={'cluster_id': model.labels_}, index=segment_summary_df.index)
    else:
        print("Unknown method in ClusterSegments. Fix me!")

    return cluster_df
