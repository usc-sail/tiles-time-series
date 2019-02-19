#!/usr/bin/env python3 
import numpy as np
import pandas as pd

def SummarizeSequenceSegments(sequence_df, segments_df, method):
    """ Produces a fixed-length summary vector of the sequence for each segment.

    Params:
    sequence_df - pandas dataframe indexed by the values in segments_df with multiple columns allowed
    segments_df - pandas dataframe with two columns: 'start' and 'end' where the values for each row correspond to the starting index and ending index in sequence_df of a segment
    method - a string indicating which summary method to use

    Returns:
    Pandas dataframe containing one row for each segment summary and where the rows correspond to the rows in segments_df
    """
    summary_df = None
    if method == 'gaussian':
        summary_df = None
        for row_idx, segment in segments_df.iterrows():
            start_index = segment['start']
            end_index = segment['end']
            subsequence = sequence_df.loc[start_index:end_index, :]
            means = subsequence.mean(axis=0)
            variances = subsequence.var(axis=0)
            statistics = [('means', means), ('variances', variances)]
            if summary_df is None:
                columns = []
                for statistic in statistics:
                    columns.extend([x+'_'+statistic[0] for x in statistic[1].index])
                summary_df = pd.DataFrame(np.nan, index=segments_df.index, columns=columns)
            col_idx = 0
            for statistic in statistics:
                summary_df.iloc[row_idx, col_idx:col_idx+len(statistic[1])] = statistic[1].values
                col_idx += len(statistic[1])
    else:
        print("Unknown method in SummarizeSequenceSegments. Fix me!")

    return summary_df
