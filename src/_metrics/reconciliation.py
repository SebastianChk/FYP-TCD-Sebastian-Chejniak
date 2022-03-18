import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix # for rollup matrix
import pickle

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
  
    
    
"""Code largely inspired by kaggle.com/chrisrichardmiles/m5-helpers/notebook
I made some small adjustments. One notable one is that I changed the naming method of get_rollup so that it matches 
sample_submission.csv better"""

def get_rollup(train_df, csr=True):
    """Gets a sparse roll up matrix for aggregation and 
    an index to align weights and scales."""    
    # Take the transpose of each dummy matrix to correctly orient the matrix
    dummy_frames = [
        pd.DataFrame({'Total': np.ones((train_df.shape[0],)).astype('int8')}, index=train_df.index).T, 
        pd.get_dummies(train_df.state_id, dtype=np.int8).T,
        pd.get_dummies(train_df.store_id, dtype=np.int8).T,
        pd.get_dummies(train_df.cat_id, dtype=np.int8).T,
        pd.get_dummies(train_df.dept_id, dtype=np.int8).T,
        pd.get_dummies(train_df.state_id + '_' + train_df.cat_id, dtype=np.int8).T,
        pd.get_dummies(train_df.state_id + '_' + train_df.dept_id, dtype=np.int8).T,
        pd.get_dummies(train_df.store_id + '_' + train_df.cat_id, dtype=np.int8).T,
        pd.get_dummies(train_df.store_id + '_' + train_df.dept_id, dtype=np.int8).T,
        pd.get_dummies(train_df.item_id, dtype=np.int8).T,
        pd.get_dummies(train_df.item_id + '_' + train_df.state_id, dtype=np.int8).T,
        pd.get_dummies(train_df.item_id + '_' + train_df.store_id, dtype=np.int8).T
    ]

    rollup_matrix = pd.concat(dummy_frames, keys=range(1,13), names=['Level', 'id'])

    # Save the index for later use 
    rollup_index = rollup_matrix.index
    if csr:
        return csr_matrix(rollup_matrix), rollup_index
    return rollup_matrix
    

def get_series_df(train_df, rollup_matrix_csr, rollup_index, num_columns=False, prediction=False):
    """Returns a dataframe with series for all 12 levels of aggregation. We also 
    replace leading zeros with np.nan. At first I thought this was dumb, but then
    I remembered that evaluation is done on the time series starting from the first
    non-zero value!"""
    if num_columns:
        series_df = pd.DataFrame(data=rollup_matrix_csr * train_df.loc[:, num_columns].values,
                             index=rollup_index, 
                             columns=num_columns)
    else: 
        series_df = pd.DataFrame(data=rollup_matrix_csr * train_df.iloc[:, 6:].values,
                             index=rollup_index, 
                             columns=train_df.iloc[:, 6:].columns)
    
    if not prediction:
        # We manipulate the data in a way so that only 
        # leading zeros will remane unchanged. 
        zero_mask = series_df.cumsum(axis=1) * 2 == series_df

        # Now set the leading zeros to np.nan

        series_df[zero_mask] = np.nan                

    return series_df


def get_stats_df(series_df):
    """Returns a dataframe that shows basic stats for all 
    series in sereis_df."""
    
    ################ Percentiles ######################
    # These will be especially useful in the uncertainty competition. 
    percentiles = [.005, .025, .165, .25, .5, .75, .835, .975, .995]


    ############# Create stats_df ########################
    stats_df = series_df.T.describe(percentiles).T

    ################## fraction 0 #######################
    # We want to know what fraction of sales are zero 
    stats_df['fraction_0'] = ((series_df == 0).sum(axis = 1) / stats_df['count'])
    
    return stats_df