import pandas as pd 
import numpy as np 
from scipy.sparse import csr_matrix # for rollup matrix 
import psutil, os # for memory functions 
from time import time
import gc
import sys



#############################################
#############################################
################ S O U R C E ################
# https://www.kaggle.com/chrisrichardmiles/m5-helpers
#############################################
# I made a slight adjustment to this code. Namely, I replaced
# w_df['weight'] = w_df.dollar_sales / w_df.dollar_sales[0]
# with
# w_df['weight'] = w_df.dollar_sales / w_df.dollar_sales.iloc[0]
#############################################
#############################################
#############################################


##################################################################################################
##################################################################################################
##############################                          ##########################################
##############################    Standard functions    ##########################################
##############################                          ##########################################
##################################################################################################
##################################################################################################

################# Plotting preferences ####################
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

# plt.rcParams['figure.figsize'] = (14,6)
# plt.rcParams['font.size'] = 16



##################################################################################################
##################################################################################################

########## Useful memory functions ################
# I learned these from https://www.kaggle.com/kyakovlev
def get_memory_usage():
    """Returns RAM usage"""
    ########### Breakdown of functions
    # getpid: gets the process id number.
    # psutil.process gets that process with a certain pid.
    # .memory_info() describes notebook memory usage.
    # [0] gets the rss resident state size of (process it think) in bytes.
    # /2.**30 converts output from bytes to gigabytes
    # Returns RAM usage
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    """Reformats num, which is num bytes"""
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)



## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    """Converts numeric columns to smallest datatype that preserves information"""
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def merge_by_concat(df1, df2, merge_on):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1

##################################################################################################
# This one is from https://www.kaggle.com/hectorhaffenden
# import sys
def top_n_mem(n):
    '''
    Function to get what variables are using how much memory in global env
        Input: n = number of variables to return
    '''
    local_vars = list(globals().items())
    name = []
    size = []
    for var, obj in local_vars:
        name.append(var)
        size.append(sys.getsizeof(obj))
    return pd.concat([pd.DataFrame({"name": name}),
                      pd.DataFrame({"size": size})],
                      axis = 1).sort_values('size', ascending=False).head(n)



##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################
##############################                          ##########################################
##############################            M5            ##########################################
##############################                          ##########################################
##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

# import pandas as pd 
# import numpy as np 
# from scipy.sparse import csr_matrix

def read_data(DATA_PATH):  
    """Reads in 'sales_train_evaluation.csv', 'calendar.csv', 'sell_prices.csv', 'sample_submission.csv'
    from DATA_PATH. 
    
    RETURNS: 4 dataframes of the csvs"""
    csvs = ['sales_train_evaluation.csv', 'calendar.csv', 'sell_prices.csv', 'sample_submission.csv']
    return [pd.read_csv(f'{DATA_PATH}' + csv) for csv in csvs]

def get_rollup(train_df):
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

    # Sparse format will save space and calculation time
    rollup_matrix_csr = csr_matrix(rollup_matrix)
    
    return rollup_matrix_csr, rollup_index

##################################################################################################
##################################################################################################

def get_w_df(train_df, cal_df, prices_df, rollup_index, rollup_matrix_csr, start_test=1914): 
    """Returns the weight, scale, and scaled weight of all series, 
    in a dataframe aligned with the rollup_index, created in get_rollup()"""
    
    d_cols = [f'd_{i}' for i in range(start_test - 28, start_test)]
    df = train_df[['store_id', 'item_id'] + d_cols]
    df = df.melt(id_vars=['store_id', 'item_id'],
                           var_name='d', 
                           value_name = 'sales')
    df = df.merge(cal_df[['d', 'wm_yr_wk']], on='d', how='left')
    df = df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    df['dollar_sales'] = df.sales * df.sell_price

    # Now we will get the total dollar sales 
    dollar_sales = df.groupby(['store_id', 'item_id'], sort=False)['dollar_sales'].sum()
    del df

    # Build a weight, scales, and scaled weight columns 
    # that are aligned with rollup_index. 
    w_df = pd.DataFrame(index = rollup_index)
    w_df['dollar_sales'] = rollup_matrix_csr * dollar_sales
    w_df['weight'] = w_df.dollar_sales / w_df.dollar_sales.iloc[0]
    del w_df['dollar_sales']

    ##################### Scaling factor #######################
    
    df = train_df.loc[:, 'd_1':f'd_{start_test-1}']
    agg_series = rollup_matrix_csr * df.values
    no_sale = np.cumsum(agg_series, axis=1) == 0
    agg_series = np.where(no_sale, np.nan, agg_series)
    scale = np.nanmean(np.diff(agg_series, axis=1) ** 2, axis=1)

    w_df['scale'] = scale
    w_df['scaled_weight'] = w_df.weight / np.sqrt(w_df.scale)
    
    ################# spl_scale ####################
    # Now we can compute the scale and add 
    # it as a column to our w_df
    scale = np.nanmean(np.abs(np.diff(agg_series)), axis = 1)
    scale.shape
    w_df['spl_scale'] = scale

    # It may also come in handy to have a scaled_weight 
    # on hand.  
    w_df['spl_scaled_weight'] = w_df.weight / w_df.spl_scale
     
    ############## sub_id for uncertainty submission ##############
    w_df = add_sub_id(w_df)
    
    return w_df

##################################################################################################
##################################################################################################
##############################                          ##########################################
##############################      EDA and plot        ##########################################
##############################                          ##########################################
##################################################################################################
##################################################################################################

def plot_item_series(item, series_df, state=None, fillna=False, start=0, end=1941): 
    """Plots the level 10-12 series containing the item"""
    item_series_all = series_df.loc[series_df.index.get_level_values(1).str.contains(item)]
    if state: 
        state_mask = item_series_all.index.get_level_values(1).str.contains(state)
        if fillna: 
            item_series_all.loc[state_mask].iloc[:, start:end].fillna(fillna).T.plot(title=f'{item} overall in {state} and by store')
        else: 
            item_series_all.loc[state_mask].iloc[:, start:end].T.plot(title=f'{item} overall in {state} and by store')
        plt.legend(bbox_to_anchor=(1,1.04), loc='lower right', ncol=1)
        for i in [1941 - 364*i for i in range(6) if start < 1941 - 364*i <= end]: 
            plt.axvline(i, ls=':')
        plt.show()
        
    else: 
        if fillna: 
            item_series_all.iloc[:4, start:end].fillna(fillna).T.plot(title=f'{item} total and by state')
        else: 
            item_series_all.iloc[:4, start:end].T.plot(title=f'{item} total and by state')
        plt.legend(bbox_to_anchor=(.5,.99), loc='upper center', ncol=1)
        for i in [1941 - 364*i for i in range(6) if start < 1941 - 364*i <= end]: 
            plt.axvline(i, ls=':')     
        plt.show()
        
def plot_all_item_series(item, series_df, fillna=False, start=0, end=1941): 
    plot_item_series(item, series_df, state=None, fillna=fillna, start=start, end=end)
    for state in ['CA', 'TX', 'WI']: 
        plot_item_series(item, series_df, state=state, fillna=fillna, start=start, end=end)

##################################################################################################
##################################################################################################
##############################                          ##########################################
##############################      out of stock        ##########################################
##############################                          ##########################################
##################################################################################################
##################################################################################################

def fix_oos(item, series_df):
    """Processes item and returns series that has np.nan
    where we think out of stock zeros occur"""
    item_series = series_df.loc[item]
    item_mean = np.nanmean(item_series) 
    x = True
    while x == True:
        item_series, new_mean, streak_length, item_streaks, limit_99 = nan_zeros(item_series, item_mean)
        x = new_mean > item_mean
        item_mean = new_mean
        
    return item_series, new_mean, streak_length, item_streaks, limit_99

def nan_zeros(item_series, item_mean):
    """Returns item_series with streaks replaced by nans, 
    the new average of item series, and max_streak_length, 
    which is the highest streak
    count that was not replaced with nans."""
    # With the mean, we can find the probability 
    # of a zero sales, given the item follows 
    # the poisson distribution 
    prob_0 = np.exp(-item_mean)
    
    # Adding this to make sure we catch long streaks that 
    # artificially decrease our starting mean, leading to 
    # an artificially large 
    lowest_prob_allowed = .000_001
    lowest_streak_allowed = 1
    while prob_0 ** lowest_streak_allowed > lowest_prob_allowed: 
        lowest_streak_allowed += 1

    # Given the probability of a zero, we can find 
    # the expected value of the total number of 
    # zeros. 
    expected_zeros = prob_0 * (~np.isnan(item_series)).sum()

    # Given the number of total zeros should 
    # follow the binomial distribution, approximated
    # by the normal distribution, we can assume 
    # that total zeros are below mean + 3 standard 
    # deviations 99.85 percent of the time.
    std = np.sqrt((~np.isnan(item_series)).sum() * prob_0 * (1-prob_0))
    limit_99 = expected_zeros + 3 * std
    item_streaks = mark_streaks(item_series)
    max_streak_length = 1
    total_zeros = (item_streaks == max_streak_length).sum()
    while (total_zeros < limit_99) & (max_streak_length < lowest_streak_allowed): 
        max_streak_length += 1
        total_zeros = (item_streaks == max_streak_length).sum()

    # Now remove the zeros in streaks greater 
    # than max_streak_length
    m = min(max_streak_length, lowest_streak_allowed)
    item_series = np.where(item_streaks > m, np.nan, item_series)
    new_mean = np.nanmean(item_series)
    
    return item_series, new_mean, max_streak_length, item_streaks, limit_99

########################## To add oos_scale to w_df ##################
def get_oos_scale(oos_train_df):
    """oos_train_df ids should be in correct order"""
    rec = oos_train_df.iloc[:, 6:-28]
    rdiff = np.diff(rec, axis=1)
    return np.sqrt(np.nanmean(rdiff**2, axis=1))

###### Mark zeros with length of consecutive zeros ######
# New version thanks to @nadare tip in sibmikes notebook, 
# where I learned about np.frompyfunc, and how it can 
# make python functions run really fast. 
def mark_streaks(ts):
    """Returns an array of the same length as ts, 
    except positive values are replaced by zero, 
    and zeros are replaced by the lenth of the zero 
    streak to which they belong.
    
    ########## Example ############
    ### in ###
    series = np.array([np.nan,3,0,0,0,2,0,0,1,0])
    mark_streaks(series)
    
    ### out ###
    array([nan,  0.,  3.,  3.,  3.,  0.,  2.,  2.,  0.,  1.])
    """
    ts_nan_mask = np.isnan(ts)
    zeros = ~(ts > 0) * 1
    accum_add_prod = np.frompyfunc(lambda x, y: int((x + y)*y), 2, 1)
    a = accum_add_prod.accumulate(zeros, dtype=np.object)
    a = np.where(a==0, 0, np.where(a < np.roll(a, -1), np.nan, a))
    a = pd.Series(a).fillna(method='bfill').to_numpy()
    item_streaks = np.where(ts_nan_mask, np.nan, a)
    
    return item_streaks

##################################################################################################
##################################################################################################
##############################                          ##########################################
##############################      WRMSSE object       ##########################################
##############################                          ##########################################
##################################################################################################
##################################################################################################
def make_main_object(path, start_test, output_path=''): 
    """Saves a few lines of code when we want a WRMSSE object"""
    train_df, cal_df, prices_df, _ = read_data(path)
    return WRMSSE(train_df, cal_df, prices_df, start_test, output_path)

class WRMSSE():
    """An object that is capable of scoring predictions for any 
    time period, provied the necessary dataframes. It will also have 
    """
    def __init__(self, train_df, cal_df, prices_df, start_test, output_path=''):
            
        self.train_df = train_df
        self.cal_df = cal_df
        self.prices_df = prices_df 
        self.start_test = start_test
        self.output_path = output_path
        if start_test != 1942: 
            self.actuals = train_df.loc[:, f'd_{self.start_test}': f'd_{self.start_test + 27}'].values

        self.rollup_matrix_csr, self.rollup_index = get_rollup(self.train_df)
        self.w_df = get_w_df(self.train_df, 
                             self.cal_df, 
                             self.prices_df, 
                             self.rollup_index, 
                             self.rollup_matrix_csr, 
                             self.start_test)

        self.level_scores = []
        
    def score(self, preds): 
        """Used only for quick scoring, possibly a custom metric 
        for early stopping."""
        if type(preds) == pd.DataFrame: 
            preds = preds.values
        diff = self.actuals - preds
        res = np.sum(
                np.sqrt(
                    np.mean((self.rollup_matrix_csr * diff)**2, axis=1)
                       ) * self.w_df.scaled_weight
                  ) / 12
        return res
    
    def score_all(self, preds, evaluator_name = 'no_name', output=True): 
        if type(preds) == pd.DataFrame: 
            preds = preds.values
        diff = self.actuals - preds
        self.scores_df = pd.DataFrame(np.sqrt(
            np.mean((self.rollup_matrix_csr * diff)**2, axis=1)
                ) * self.w_df.scaled_weight).rename(mapper = {'scaled_weight': 'WRMSSE'}, axis = 1)
        
        if output: 
            str_score = str(round(self.scores_df.groupby(level=0).sum().mean()[0], 4)).replace('.', '')
            ############### Dump scores_df and preds_df #######################
            self.scores_df.to_csv(f'{self.output_path}scores_df_' + evaluator_name + '.csv')

            pd.DataFrame(preds).to_csv(f'{self.output_path}preds_df_' +  evaluator_name + '.csv')
        self.plot_scores(evaluator_name)
        
    def plot_scores(self, evaluator_name): 
        fig, ax = plt.subplots()
        wrmsses = self.scores_df.groupby(level=0).sum()
        sns.barplot(x=wrmsses.index, y=wrmsses['WRMSSE'])
        plt.axhline(wrmsses.mean()[0], color='blue', alpha=.5, ls=':')
        plt.title(f'{evaluator_name} by Level: {round(wrmsses.mean()[0], 4)} WRMSSE score', fontsize=20, fontweight='bold')
        for i in range(12): 
            ax.text(i, wrmsses['WRMSSE'][i+1], 
                    str(round(wrmsses['WRMSSE'][i+1], 4)), 
                    color='black', ha='center', fontsize=15)
        plt.show()

        

##################################################################################################
##################################################################################################
##############################                          ##########################################
##############################    WRMSSE metrics/loss   ##########################################
##############################                          ##########################################
##################################################################################################
##################################################################################################

def get_L12_metric(e, grid_df, p_horizon=28):
    """
    Gets a level 12 WRMSSE custom metric function that is 
    fit to the 'grid_df' training data.

    Parameters
    ----------
    e : WRMSSE object from this script 
        It is used to get the weights and scales
    p_horizon : int
        the number of days in the prediciton horizon. 
    grid_df : pandas DataFrame
        Training data with an id column 

    Returns
    -------
    custom_metric : function
        custom_metric(preds, train_data) is meant to be put 
        into the feval of an Lgbm estimator.

    """
    ########## Getting the scales and weights #############
    w_df = e.w_df.copy()
    w_df = w_df.loc[12].reset_index()
    w_df['id'] = w_df.id + '_evaluation'
    w_df = pd.merge(grid_df[['id']], w_df, on='id', how='left')

    ########### Setting horizon and num_products ##########
    P_HORIZON = p_horizon                       # Prediction horizon 
    NUM_PRODUCTS = grid_df.id.nunique()  # Number of products 

    ############ scale and weight vectors #################
    scale = w_df[-P_HORIZON * NUM_PRODUCTS:].scale
    weight = w_df[-P_HORIZON * NUM_PRODUCTS:].weight[:NUM_PRODUCTS]
    weight = weight/weight.sum()

    ################### Custom metric #####################
    def custom_metric(preds, train_data):
        actuals = train_data.get_label()
        res = L12_WRMSSE(preds, actuals, P_HORIZON, NUM_PRODUCTS, scale, weight)
        return 'L12_WRMSSE', res, False
    
    return custom_metric


def L12_WRMSSE(preds, actuals, p_horizon, num_products, scale, weight): 
    """
    Used inside a custom 
    Steps guide: 
    -----------
    1)  The first step in calculating the wrmsse is 
        squareing the daily error.
        
    2)  Now divide the result by the appropriate scale
        take values of scale to avoid converting res 
        to a pandas series
        
    3)  The next step is summing accross the horizon
        We must reshape our data to get the products
        in line. 
        
    4)  Now we take the mean accross the prediction horizon
        and take the square root of the result.
        
    5)  Now we multiply each result with the appropriate
        scaled_weight. We just need the first 30490 entries 
        of the scaled_weight column
    """
    actuals = actuals[-(p_horizon * num_products):]
    preds = preds[-(p_horizon * num_products):]
    diff = actuals - preds
    res = diff ** 2                             # step 1
    res = res/scale.values                      # step 2
    res = res.reshape(p_horizon, num_products)  # step 3
    res = res.mean(axis=0)                      # step 4
    res = np.sqrt(res)
    res = res * weight                          # step 5
    res = res.sum()
    return res
    
##################################################################################################
##################################################################################################
#####################                                              ###############################
#####################     Series Aggregation and Basic Stats       ###############################
#####################                                              ###############################
##################################################################################################
##################################################################################################

################### series_df function #####################
def get_series_df(train_df, rollup_matrix_csr, rollup_index, cal_df=None, noxmas=False):
    """Returns a dataframe with series for all 12 levels of aggregation. We also 
    replace leading zeros with np.nan and if noxmas, replace christmas sales with average 
    of the day before and day after christmas"""
    
    series_df = pd.DataFrame(data=rollup_matrix_csr * train_df.iloc[:, 6:].values,
                         index=rollup_index, 
                         columns=train_df.iloc[:, 6:].columns)
    
    # We manipulate the data in a way so that only 
    # leading zeros will remane unchanged. 
    zero_mask = series_df.cumsum(axis=1) * 2 == series_df

    # Now set the leading zeros to np.nan
    series_df[zero_mask] = np.nan
    
    ################## Christmas closure ####################
    if noxmas: 
        
        # First find all x where 'd_x' represents christmas. 
        xmas_days = cal_df[cal_df.date.str[-5:] == '12-25'].d.str[2:].astype('int16')

        # I will choose to replace sales for every christmas with 
        # the average of the day before and the day after. 
        for x in xmas_days: 
            series_df[f'd_{x}'] = (series_df[f'd_{x-1}'] + series_df[f'd_{x+1}']) / 2
                
    
    return series_df 


################## stats_df function #######################
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

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
# |||||||||   |||||||||                            
# |           |
# ||||||      ||||||                             Feature Engineering
# |           |
# |           |||||||||
##########################################################################################################################


################## Load data ####################
# train_df = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

############################################################
######################### Imports ##########################
# import numpy as np 
# import pandas as pd
# from time import time
# import gc

############################################################
#################### Making grid_df ########################

def nan_leading_zeros(rec):
    rec = rec.astype(np.float64)
    zero_mask = rec.cumsum(axis=1) == 0
    rec[zero_mask] = np.nan
    return rec

def make_grid_df(train_df, use_index=False, pred_horizon=True): 
    """Returns a grid_df and a numpy array (rectangle) """
    
    # Don't want to change global train_df variable 
    train_df = train_df.copy()
    
    start_time = time()
    print("#" * 72, "\nMaking grid_df")
    # Add 28 days for the predicton horizon 
    
    last_day = int(train_df.columns[-1][2:])
    if pred_horizon: 
        for i in range(last_day + 1, last_day + 29): 
            train_df[f'd_{i}'] = np.nan
            
            
    d_cols = [col for col in train_df.columns if 'd_' in col]
    if use_index: 
        index_df = train_df[[]]
    else: 
        index = train_df.id
        index = pd.Series(np.tile(index, last_day + 28)).astype('category')
    
    # Turn leading zeros into np.nan
    rec = nan_leading_zeros(train_df[d_cols].values)
    sales = rec.T.reshape(-1,)

    if use_index: 
        grid_df = pd.concat([index_df for i in range(train_df.shape[1])])
        grid_df['d'] = np.concatenate([[i] * train_df.shape[0] for i in range(1, last_day + 28 + 1)]).astype(np.int16)
        grid_df['sales'] = sales
    else: 
        grid_df = pd.DataFrame({'id': index, 
                          'd': np.concatenate([[i] * 30490 for i in range(1, last_day + 28 + 1)]).astype(np.int16), 
                          'sales': sales})
    print(f'Time: {(time() - start_time):.2f} seconds')
    return grid_df, rec

############################################################
#####################@ Basic lags ##########################

def add_lags(grid_df, lags = range(1,16), num_series=30490):
    """Adds lag feature columns to grid_df. 'num_series' is 
    the number of distinct series in grid_df"""
    start_time = time()
    print( 72 * '#', '\nAdding lag columns')
    for i in lags:
        grid_df[f'lag_{i}'] = grid_df['sales'].shift(num_series * i).astype(np.float16)
    
    print(f'Time: {(time() - start_time):.2f} seconds')
        
        
############################################################       
################# Rolling window features ##################

def rolling_window(a, window):
    """Reference: https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    A super fast way of getting rolling windows on a numpy array. """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def split_array(ary, sections, axis=0):
    """Works just like np.split, but sections must be a 
    single integer. It will work, even when sections doesn't 
    evenly divide the length of ary.
    
    Examples
    --------
    >>> x = np.arange(9)
    >>> split_array(x, 4)
    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7, 8])]

    >>> split_array(x, 5)
    [array([0, 1]), array([2, 3]), array([4, 5]), array([6, 7, 8])]"""
    w = np.int(np.ceil(len(ary)/sections))
    return np.split(ary, [w * i for i in range(1, int(np.ceil(len(ary)//w)))], axis=axis)

def make_rolling_col(rw, function, split=1, num_series=30490): 
    # We need to take off the last columns to
    # get the rolling feature shifted one day.
    
    split_rw = split_array(rw, split, axis=0)
    split_col = [function(rw, -1) for rw in split_rw]
    col = np.concatenate(split_col)
    col = col[:, :-1].T.reshape(-1,)

    # The new column must be prepended with np.nans 
    # to account for missing gaps
    return np.append(np.zeros(num_series * rw.shape[-1]) + np.nan, col).astype(np.float16)


def add_rolling_cols(df: pd.DataFrame, rec: np.array, windows: list, functions: list, function_names: list, splits=None): 
    """Adds rolling features to df."""
    
    print( 72 * '#', '\nAdding rolling columns\n',  )
    start_time = time()
    if splits == None: 
        splits = [1] * len(functions)
    f = list(zip(functions, function_names, splits))
    
    for window in windows: 
        rw = rolling_window(rec, window)
        for function in f: 
            s_time = time()
            df[f'shift_1_rolling_{function[1]}_{str(window)}'] = make_rolling_col(rw, function[0], split=function[2], num_series=rec.shape[0])
            print(f'{function[1]} with window {window} time: {(time() - s_time):.2f} seconds')
            gc.collect()
            
    print(f'Total time for rolling cols: {(time() - start_time)/60:.2f} minutes')
    
################## Custom rolling functions ###################
# These functions are designed to act on a rolling_window array
# created by the rolling_window function, similar to np.mean
# or np.std. 
def diff_mean(rolling_window, axis=-1): 
    """For M5 purposes, used on an object generated by the 
    rolling_window function. Returns the mean of the first 
    difference of a window of sales."""
    return np.diff(rolling_window, axis=axis).mean(axis=axis)

def diff_nanmean(rolling_window, axis=-1): 
    """For M5 purposes, used on an object generated by the 
    rolling_window function. Returns the mean of the first 
    difference of a window of sales."""
    return np.nanmean(np.diff(rolling_window, axis=axis), axis = axis)

def mean_decay(rolling_window, axis=-1): 
    """Returns the mean_decay along an axis of a rolling window object, 
    which is created by the rolling_window() function."""
    
    # decay window must be as long as the last 
    # dimension in the rolling window
    decay_window = np.power(.9, np.arange(rolling_window.shape[-1]))[::-1]
    decay_sum = decay_window.sum()
    return (rolling_window * decay_window).sum(axis = -1) / decay_sum
    
    
############################################################       
################# Shifting function ########################
def add_shift_cols(grid_df, shifts, cols, num_series=30490): 
    
    print( 72 * '#', '\nAdding shift columns',  )
    start_time = time()
    for shift in shifts: 
        for col in cols: 
            grid_df[f"{col.replace('shift_1', f'shift_{shift}')}"] = grid_df[col].shift((shift - 1) * num_series)
    print(f'Time: {(time() - start_time):.2f} seconds')

            
            
            
############################################################       
################# Create lags df ###########################
def make_lags_df(train_df): 
    
    start_time = time()
    grid_df, rec = make_grid_df(train_df)
    add_lags(grid_df)
    add_rolling_cols(grid_df, 
                     rec, 
                     windows=[7, 14, 30, 60, 180], 
                     functions=[np.mean, np.std], 
                     function_names=['mean', 'std'])
    
    
    shifts = [8, 15]
    cols = [f'shift_1_rolling_mean_{i}' for i in [7, 14, 30, 60]]
    add_shift_cols(grid_df, shifts, cols, num_series=30490)
    
    print(72 * '#', f'Total time: {(time() - start_time)//60:} : {(time() - start_time)%60:.2f}')
    return grid_df

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################


##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
#####################                                                                                ##################### 
#####################                                                                                #####################
#####################                        Uncertainty   Competition                               #####################
#####################                                                                                #####################
#####################                                                                                #####################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

####################################################################################
############################ WSPL and helpers ######################################
# Find the notebook for these functions at this link
# https://www.kaggle.com/chrisrichardmiles/m5u-wsplevaluator-weighted-scaled-pinball-loss/edit

############################ WSPLEvaluator Object ##################################
class WSPLEvaluator(): 
    """ Will generate w_df and ability to score prediction for any start_test period """
    def __init__(self, train_df, cal_df, prices_df, start_test=1914):
        self.rollup_matrix_csr, self.rollup_index = get_rollup(train_df)
                        
        self.w_df = get_w_df(
                        train_df,
                        cal_df,
                        prices_df,
                        self.rollup_index,
                        self.rollup_matrix_csr,
                        start_test=start_test,
                    )
        
        self.quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
        level_12_actuals = train_df.loc[:, f'd_{start_test}': f'd_{start_test + 27}']
        self.actuals = self.rollup_matrix_csr * level_12_actuals.values
        self.actuals_tiled = np.tile(self.actuals.T, 9).T
        
        
    def score_all(self, preds): 
        scores_df, total = wspl(self.actuals_tiled, preds, self.w_df)
        self.scores_df = scores_df
        self.total_score = total
        print(f"Total score is {total}")


############## spl scaling factor function ###############
def add_spl_scale(w_df, train_df, rollup_matrix_csr): 
    # We calculate scales for days preceding 
    # the start of the testing/scoring period. 
    start_test = 1914
    df = train_df.loc[:, 'd_1':f'd_{start_test-1}']

    # We will need to aggregate all series 
    agg_series = rollup_matrix_csr * df.values

    # Make sure leading zeros are not included in calculations
    agg_series = h.nan_leading_zeros(agg_series)

    # Now we can compute the scale and add 
    # it as a column to our w_df
    scale = np.nanmean(np.abs(np.diff(agg_series)), axis = 1)
    scale.shape
    w_df['spl_scale'] = scale

    # It may also come in handy to have a scaled_weight 
    # on hand.  
    w_df['spl_scaled_weight'] = w_df.weight / w_df.spl_scale
    
    return w_df

########## Function for all level pinball loss for quantile u ############
def spl_u(actuals, preds, u, w_df):
    """Returns the scaled pinball loss for each series"""
    pl = np.where(actuals >= preds, (actuals - preds) * u, (preds - actuals) * (1 - u)).mean(axis=1)

    # Now calculate the scaled pinball loss.  
    all_series_spl = pl / w_df.spl_scale
    return all_series_spl

########## wspl for all quantiles ############
def wspl(actuals, preds, w_df): 
    """
    :acutals:, 9 vertical copies of the ground truth for all series. 
    :preds: predictions for all series and all quantiles. Same 
    shape as actuals"""
    quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]
    scores = []
    
    # In this case, preds has every series for every  
    # quantile T, so it has 42840 * 9 rows. We first 
    # break it up into 9 parts to get the wspl_T for each.
    # We also do the same for actuals. 
    preds_list = np.split(preds, 9)
    actuals_list = np.split(actuals, 9)
    
    for i in range(9):
        scores.append(spl_u(actuals_list[i], preds_list[i], quantiles[i], w_df))
        
    # Store all our results in a dataframe
    scores_df = pd.DataFrame(dict(zip(quantiles, [w_df.weight * score for score in scores])))
    
    #  We divide score by 9 
    # to get the average wspl of each quantile. 
    spl = sum(scores) / 9
    wspl_by_series = (w_df.weight * spl)
    total = wspl_by_series.sum() / 12
    
    return scores_df, total

####################################################################################
############################ formatting for submission #############################

############## sub_id function ################
def add_sub_id(w_df):
    """ adds a column 'sub_id' which will match the 
    labels in the sample_submission 'id' column. Next 
    step will be adding '_{quantile}_validation/evaluation'
    onto the sub_id column. This will be done in another 
    function. 
    
    :w_df: dataframe with the multi-index that is 
    genereated by get_rollup()
    
    Returns w_df with added 'sub_id' column"""
    # Lets add a sub_id col to w_df that 
    # we will build to match the submission file. 
    w_df['sub_id'] = w_df.index.get_level_values(1)

    ###### level 1-5, 10 change ########
    w_df.loc[1:5, 'sub_id'] = w_df.sub_id + '_X'
    w_df.loc[10, 'sub_id'] = w_df.sub_id + '_X'

    ######## level 11 change ##########
    splits = w_df.loc[11, 'sub_id'].str.split('_')
    w_df.loc[11, 'sub_id'] = (splits.str[3] + '_' + \
                              splits.str[0] + '_' + \
                              splits.str[1] + '_' + \
                              splits.str[2]).values
    
    return w_df



################## add quantile function ################
def add_quantile_to_sub_id(w_df, u): 
    """Used to format 'sub_id' column in w_df. w_df must 
    already have a 'sub_id' column. This used to match 
    the 'id' column of the submission file."""
    # Make sure not to affect global variable if we 
    # don't want to. 
    w_df = w_df.copy()
    w_df['sub_id'] = w_df.sub_id + f"_{u:.3f}_validation"
    return w_df