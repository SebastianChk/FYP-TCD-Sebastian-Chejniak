import pandas as pd
import numpy as np
import properscoring as ps
import os
from sktime.performance_metrics.forecasting import (
    mean_squared_scaled_error,
    mean_absolute_error,
    mean_absolute_scaled_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from utils.evaluation import WRMSSEEvaluator
#rmsse = MeanSquaredScaledError(square_root=True)

# experimentation
# from gluonts.evaluation import metrics.mse, metrics.mase, metrics.smape





def create_wrmsse(DATA_PATH='../../data/external/'):
    '''Only works for all of the data, ordered as given in sale_train_evaluation.csv'''
    
    key_names = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    date_names = ["d_" + str(i) for i in range(1, 1942)]
    test_steps = 28
    
    calendar = pd.read_csv(os.path.join(DATA_PATH, "calendar.csv"))
    selling_prices = pd.read_csv(os.path.join(DATA_PATH, "sell_prices.csv"))
    df_train_eval = pd.read_csv(os.path.join(DATA_PATH, "sales_train_evaluation.csv"))
    
    df_train = df_train_eval.loc[:, key_names + date_names[:-test_steps]]
    df_test = df_train_eval.loc[:, date_names[-test_steps:]]
    wrmsse_evaluator = WRMSSEEvaluator(
        df_train, df_test, calendar, selling_prices, test_steps
    )
    
    def metric(y_preds):
        df_pred = pd.DataFrame(data=y_preds, 
                               columns=date_names[-test_steps:])
        return wrmsse_evaluator.score(df_pred)
    
    return metric


def create_mae(y_tests, y_trains=None):
    def metric(y_preds):
        metrics = np.zeros(y_preds.shape[0])
        
        for idx, y_test in enumerate(y_tests):
            metrics[idx] = mean_absolute_error(y_test,
                                               y_preds[idx,:])
        
        return np.mean(metrics)
    return metric


def create_rmse(y_tests, y_trains=None, gluon=True):
    def metric(y_preds):
        metrics = np.zeros(y_preds.shape[0])
        
        for idx, y_test in enumerate(y_tests):
            metrics[idx] = mean_squared_error(y_test,
                                              y_preds[idx,:])
            # if not gluon:
            #     metrics[idx] = np.sqrt(metrics[idx])
        # gluon is true if we are to find the rmse as is done by gluonts (noting that gluonts does this for the mean forecasts)
        if not gluon:
            metrics = np.sqrt(metrics)
            return np.mean(metrics)
        
        if gluon:
            metrics = np.mean(metrics)
            return np.sqrt(metrics)
            
        # print(metrics)
        
        return np.mean(metrics)
    return metric


def create_smape(y_tests, y_trains=None):
    def metric(y_preds):
        metrics = np.zeros(y_preds.shape[0])
        
        for idx, y_test in enumerate(y_tests):
            metrics[idx] = mean_absolute_percentage_error(y_test,
                                                          y_preds[idx,:],
                                                          symmetric=True)
        return np.mean(metrics)
    return metric

######################################################
def create_mase(y_tests, y_trains):
    def metric(y_preds):
        metrics = np.zeros(y_preds.shape[0])
        # print(y_preds.shape, y_tests.shape, y_trains.shape)
        for idx, y_test in enumerate(y_tests):
            # print(idx, y_test.shape, y_preds[idx,:].shape, y_trains[idx,:].shape)
            _mase = mean_absolute_scaled_error(y_test,
                                                      y_preds[idx,:],
                                                      y_train=y_trains[idx,:])
            metrics[idx] = _mase
        # print(metrics[:1000])
        # return metrics
        return np.mean(metrics)
    return metric
######################################################

# def gluon_create_mase(y_trains, y_tests):
#     def metric(y_preds):
#         metrics = np.zeros(y_preds.shape[0])
        
#         for idx, y_test in enumerate(y_tests):
#             metrics[idx] = mean_absolute_scaled_error(y_test,
#                                                       y_preds[idx,:],
#                                                       y_train=y_trains[idx,:])
#         return np.mean(metrics)
#     return metric
# def create_metric()
    
def compute_point_metrics(y_trains: np.ndarray,
                          y_tests: np.ndarray,
                          predictions: dict,
                          exclude_wrmsse=False,
                          include=['MASE', 'sMAPE', 'RMSE', 'WRMSSE']):
    """ TEMPORARILY EXCLUDING MAE FROM HERE include IN THE ABOVE"""
    mae = create_mae(y_tests, y_trains)
    rmse = create_rmse(y_tests, y_trains)
    smape = create_smape(y_tests, y_trains)
    mase = create_mase(y_tests, y_trains)
    
    if exclude_wrmsse:
        include = [i for i in include if i != 'WRMSSE']
        wrmsse=None
    else:
        wrmsse = create_wrmsse()
    metric_dict = {
                   'WRMSSE':wrmsse,
                   'MAE':mae,
                   'RMSE':rmse,
                   'sMAPE':smape,
                   'MASE':mase
                   }
    if exclude_wrmsse:
        metric_dict.pop('WRMSSE')

    
    """
    
    """
    metrics = dict()
    result_df = pd.DataFrame(columns=['WRMSSE', 'MAE', 'RMSE', 'sMAPE', 'MASE', 'WSPL'],
                             index=predictions.keys())
    for model in predictions:
        print('#' * 30)
        print(f'Metrics for {model} method:')
        y_pred = predictions[model]
        for metric in include: #metric_dict:
            # print(metric)
            result_df.loc[model, metric] = metric_dict[metric](y_pred)
            print(f'{metric} = {np.round(result_df.loc[model, metric], 4)}')
    
    return result_df


def CRPSs(ytrues, prob_preds, use_ps=True):
    """
    ytrues.shape=(30490, 28)
    prob_preds.shape = (30490, 100, 28)
    """
    assert(ytrues.shape[0]==prob_preds.shape[0])
    assert(ytrues.shape[1]==prob_preds.shape[2])

    if use_ps:
        CRPSs = - np.ones(prob_preds.shape[0])
        for i in range(prob_preds.shape[0]):
            CRPSs[i] = np.mean(ps.crps_ensemble(ytrues[i], prob_preds[i].transpose()))
    else:
        CRPSs = - np.ones(prob_preds.shape[0])
        for i in range(prob_preds.shape[0]):
            CRPSs[i] = np.mean(crps(ytrues[i], prob_preds[i]))
    return CRPSs


####################################################################################
####################################################################################
######################## PROBABILISTIC METRIC EVALUATION ###########################
####################################################################################
####################################################################################

# This implementation is not as accurate as that from properscoring for high values
# so I'm just gonna use the properscoring one
# def crps(ytrue: np.ndarray, prob_pred: np.ndarray):
#     assert (ytrue.shape[0] == prob_pred.shape[1])
#     nsamples = prob_pred.shape[0]
#     ndays = prob_pred.shape[1]
    
#     crps = - np.ones(ndays)
#     for i in range(ndays): 
#         # Notation in the below is chosen so that it matches 
#         # https://www.lokad.com/continuous-ranked-probability-score
#         x = ytrue[i]
#         ypreds = prob_pred[:,i]

#         uniques = np.unique(ypreds)
#         int_end = uniques[-1]

#         temp_crps = 0
#         pmf_cumsum = 0
#         # This for-loop will iteratively add the contributions of each unique
#         # sample prediction to the crps by splitting the integrals up into 
#         # the parts where F(y) is constant
#         for j, y in enumerate(uniques):
#             proportion = np.sum( ypreds == y ) / nsamples
#             pmf_cumsum += proportion
#             F = pmf_cumsum
#             # contribution of this cdf value to the integral over the length if found in the if-else
#             # series of statements

#             # from -infinity to x
#             if y < x and x <= int_end:
#                 mini_int_length = ( uniques[j+1] - y )
#                 temp_crps += ( F ** 2 ) * mini_int_length

#             # from x to the highest unique y value, int_end
#             elif y < int_end:
#                 mini_int_length = ( uniques[j+1] - y )
#                 temp_crps += ( ( F - 1 ) ** 2 ) * mini_int_length
#         crps[i] = temp_crps
#     return crps    