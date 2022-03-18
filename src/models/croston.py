import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sktime.performance_metrics.forecasting import MeanSquaredScaledError


def Croston(ts, extra_periods=1, alpha=0.4, only_forecast=False):
#     print(ts)
    d = np.array(ts)
#     print(d)
    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)
#     print(d)
    
    q, a, f = np.full((3, cols+extra_periods), np.nan)
    a_t = 1 #periods since last demand observation
    
    #initialisation
    first_non0_demand_idx = np.argmax(d[:cols]>0)
    q[0] = d[first_non0_demand_idx]
    a[0] = 1 + first_non0_demand_idx #1+no. of inital 0s
    f[0] = q[0]/a[0]
#     q[0] = d[0]
#     a[0] = 1 doesn't make much of a difference it seems
#     f[0] = q[0]/a[0]


    for t in range(0, cols):
        if d[t] == 0:
            q[t+1] = q[t]
            a[t+1] = a[t]
            f[t+1] = f[t]
            a_t += 1
        elif d[t] > 0:
            q[t+1] = alpha * d[t] +(1-alpha) * q[t]
            a[t+1] = alpha * a_t +(1-alpha) * a[t]
            f[t+1] = q[t+1] / a[t+1]
            a_t = 1
    
    
    # future forecasts
    q[cols+1: cols+extra_periods] = q[cols]
    a[cols+1: cols+extra_periods] = a[cols]
    f[cols+1: cols+extra_periods] = f[cols]
    
    df = pd.DataFrame.from_dict({"Demand":d, "Forecast":f,
                                "Inter-arrival time estimate":a, "Demand estimate":q,
                                "Error":d-f})
    if only_forecast:
        return np.array(df['Forecast'])
    return df


def Croston_TSB(ts, extra_periods=1, alpha=0.4, beta=0.4):
    d = np.array(ts)
    cols = len(d)
    d = np.append(d, [np.nan] * extra_periods)
    
    q, p, f = np.full((3, cols+extra_periods), np.nan)
    
    #initialisation
    first_non0_demand_idx = np.argmax(d[:cols]>0)
    q[0] = d[first_non0_demand_idx]
    p[0] = 1/(1 + first_non0_demand_idx) #1/ (1+no. of inital 0s)
    f[0] = q[0] * p[0]
    
    for t in range(0, cols):
        if d[t] == 0:
            q[t+1] = q[t]
            p[t+1] = p[t] * (1 - beta)
        elif d[t] > 0:
            q[t+1] = alpha * d[t] +(1-alpha) * q[t]
            p[t+1] = beta + (1-beta) * p[t]
        f[t+1] = q[t+1] * p[t+1]
            
    # future forecasts
    q[cols+1: cols+extra_periods] = q[cols]
    p[cols+1: cols+extra_periods] = p[cols]
    f[cols+1: cols+extra_periods] = f[cols]
    
    df = pd.DataFrame.from_dict({"Demand":d, "Forecast":f,
                                "Periodicity":p, "Demand estimate":q,
                                "Error":d-f})
    return df


def Croston_cost(params=[0.4], method='std', ts=[1,2,3,0,5,0,0,2,1]):
    if method=='std':
        error = Croston(alpha=params[0], ts=ts)['Error']
    elif method=='tsb':
        error = Croston_TSB(alpha=params[0], beta=params[1], ts=ts)['Error']
    else:
        raise ValueError('method must either be \'std\' or \'tsb\' ')
    return np.sum(error ** 2)

def Croston_least_squares(method='std', initial_params=[0.1,0.1], ts=[1,2,3,0,5,0,0,2,1], extra_periods=1):
    """
    example 
    params, model = croston_least_squares(extra_periods=5,method='tsb',initial_params=[0.5,0.5], ts=[1,2,3,0,5,0,0,2,0,1])
    """
    
    
    minimized = minimize(Croston_cost, initial_params, args=(method, ts), method='Nelder-Mead')
    param_opt = minimized.x
    constrained_param_opt = np.minimum([1], np.maximum([0], param_opt))
    alpha = constrained_param_opt[0]
    if method=='tsb':
        beta = constrained_param_opt[1]
        return [alpha, beta], Croston_TSB(alpha=constrained_param_opt[0],
                                          beta=constrained_param_opt[1],
                                          ts=ts,
                                          extra_periods=extra_periods)
    
    return [alpha], Croston(alpha=constrained_param_opt[0],
                            ts=ts,
                            extra_periods=extra_periods)


def Croston_Forecasts(ste, method='tsb', prediction_length=28, context_length=2*28, ndays=1942):
    """ Returns array of shape (len(ste), 28) where each row is the forecast
    of the last 28 entries of the time series in the corresponding row of ste."""
    # os.chdir('../Code Summaries')
    # from TS_Models import Croston_Forecasts
    IDs = ste.index
    forecasts = pd.DataFrame(np.zeros((len(ste), prediction_length)),
                            columns=[f'd_{i}' for i in range(ndays-prediction_length, ndays)],
                            index=ste.index)
    d_cols = forecasts.columns
    
    tss = np.array(ste.loc[:, [f'd_{i}' for i in range(ndays - prediction_length - context_length, ndays - prediction_length)]])
    
    forecasts['alpha'] = 0.1
    if method == 'tsb':
        forecasts['beta'] = 0.1
    
    for (idx, ts) in enumerate(tss):
        item = IDs[idx]
        y_train = ts[:context_length]
        if method == 'tsb':
            [alpha, beta], model = Croston_least_squares(extra_periods=prediction_length,
                                                         method='tsb',
                                                         initial_params=[0.1,0.1],
                                                         ts=y_train)
            y_pred = np.array(model["Forecast"])[context_length:]
            forecasts.loc[item,'beta'] = beta
            
        elif method == 'std':
            [alpha], model = Croston_least_squares(extra_periods=prediction_length, method='std',
                                             initial_params=[0.1,0.1], ts=y_train)
            y_pred = np.array(model["Forecast"])[context_length:]
        forecasts.loc[item, d_cols] = y_pred
        forecasts.loc[item,'alpha'] = alpha
        if (idx + 1) % 1000 == 0 or (idx+1)==30490:
            print(f'{idx+1}/30490 time series forecasted')
    return forecasts


# def CrostonRMSSEs2(ste, cal, prices, method='tsb'):
#     forecasts = Croston_Forecasts(ste, method=method)
#     w_df = create_weights(ste, cal, prices)
    


# def CrostonRMSSEs(ste,
#                   items=range(0,30490),
#                   method='tsb'):
#     rmsse = MeanSquaredScaledError(square_root=True)
#     _ = 0
#     RMSSEs = np.empty(len(items))
#     tss = np.array(ste.iloc[:, 6:])
#     for i in items:
#         # split data into train, valid and pred for use in rmsse metric
#         y_train = tss[i, :1913]
#         y_true = tss[i, 1913:]
#         if method == 'tsb':
#             __, model = Croston_least_squares(extra_periods=28, method='tsb',
#                                              initial_params=[0.1,0.1], ts=y_train)
#             y_pred = model["Forecast"][1913:]
#         elif method == 'std':
#             __, model = Croston_least_squares(extra_periods=28, method='std',
#                                              initial_params=[0.1,0.1], ts=y_train)
#             y_pred = model["Forecast"][1913:]
        
#         # store RMSSEs in array
#         RMSSEs[_] = rmsse(y_true, y_pred, y_train=y_train)
#         _ += 1
#     return RMSSEs