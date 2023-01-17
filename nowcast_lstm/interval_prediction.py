import math
import numpy as np
import pandas as pd


# normal distribution CDF without scipy, from https://gist.github.com/luhn/2649874
def erfcc(x):
    """Complementary error function."""
    z = abs(x)
    t = 1. / (1. + 0.5*z)
    r = t * math.exp(-z*z-1.26551223+t*(1.00002368+t*(.37409196+
        t*(.09678418+t*(-.18628806+t*(.27886807+
        t*(-1.13520398+t*(1.48851587+t*(-.82215223+
        t*.17087277)))))))))
    if (x >= 0.):
        return r
    else:
        return 2. - r

def cdf(x):
    """Cumulative normal distribution function."""
    return 1. - 0.5*erfcc(x/(2**0.5))

def calc_sds(x):
    """calculate number of standard deviations for x% of data to be covered in a normal distribution"""
    key = pd.DataFrame({
        "sds": np.array(range(400)) / 100,
        "percs": [1 - ((1 - cdf(i/100)) * 2) for i in range(400)]
    })
    sds = key.iloc[(key["percs"]-x).abs().argsort()[0]].sds
    return (sds)

def calc_perc_available(model, data, target_period):
    """Calculate the proportion of data available to the model, ratio of actually available data versus theoretical maximum data available
    parameters:
        :model: LSTM.LSTM: trained LSTM model
        :data: pandas DataFrame: data to predict fitted model on
        :target_period: target period, how much data is available can depend on the target period
    output: 
        :return: float of proportion of data available versus theoretical maximum
    """
    # number of unique periods (months) in the total dataset
    n_unique_total = len(np.unique([str(x)[5:] for x in model.data.loc[:, "date"]]))
    
    # only data before target peiod
    tmp = data.loc[data.date <= target_period, :].reset_index(drop=True)
    
    n_max = 0 # max number of datapoints if all were present
    n_available = 0 # number actually present in the data
    for col in tmp.columns:
        if not(col in ["date", model.target_variable]):
            # unique months for this column
            try: 
                n_unique_col = len(np.unique([str(x)[5:] for x in model.data.loc[~pd.isna(data[col]), "date"]]))
            except:
                # if no data for this time period, just assume periodicity of max of the data
                n_unique_col = n_unique_total
            # divide by this adjustment to know how many values that variable should have given its periodicity
            adjustment = int(n_unique_total / n_unique_col)

            col_max = int(model.n_timesteps / adjustment) # max data this column could have           
            col_available = sum(~pd.isna(tmp.iloc[-(model.n_timesteps):,:].loc[:, col].values)) # number of obvs actually available

            n_max += col_max
            n_available += col_available
            
    return (n_available / n_max)

def single_interval_predict(
        sample, # list of points predictions of each of the models
        sds, # how many sds the chosen interval has
        target_sd, # sd of the target variable
        availability, # of data available to the model
        interval # uncertainty interval, e.g., 95%
):
    point_pred = np.mean(sample)
    lower_bound = np.quantile(sample, 0+(1-interval)/2)
    upper_bound = np.quantile(sample, 1-(1-interval)/2)
    
    adjustment = target_sd * sds * (1 - availability)
    
    lower_adj = lower_bound - adjustment
    upper_adj = upper_bound + adjustment
    
    return (point_pred, lower_adj, upper_adj)