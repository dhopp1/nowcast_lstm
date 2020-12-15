import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def convert_float(rawdata):
    # converting all columns to float64 if numeric
    for col in rawdata.columns:
        if is_numeric_dtype(rawdata[col]):
            rawdata[col] = rawdata[col].astype("float64")
    rawdata = rawdata.loc[
        :, [x == "float64" for x in rawdata.dtypes]
    ].copy()  # only keep numeric columns
    return rawdata

def gen_dataset(rawdata, target_variable, fill_na_func=np.mean, fill_na_other_df=None):
    """Intermediate step to generate a raw dataset the model will accept
	Input should be a pandas dataframe of of (n observations) x (m features + 1 target column). Non-numeric columns will be dropped, missing values replaced by the fill_na_func.
	The data should be fed in in the time of the most granular series. E.g. 3 monthly series and 2 quarterly should be given as a monthly dataframe, with NAs for the two intervening months for the quarterly variables. Apply the same logic to yearly  or daily variables (untested).
	
	parameters:
		:rawdata: pandas DataFrame: n x m+1 dataframe
        :target_variable: str: name of the target variable column
        :fill_na_func: function: a column-wise function to replace NAs with (e.g. np.mean, np.nanmedian). Can also be a lambda function for replacing NAs with a scalar, `fill_na_func=lambda x: -999`
        :fill_na_other_df: pandas DataFrame: A dataframe with the exact same columns as the rawdata dataframe. For use with filling NAs based on a different dataset (e.g. the train dataset). E.g. `train=LSTM(...)`, `gen_dataset(test_data, target_variable, fill_na_other_df=train.data)`
	
	output:
		:return: numpy array: n x m+1 array
	"""
    
    rawdata = convert_float(rawdata)
    # to get fill_na values based on either this dataframe or another (training)
    if fill_na_other_df is not None:
        fill_na_df = convert_float(fill_na_other_df)
    else:
        fill_na_df = rawdata
    
    variables = list(
        rawdata.columns[rawdata.columns != target_variable]
    )  # features, excluding target variable
    
    # fill nas with a function
    for col in rawdata.columns[rawdata.columns != target_variable]: # leave target as NA
        rawdata.loc[pd.isna(rawdata[col]), col] = fill_na_func(fill_na_df[col])
    
    # returning array, target variable at the end
    data_dict = {}
    for variable in variables:
        data_dict[variable] = rawdata.loc[:, variable].values
        data_dict[variable] = data_dict[variable].reshape((len(data_dict[variable]), 1))
    target = rawdata.loc[:, target_variable].values
    target = target.reshape((len(target), 1))
    dataset = np.hstack(([data_dict[k] for k in data_dict] + [target]))
    return dataset


def gen_model_input(dataset, n_timesteps, drop_missing_ys=True):
    """Final step in generating a dataset the model will accept
	Input should be output of the `gen_dataset` function. Creates two series, X for input and y for target. 
	y is a one-dimensional np array equivalent to a list of target values. 
	X is an n x n_timesteps x m matrix. 
	Essentially the input data for each test observation becomes an n_steps x m matrix instead of a single row of data. In this way the LSTM network can learn from each variables past, not just its current value.
	Observations that don't have enough n_steps history will be dropped.
	
	parameters:
		:dataset: numpy array: n x m+1 array
		:n_timesteps: int: how many historical periods to consider when training the model. For example if the original data is monthly, n_timesteps=12 would consider data for the last year.
        :drop_missing_ys: boolean: whether or not to filter out missing ys. Set to true when creating training data, false when want to run predictions on data that may not have a y.
	
	output:
		:return: numpy tuple of:
			X: `n_obs x n_timesteps x n_features`
			y: `n_obs`
	"""

    X, y = list(), list()
    for i in range(len(dataset)):
        # find the end of this pattern
        end_ix = i + n_timesteps
        # check if we are beyond the dataset
        if end_ix > len(dataset):
            break
            # gather input and output parts of the pattern
        seq_x, seq_y = dataset[i:end_ix, :-1], dataset[end_ix - 1, -1]
        X.append(seq_x)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)
    if drop_missing_ys:
        X = X[y != 0.0, :, :]  # delete na ys, no useful training data
        y = y[y != 0.0]

    return X, y


def gen_ragged_X(X, pub_lags, lag):
    """Produce vintage model inputs given the period lag of different variables, for use when testing historical performance (model evaluation, etc.)
	
	parameters:
		:X: numpy array: n x m+1 array, output of `gen_model_input` function, or an instantiated LSTM object, `LSTM().X`
		:pub_lags: list[int]: list of periods back each input variable is set to missing. I.e. publication lag of the variable.
        :lag: int: simulated periods back, interpretable as last complete period relative to target period. E.g. -2 = simulating data as it would have been 1 month before target period, i.e. 2 months ago is last complete period. 
	
	output:
		:return: numpy array equivalent in shape to X input, but with trailing edges set to missing/0
	"""

    X_ragged = np.array(X)
    for obs in range(X_ragged.shape[0]):  # go through every observation
        for var in range(len(pub_lags)):  # every variable (and its corresponding lag)
            for ragged in range(
                1, pub_lags[var] + 1 - lag
            ):  # setting correct lags (-lag because input -2 for -2 months, so +2 additional months of lag)
                X_ragged[
                    obs, X_ragged.shape[1] - ragged, var
                ] = 0.0  # setting to missing data

    return X_ragged
