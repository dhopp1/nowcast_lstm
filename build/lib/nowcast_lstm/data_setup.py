import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


data = pd.read_csv(
    "/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/2020-08-07_database_tf.csv"
)
rawdata = data.loc[:, ["date", "x_world", "x_cn", "bci_jp", "fc_x_de"]]


def gen_dataset(rawdata, target_variable):
    """Intermediate step to generate a raw dataset the model will accept
	Input should be a pandas dataframe of of (n observations) x (m features + 1 target column). Non-numeric columns will be dropped, missing values replaced by 0.
	The data should be fed in in the time of the most granular series. E.g. 3 monthly series and 2 quarterly should be given as a monthly dataframe, with NAs for the two intervening months for the quarterly variables. Apply the same logic to yearly  or daily variables (untested).
	
	parameters:
		:rawdata: pandas DataFrame: n x m+1 dataframe
        :target_variable: str: name of the target variable column
	
	output:
		:return: numpy array: n x m+1 array
	"""

    rawdata = rawdata.fillna(0.0)  # fill nas with 0's
    # converting all columns to float64 if numeric
    for col in rawdata.columns:
        if is_numeric_dtype(rawdata[col]):
            rawdata[col] = rawdata[col].astype("float64")
    rawdata = rawdata.loc[
        :, [x == "float64" for x in rawdata.dtypes]
    ]  # only keep numeric columns
    variables = list(
        rawdata.columns[rawdata.columns != target_variable]
    )  # features, excluding target variable

    data_dict = {}
    for variable in variables:
        data_dict[variable] = np.array(rawdata.loc[:, variable])
        data_dict[variable] = data_dict[variable].reshape((len(data_dict[variable]), 1))
    target = np.array(rawdata.loc[:, target_variable])
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
        :lag: int: simulated periods back. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc.
	
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
