import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_datetime64_any_dtype
from pmdarima.arima import auto_arima, ARIMA


def convert_float(rawdata):
    result = rawdata.copy()
    # converting all columns to float64 if numeric
    for col in result.columns:
        if is_numeric_dtype(result[col]):
            result[col] = result[col].astype("float64")
    result = result.loc[
        :, [x == "float64" for x in result.dtypes]
    ].copy()  # only keep numeric columns
    return result


def estimate_arma(series):
    """Estimate ARMA parameters on a series"""
    series = pd.Series(series)
    series = series[~pd.isna(series)]
    arma_model = auto_arima(
        series,
        start_p=0,
        d=0,
        start_q=0,
        D=0,
        stationary=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    return arma_model


def ragged_fill_series(
    series,
    function=np.nanmean,
    backup_fill_method=np.nanmean,
    est_series=None,
    fitted_arma=None,
    arma_full_series=None,
):
    """Filling in the ragged ends of a series, adhering to the periodicity of the series. If there is only one observation and periodicity cannot be determined, series will be returned unchanged.
	
	parameters:
		:series: list/pandas Series: the series to fill the ragged edges of. Missings should be np.nans
        :function: the function to fill nas with (e.g. np.nanmean, etc.). Use "ARMA" for ARMA filling
        :backup_fill_method: function: which function to fill ragged edges with in case ARMA can't be estimated
        :est_series: list/pandas Series: optional, the series to calculate the fillna and/or ARMA function on. Should not have nas filled in yet by any method. E.g. a train set. If None, will calculated based on itself.
        :fitted_arma: optional, fitted ARMA model if available to avoid reestimating every time in the `gen_ragged_X` function
        :arma_full_series: optional, for_full_arma_dataset output of `gen_dataset` function. Fitting the ARMA model on the full series history rather than just the series provided
	
	output:
		:return: pandas Series with filled ragged edges
	"""
    result = pd.Series(series).copy()
    if est_series is None:
        est_series = result.copy()

    # periodicity of the series, to see which to fill in
    nonna_bools = ~pd.isna(series)
    nonna_indices = list(nonna_bools.index[nonna_bools])  # existing indices with values
    # if there is only one non-na observation, can't determine periodicity or position in full series, don't fill anything
    if len(nonna_indices) > 1:
        periodicity = int(
            (
                pd.Series(result[~pd.isna(result)].index)
                - (pd.Series(result[~pd.isna(result)].index)).shift()
            ).mode()[0]
        )  # how often data comes (quarterly, monthly, etc.)
        last_nonna = result.index[result.notna()][-1]
        fill_indices = nonna_indices + [
            int(nonna_indices[-1] + periodicity * i)
            for i in range(1, (len(series) - last_nonna))
        ]  # indices to be filled in, including only the correct periodicity
        fill_indices = [
            x for x in fill_indices if x in series.index
        ]  # cut down on the indices if went too long

        if function == "ARMA":
            # estimate the model if not given
            if fitted_arma is None:
                fitted_arma = estimate_arma(est_series)
            # instantiate model with previously estimated parameters (i.e. on train set)
            arma = ARIMA(order=fitted_arma.order)
            arma.set_params(**fitted_arma.get_params())

            # refit the model on the full series to this point
            if arma_full_series is not None:
                y = list(arma_full_series[~pd.isna(arma_full_series)])
                present = list(result[~pd.isna(result)])
                # limit the series to the point where actuals are
                end_index = 0
                for i in range(len(present), len(y) + 1):
                    if list(y[(i - len(present)) : i]) == list(present):
                        end_index = i
                y = y[:end_index]
            # refit model on just this series
            else:
                y = list(result[~pd.isna(result)])  # refit the model on data
                present = y.copy()
            # can fail if not enough datapoints for order of ARMA process
            try:
                arma.fit(y, error_action="ignore")
                preds = arma.predict(n_periods=int(len(series) - last_nonna))
                fills = list(present) + list(preds)
                fills = fills[: len(fill_indices)]
            except:
                fills = list(result[~pd.isna(result)]) + [
                    backup_fill_method(est_series)
                ] * (len(series) - last_nonna)
                fills = fills[: len(fill_indices)]
            result[fill_indices] = fills
        else:
            fills = list(result[~pd.isna(result)]) + [function(est_series)] * (
                len(series) - last_nonna
            )
            fills = fills[: len(fill_indices)]
            result[fill_indices] = fills

    return result


def gen_dataset(
    rawdata,
    target_variable,
    fill_na_func=np.nanmean,
    fill_ragged_edges=None,
    fill_na_other_df=None,
    arma_full_df=None,
):
    """Intermediate step to generate a raw dataset the model will accept
	Input should be a pandas dataframe of of (n observations) x (m features + 1 target column). Non-numeric columns will be dropped. Missing values should be `np.nan`s.
	The data should be fed in in the time of the most granular series. E.g. 3 monthly series and 2 quarterly should be given as a monthly dataframe, with NAs for the two intervening months for the quarterly variables. Apply the same logic to yearly or daily variables.
	
	parameters:
		:rawdata: pandas DataFrame: n x m+1 dataframe
        :target_variable: str: name of the target variable column
        :fill_na_func: function: function to replace within-series NAs. Given the column, the function should return a scalar. 
        :fill_ragged_edges: function to replace NAs in ragged edges (data missing at end of series). Pass "ARMA" for ARMA filling
        :fill_na_other_df: pandas DataFrame: A dataframe with the exact same columns as the rawdata dataframe. For use with filling NAs based on a different dataset (e.g. the train dataset). E.g. `train=LSTM(...)`, `gen_dataset(test_data, target_variable, fill_na_other_df=train.data)`
        :arma_full_df: pandas DataFrame: A dataframe with the exact same columns as the rawdata dataframe. For use with ARMA filling on a full-series history, rather than just the history present in the train set
	
	output:
		:return: Dict of numpy arrays: n x m+1 arrays.
            na_filled_dataset: NA filled dataset the model will be trained on
            for_ragged_dataset: dataset with NAs maintained, for knowing periodicity in `gen_ragged_X` function
            for_full_arma_dataset: optional, full dataset for calculating ARMA on full history of series
            other_dataset: other dataset (i.e. train) on which to base NA filling
	"""
    if fill_ragged_edges is None:
        fill_ragged_edges = fill_na_func

    date_series = rawdata[
        [
            column
            for column in rawdata.columns
            if is_datetime64_any_dtype(rawdata[column])
        ]
    ]
    rawdata = convert_float(rawdata)
    # to get fill_na values based on either this dataframe or another (training)
    if fill_na_other_df is None:
        fill_na_df = rawdata.copy()
    else:
        fill_na_df = convert_float(fill_na_other_df)

    variables = list(
        rawdata.columns[rawdata.columns != target_variable]
    )  # features, excluding target variable

    # fill NAs with a function
    for_ragged = rawdata.copy()  # needs to be kept for generating ragged data
    for col in rawdata.columns[
        rawdata.columns != target_variable
    ]:  # leave target as NA
        # ragged edges
        if arma_full_df is not None:
            rawdata[col] = ragged_fill_series(
                rawdata[col],
                function=fill_ragged_edges,
                est_series=fill_na_df[col],
                arma_full_series=arma_full_df[col],
            )
        else:
            rawdata[col] = ragged_fill_series(
                rawdata[col], function=fill_ragged_edges, est_series=fill_na_df[col]
            )

        # within-series missing
        rawdata[col] = rawdata[col].fillna(fill_na_func(fill_na_df[col]))

    # drop any rows still with missing X data, in case fill_na_func doesn't get full coverage
    rawdata = rawdata.loc[rawdata.loc[:, variables].dropna().index, :].reset_index(
        drop=True
    )
    for_ragged = for_ragged.loc[
        rawdata.loc[:, variables].dropna().index, :
    ].reset_index(drop=True)

    # returning array, target variable at the end
    def order_dataset(rawdata, variables, target_variable):
        data_dict = {}
        for variable in variables:
            data_dict[variable] = rawdata.loc[:, variable].values
            data_dict[variable] = data_dict[variable].reshape(
                (len(data_dict[variable]), 1)
            )
        target = rawdata.loc[:, target_variable].values
        target = target.reshape((len(target), 1))
        dataset = np.hstack(([data_dict[k] for k in data_dict] + [target]))
        return dataset

    # final datasets
    dataset = order_dataset(rawdata, variables, target_variable)
    for_ragged_dataset = order_dataset(for_ragged, variables, target_variable)
    if arma_full_df is not None:
        for_arma_full = order_dataset(
            convert_float(arma_full_df), variables, target_variable
        )
    else:
        for_arma_full = None
    fill_na_other = order_dataset(fill_na_df, variables, target_variable)

    return {
        "na_filled_dataset": dataset,
        "for_ragged_dataset": for_ragged_dataset,
        "for_full_arma_dataset": for_arma_full,
        "other_dataset": fill_na_other,
        "date_series": date_series,
    }


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
        X = X[~pd.isna(y), :, :]  # delete na ys, no useful training data
        y = y[~pd.isna(y)]

    return X, y


def gen_ragged_X(
    X,
    pub_lags,
    lag,
    for_ragged_dataset,
    target_variable,
    fill_ragged_edges=np.nanmean,
    backup_fill_method=np.nanmean,
    other_dataset=None,
    for_full_arma_dataset=None,
):
    """Produce vintage model inputs given the period lag of different variables, for use when testing historical performance (model evaluation, etc.)
	
	parameters:
		:X: numpy array: n x m+1 array, second output of `gen_model_input` function, `for_ragged_dataset`, passed through the `gen_model_input` function
		:pub_lags: list[int]: list of periods back each input variable is set to missing. I.e. publication lag of the variable.
        :lag: int: simulated periods back, interpretable as last complete period relative to target period. E.g. -2 = simulating data as it would have been 1 month before target period, i.e. 2 months ago is last complete period. 
        :for_ragged_dataset: numpy array: the original full ragged dataset, output of `gen_dataset` function, `for_ragged_dataset`
        :target_variable: str: the target variable of this dataset
        :fill_ragged_edges: function: which function to fill ragged edges with, "ARMA" for ARMA model
        :backup_fill_method: function: which function to fill ragged edges with in case ARMA can't be estimated. Should be the same as originally passed to `gen_dataset` function
        :other_dataset: numpy array: other dataframe from which to calculate the fill NA values, i.e. a training dataset. Output of `gen_dataset` function, `other_dataset`
        :for_full_arma_dataset: numpy array: 
	
	output:
		:return: numpy array equivalent in shape to X input, but with trailing edges set to NA then filled
	"""
    # to get fill_na values based on either this dataframe or another (training)
    if other_dataset is None:
        fill_na_dataset = for_ragged_dataset
    else:
        fill_na_dataset = other_dataset

    # if no ragged edges fill provided, just do same as backup method
    if fill_ragged_edges is None:
        fill_ragged_edges = backup_fill_method

    # estimating ARMA models just once per variable instead of every observation
    if fill_ragged_edges == "ARMA":
        arma_models = []
        for var in range(X.shape[2]):
            arma_models = arma_models + [estimate_arma(fill_na_dataset[:, var])]

    # clearing ragged data
    X_ragged = np.array(X)
    for obs in range(X_ragged.shape[0]):  # go through every observation
        for var in range(len(pub_lags)):  # every variable (and its corresponding lag)
            for ragged in range(
                1, pub_lags[var] + 1 - lag
            ):  # setting correct lags (-lag because input -2 for -2 months, so +2 additional months of lag)
                X_ragged[
                    obs, X_ragged.shape[1] - ragged, var
                ] = np.nan  # setting to missing data
            if fill_ragged_edges == "ARMA":
                # pass the full ARMA series if available
                if for_full_arma_dataset is None:
                    X_ragged[obs, :, var] = ragged_fill_series(
                        pd.Series(X_ragged[obs, :, var]),
                        function=fill_ragged_edges,
                        backup_fill_method=backup_fill_method,
                        est_series=fill_na_dataset[:, var],
                        fitted_arma=arma_models[var],
                    )
                else:
                    X_ragged[obs, :, var] = ragged_fill_series(
                        pd.Series(X_ragged[obs, :, var]),
                        function=fill_ragged_edges,
                        backup_fill_method=backup_fill_method,
                        est_series=fill_na_dataset[:, var],
                        fitted_arma=arma_models[var],
                        arma_full_series=for_full_arma_dataset[:, var],
                    )
            else:
                X_ragged[obs, :, var] = ragged_fill_series(
                    pd.Series(X_ragged[obs, :, var]),
                    function=fill_ragged_edges,
                    est_series=fill_na_dataset[:, var],
                )
            X_ragged[obs, :, var] = pd.Series(X_ragged[obs, :, var]).fillna(
                backup_fill_method(fill_na_dataset[:, var])
            )

    return X_ragged
