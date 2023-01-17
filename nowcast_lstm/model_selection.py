import math
import numpy as np
import pandas as pd

from nowcast_lstm.LSTM import LSTM


def AICc(n, n_regressors, SSE):
    return n * math.log(SSE / n) + (n + n_regressors) / (1 - (n_regressors + 2) / n)


def adj_metric(alpha, n, n_regressors, metric):
    """Penalize the performance metric based on the number of observations vs. regressors using a modified R2 penalization term

    parameters:
            :alpha: float: ϵ [0,1]. 0 implies no penalization for additional regressors, 1 implies most severe penalty for additional regressors

    output:
            :return: float: the adjusted/penalized performance metric
    """
    if alpha == 0.0:
        adjustment = 0.0
    elif (n - n_regressors - 1) > 0:
        adjustment = (((n - 1) / (n - n_regressors - 1)) - 1) * alpha
    else:
        raise ZeroDivisionError(
            "More input variables than observations. Try rerunning with `alpha=0`"
        )
    return metric * (1 + adjustment)


def gen_folds(data, n_folds=3, init_test_size=0.2):
    """Generate the last training indices for rolling folds. The size of successive test sets is reduced incrementally by init_test_size / n_folds.

    parameters:
            :data: pandas DataFrame: data, ordered temporally from earliest observation to latest
        :n_folds: int: how many folds to produce
        :init_test_size: float: what proportion of the data to use for testing at the first fold

    output:
            :return: list[int]: the last training indices of the folds
    """
    test_size_step = init_test_size / n_folds

    end_train_indices = []
    for fold in range(n_folds):
        train_end_row = round(len(data) * (1 - init_test_size + test_size_step * fold))
        end_train_indices.append(train_end_row)

    return end_train_indices


def univariate_order(
    data,
    target_variable,
    n_timesteps,
    fill_na_func=np.nanmean,
    fill_ragged_edges_func=np.nanmean,
    n_models=1,
    train_episodes=200,
    batch_size=30,
    decay=0.98,
    n_hidden=20,
    n_layers=2,
    dropout=0,
    criterion="",
    optimizer="",
    optimizer_parameters={"lr": 1e-2},
    n_folds=1,
    init_test_size=0.2,
    pub_lags=[],
    lags=[],
    performance_metric="RMSE",
    quiet=False,
):
    "univariate runs to determine order for additive variable selection"

    data = data.copy()

    # columns to assess, excluding date column and target variable
    columns = list(data.columns[data.columns != target_variable][1:])

    # initializing performance dictionary
    performance = dict.fromkeys(columns, [])

    # fold train indices
    end_train_indices = gen_folds(data, n_folds=n_folds, init_test_size=init_test_size)

    # defining RMSE and MAE
    if performance_metric == "RMSE":

        def performance_metric(preds, actuals):
            return np.sqrt(np.nanmean((preds - actuals) ** 2))

    elif performance_metric == "MAE":

        def performance_metric(preds, actuals):
            return np.nanmean(np.abs(preds - actuals))

    counter = 0
    for col in columns:
        if quiet == False:
            print(f"univariate stage: {counter} / {len(columns)} columns")
        counter += 1

        for fold in range(n_folds):
            train = data.loc[: end_train_indices[fold], ["date", target_variable, col]]

            n_obs = len(
                train.loc[~pd.isna(train[target_variable]), :].reset_index(drop=True)
            )  # only count non-missing target variables as an observation for the metric penalty

            # first date in the test set
            first_test_date = data.iloc[end_train_indices[fold] + 1, 0]

            model = LSTM(
                train,
                target_variable,
                n_timesteps,
                fill_na_func,
                fill_ragged_edges_func,
                n_models,
                train_episodes,
                batch_size,
                decay,
                n_hidden,
                n_layers,
                dropout,
                criterion,
                optimizer,
                optimizer_parameters,
            )
            model.train(quiet=True)

            # assess on full data performance for all cases
            test_set = data.loc[:, ["date", target_variable, col]]
            preds_df = model.predict(test_set, only_actuals_obs=True)
            preds_df = preds_df.loc[
                preds_df.iloc[:, 0] >= first_test_date, :
            ].reset_index(drop=True)
            actuals = preds_df.actuals
            preds = preds_df.predictions

            if performance_metric == "AICc":
                performance[col] = performance[col] + [
                    AICc(n_obs, 1, np.sum((preds - actuals) ** 2))
                ]
            else:
                performance[col] = performance[col] + [
                    performance_metric(preds, actuals)
                ]

            # assessing on lags, if applicable
            if (len(pub_lags) > 0) & (len(lags) > 0):
                for lag in lags:
                    preds_df = model.ragged_preds(
                        [pub_lags[counter - 1]],
                        lag,
                        test_set,
                        start_date=first_test_date.strftime("%Y-%m-%d"),
                    )
                    actuals = preds_df.actuals
                    preds = preds_df.predictions

                    if performance_metric == "AICc":
                        performance[col] = performance[col] + [
                            AICc(n_obs, 1, np.sum((preds - actuals) ** 2))
                        ]
                    else:
                        performance[col] = performance[col] + [
                            performance_metric(preds, actuals)
                        ]

    # getting average perforomance over the folds
    for key in performance.keys():
        performance[key] = np.nanmean(performance[key])

    # return variables in order of best-performing to worst
    return [k for k, v in sorted(performance.items(), key=lambda item: item[1])]


def variable_selection(
    data,
    target_variable,
    n_timesteps,
    fill_na_func=np.nanmean,
    fill_ragged_edges_func=np.nanmean,
    n_models=1,
    train_episodes=200,
    batch_size=30,
    decay=0.98,
    n_hidden=20,
    n_layers=2,
    dropout=0,
    criterion="",
    optimizer="",
    optimizer_parameters={"lr": 1e-2},
    n_folds=1,
    init_test_size=0.2,
    pub_lags=[],
    lags=[],
    performance_metric="RMSE",
    alpha=0.0,
    initial_ordering="feature_contribution",
    quiet=False,
):
    """Pick best-performing variables for a given set of hyperparameters

    parameters:
        All parameters up to `optimizer_parameters` exactly the same as for any LSTM() model
        :n_folds: int: how many folds for rolling fold validation to do
        :init_test_size: float: ϵ [0,1]. What proportion of the data to use for testing at the first fold
        :pub_lags: list[int]: list of periods back each input variable is set to missing. I.e. publication lag of the variable. Leave empty to pick variables only on complete information, no synthetic vintages.
        :lags: list[int]: simulated periods back to test when selecting variables. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc. So [-2, 0, 2] will account for those vintages in model selection. Leave empty to pick variables only on complete information, no synthetic vintages.
        :performance_metric: performance metric to use for variable selection. Pass "RMSE" for root mean square error, "MAE" for mean absolute error, or "AICc" for correctd Akaike Information Criterion. Alternatively can pass a function that takes arguments of a pandas Series of predictions and actuals and returns a scalar. E.g. custom_function(preds, actuals).
        :alpha: float: ϵ [0,1]. 0 implies no penalization for additional regressors, 1 implies most severe penalty for additional regressors. Not used for "AICc" performance metric.
        :initial_ordering: str: ["feature_contribution", "univariate"]. How to get initial order of features to check additively. "feature_contribution" uses the feature contribution of one model, "univariate" calculates univariate models of all features and orders by performance metric. Feature contribution is about twice as fast.
        :quiet: bool: whether or not to print progress
    output:
        :return: tuple
            list[str]: list of best-performing column names
            float: performance metric of these variables (i.e. best performing)
    """

    data = data.copy()

    if initial_ordering == "univariate":
        column_order = univariate_order(
            data,
            target_variable,
            n_timesteps,
            fill_na_func,
            fill_ragged_edges_func,
            n_models,
            train_episodes,
            batch_size,
            decay,
            n_hidden,
            n_layers,
            dropout,
            criterion,
            optimizer,
            optimizer_parameters,
            n_folds,
            init_test_size,
            pub_lags,
            lags,
            performance_metric,
            quiet,
        )
    elif initial_ordering == "feature_contribution":
        model = LSTM(
            data,
            target_variable,
            n_timesteps,
            fill_na_func,
            fill_ragged_edges_func,
            n_models,
            train_episodes,
            batch_size,
            decay,
            n_hidden,
            n_layers,
            dropout,
            criterion,
            optimizer,
            optimizer_parameters,
        )
        model.train(quiet=True)
        column_order = list(model.feature_contribution().feature.values)

    # columns to assess, excluding date column and target variable, used for pub_lags
    all_columns = list(data.columns[data.columns != target_variable][1:])

    # fold train indices
    end_train_indices = gen_folds(data, n_folds=n_folds, init_test_size=init_test_size)

    # defining RMSE and MAE
    if performance_metric == "RMSE":

        def performance_metric(preds, actuals):
            return np.sqrt(np.nanmean((preds - actuals) ** 2))

    elif performance_metric == "MAE":

        def performance_metric(preds, actuals):
            return np.nanmean(np.abs(preds - actuals))

    # final list of selected variables
    end_variables = [column_order[0]]
    performance = []  # storage of each run's performance to check if new run is better

    counter = 0
    for col in column_order:
        if quiet == False:
            print(f"multivariate stage: {counter} / {len(column_order)} columns")
        counter += 1

        col_performance = []  # storage of all fold/vintage performances

        for fold in range(n_folds):
            if col == end_variables[0]:
                train = data.loc[
                    : end_train_indices[fold], ["date", target_variable] + end_variables
                ]  # don't add the initial variable again
            else:
                train = data.loc[
                    : end_train_indices[fold],
                    ["date", target_variable] + end_variables + [col],
                ]  # end variables + this new one

            n_obs = len(
                train.loc[~pd.isna(train[target_variable]), :].reset_index(drop=True)
            )  # only count non-missing target variables as an observation for the metric penalty

            # first date in the test set
            first_test_date = data.iloc[end_train_indices[fold] + 1, 0]

            model = LSTM(
                train,
                target_variable,
                n_timesteps,
                fill_na_func,
                fill_ragged_edges_func,
                n_models,
                train_episodes,
                batch_size,
                decay,
                n_hidden,
                n_layers,
                dropout,
                criterion,
                optimizer,
                optimizer_parameters,
            )
            model.train(quiet=True)

            # assess on full data performance for all cases
            if col == end_variables[0]:
                test_set = data.loc[
                    :, ["date", target_variable] + end_variables
                ]  # don't add the initial variable again
            else:
                test_set = data.loc[
                    :, ["date", target_variable] + end_variables + [col]
                ]  # end variables + this new one
            preds_df = model.predict(test_set, only_actuals_obs=True)
            preds_df = preds_df.loc[
                preds_df.iloc[:, 0] >= first_test_date, :
            ].reset_index(drop=True)
            actuals = preds_df.actuals
            preds = preds_df.predictions

            if performance_metric == "AICc":
                col_performance.append(
                    AICc(n_obs, len(end_variables), np.sum((preds - actuals) ** 2))
                )
            else:
                col_performance.append(
                    adj_metric(
                        alpha,
                        n_obs,
                        len(end_variables),
                        performance_metric(preds, actuals),
                    )
                )

            # assessing on lags, if applicable
            if (len(pub_lags) > 0) & (len(lags) > 0):
                for lag in lags:
                    if col == end_variables[0]:
                        which_vars = (
                            end_variables  # don't add the initial variable again
                        )
                    else:
                        which_vars = end_variables + [
                            col
                        ]  # end variables + this new one
                    preds_df = model.ragged_preds(
                        [
                            pub_lags[i]
                            for i in [all_columns.index(x) for x in which_vars]
                        ],  # pub_lags of chosen variables
                        lag,
                        test_set,
                        start_date=first_test_date.strftime("%Y-%m-%d"),
                    )
                    actuals = preds_df.actuals
                    preds = preds_df.predictions
                    if performance_metric == "AICc":
                        col_performance.append(
                            AICc(
                                n_obs,
                                len(end_variables),
                                np.sum((preds - actuals) ** 2),
                            )
                        )
                    else:
                        col_performance.append(
                            adj_metric(
                                alpha,
                                n_obs,
                                len(end_variables),
                                performance_metric(preds, actuals),
                            )
                        )

        performance.append(np.nanmean(col_performance))

        # add this column to final list of performance improved over previous minimum
        if col != column_order[0]:  # only relevant if not first column
            if performance[-1] < np.min(performance[:-1]):
                end_variables = end_variables + [col]

    return end_variables, np.min(performance)


def hyperparameter_tuning(
    data,
    target_variable,
    n_models=1,
    n_timesteps_grid=[6, 12],
    fill_na_func_grid=[np.nanmean],
    fill_ragged_edges_func_grid=[np.nanmean],
    train_episodes_grid=[50, 100, 200],
    batch_size_grid=[30, 100, 200],
    decay_grid=[0.98],
    n_hidden_grid=[10, 20, 40],
    n_layers_grid=[1, 2, 4],
    dropout_grid=[0],
    criterion_grid=[""],
    optimizer_grid=[""],
    optimizer_parameters_grid=[{"lr": 1e-2}],
    n_folds=1,
    init_test_size=0.2,
    pub_lags=[],
    lags=[],
    performance_metric="RMSE",
    quiet=False,
):
    """Pick best-performing hyperparameters for a given dataset. n_timesteps_grid has default grid for predicting quarterly variable with monthly series, may have to change per use case. E.g. [12,24] for a yearly target with monthly indicators.

    parameters:
        All parameters up to `optimizer_parameters` exactly the same as for any LSTM() model, provide a list with the values to check
        :n_folds: int: how many folds for rolling fold validation to do
        :init_test_size: float: ϵ [0,1]. What proportion of the data to use for testing at the first fold
        :pub_lags: list[int]: list of periods back each input variable is set to missing. I.e. publication lag of the variable. Leave empty to pick variables only on complete information, no synthetic vintages.
        :lags: list[int]: simulated periods back to test when selecting variables. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc. So [-2, 0, 2] will account for those vintages in model selection. Leave empty to pick variables only on complete information, no synthetic vintages.
        :performance_metric: performance metric to use for variable selection. Pass "RMSE" for root mean square error or "MAE" for mean absolute error, "AICc" for corrected Akaike Information Criterion. Alternatively can pass a function that takes arguments of a pandas Series of predictions and actuals and returns a scalar. E.g. custom_function(preds, actuals).
        :quiet: bool: whether or not to print progress
    output:
        :return: Pandas DataFrame: hyperparameters sorted by best-performing model to least
    """

    data = data.copy()

    alpha = 0.0  # legacy to work with same variable selection code

    # fold train indices
    end_train_indices = gen_folds(data, n_folds=n_folds, init_test_size=init_test_size)

    # defining RMSE and MAE
    if performance_metric == "RMSE":

        def performance_metric(preds, actuals):
            return np.sqrt(np.nanmean((preds - actuals) ** 2))

    elif performance_metric == "MAE":

        def performance_metric(preds, actuals):
            return np.nanmean(np.abs(preds - actuals))

    # running list of hyperparameters
    end_params = []
    performance = []  # storage of each run's performance

    counter = 0
    n_runs = np.prod(
        [
            len(x)
            for x in [
                n_timesteps_grid,
                fill_na_func_grid,
                fill_ragged_edges_func_grid,
                train_episodes_grid,
                batch_size_grid,
                decay_grid,
                n_hidden_grid,
                n_layers_grid,
                dropout_grid,
                criterion_grid,
                optimizer_grid,
                optimizer_parameters_grid,
            ]
        ]
    )
    for i in [0]:  # for some reason first loop not working
        for fill_na_func in fill_na_func_grid:
            for n_timesteps in n_timesteps_grid:
                for fill_ragged_edges_func in fill_ragged_edges_func_grid:
                    for train_episodes in train_episodes_grid:
                        for batch_size in batch_size_grid:
                            for decay in decay_grid:
                                for n_hidden in n_hidden_grid:
                                    for n_layers in n_layers_grid:
                                        for dropout in dropout_grid:
                                            for criterion in criterion_grid:
                                                for optimizer in optimizer_grid:
                                                    for (
                                                        optimizer_parameters
                                                    ) in optimizer_parameters_grid:
                                                        if quiet == False:
                                                            print(
                                                                f"tuning: {counter} / {n_runs}"
                                                            )
                                                        counter += 1

                                                        run_performance = (
                                                            []
                                                        )  # storage of all fold/vintage performances

                                                        for fold in range(n_folds):
                                                            train = data.loc[
                                                                : end_train_indices[
                                                                    fold
                                                                ],
                                                                :,
                                                            ]

                                                            n_obs = len(
                                                                train.loc[
                                                                    ~pd.isna(
                                                                        train[
                                                                            target_variable
                                                                        ]
                                                                    ),
                                                                    :,
                                                                ].reset_index(drop=True)
                                                            )  # only count non-missing target variables as an observation for the metric penalty

                                                            # first date in the test set
                                                            first_test_date = data.iloc[
                                                                end_train_indices[fold]
                                                                + 1,
                                                                0,
                                                            ]

                                                            model = LSTM(
                                                                train,
                                                                target_variable,
                                                                n_timesteps,
                                                                fill_na_func,
                                                                fill_ragged_edges_func,
                                                                n_models,
                                                                train_episodes,
                                                                batch_size,
                                                                decay,
                                                                n_hidden,
                                                                n_layers,
                                                                dropout,
                                                                criterion,
                                                                optimizer,
                                                                optimizer_parameters,
                                                            )
                                                            model.train(quiet=True)

                                                            # assess on full data performance for all cases
                                                            test_set = data.copy()
                                                            preds_df = model.predict(
                                                                test_set,
                                                                only_actuals_obs=True,
                                                            )
                                                            preds_df = preds_df.loc[
                                                                preds_df.iloc[:, 0]
                                                                >= first_test_date,
                                                                :,
                                                            ].reset_index(drop=True)
                                                            actuals = preds_df.actuals
                                                            preds = preds_df.predictions

                                                            if (
                                                                performance_metric
                                                                == "AICc"
                                                            ):
                                                                run_performance.append(
                                                                    AICc(
                                                                        n_obs,
                                                                        len(
                                                                            data.columns
                                                                        )
                                                                        - 2,  # minus two for date column and target variable column
                                                                        np.sum(
                                                                            (
                                                                                preds
                                                                                - actuals
                                                                            )
                                                                            ** 2
                                                                        ),
                                                                    )
                                                                )
                                                            else:
                                                                run_performance.append(
                                                                    adj_metric(
                                                                        alpha,
                                                                        n_obs,
                                                                        len(
                                                                            data.columns
                                                                        )
                                                                        - 2,  # minus two for date column and target variable column
                                                                        performance_metric(
                                                                            preds,
                                                                            actuals,
                                                                        ),
                                                                    )
                                                                )

                                                            # assessing on lags, if applicable
                                                            if (len(pub_lags) > 0) & (
                                                                len(lags) > 0
                                                            ):
                                                                for lag in lags:
                                                                    preds_df = model.ragged_preds(
                                                                        pub_lags,
                                                                        lag,
                                                                        test_set,
                                                                        start_date=first_test_date.strftime(
                                                                            "%Y-%m-%d"
                                                                        ),
                                                                    )
                                                                    actuals = (
                                                                        preds_df.actuals
                                                                    )
                                                                    preds = (
                                                                        preds_df.predictions
                                                                    )
                                                                    if (
                                                                        performance_metric
                                                                        == "AICc"
                                                                    ):
                                                                        run_performance.append(
                                                                            AICc(
                                                                                n_obs,
                                                                                len(
                                                                                    data.columns
                                                                                )
                                                                                - 2,  # minus two for date column and target variable column
                                                                                np.sum(
                                                                                    (
                                                                                        preds
                                                                                        - actuals
                                                                                    )
                                                                                    ** 2
                                                                                ),
                                                                            )
                                                                        )
                                                                    else:
                                                                        run_performance.append(
                                                                            adj_metric(
                                                                                alpha,
                                                                                n_obs,
                                                                                len(
                                                                                    data.columns
                                                                                )
                                                                                - 2,  # minus two for date column and target variable column
                                                                                performance_metric(
                                                                                    preds,
                                                                                    actuals,
                                                                                ),
                                                                            )
                                                                        )

                                                        performance.append(
                                                            np.nanmean(run_performance)
                                                        )

                                                        # dict of params in model
                                                        tmp_end_params = {
                                                            "n_models": n_models,
                                                            "n_timesteps": n_timesteps,
                                                            "fill_na_func": fill_na_func,
                                                            "fill_ragged_edges_func": fill_ragged_edges_func,
                                                            "train_episodes": train_episodes,
                                                            "batch_size": batch_size,
                                                            "decay": decay,
                                                            "n_hidden": n_hidden,
                                                            "n_layers": n_layers,
                                                            "dropout": dropout,
                                                            "criterion": criterion,
                                                            "optimizer": optimizer,
                                                            "optimizer_parameters": optimizer_parameters,
                                                        }

                                                        end_params.append(
                                                            tmp_end_params
                                                        )

        results = pd.DataFrame({"hyper_params": end_params, "performance": performance})
        results = results.sort_values(["performance"]).reset_index(drop=True)

        return results


def select_model(
    data,
    target_variable,
    n_models=1,
    n_timesteps_grid=[12],
    fill_na_func_grid=[np.nanmean],
    fill_ragged_edges_func_grid=[np.nanmean],
    train_episodes_grid=[200],
    batch_size_grid=[50],
    decay_grid=[0.98],
    n_hidden_grid=[20],
    n_layers_grid=[2],
    dropout_grid=[0],
    criterion_grid=[""],
    optimizer_grid=[""],
    optimizer_parameters_grid=[{"lr": 1e-2}],
    n_folds=1,
    init_test_size=0.2,
    pub_lags=[],
    lags=[],
    performance_metric="RMSE",
    alpha=0.0,
    initial_ordering="feature_contribution",
    quiet=False,
):
    """Pick best-performing hyperparameters and variables for a given dataset. Given all permutations of hyperparameters (k), and p variables in the data,
    this function will run k * p * 2 models if `initial_ordering` set to "univariate", k * p if set to default of `feature_contribution`. This can take a very long time. To cut down on this time, run it with a highly reduced hyperparameter grid,
    i.e., a very small k, then record the selected variables, then run the `hyperparameter_tuning` function with these selected variables
    with a much more detailed grid.

    parameters:
        All parameters up to `optimizer_parameters` exactly the same as for any LSTM() model, provide a list with the values to check
        :n_folds: int: how many folds for rolling fold validation to do
        :init_test_size: float: ϵ [0,1]. What proportion of the data to use for testing at the first fold
        :pub_lags: list[int]: list of periods back each input variable is set to missing. I.e. publication lag of the variable. Leave empty to pick variables only on complete information, no synthetic vintages.
        :lags: list[int]: simulated periods back to test when selecting variables. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc. So [-2, 0, 2] will account for those vintages in model selection. Leave empty to pick variables only on complete information, no synthetic vintages.
        :performance_metric: performance metric to use for variable selection. Pass "RMSE" for root mean square error or "MAE" for mean absolute error, "AICc" for corrected Akaike Information Criterion. Alternatively can pass a function that takes arguments of a pandas Series of predictions and actuals and returns a scalar. E.g. custom_function(preds, actuals).
        :alpha: float: ϵ [0,1]. 0 implies no penalization for additional regressors, 1 implies most severe penalty for additional regressors.
        :initial_ordering: str: ["feature_contribution", "univariate"]. How to get initial order of features to check additively. "feature_contribution" uses the feature contribution of one model, "univariate" calculates univariate models of all features and orders by performance metric. Feature contribution is about twice as fast.
        :quiet: bool: whether or not to print progress
    output:
        :return: Pandas DataFrame: hyperparameters and variables sorted by best-performing model to least
    """
    performance = []
    variables = []
    hyperparams = []

    counter = 0
    n_runs = np.prod(
        [
            len(x)
            for x in [
                n_timesteps_grid,
                fill_na_func_grid,
                fill_ragged_edges_func_grid,
                train_episodes_grid,
                batch_size_grid,
                decay_grid,
                n_hidden_grid,
                n_layers_grid,
                dropout_grid,
                criterion_grid,
                optimizer_grid,
                optimizer_parameters_grid,
            ]
        ]
    )

    for i in [0]:  # for some reason first loop not working
        for fill_na_func in fill_na_func_grid:
            for n_timesteps in n_timesteps_grid:
                for fill_ragged_edges_func in fill_ragged_edges_func_grid:
                    for train_episodes in train_episodes_grid:
                        for batch_size in batch_size_grid:
                            for decay in decay_grid:
                                for n_hidden in n_hidden_grid:
                                    for n_layers in n_layers_grid:
                                        for dropout in dropout_grid:
                                            for criterion in criterion_grid:
                                                for optimizer in optimizer_grid:
                                                    for (
                                                        optimizer_parameters
                                                    ) in optimizer_parameters_grid:
                                                        if quiet == False:
                                                            print(
                                                                f"tuning: {counter} / {n_runs}"
                                                            )
                                                        counter += 1

                                                        (
                                                            selected_variables,
                                                            run_performance,
                                                        ) = variable_selection(
                                                            data,
                                                            target_variable,
                                                            n_timesteps,
                                                            fill_na_func,
                                                            fill_ragged_edges_func,
                                                            n_models,
                                                            train_episodes,
                                                            batch_size,
                                                            decay,
                                                            n_hidden,
                                                            n_layers,
                                                            dropout,
                                                            criterion,
                                                            optimizer,
                                                            optimizer_parameters,
                                                            n_folds,
                                                            init_test_size,
                                                            pub_lags,
                                                            lags,
                                                            performance_metric,
                                                            alpha,
                                                            initial_ordering,
                                                            quiet,
                                                        )

                                                        # dict of params in model
                                                        tmp_end_params = {
                                                            "n_models": n_models,
                                                            "n_timesteps": n_timesteps,
                                                            "fill_na_func": fill_na_func,
                                                            "fill_ragged_edges_func": fill_ragged_edges_func,
                                                            "train_episodes": train_episodes,
                                                            "batch_size": batch_size,
                                                            "decay": decay,
                                                            "n_hidden": n_hidden,
                                                            "n_layers": n_layers,
                                                            "dropout": dropout,
                                                            "criterion": criterion,
                                                            "optimizer": optimizer,
                                                            "optimizer_parameters": optimizer_parameters,
                                                        }

                                                        hyperparams.append(
                                                            tmp_end_params
                                                        )
                                                        performance.append(
                                                            run_performance
                                                        )
                                                        variables.append(
                                                            selected_variables
                                                        )

    results = (
        pd.DataFrame(
            {
                "variables": variables,
                "hyperparameters": hyperparams,
                "performance": performance,
            }
        )
        .sort_values(["performance"])
        .reset_index(drop=True)
    )
    return results
