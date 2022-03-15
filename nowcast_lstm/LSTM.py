from importlib import import_module
import numpy as np
import pandas as pd
import datetime

import nowcast_lstm.data_setup
import nowcast_lstm.modelling


class LSTM:
    """Primary class of the library, used for transforming data, training the model, and making predictions.
    `model = LSTM()`
    `model.train()` to train the model
    `model.X` to see model inputs
    `model.y` to see actual ys
    `model.predict(model.X)` to get predictions on the train set
    `model.predict(LSTM(new_data, target, n_timesteps, False).X)` to test on a totally new set of data
    `model.mv_lstm` to get a list of n_models length of torch networks
    `model.train_loss` to get a list of lists (len = n_models) of training losses per epoch
    `model.gen_ragged_X(pub_lags, lag)` to generate a data vintage of X model input, useful for evaluation, how would this model have performed historically with missing data.


        parameters:
        :data: pandas DataFrame: n x m+1 dataframe
        :target_variable: str: name of the target var
        :n_timesteps: how many historical periods to consider when training the model. For example if the original data is monthly, n_steps=12 would consider data for the last year.
        :fill_na_func: function: function to replace within-series NAs. Given a column, the function should return a scalar.
        :fill_ragged_edges_func: function to replace NAs in ragged edges (data missing at end of series). Pass "ARMA" for ARMA filling. Not ARMA filling will be significantly slower as models have to be estimated for each variable to fill ragged edges.
        :n_models: int: number of models to train and take the average of for more robust estimates
        :train_episodes: int: number of epochs/episodes to train the model
        :batch_size: int: number of observations per training batch
        :decay: float: learning rate decay
        :n_hidden: int: number of hidden states in the network
        :n_layers: int: number of LSTM layers in the network
        :dropout: float: dropout rate between the LSTM layers
        :criterion: torch loss criterion, defaults to MAE
        :optimizer: torch optimizer, defaults to Adam
        :optimizer_parameters: dictionary: list of parameters for optimizer, including learning rate. E.g. {"lr": 1e-2}
    """

    def __init__(
        self,
        data,
        target_variable,
        n_timesteps,
        fill_na_func=np.nanmean,
        fill_ragged_edges_func=None,
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
    ):
        self.data_setup = import_module("nowcast_lstm.data_setup")
        self.modelling = import_module("nowcast_lstm.modelling")

        self.data = data.reset_index(drop=True)
        self.target_variable = target_variable
        self.n_timesteps = n_timesteps

        self.fill_na_func = fill_na_func
        self.fill_ragged_edges_func = fill_ragged_edges_func

        self.train_episodes = train_episodes
        self.n_models = n_models

        self.batch_size = batch_size
        self.decay = decay
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer_parameters = optimizer_parameters

        self.dataset = self.data_setup.gen_dataset(
            self.data,
            self.target_variable,
            self.fill_na_func,
            self.fill_ragged_edges_func,
            fill_na_other_df=self.data,
            arma_full_df=self.data,
        )
        self.na_filled_dataset = self.dataset["na_filled_dataset"]
        self.for_ragged_dataset = self.dataset["for_ragged_dataset"]
        self.for_full_arma_dataset = self.dataset["for_full_arma_dataset"]
        self.other_dataset = self.dataset["other_dataset"]
        self.date_series = self.dataset["date_series"]
        self.arma_models = self.dataset["arma_models"]

        self.model_input = self.data_setup.gen_model_input(
            self.na_filled_dataset, self.n_timesteps, drop_missing_ys=True
        )
        self.X = self.model_input[0]
        self.y = self.model_input[1]

        self.ragged_input = self.data_setup.gen_model_input(
            self.for_ragged_dataset, self.n_timesteps, drop_missing_ys=True
        )
        self.ragged_X = self.ragged_input[0]

        self.mv_lstm = []
        self.train_loss = []

    def train(self, num_workers=0, shuffle=False, quiet=False):
        """train the model

        :num_workers: int: number of workers for multi-process data loading
        :shuffle: boolean: whether to shuffle data at every epoch
                :quiet: boolean: whether or not to print the losses in the epoch loop
        """
        for i in range(self.n_models):
            if quiet == False:
                print(f"Training model {i+1}")
            # instantiate the model
            instantiated = self.modelling.instantiate_model(
                self.X,
                n_timesteps=self.n_timesteps,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                dropout=self.dropout,
                criterion=self.criterion,
                optimizer=self.optimizer,
                optimizer_parameters=self.optimizer_parameters,
            )
            mv_lstm = instantiated["mv_lstm"]
            criterion = instantiated["criterion"]
            optimizer = instantiated["optimizer"]
            # train the model
            trained = self.modelling.train_model(
                self.X,
                self.y,
                mv_lstm,
                criterion,
                optimizer,
                train_episodes=self.train_episodes,
                batch_size=self.batch_size,
                decay=self.decay,
                num_workers=num_workers,
                shuffle=shuffle,
                quiet=quiet,
            )
            self.mv_lstm.append(trained["mv_lstm"])
            self.train_loss.append(trained["train_loss"])

    def predict(self, data, only_actuals_obs=False):
        """Make predictions on a dataframe with same columns and order as the model was trained on, including the target variable.

        parameters:
                :data: pandas DataFrame: data to predict fitted model on
            :only_actuals_obs: boolean: whether or not to predict observations without a target actual

        output:
                :return: pandas DataFrame of dates and predictions
        """
        dataset = self.data_setup.gen_dataset(
            data,
            self.target_variable,
            self.fill_na_func,
            self.fill_ragged_edges_func,
            fill_na_other_df=self.data,  # use training data to calculate fill na means, etc.
            arma_full_df=data,  # use this data to fit ARMA model on
        )
        na_filled_dataset = dataset["na_filled_dataset"]
        date_series = dataset["date_series"]
        model_input = self.data_setup.gen_model_input(
            na_filled_dataset, self.n_timesteps, drop_missing_ys=False
        )
        X = model_input[0]
        y = model_input[1]

        # predictions on every model
        preds = []
        for i in range(self.n_models):
            preds.append(self.modelling.predict(X, self.mv_lstm[i]))
        preds = list(np.mean(preds, axis=0))

        prediction_df = pd.DataFrame(
            {
                "date": date_series[
                    (len(date_series) - len(preds)) :
                ].values.flatten(),  # may lose some observations at the beginning depending on n_timeperiods, account for that
                "actuals": y,
                "predictions": preds,
            }
        )
        if only_actuals_obs:
            prediction_df = prediction_df.loc[~pd.isna(y), :].reset_index(drop=True)
        return prediction_df

    def gen_ragged_X(self, pub_lags, lag, data=None, start_date=None, end_date=None):
        """Produce vintage model inputs X given the period lag of different variables, for use when testing historical performance (model evaluation, etc.)

        parameters:
                :pub_lags: list[int]: list of periods back each input variable is set to missing. I.e. publication lag of the variable.
            :lag: int: simulated periods back. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc.
            :data: pandas DataFrame: dataframe to generate the ragged datasets on, if none will calculate on training data
            :start_date: str in "YYYY-MM-DD" format: start date of generating ragged preds. To save calculation time, i.e. just calculating after testing date instead of all dates
            :end_date: str in "YYYY-MM-DD" format: end date of generating ragged preds

        output:
                :return: numpy array equivalent in shape to X input, but with trailing edges set to missing/0, ys, and dates
        """
        if data is None:
            data = self.data
        data = data.reset_index(drop=True)

        # converting format of start and end dates
        if start_date != None:
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        if end_date != None:
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        dataset = self.data_setup.gen_dataset(
            data,
            self.target_variable,
            self.fill_na_func,  # don't need to pass fill_ragged_edges_func because will be overwritten anyway
            fill_na_other_df=self.data,
            arma_full_df=data,
        )
        for_ragged_dataset = dataset["for_ragged_dataset"]
        for_full_arma_dataset = dataset["for_full_arma_dataset"]
        date_series = dataset["date_series"][
            ~pd.isna(data[self.target_variable])
        ].reset_index(
            drop=True
        )  # only keep dates where there's a target value
        model_input = self.data_setup.gen_model_input(
            for_ragged_dataset, self.n_timesteps, drop_missing_ys=True
        )
        X = model_input[0]
        y = model_input[1]
        dates = date_series[
            (len(date_series) - len(y)) :
        ].values.flatten()  # may lose some observations at the beginning depending on n_timeperiods, account for that

        # do all if no start/end dates given
        if start_date is None:
            start_date = dates[0]
        if end_date is None:
            end_date = dates[-1]

        ragged_X = self.data_setup.gen_ragged_X(
            X=X,
            pub_lags=pub_lags,
            lag=lag,
            for_ragged_dataset=self.for_ragged_dataset,
            target_variable=self.target_variable,
            fill_ragged_edges=self.fill_ragged_edges_func,
            backup_fill_method=self.fill_na_func,
            other_dataset=self.other_dataset,
            for_full_arma_dataset=for_full_arma_dataset,
            arma_models=self.arma_models,
            dates=pd.Series(dates),
            start_date=start_date,
            end_date=end_date,
        )
        return ragged_X, y, dates

    def ragged_preds(self, pub_lags, lag, data=None, start_date=None, end_date=None):
        """Get predictions on artificial vintages

        parameters:
                :pub_lags: list[int]: list of periods back each input variable is set to missing. I.e. publication lag of the variable.
            :lag: int: simulated periods back. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc.
            :data: pandas DataFrame: dataframe to generate the ragged datasets on, if none will calculate on training data
            :start_date: str in "YYYY-MM-DD" format: start date of generating ragged preds. To save calculation time, i.e. just calculating after testing date instead of all dates
            :end_date: str in "YYYY-MM-DD" format: end date of generating ragged preds

        output:
                :return: pandas DataFrame of actuals, predictions, and dates
        """
        X, y, dates = self.gen_ragged_X(pub_lags, lag, data, start_date, end_date)
        # predictions on every model
        preds = []
        for i in range(self.n_models):
            preds.append(self.modelling.predict(X, self.mv_lstm[i]))
        preds = list(np.mean(preds, axis=0))
        pred_df = pd.DataFrame(
            {
                "date": dates,
                "actuals": y,
                "predictions": preds,
            }
        )
        # filter rows with no prediction
        pred_df = pred_df.loc[~pd.isna(pred_df.predictions), :].reset_index(drop=True)
        # filter dates
        if start_date != None:
            pred_df = pred_df.loc[pred_df.date >= start_date, :].reset_index(drop=True)
        if end_date != None:
            pred_df = pred_df.loc[pred_df.date <= end_date, :].reset_index(drop=True)

        return pred_df

    def gen_news(self, target_period, old_data, new_data):
        """Generate the news between two data releases using the method of holding out new data feature by feature and recording the differences in model output
        Make sure both the old and new dataset have the target period in them to allow for predictions and news generation.

        parameters:
            :target_period: str: target prediction date
            :old_data: pd.DataFrame: previous dataset
            :new_data: pd.DataFrame: new dataset

        output: Dict
            :news: dataframe of news contribution of each column with updated data. scaled_news is news scaled to sum to actual prediction delta.
            :old_pred: prediction on the previous dataset
            :new_pred: prediction on the new dataset
            :holdout_discrepency: % difference between the sum of news via the holdout method and the actual prediction delta
        """
        return self.modelling.gen_news(self, target_period, old_data, new_data)

    def feature_contribution(self):
        """Obtain permutation feature contribution via RMSE on the train set

        output: Pandas DataFrame
            :feature: column name
            :scaled_contribution: contribution of feature to the model, scaled to 1 = most important feature
        """
        return self.modelling.feature_contribution(self)
