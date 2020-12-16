from importlib import import_module
import numpy as np

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
        :fill_ragged_edges_func: function to replace NAs in ragged edges (data missing at end of series). Pass "ARMA" for ARMA filling
        :fill_na_other_df: pandas DataFrame: A dataframe with the exact same columns as the rawdata dataframe. For use with filling NAs based on a different dataset (e.g. the train dataset). E.g. `train=LSTM(...)`, `gen_dataset(test_data, target_variable, fill_na_other_df=train.data)`
        :arma_full_df: pandas DataFrame: A dataframe with the exact same columns as the rawdata dataframe. For use with ARMA filling on a full-series history, rather than just the history present in the train set
        :drop_missing_ys: boolean: whether or not to filter out missing ys. Set to true when creating training data, false when want to run predictions on data that may not have a y.
        :n_models: int: number of models to train and take the average of for more robust estimates
        :train_episodes: int: number of epochs/episodes to train the model
        :batch_size: int: number of observations per training batch
        :lr: float: learning rate
        :decay: float: learning rate decay
        :n_hidden: int: number of hidden states in the network
        :n_layers: int: number of LSTM layers in the network
        :dropout: float: dropout rate between the LSTM layers
        :criterion: torch loss criterion, defaults to MAE
        :optimizer: torch optimizer, defaults to Adam
	"""

    def __init__(
        self,
        data,
        target_variable,
        n_timesteps,
        fill_na_func=np.nanmean,
        fill_ragged_edges_func=None,
        fill_na_other_df=None,
        arma_full_df=None,
        drop_missing_ys=True,
        n_models=1,
        train_episodes=200,
        batch_size=30,
        lr=1e-2,
        decay=0.98,
        n_hidden=20,
        n_layers=2,
        dropout=0,
        criterion="",
        optimizer="",
    ):
        self.data_setup = import_module("nowcast_lstm.data_setup")
        self.modelling = import_module("nowcast_lstm.modelling")

        self.data = data
        self.target_variable = target_variable
        self.n_timesteps = n_timesteps

        self.fill_na_func = fill_na_func
        self.fill_ragged_edges_func = fill_ragged_edges_func
        self.fill_na_other_df = fill_na_other_df
        self.arma_full_df = arma_full_df

        self.train_episodes = train_episodes
        self.n_models = n_models

        self.batch_size = batch_size
        self.lr = lr
        self.decay = decay
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.criterion = criterion
        self.optimizer = optimizer
        self.drop_missing_ys = drop_missing_ys

        self.dataset = self.data_setup.gen_dataset(
            self.data,
            self.target_variable,
            self.fill_na_func,
            self.fill_ragged_edges_func,
            self.fill_na_other_df,
            self.arma_full_df,
        )
        self.na_filled_dataset = self.dataset["na_filled_dataset"]
        self.for_ragged_dataset = self.dataset["for_ragged_dataset"]
        self.for_full_arma_dataset = self.dataset["for_full_arma_dataset"]
        self.other_dataset = self.dataset["other_dataset"]

        self.model_input = self.data_setup.gen_model_input(
            self.na_filled_dataset, self.n_timesteps, self.drop_missing_ys
        )
        self.X = self.model_input[0]
        self.y = self.model_input[1]
        
        self.ragged_input = self.data_setup.gen_model_input(
            self.for_ragged_dataset, self.n_timesteps, self.drop_missing_ys
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
            print(f"Training model {i+1}")
            # instantiate the model
            instantiated = self.modelling.instantiate_model(
                self.X,
                n_timesteps=self.n_timesteps,
                n_hidden=self.n_hidden,
                n_layers=self.n_layers,
                dropout=self.dropout,
                lr=self.lr,
                criterion=self.criterion,
                optimizer=self.optimizer,
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

    def predict(self, X):
        preds = []
        for i in range(self.n_models):
            preds.append(self.modelling.predict(X, self.mv_lstm[i]))

        return list(np.mean(preds, axis=0))

    def gen_ragged_X(self, pub_lags, lag):
        """Produce vintage model inputs X given the period lag of different variables, for use when testing historical performance (model evaluation, etc.)
	
    	parameters:
    		:pub_lags: list[int]: list of periods back each input variable is set to missing. I.e. publication lag of the variable.
            :lag: int: simulated periods back. E.g. -2 = simulating data as it would have been 2 months before target period, 1 = 1 month after, etc.
    	
    	output:
    		:return: numpy array equivalent in shape to X input, but with trailing edges set to missing/0
    	"""
        return self.data_setup.gen_ragged_X(
            X=self.ragged_X,
            pub_lags=pub_lags,
            lag=lag,
            for_ragged_dataset=self.for_ragged_dataset,
            target_variable=self.target_variable,
            fill_ragged_edges=self.fill_ragged_edges_func,
            backup_fill_method=self.fill_na_func,
            other_dataset=self.other_dataset,
            for_full_arma_dataset=self.for_full_arma_dataset,
        )
