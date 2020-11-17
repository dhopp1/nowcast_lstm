from importlib import import_module

import nowcast_lstm.data_setup
import nowcast_lstm.modelling


class LSTM:
    """Primary class of the library, used for transforming data, training the model, and making predictions.
    `model = LSTM()`
    `model.train()` to train the model
    `model.X` to see model inputs
    `model.y` to see actual ys
    `model.predict(model.X)` to get predictions on the train set
    To test on a totally new set of data: `model.predict(LSTM(new_data, target, n_timesteps).X)`
	
	parameters:
		:data: pandas DataFrame: n x m+1 dataframe
        :target_variable: str: name of the target var
        :n_timesteps: how many historical periods to consider when training the model. For example if the original data is monthly, n_steps=12 would consider data for the last year.
        :train_episodes: int: number of epochs/episodes to train the model
        :batch_size: int: number of observations per training batch
        :lr: float: learning rate
		:decay: float: learning rate decay
        :n_hidden: int: number of hidden states in the network
		:n_layers: int: number of LSTM layers in the network
		:dropout: float: dropout rate between the LSTM layers
		:criterion: torch loss criterion, defaults to MAE
		:optimizer: torch optimizer, defaults to Adam
		
	output:
		:mv_lstm: torch network
		:criterion: torch criterion
		:optimizer: torch optimizer
	"""

    def __init__(
        self,
        data,
        target_variable,
        n_timesteps,
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
        self.train_episodes = train_episodes
        self.batch_size = batch_size
        self.lr = lr
        self.decay = decay
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.criterion = criterion
        self.optimizer = optimizer

        self.dataset = self.data_setup.gen_dataset(self.data, self.target_variable)
        self.model_input = self.data_setup.gen_model_input(
            self.dataset, self.n_timesteps
        )
        self.X = self.model_input[0]
        self.y = self.model_input[1]

    def train(self, quiet=False):
        "quiet is whether or not to print output of loss during training"
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
            quiet=quiet,
        )
        self.mv_lstm = trained["mv_lstm"]
        self.train_loss = trained["train_loss"]

    def predict(self, X):
        return self.modelling.predict(X, self.mv_lstm)
