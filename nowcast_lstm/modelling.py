import torch

import nowcast_lstm.data_setup
import nowcast_lstm.mv_lstm


def instantiate_model(
    model_x_input,
    n_timesteps,
    n_hidden=20,
    n_layers=2,
    dropout=0,
    lr=1e-2,
    criterion="",
    optimizer="",
):
    """Create the network, criterion, and optimizer objects necessary for training a model
	
	parameters:
		:model_input: numpy array: output of `gen_model_input` function, first entry in tuple (X)
		:n_timesteps: how many historical periods to consider when training the model. For example if the original data is monthly, n_steps=12 would consider data for the last year.
		:n_hidden: int: number of hidden states in the network
		:n_layers: int: number of LSTM layers in the network
		:dropout: float: dropout rate between the LSTM layers
		:lr: float: learning rate
		:criterion: torch loss criterion, defaults to MAE
		:optimizer: torch optimizer, defaults to Adam
	
	output: Dict
		:mv_lstm: torch network
		:criterion: torch criterion
		:optimizer: torch optimizer
	"""

    n_features = model_x_input.shape[
        2
    ]  # 3rd axis of the matrix is the number of features
    mv_lstm = nowcast_lstm.mv_lstm.MV_LSTM(
        n_features, n_timesteps, n_hidden, n_layers, dropout
    )
    if criterion == "":
        criterion = torch.nn.L1Loss()
    if optimizer == "":
        optimizer = torch.optim.Adam(mv_lstm.parameters(), lr=lr)
    return {
        "mv_lstm": mv_lstm,
        "criterion": criterion,
        "optimizer": optimizer,
    }


def train_model(
    X,
    y,
    mv_lstm,
    criterion,
    optimizer,
    train_episodes=200,
    batch_size=30,
    decay=0.98,
    quiet=False,
):
    """Train the network
	
	parameters:
		:X: numpy array: output of `gen_model_input` function, first entry in tuple (X), input variables
		:y: numpy array: output of `gen_model_input` function, second entry in tuple (y), targets
		:mv_lstm: torch network: output of `instantiate_model` function, "mv_lstm" entry
		:criterion: torch criterion: output of `instantiate_model` function, "criterion" entry, MAE is default
		:optimizer: torch optimizer: output of `instantiate_model` function, "optimizer" entry, Adam is default
		:train_episodes: int: number of epochs/episodes to train the model
		:batch_size: int: number of observations per training batch
		:decay: float: learning rate decay
		:quiet: boolean: whether or not to print the losses in the epoch loop
	
	output:
		:return: Dict
			:mv_lstm: trained network
			:train_loss: list of losses per epoch, for informational purposes
	"""
    train_loss = []  # for plotting train loss
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decay
    )  # for learning rate decay

    mv_lstm.train()
    for t in range(train_episodes):
        for b in range(0, len(X), batch_size):
            inpt = X[b : b + batch_size, :, :]
            target = y[b : b + batch_size]

            x_batch = torch.tensor(inpt, dtype=torch.float32)
            y_batch = torch.tensor(target, dtype=torch.float32)

            mv_lstm.init_hidden(x_batch.size(0))
            output = mv_lstm(x_batch)
            loss = criterion(output.view(-1), y_batch)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        my_lr_scheduler.step()  # learning rate decay

        if not (quiet):
            print("step : ", t, "loss : ", loss.item())

        train_loss.append(loss.item())

    return {
        "mv_lstm": mv_lstm,
        "train_loss": train_loss,
    }


def predict(X, mv_lstm):
    """Make predictions on a trained network
	
	parameters:
		:X: numpy array: output of `gen_model_input` function, first entry in tuple (X), input variables
		:mv_lstm: torch network: output of `train_model` function, "mv_lstm" entry, trained network
	
	output:
		:return: np array: array of predictions
	"""
    with torch.no_grad():
        inpt = torch.tensor(X, dtype=torch.float32)
    mv_lstm.init_hidden(inpt.size(0))
    preds = mv_lstm(inpt).view(-1).detach().numpy()

    return preds
