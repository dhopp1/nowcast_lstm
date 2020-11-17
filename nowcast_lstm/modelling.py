import torch
import nowcast_lstm.mv_lstm as mv_lstm


def instantiate_model(
    model_x_input,
    n_timesteps,
    n_hidden=20,
    n_layers=2,
    dropout=0.98,
    lr=1e-2,
    criterion="",
    optimizer="",
):
    """Create the network, criterion, and optimizer objects necessary for training a model
	
	parameters:
		:model_input: pandas DataFrame: output of `gen_model_input` function, first entry in tuple (X)
		:n_timesteps: how many historical periods to consider when training the model. For example if the original data is monthly, n_steps=12 would consider data for the last year.
		:n_hidden: int: number of hidden states in the network
		:n_layers: int: number of LSTM layers in the network
		:dropout: float: learning rate decay rate
		:lr: float: learning rate
		:criterion: torch loss criterion, defaults to MAE
		:optimizer: torch optimizer, defaults to Adam
	
	output:
		:return: numpy array: n x m+1 array
	"""

    n_features = model_x_input.shape[
        2
    ]  # 3rd axis of the matrix is the number of features
    net = mv_lstm.MV_LSTM(n_features, n_timesteps, n_hidden, n_layers, dropout)
    if criterion == "":
        criterion = torch.nn.L1Loss()
    if optimizer == "":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    return {
        "net": net,
        "criterion": criterion,
        "optimizer": optimizer,
    }


def train_model(
    X, y, n_timesteps, mv_lstm, criterion, optimizer, train_episodes, batch_size, decay
):
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

        my_lr_scheduler.step()

        print("step : ", t, "loss : ", loss.item())
        train_loss.append(loss.item())
    return {
        "mv_lstm": mv_lstm,
        "train_loss": train_loss,
    }


def gen_preds(X, net):
    inpt = torch.tensor(X, dtype=torch.float32)
    net.init_hidden(inpt.size(0))
    preds = net(inpt).view(-1).detach().numpy()
    return preds
