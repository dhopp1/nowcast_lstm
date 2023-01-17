import torch
import numpy as np
import pandas as pd

import nowcast_lstm.data_setup
import nowcast_lstm.mv_lstm


def instantiate_model(
    model_x_input,
    n_timesteps,
    n_hidden=20,
    n_layers=2,
    dropout=0,
    criterion="",
    optimizer="",
    optimizer_parameters={"lr": 1e-2},
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

    # for generating the optimizer
    def generate_optimizer(model, opt_fn=None, opt_kwargs=optimizer_parameters):
        optimizer = opt_fn(model.parameters(), **opt_kwargs)
        return optimizer

    if optimizer == "":
        optim = generate_optimizer(mv_lstm, torch.optim.Adam, optimizer_parameters)
    else:
        optim = generate_optimizer(mv_lstm, optimizer, optimizer_parameters)

    return {
        "mv_lstm": mv_lstm,
        "criterion": criterion,
        "optimizer": optim,
    }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index, :, :]
        y = self.y[index]

        return X, y


def train_model(
    X,
    y,
    mv_lstm,
    criterion,
    optimizer,
    train_episodes=200,
    batch_size=30,
    decay=0.98,
    num_workers=0,
    shuffle=False,
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
        :num_workers: int: number of workers for multi-process data loading
        :shuffle: boolean: whether to shuffle data at every epoch
        :quiet: boolean: whether or not to print the losses in the epoch loop

    output:
        :return: Dict
            :mv_lstm: trained network
            :train_loss: list of losses per epoch, for informational purposes
    """

    # CUDA if available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    params = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": num_workers}

    # PyTorch dataset
    data_generator = torch.utils.data.DataLoader(
        Dataset(
            torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        ),
        **params
    )

    train_loss = []  # for plotting train loss
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=decay
    )  # for learning rate decay

    mv_lstm.train()
    for t in range(train_episodes):
        for batch_X, batch_y in data_generator:
            # to GPU
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            mv_lstm.init_hidden(batch_X.size(0))
            output = mv_lstm(batch_X)
            loss = criterion(output.view(-1), batch_y)

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


def gen_news(model, target_period, old_data, new_data):
    """Generate the news between two data releases using the method of holding out new data feature by feature and recording the differences in model output
    Make sure both the old and new dataset have the target period in them to allow for predictions and news generation.

    parameters:
        :model: LSTM.LSTM: trained LSTM model
        :target_period: str: target prediction date
        :old_data: pd.DataFrame: previous dataset
        :new_data: pd.DataFrame: new dataset

    output: Dict
        :news: dataframe of news contribution of each column with updated data. scaled_news is news scaled to sum to actual prediction delta.
        :old_pred: prediction on the previous dataset
        :new_pred: prediction on the new dataset
        :holdout_discrepency: % difference between the sum of news via the holdout method and the actual prediction delta
    """
    data = {}
    data["old"] = old_data.copy()
    data["new"] = new_data.copy()
    data["vintage"] = new_data.copy()
    # force all dataframes to have same number of rows
    data["old"] = data["new"].loc[:, ["date"]].merge(data["old"], on="date", how="left")

    # vintage data is new data, but with missings where old had missings. For revisions contribution.
    data["vintage"] = data["new"].copy()
    for col in data["vintage"].columns:
        data["vintage"].loc[pd.isna(data["old"][col]), col] = np.nan

    # predictions on each dataframe, subsequently getting revisions contribution
    preds = pd.DataFrame(columns=["column", "prediction"])
    for dataset in ["old", "new", "vintage"]:
        preds = preds.append(
            pd.DataFrame(
                {
                    "column": dataset,
                    "prediction": model.predict(data[dataset])
                    .loc[lambda x: x.date == target_period]
                    .predictions.values[0],
                },
                index=[0],
            )
        ).reset_index(drop=True)

    # looping through each column
    for col in data["vintage"].columns:
        if col != model.target_variable:
            # any new values?
            if not all(
                list(
                    (data["vintage"][col] == data["new"][col])
                    | (pd.isna(data["vintage"][col]) & pd.isna(data["new"][col]))
                )
            ):
                # predictions on new - new value (subtractive method)
                subtractive = data["new"].copy()
                subtractive[col] = data["vintage"][col]
                preds = preds.append(
                    pd.DataFrame(
                        {
                            "column": col,
                            "prediction": model.predict(subtractive)
                            .loc[lambda x: x.date == target_period]
                            .predictions.values[0],
                        },
                        index=[0],
                    )
                ).reset_index(drop=True)

    # scale the news so it adds to actual delta
    old_pred = preds.loc[preds.column == "old", "prediction"].values[0]
    new_pred = preds.loc[preds.column == "new", "prediction"].values[0]
    revisions = preds.loc[preds.column == "vintage", "prediction"].values[0] - old_pred
    delta = new_pred - old_pred
    subtractive = (
        preds.loc[preds.column == "new", "prediction"].values[0]
        - preds.loc[~preds.column.isin(["new", "old", "vintage"]), "prediction"]
    )
    subtractive = pd.DataFrame(
        {
            "column": preds.loc[subtractive.index, "column"].values,
            "news": subtractive.values,
        }
    )
    news = subtractive.copy()
    news.loc[len(news), "news"] = revisions
    news.loc[len(news) - 1, "column"] = "revisions"
    if delta != 0:
        diff = news.news.sum() / delta
        news["scaled_news"] = news.news / diff
    else:
        diff = 1
        news["scaled_news"] = news.news
    return {
        "news": news,
        "old_pred": old_pred,
        "new_pred": new_pred,
        "holdout_discrepency": diff,
    }


def feature_contribution(model):
    """Obtain permutation feature contribution via RMSE on the train set

    parameters:
        :model: LSTM.LSTM: trained LSTM model

    output: Pandas DataFrame
        :feature: column name
        :scaled_contribution: contribution of feature to the model, scaled to 1 = most important feature
    """
    train_rmse = np.sqrt(
        np.nanmean(
            (model.y - model.predict(model.data, only_actuals_obs=True).predictions)
            ** 2
        )
    )

    # iteratively go through columns recording change in RMSE
    col_rmses = []
    cols = []
    for col in model.data.columns:
        if (col != model.target_variable) & (col != model.date_series.columns[0]):
            tmp_data = model.data.copy()
            tmp_data[col] = np.nanmean(
                tmp_data[col]
            )  # set column to mean, effectively removing it
            tmp_rmse = np.sqrt(
                np.nanmean(
                    (
                        model.y
                        - model.predict(tmp_data, only_actuals_obs=True).predictions
                    )
                    ** 2
                )
            )
            col_rmses.append(tmp_rmse)
            cols.append(col)
    importance = (
        pd.DataFrame(
            {"feature": cols, "scaled_contribution": col_rmses / train_rmse - 1}
        )
        .sort_values(["scaled_contribution"], ascending=False)
        .reset_index(drop=True)
    )

    # scaling importance relataive to the most important
    importance.scaled_contribution = importance.scaled_contribution / np.max(
        importance.scaled_contribution
    )

    return importance
