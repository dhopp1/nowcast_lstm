# nowcast_lstm
**Installation**: `pip install nowcast-lstm`
<br>
**Example**: `nowcast_lstm_example.zip` contains a jupyter notebook file with a dataset and more detailed example of usage.
<br><br>
[LSTM neural networks](https://en.wikipedia.org/wiki/Long_short-term_memory) have been used for nowcasting [before](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), combining the strengths of artificial neural networks with a temporal aspect. However their use in nowcasting economic indicators remains limited, no doubt in part due to the difficulty of obtaining results in existing deep learning frameworks. This library seeks to streamline the process of obtaining results in the hopes of expanding the domains to which LSTM can be applied.

While neural networks are flexible and this framework may be able to get sensible results on levels, the model architecture was developed to nowcast growth rates of economic indicators. As such training inputs should ideally be stationary and seasonally adjusted.

Further explanation of the background problem can be found in [this UNCTAD research paper](https://unctad.org/system/files/official-document/ser-rp-2018d9_en.pdf). Further explanation and results to be published in an upcoming [UNCTAD](https://unctad.org/) research paper.


## Quick usage
The main object and functionality of the library comes from the `LSTM` object. Given `data` = a pandas DataFrame of a date column + monthly data + a quarterly target series to run the model on, usage is as follows:
```python
from nowcast_lstm.LSTM import LSTM

model = LSTM(data, "target_col_name", n_timesteps=12) # default parameters with 12 timestep history

model.X # array of the transformed training dataset
model.y # array of the target values

model.mv_lstm # list of trained PyTorch network(s)
model.train_loss # list of training losses for the network(s)

model.train()
model.predict(model.data) # predictions on the training set

# predicting on a testset, which is the same dataframe as the training data + newer data
# this will give predictions for all dates, but only predictions after the training data ends should be considered for testing
model.predict(test_data)

# to gauge performance on artificial data vintages
model.ragged_preds(pub_lags, lag, test_data)

# save a trained model using dill
import dill
dill.dump(model, open("trained_model.pkl", mode="wb"))

# load a previously trained model using dill
trained_model = dill.load(open("trained_model.pkl", "rb", -1))

```

## LSTM parameters
- `data`: `pandas DataFrame` of the data to train the model on. Should contain a target column. Any non-numeric columns will be dropped. It should be in the most frequent period of the data. E.g. if I have three monthly variables, two quarterly variables, and a quarterly series, the rows of the dataframe should be months, with the quarterly values appearing every three months (whether Q1 = Jan 1 or Mar 1 depends on the series, but generally the quarterly value should come at the end of the quarter, i.e. Mar 1), with NAs or 0s in between. The same logic applies for yearly variables.
- `target_variable`: a `string`, the name of the target column in the dataframe.
- `n_timesteps`: an `int`, corresponding to the "memory" of the network, i.e. the target value depends on the x past values of the independent variables. For example, if the data is monthly, `n_timesteps=12` means that the estimated target value is based on the previous years' worth of data, 24 is the last two years', etc. This is a hyper parameter that can be evaluated.
- `fill_na_func`: a function used to replace missing values. Should take a column as a parameter and return a scalar, e.g. `np.nanmean` or `np.nanmedian`.
- `fill_ragged_edges_func`: a function used to replace missing values at the end of series. Leave blank to use the same function as `fill_na_func`, pass `"ARMA"` to use ARMA estimation using `pmdarima.arima.auto_arima`.
- `n_models`: `int` of the number of networks to train and predict on. Because neural networks are inherently stochastic, it can be useful to train multiple networks with the same hyper parameters and take the average of their outputs as the model's prediction, to smooth output.
- `train_episodes`: `int` of the number of training episodes/epochs. A short discussion of the topic can be found [here](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/).
- `batch_size`: `int` of the number of observations per batch. Discussed [here](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
- `lr`: `float` of the learning rate of network. Discussed [here](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/).
- `decay`: `float` of the rate of decay of the learning rate. Also discussed [here](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/). Set to `0` for no decay.
- `n_hidden`: `int` of the number of hidden states in the LSTM network. Discussed [here](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/).
- `n_layers`: `int` of the number of LSTM layers to include in the network. Also discussed [here](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/).
- `dropout`: `float` of the proportion of layers to drop in between LSTM layers. Discussed [here](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/).
- `criterion`: `PyTorch loss function`. Discussed [here](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/), list of available options in PyTorch [here](https://pytorch.org/docs/stable/nn.html#loss-functions).
- `optimizer`: `PyTorch optimizer`. Discussed [here](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6), list of available options in PyTorch [here](https://pytorch.org/docs/stable/optim.html)

## LSTM outputs
Assuming a model has been instantiated and trained with `model = LSTM(...)`:
- `model.train()`: trains the network. Set `quiet=True` to suppress printing of losses per epoch during training.
- `model.X`: transformed data in the format the model was/will actually be trained on. A `numpy array` of dimensions `n observations x n timesteps x n features`.
- `model.y`: one-dimensional list target values the model was/will be trained on.
- `model.predict(model.data)`: given a dataframe with the same columns the model was trained on, returns a dataframe with date, actuals, and predictions, pass `model.data` for performance on the training set.
- `model.predict(new_data)`: generate dataframe of predictions on a new dataset. Generally should be the same dataframe as the training set, plus additional dates/datapoints.
- `model.mv_lstm`: a `list` of length `n_models` containing the PyTorch networks. 
- `model.train_loss`: a `list` of length `n_models` containing the training losses of each of the trained networks.
- `model.ragged_preds(pub_lags, lag, new_data)`: adds artificial missing data then returns a dataframe with date, actuals, and predictions. This is especially useful as a testing mechanism, to generate datasets to see how a trained model would have performed at different synthetic vintages or periods of time in the past. `pub_lags` should be a list of `ints` (in the same order as the columns of the original data) of length `n_features` (i.e. excluding the target variable) dictating the normal publication lag of each of the variables. `lag` is an int of how many periods back we want to simulate being, interpretable as last period relative to target period. E.g. if we are nowcasting June, `lag = -1` will simulate being in May, where May data is published for variables with a publication lag of 0. It will fill with missings values that wouldn't have been available yet according to the publication lag of the variable + the `lag` parameter. It will fill missings with the same method specified in the `fill_ragged_edges_func` parameter in model instantiation.
