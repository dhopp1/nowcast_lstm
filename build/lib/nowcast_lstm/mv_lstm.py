import torch


class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, n_timesteps, n_hidden, n_layers, dropout):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        # model layers
        # n_layers stacked LSTM layers + one linear dense layer to get final prediction
        self.l_lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.l_linear = torch.nn.Linear(self.n_hidden * self.n_timesteps, 1)

    # model layers

    def init_hidden(self, batch_size):  # initializing hidden layers
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, n_timesteps, _ = x.size()

        # model layers
        x, self.hidden = self.l_lstm(x, self.hidden)
        x = x.contiguous().view(batch_size, -1)  # make tensor of right dimensions
        x = self.l_linear(x)
        # model layers

        return x
