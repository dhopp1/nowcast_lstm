import unittest
import pandas as pd
import numpy as np
from nowcast_lstm import LSTM
from nowcast_lstm.model_selection import (
    variable_selection,
    hyperparameter_tuning,
    select_model,
)


class TestDataSetup(unittest.TestCase):
    x = pd.DataFrame(
        {
            "date": [
                pd.to_datetime("2020-03-01"),
                pd.to_datetime("2020-04-01"),
                pd.to_datetime("2020-05-01"),
            ],
            "var1": [1, 2, 3],
            "var2": [4, np.nan, 6],
            "target": [7, 8, 9],
        }
    )
    x.date = pd.to_datetime(x.date)
    model = LSTM.LSTM(data=x, target_variable="target", n_timesteps=2)
    model.train(quiet=True)

    long_x = pd.DataFrame(
        {
            "date": [
                pd.to_datetime("2019-01-01"),
                pd.to_datetime("2019-02-01"),
                pd.to_datetime("2019-03-01"),
                pd.to_datetime("2019-04-01"),
                pd.to_datetime("2019-05-01"),
                pd.to_datetime("2019-06-01"),
                pd.to_datetime("2019-07-01"),
            ],
            "var1": [1, 2, 3, 4, 5, 6, 7],
            "var2": [4, np.nan, 6, 7, 8, 9, 10],
            "target": [7, 8, 9, 10, 11, 12, 13],
        }
    )
    long_x.date = pd.to_datetime(long_x.date)

    def test_LSTM(self):
        self.assertEqual(len(self.model.predict(self.model.data)), 2)

    def test_LSTM_newdata(self):
        new_x = self.x.copy()
        new_x.iloc[
            1:, 3
        ] = np.nan  # simulating no actuals for this, still able to predict
        preds = self.model.predict(data=new_x)

        self.assertEqual(len(preds), 2)

    def test_LSTM_multiple_models(self):
        model2 = LSTM.LSTM(
            data=self.x, target_variable="target", n_timesteps=2, n_models=3
        )
        model2.train(quiet=True)
        preds = model2.predict(model2.data)

        self.assertEqual(len(preds), 2)

    def test_LSTM_gen_ragged(self):
        model3 = LSTM.LSTM(
            data=self.long_x,
            target_variable="target",
            n_timesteps=2,
            fill_na_func=np.nanmedian,
            fill_ragged_edges_func="ARMA",
        )
        result = model3.gen_ragged_X([1, 2], 0)[0]

        self.assertTrue(
            (
                result
                == np.array(
                    [
                        [[1.0, 7.5], [4.0, 7.5]],
                        [[2.0, 7.5], [4.0, 7.5]],
                        [[3.0, 7.5], [4.0, 7.5]],
                        [[4.0, 7.5], [4.0, 7.5]],
                        [[5.0, 7.5], [4.0, 7.5]],
                        [[6.0, 7.5], [4.0, 7.5]],
                    ]
                )
            ).all()
        )

    def test_LSTM_feature_contribution(self):
        model2 = LSTM.LSTM(
            data=self.x, target_variable="target", n_timesteps=2, n_models=3
        )
        model2.train(quiet=True)

        self.assertEqual(model2.feature_contribution().iloc[0, 1], 1)
        self.assertEqual(len(model2.feature_contribution()), 2)

    def test_LSTM_variable_selection(self):
        variables, perf = variable_selection(
            self.long_x,
            "target",
            2,
            init_test_size=0.5,
            quiet=True,
            initial_ordering="feature_contribution",
        )
        self.assertEqual(variables, ["var1"])

    def test_LSTM_variable_selection_univariate(self):
        variables, perf = variable_selection(
            self.long_x,
            "target",
            2,
            init_test_size=0.5,
            quiet=True,
            initial_ordering="univariate",
        )
        self.assertEqual(variables, ["var1"])

    def test_LSTM_hyperparameter_tuning(self):
        results = hyperparameter_tuning(
            self.long_x,
            "target",
            2,
            init_test_size=0.5,
            quiet=True,
            n_timesteps_grid=[2],
            train_episodes_grid=[10, 20],
            batch_size_grid=[30],
            n_hidden_grid=[10],
            n_layers_grid=[1],
        )
        self.assertEqual(len(results), 2)

    def test_LSTM_select_model(self):
        results = select_model(
            self.long_x,
            "target",
            n_timesteps_grid=[2],
            n_layers_grid=[1, 2],
            init_test_size=0.5,
            quiet=True,
            initial_ordering="feature_contribution",
        )
        self.assertEqual(len(results), 2)

    def test_LSTM_select_model_univariate(self):
        results = select_model(
            self.long_x,
            "target",
            n_timesteps_grid=[2],
            n_layers_grid=[1, 2],
            init_test_size=0.5,
            quiet=True,
            initial_ordering="univariate",
        )
        self.assertEqual(len(results), 2)

    def test_LSTM_interval_predict(self):
        model2 = LSTM.LSTM(
            data=self.x, target_variable="target", n_timesteps=2, n_models=3
        )
        model2.train(quiet=True)

        results = model2.interval_predict(model2.data)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(results.columns), 5)

    def test_LSTM_ragged_interval_predict(self):
        model2 = LSTM.LSTM(
            data=self.x, target_variable="target", n_timesteps=2, n_models=3
        )
        model2.train(quiet=True)

        results = model2.ragged_interval_predict([0, 0], 0, model2.data)

        self.assertEqual(len(results), 2)
        self.assertEqual(len(results.columns), 5)


if __name__ == "__main__":
    unittest.main()
