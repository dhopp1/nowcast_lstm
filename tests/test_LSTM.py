import unittest
import pandas as pd
import numpy as np
from nowcast_lstm import LSTM


class TestDataSetup(unittest.TestCase):
    x = pd.DataFrame(
        {
            "date": ["a", "b", "c"],
            "var1": [1, 2, 3],
            "var2": [4, np.nan, 6],
            "target": [7, 8, 9],
        }
    )
    model = LSTM.LSTM(x, "target", 2)
    model.train(quiet=True)

    long_x = pd.DataFrame(
        {
            "date": ["a", "b", "c", "c", "c", "c", "c"],
            "var1": [1, 2, 3, 4, 5, 6, 7],
            "var2": [4, np.nan, 6, 7, 8, 9, 10],
            "target": [7, 8, 9, 10, 11, 12, 13],
        }
    )

    def test_LSTM(self):
        self.assertEqual(len(self.model.predict(self.model.X)), 2)

    def test_LSTM_newdata(self):
        new_x = self.x
        new_x.iloc[1:, 3] = 0.0  # simulating no actuals for this, still able to predict
        preds = self.model.predict(LSTM.LSTM(data=new_x, target_variable="target", n_timesteps=2, drop_missing_ys=False).X)

        self.assertEqual(len(preds), 2)

    def test_LSTM_multiple_models(self):
        model2 = LSTM.LSTM(data=self.x, target_variable="target", n_timesteps=2, n_models=3)
        model2.train(quiet=True)
        preds = model2.predict(self.model.X)

        self.assertEqual(len(preds), 2)

    def test_LSTM_gen_ragged(self):
        model3 = LSTM.LSTM(data=self.long_x, target_variable="target", n_timesteps=2, fill_na_func=np.nanmedian, fill_ragged_edges_func="ARMA")
        result = model3.gen_ragged_X([1, 2], 0)

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


if __name__ == "__main__":
    unittest.main()
