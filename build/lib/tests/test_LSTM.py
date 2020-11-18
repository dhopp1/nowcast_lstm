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
        preds = self.model.predict(LSTM.LSTM(new_x, "target", 2, False).X)

        self.assertEqual(len(preds), 2)

    def test_LSTM_multiple_models(self):
        model2 = LSTM.LSTM(self.x, "target", 2, n_models=3)
        model2.train(quiet=True)
        preds = model2.predict(self.model.X)

        self.assertEqual(len(preds), 2)

    def test_LSTM_gen_ragged(self):
        model3 = LSTM.LSTM(self.long_x, "target", 2)
        result = model3.gen_ragged_X([1, 2], 0)

        self.assertTrue(
            (
                result
                == np.array(
                    [
                        [[1.0, 0.0], [0.0, 0.0]],
                        [[2.0, 0.0], [0.0, 0.0]],
                        [[3.0, 0.0], [0.0, 0.0]],
                        [[4.0, 0.0], [0.0, 0.0]],
                        [[5.0, 0.0], [0.0, 0.0]],
                        [[6.0, 0.0], [0.0, 0.0]],
                    ]
                )
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
