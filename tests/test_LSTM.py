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

    def test_LSTM(self):
        model = LSTM.LSTM(self.x, "target", 2)
        model.train(quiet=True)
        self.assertEqual(len(model.predict(model.X)), 2)


if __name__ == "__main__":
    unittest.main()
