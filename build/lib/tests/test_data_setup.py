import unittest
import pandas as pd
import numpy as np
from nowcast_lstm import data_setup


class TestDataSetup(unittest.TestCase):
    x = pd.DataFrame(
        {
            "date": ["a", "b", "c"],
            "var1": [1, 2, 3],
            "var2": [4, np.nan, 6],
            "target": [7, 8, 9],
        }
    )

    long_x = pd.DataFrame(
        {
            "date": ["a", "b", "c", "c", "c", "c", "c"],
            "var1": [1, 2, 3, 4, 5, 6, 7],
            "var2": [4, np.nan, 6, 7, 8, 9, 10],
            "target": [7, 8, 9, 10, 11, 12, 13],
        }
    )

    def test_gen_dataset(self):
        self.assertTrue(
            (
                data_setup.gen_dataset(self.x, "target")
                == np.array([[1, 4, 7], [2, 0, 8], [3, 6, 9]], np.float64)
            ).all()
        )

    def test_gen_model_input(self):
        result = data_setup.gen_model_input(
            data_setup.gen_dataset(self.x, "target"), n_timesteps=2
        )
        self.assertTrue(
            (
                result[0] == np.array([[[1, 4], [2, 0]], [[2, 0], [3, 6]]], np.float64)
            ).all()
        )

    def test_gen_ragged_X(self):
        X = data_setup.gen_dataset(self.long_x, "target")
        X = data_setup.gen_model_input(X, n_timesteps=2)
        result = data_setup.gen_ragged_X(X[0], [1, 2], 0)

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
