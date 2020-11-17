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


if __name__ == "__main__":
    unittest.main()
