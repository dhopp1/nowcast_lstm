import unittest
import numpy as np
import pandas as pd

from nowcast_lstm import data_setup, modelling


class TestModelling(unittest.TestCase):
    x = pd.DataFrame(
        {
            "date": ["a", "b", "c"],
            "var1": [1, 2, 3],
            "var2": [4, np.nan, 6],
            "target": [7, 8, 9],
        }
    )

    def test_instantiate_model(self):
        model_input = data_setup.gen_model_input(
            data_setup.gen_dataset(self.x, "target"), n_timesteps=2
        )[
            0
        ]  # first tuple of the function, X
        result = modelling.instantiate_model(model_input, n_timesteps=2)

        self.assertEqual(result["net"].n_layers, 2)
        self.assertEqual(
            str(type(result["criterion"])), "<class 'torch.nn.modules.loss.L1Loss'>"
        )
        self.assertEqual(
            str(type(result["optimizer"])), "<class 'torch.optim.adam.Adam'>"
        )


if __name__ == "__main__":
    unittest.main()
