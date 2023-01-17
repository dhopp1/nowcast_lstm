import unittest
import numpy as np
import pandas as pd
import torch

from nowcast_lstm import data_setup, modelling


class TestModelling(unittest.TestCase):
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

    def test_instantiate_model(self):
        model_input = data_setup.gen_model_input(
            data_setup.gen_dataset(self.x, "target")["na_filled_dataset"], n_timesteps=2
        )[
            0
        ]  # first tuple of the function, X
        result = modelling.instantiate_model(model_input, n_timesteps=2)

        self.assertEqual(result["mv_lstm"].n_layers, 2)
        self.assertEqual(
            str(type(result["criterion"])), "<class 'torch.nn.modules.loss.L1Loss'>"
        )
        self.assertEqual(
            str(type(result["optimizer"])), "<class 'torch.optim.adam.Adam'>"
        )

    def test_different_optimizer(self):
        model_input = data_setup.gen_model_input(
            data_setup.gen_dataset(self.x, "target")["na_filled_dataset"], n_timesteps=2
        )[
            0
        ]  # first tuple of the function, X
        result = modelling.instantiate_model(
            model_input,
            n_timesteps=2,
            optimizer=torch.optim.SGD,
            optimizer_parameters={"lr": 1e-3, "weight_decay": 0.96},
        )

        self.assertEqual(result["mv_lstm"].n_layers, 2)
        self.assertEqual(
            str(type(result["criterion"])), "<class 'torch.nn.modules.loss.L1Loss'>"
        )
        self.assertEqual(
            str(type(result["optimizer"])), "<class 'torch.optim.sgd.SGD'>"
        )
        self.assertEqual(
            str((result["optimizer"])).strip("\n"),
            "SGD (\nParameter Group 0\n    dampening: 0\n    lr: 0.001\n    momentum: 0\n    nesterov: False\n    weight_decay: 0.96\n)".strip(
                "\n"
            ),
        )

    def test_train_model(self):
        model_input = data_setup.gen_model_input(
            data_setup.gen_dataset(self.x, "target")["na_filled_dataset"], n_timesteps=2
        )
        result = modelling.instantiate_model(model_input[0], n_timesteps=2)
        model = result["mv_lstm"]
        crit = result["criterion"]
        opt = result["optimizer"]
        model_result = modelling.train_model(
            model_input[0], model_input[1], model, crit, opt, quiet=True
        )
        model = model_result["mv_lstm"]
        loss = model_result["train_loss"]

        self.assertEqual(len(loss), 200)

    def test_predict(self):
        model_input = data_setup.gen_model_input(
            data_setup.gen_dataset(self.x, "target")["na_filled_dataset"], n_timesteps=2
        )
        result = modelling.instantiate_model(model_input[0], n_timesteps=2)
        model = result["mv_lstm"]
        crit = result["criterion"]
        opt = result["optimizer"]
        model_result = modelling.train_model(
            model_input[0], model_input[1], model, crit, opt, quiet=True
        )
        model = model_result["mv_lstm"]
        preds = modelling.predict(model_input[0], model)

        self.assertEqual(len(preds), 2)


if __name__ == "__main__":
    unittest.main()
