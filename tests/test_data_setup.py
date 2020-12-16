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

    x_ragged = pd.DataFrame(
        {
            "date": ["a", "b", "c"],
            "var1": [1, 2, np.nan],
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

    missing_y = pd.DataFrame(
        {
            "date": ["a", "b", "c", "c", "c", "c", "c"],
            "var1": [1, 2, 3, 4, 5, 6, 7],
            "var2": [4, np.nan, 6, 7, 8, 9, 10],
            "target": [7, 8, np.nan, 10, 11, np.nan, 13],
        }
    )

    def test_gen_dataset(self):
        # fill nas with mean
        # def gen_dataset(rawdata, target_variable, fill_na_func=lambda x: x.fillna(np.mean(x)), fill_na_other_df=None, fill_ragged_edges=np.mean):

        self.assertTrue(
            (
                data_setup.gen_dataset(self.x, "target", np.mean)["na_filled_dataset"]
                == np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]], np.float64)
            ).all()
        )
        # fill nas with scalar, lambda
        self.assertTrue(
            (
                data_setup.gen_dataset(self.x, "target", lambda x: -999)[
                    "na_filled_dataset"
                ]
                == np.array([[1, 4, 7], [2, -999, 8], [3, 6, 9]], np.float64)
            ).all()
        )
        # fill nas with median from another dataframe
        self.assertTrue(
            (
                data_setup.gen_dataset(
                    self.x,
                    "target",
                    fill_na_func=np.nanmedian,
                    fill_na_other_df=self.long_x,
                )["na_filled_dataset"]
                == np.array([[1, 4, 7], [2, 7.5, 8], [3, 6, 9]], np.float64)
            ).all()
        )
        # ragged edges mean
        self.assertTrue(
            (
                data_setup.gen_dataset(
                    self.x_ragged,
                    "target",
                    fill_na_func=np.nanmedian,
                    fill_ragged_edges=np.mean,
                    fill_na_other_df=self.long_x,
                )["na_filled_dataset"]
                == np.array([[1, 4, 7], [2, 7.5, 8], [4, 6, 9]], np.float64)
            ).all()
        )
        # ragged edges ARMA
        self.assertTrue(
            (
                data_setup.gen_dataset(
                    self.x_ragged,
                    "target",
                    fill_na_func=np.nanmedian,
                    fill_ragged_edges="ARMA",
                    fill_na_other_df=self.long_x,
                )["na_filled_dataset"].round(0)
                == np.array([[1, 4, 7], [2, 8, 8], [1, 6, 9]], np.float64)
            ).all()
        )

    def test_gen_model_input(self):
        # normal X
        result = data_setup.gen_model_input(
            data_setup.gen_dataset(self.x, "target", np.mean)["na_filled_dataset"],
            n_timesteps=2,
        )
        self.assertTrue(
            (
                result[0] == np.array([[[1, 4], [2, 5]], [[2, 5], [3, 6]]], np.float64)
            ).all()
        )
        # missing ys
        result = data_setup.gen_model_input(
            data_setup.gen_dataset(self.missing_y, "target", np.nanmedian)[
                "na_filled_dataset"
            ],
            n_timesteps=2,
        )
        self.assertTrue(
            (
                result[0]
                == np.array(
                    [
                        [[1, 4], [2, 7.5]],
                        [[3, 6], [4, 7]],
                        [[4, 7], [5, 8]],
                        [[6, 9], [7, 10]],
                    ],
                    np.float64,
                )
                # result[1] == np.array([8, 10, 11, 13], np.float64)
            ).all()
        )

    def test_gen_ragged_X(self):
        ragged_dataset = data_setup.gen_dataset(self.long_x, "target")[
            "for_ragged_dataset"
        ]
        X = data_setup.gen_model_input(ragged_dataset, n_timesteps=2)
        result = data_setup.gen_ragged_X(
            X[0],
            [1, 2],
            0,
            ragged_dataset,
            "target",
            fill_ragged_edges=np.nanmedian,
            backup_fill_method=np.nanmedian,
        )

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
