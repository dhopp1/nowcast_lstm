import unittest
import pandas as pd
import numpy as np
from nowcast_lstm import data_setup

class TestDataSetup(unittest.TestCase):
    def test_gen_dataset(self):
		# testing gen_dataset function
        x = pd.DataFrame({
				"date": ["a", "b", "c"],
				"var1": [1,2,3],
				"var2": [4,np.nan,6],
				"target": [7,8,9],
		})
        self.assertTrue(
				(data_setup.gen_dataset(x, "target") == np.array([[1,4,7], [2,0,8], [3,6,9]], np.float64)).all()
		)
	
if __name__ == "__main__":
    unittest.main()
