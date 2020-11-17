import unittest
from nowcast_lstm import test

class TestTest(unittest.TestCase):
    def test_test(self):
	# test 1
        self.assertEqual(test.test("hi"), "hi")

	# test 2
	self.assertEqual(1, 1)
	
if __name__ == "__main__":
    unittest.main()
