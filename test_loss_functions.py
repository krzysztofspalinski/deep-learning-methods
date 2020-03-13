import unittest
import loss_functions as lf
import numpy as np


class TestLossFunctions(unittest.TestCase):

    def test_logistic_loss(self):
        result = lf.logistic_loss(np.array([.9, 0.02, .8, .73]), np.array([1, 0, 1, 1]), 4)
        expected_result = 0.165854
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_logistic_loss_derivative(self):
        result = lf.logistic_loss_derivative(np.array([.9, 0.02, .8, .73]), np.array([1, 0, 1, 1]))
        expected_result = np.array([-1.111111, 1.020408, -1.25, -1.369863])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_likelihood_loss(self):
        result = lf.max_likelihood_loss(np.array([.9, 0.02, .8, .73]), np.array([1, 0, 1, 1]), 4)
        expected_result = 0.160803
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_likelihood_loss_derivative(self):
        result = lf.max_likelihood_loss_derivative(np.array([[.9, 0.1, .2], [.1, .9, .8]]),
                                                   np.array([[1, 0, 0], [0, 1, 1]]))
        expected_result = np.array([[-1.111111, 0, 0], [0, -1.111111, -1.25]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_text2func(self):
        with self.assertRaises(NameError):
            lf.text2func('notALossFunction')


if __name__ == "__main__":
    unittest.main()
