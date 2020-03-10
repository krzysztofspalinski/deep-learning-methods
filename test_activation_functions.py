import unittest
import activation_functions as af
import numpy as np


class TestActivationFunctions(unittest.TestCase):

    def test_sigmoid(self):
        result = af.sigmoid(np.array([[1, 2], [3, 4]]))
        expected_result = np.array([[0.731058, 0.880797], [0.952574, 0.982013]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_sigmoid_derivative(self):
        result = af.sigmoid_derivative(np.array([[1, 2], [3, 4]]))
        expected_result = np.array([[0.365529, 0.236882], [0.113549, 0.046572]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_relu(self):
        result = af.relu(np.array([[10, -5.3], [-195, 13.6]]))
        expected_result = np.array([[10, 0], [0, 13.6]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_relu_derivative(self):
        result = af.relu_derivative(np.array([[10, -5.3], [-195, 13.6]]))
        expected_result = np.array([[1, 0], [0, 1]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_leaky_relu(self):
        result = af.leaky_relu(np.array([[10, -5.3], [-195, 13.6]]))
        expected_result = np.array([[10, -0.053], [-1.95, 13.6]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_leaky_relu_derivative(self):
        result = af.leaky_relu_derivative(np.array([[10, -5.3], [-195, 13.6]]))
        expected_result = np.array([[1, 0.01], [0.01, 1]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_tanh(self):
        result = af.tanh(np.array([[1, 2], [3, 4]]))
        expected_result = np.array([[0.761594, 0.964027], [0.995054, 0.99932]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_tanh_derivative(self):
        result = af.tanh_derivative(np.array([[1, 2], [3, 4]]))
        expected_result = np.array([[0.419974, 0.070650], [0.009866, 0.001340]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

    def test_softmax(self):
        result = af.softmax(np.array([[1, -1], [3, 4]]))
        expected_result = np.array([[0.119202, 0.006692], [0.880797, 0.993307]])
        difference = result - expected_result
        self.assertTrue(np.linalg.norm(difference) < 1e-4)

        A = af.softmax(np.random.randn(5, 10))
        self.assertTrue(np.linalg.norm(np.sum(A, axis=0, keepdims=True) - 1) < 1e-4)

    def test_text2func(self):
        with self.assertRaises(NameError):
            af.text2func('notAnActivationFunction')


if __name__ == "__main__":
    unittest.main()
