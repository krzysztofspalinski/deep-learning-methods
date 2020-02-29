import nn_basic_functions as func
import numpy as np


class Layer:
    def __init__(self, n_of_neurons: int, n_of_neurons_prev, activation_function_and_derivative):
        self.activation_function, self.activation_function_derivative = activation_function_and_derivative
        self.weights = func.initialize_weights(n_of_neurons, n_of_neurons_prev)

    def propagate(self, A):
        Z = np.dot(self.weights['W'], A) + self.weights['b']
        A = self.activation_function(Z)
        return A, Z

    def back_propagate(self, dA, Z, A_prev, m):
        dZ = dA * self.activation_function_derivative(Z)
        dW = (1 / m) * np.dot(dZ, np.transpose(A_prev))
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(np.transpose(self.weights['W']), dZ)
        return dW, db, dA_prev

    def update_weights(self, dW, db, learning_rate):
        self.weights['W'] = self.weights['W'] - learning_rate * dW
        self.weights['b'] = self.weights['b'] - learning_rate * db


class NeuralNetwork:
    def __init__(self, input_dim: int, neuron_numbers: list, activation_functions: list, learning_rate: float):
        self.learning_rate = learning_rate
        self.neuron_numbers = neuron_numbers
        self.activation_functions = activation_functions
        self.cache = {}
        self.loss_on_iteration = None
        self.X = None
        self.y = None
        self.n = input_dim  # dimension of observations
        self.m = None  # number of observations
        self.layers = None
        self.create_layers()

    def create_layers(self):
        self.layers = [Layer(self.neuron_numbers[0], self.n, func.af_text2func(self.activation_functions[0]))]
        for i in range(1, len(self.neuron_numbers)):
            self.layers.append(Layer(self.neuron_numbers[i], self.neuron_numbers[i - 1],
                                     func.af_text2func(self.activation_functions[i])))

    def train(self, X_train, y_train, iterations):
        """
        Trains the network to fit the dataset.
        :param X_train: training set, observations in rows.
        :param y_train: test set to compare with last layer, observations in rows.
        :param iterations: number of iterations.
        """
        self.X = X_train.T
        self.y = y_train.T

        self.m = self.X.shape[1]

        loss_on_iteration = []
        for iteration in range(iterations):
            A = self.X
            self.cache['A0'] = A
            for i in range(len(self.layers)):
                A, Z = self.layers[i].propagate(A)
                self.cache['Z' + str(i + 1)] = Z
                self.cache['A' + str(i + 1)] = A
            loss_on_iteration.append(func.loss_function(A, self.y, self.m))

            dA = - (np.divide(self.y, A) - np.divide(1 - self.y, 1 - A))

            for i in range(len(self.layers) - 1, -1, -1):
                Z = self.cache['Z' + str(i + 1)]
                A_prev = self.cache['A' + str(i)]
                dW, db, dA_prev = self.layers[i].back_propagate(dA, Z, A_prev, self.m)
                self.layers[i].update_weights(dW, db, self.learning_rate)
                dA = dA_prev

        self.loss_on_iteration = loss_on_iteration
        return loss_on_iteration

    def predict(self, X):
        for i in range(len(self.layers)):
            output: np.ndarray
            X, output = self.layers[i].propagate(X)
        return output

