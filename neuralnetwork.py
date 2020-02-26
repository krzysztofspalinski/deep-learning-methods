import nn_basic_functions as func
import numpy as np
from matplotlib import pyplot as plt


class NeuralNetworkClassifier:
    class Layer:
        def __init__(self, m, n_of_neurons: int, n_of_neurons_prev, activation_function_and_derivative):
            self.m = m
            self.activation_function, self.activation_function_derivative = activation_function_and_derivative
            self.weights = func.initialize_weights(n_of_neurons, n_of_neurons_prev)

        def propagate(self, A):
            Z = np.dot(self.weights['W'], A) + self.weights['b']
            A = self.activation_function(Z)
            return A, Z

        def back_propagate(self, dA, Z, A_prev):
            dZ = dA * self.activation_function_derivative(Z)
            dW = (1 / self.m) * np.dot(dZ, np.transpose(A_prev))
            db = (1 / self.m) * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(np.transpose(self.weights['W']), dZ)
            return dW, db, dA_prev

        def update_weights(self, dW, db, learning_rate):
            self.weights['W'] = self.weights['W'] - learning_rate * dW
            self.weights['b'] = self.weights['b'] - learning_rate * db

    def __init__(self, X_train, y_train, neuron_numbers: list, activation_functions: list, learning_rate):
        self.n, self.m = X_train.shape
        self.X_train, self.y_train = X_train, y_train
        self.learning_rate = learning_rate
        self.cache = {}
        self.layers = [self.Layer(self.m, neuron_numbers[0], self.n, func.af_text2func(activation_functions[0]))]
        self.loss_on_iteration = None
        for i in range(1, len(neuron_numbers)):
            self.layers.append(self.Layer(self.m, neuron_numbers[i], neuron_numbers[i - 1],
                                          func.af_text2func(activation_functions[i])))

    def train(self, iterations):
        loss_on_iteration = []
        for iteration in range(iterations):
            A = self.X_train
            self.cache['A0'] = A
            for i in range(len(self.layers)):
                A, Z = self.layers[i].propagate(A)
                self.cache['Z' + str(i + 1)] = Z
                self.cache['A' + str(i + 1)] = A
            loss_on_iteration.append(func.loss_function(A, self.y_train, self.m))

            dA = - (np.divide(self.y_train, A) - np.divide(1 - self.y_train, 1 - A))

            for i in range(len(self.layers) - 1, -1, -1):
                Z = self.cache['Z' + str(i + 1)]
                A_prev = self.cache['A' + str(i)]
                dW, db, dA_prev = self.layers[i].back_propagate(dA, Z, A_prev)
                self.layers[i].update_weights(dW, db, self.learning_rate)
                dA = dA_prev

            if iteration % 100 == 0:
                print(f'Cost after {iteration} iteration: {loss_on_iteration[iteration]}')
        self.loss_on_iteration = loss_on_iteration

    def plot_loss(self):
        iterations = len(self.loss_on_iteration)
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(list(range(iterations)), self.loss_on_iteration)
        plt.show()


def normalize_data(X_train, X_test):
    mi = np.mean(X_train, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean(X_train ** 2, axis=1, keepdims=True))
    assert (mi.shape == (X_train.shape[0], 1))
    assert (sigma.shape == (X_train.shape[0], 1))
    X_train = (X_train - mi) / sigma
    X_test = (X_test - mi) / sigma
    return X_train, X_test
