import activation_functions as af
import loss_functions as lf
import numpy as np
import optimizers

class Layer:
    """
    MLP's Layer class
    """
    def __init__(self,
                 n_of_neurons: int,
                 n_of_neurons_prev,
                 activation_function_and_derivative,
                 bias=True):
        """
        :param n_of_neurons: output size
        :param n_of_neurons_prev: input size
        :param activation_function_and_derivative: activation function name
        :param bias: trigger defining if bias should be fitted
        """
        self.activation_function, self.activation_function_derivative = activation_function_and_derivative
        self.weights = Layer.initialize_weights(n_of_neurons, n_of_neurons_prev)
        self.bias = bias
        if not self.bias: self.weights['b'] *= 0

    def propagate(self, A):
        Z = np.dot(self.weights['W'], A) + self.weights['b']
        A = self.activation_function(Z)
        return A, Z

    def back_propagate(self, dA, Z, A_prev, m, softmax_backpropagation=False, A=None, y=None):
        if softmax_backpropagation:
            dZ = A - y
        else:
            dZ = dA * self.activation_function_derivative(Z)
        dW = (1 / m) * np.dot(dZ, np.transpose(A_prev))
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(np.transpose(self.weights['W']), dZ)
        return dW, db, dA_prev

    def update_weights(self, dW, db, learning_rate):
        self.weights['W'] = self.weights['W'] - learning_rate * dW
        self.weights['b'] = self.weights['b'] - learning_rate * db
        self.weights['dW'] = dW
        self.weights['db'] = db

        if not self.bias: self.weights['b'] *= 0

    def get_weights(self):
        return self.weights

    @staticmethod
    def initialize_weights(n_of_neurons: int, n_of_neurons_prev: int):
        W = np.random.randn(n_of_neurons, n_of_neurons_prev) * np.sqrt(2 / n_of_neurons_prev)
        b = np.random.randn(n_of_neurons, 1)
        weights = {
            'W': W,
            'b': b,
            'dW': 0 * W,
            'db': 0 * b}
        return weights


class NeuralNetwork:
    """
    Multilayer perceptron NumPy implementation
    """
    def __init__(self,
                 input_dim: int,
                 neuron_numbers: list,
                 activation_functions: list,
                 loss_function,
                 learning_rate: float,
                 optimizer,
                 bias=True):
        """
        :param input_dim:
        :param neuron_numbers:
        :param activation_functions:
        :param loss_function:
        :param learning_rate:
        :param bias:
        """
        self.learning_rate = learning_rate
        self.neuron_numbers = neuron_numbers
        self.activation_functions = activation_functions
        self.loss_function, self.loss_function_derivative = lf.text2func(loss_function)
        self.cache = {}
        self.loss_on_iteration = None
        self.X = None
        self.y = None
        self.n = input_dim  # dimension of observations
        self.m = None  # number of observations
        self.layers = None
        self.bias = bias
        self.optimizer = optimizer
        self.create_layers(self.bias)

    def create_layers(self, bias):
        self.layers = [Layer(self.neuron_numbers[0],
                             self.n,
                             af.text2func(self.activation_functions[0]),
                             bias)]
        for i in range(1, len(self.neuron_numbers)):
            self.layers.append(Layer(self.neuron_numbers[i],
                                     self.neuron_numbers[i - 1],
                                     af.text2func(self.activation_functions[i]),
                                     bias))

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
            loss_on_iteration.append(self.loss_function(A, self.y, self.m))

            # iterator for upcoming while loop
            i = len(self.layers) - 1

            # dA is dL/dA, dZ is dL/dZ, etc.
            if self.activation_functions[-1] == 'softmax':
                Z = self.cache['Z' + str(i + 1)]
                A_prev = self.cache['A' + str(i)]
                dW, db, dA_prev = self.layers[i].back_propagate(None, Z, A_prev, self.m, True, A, self.y)

                # dict of W, b, dW, db in i-th layer
                old_weights = self.layers[i].get_weights()
                dW = self.optimizer.optimize(old_weights['dW'], dW)
                db = self.optimizer.optimize(old_weights['db'], db)

                self.layers[i].update_weights(dW, db, self.learning_rate)
                dA = dA_prev
                i = i - 1
            else:
                dA = self.loss_function_derivative(A, self.y)

            while i >= 0:
                Z = self.cache['Z' + str(i + 1)]
                A_prev = self.cache['A' + str(i)]
                dW, db, dA_prev = self.layers[i].back_propagate(dA, Z, A_prev, self.m)

                # dict of W, b, dW, db in i-th layer
                old_weights = self.layers[i].get_weights()
                dW = self.optimizer.optimize(old_weights['dW'], dW)
                db = self.optimizer.optimize(old_weights['db'], db)

                self.layers[i].update_weights(dW, db, self.learning_rate)
                dA = dA_prev
                i = i - 1

        self.loss_on_iteration = loss_on_iteration
        return loss_on_iteration

    def predict(self, X):
        A = X.T
        for i in range(len(self.layers)):
            output: np.ndarray
            A, _ = self.layers[i].propagate(A)
        return A

