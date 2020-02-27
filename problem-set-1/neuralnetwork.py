import nn_basic_functions as func
import numpy as np
from matplotlib import pyplot as plt
import random

class Layer:
    def __init__(self, n_of_neurons: int, n_of_neurons_prev, activation_function_and_derivative):
        self.activation_function, self.activation_function_derivative = func.af_text2func(activation_function_and_derivative)
        self.weights = func.initialize_weights(n_of_neurons, n_of_neurons_prev)
        return

    def propagate(self, A):
        Z = np.dot(self.weights['W'], A) + self.weights['b']
        A = self.activation_function(Z)
        return A, Z

    def back_propagate(self, dA, Z, A_prev):
        dZ = dA * self.activation_function_derivative(Z)
        dW = np.dot(dZ, np.transpose(A_prev))
        db = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(np.transpose(self.weights['W']), dZ)
        return dW, db, dA_prev

    def update_weights(self, dW, db, learning_rate):
        self.weights['W'] = self.weights['W'] - learning_rate * dW
        self.weights['b'] = self.weights['b'] - learning_rate * db
        return

class NeuralNetworkClassifier:
    #TODO: what about regression?

    def __init__(self, learning_rate=1e-3, batch_size=1):

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.cache = {}
        self.layers = []
        #self.layers = [self.Layer(self.m, neuron_numbers[0], self.n, func.af_text2func(activation_functions[0]))]
        return

    def add_layer(self, layer):
        #TODO: bias trigger
        assert isinstance(layer, Layer), "Input is not a nn layer!"
        self.layers.append(layer)

    def train(self, X, y, epochs):
        loss_on_epoch = []

        def _create_batches(n_obs, batch_size):
            idxs = [i for i in range(n_obs)]
            batches = []
            while len(idxs) > batch_size:
                batch = random.sample(idxs, batch_size)
                batches.append(batch)
                for i in batch: idxs.remove(i)
            if len(idxs) != 0: batches.append(idxs)
            return batches
        batches = _create_batches(X.shape[0], self.batch_size)

        for epoch in range(epochs):

            for batch in batches:
                X_train = X[batch, :]
                y_train = y[:, batch]
                # propagation

                self.cache['A0'] = X_train
                A = X_train
                for i in range(len(self.layers)):
                    A, Z = self.layers[i].propagate(A)
                    self.cache['Z' + str(i + 1)] = Z
                    self.cache['A' + str(i + 1)] = A
                loss_on_iteration.append(func.loss_function(A, y_train, self.batch_size))

                # TODO: it should work for regression tasks also;
                dA = - (np.divide(y_train, A) - np.divide(1 - y_train, 1 - A))

                # back propagation
                for i in range(len(self.layers) - 1, -1, -1):
                    Z = self.cache['Z' + str(i + 1)]
                    A_prev = self.cache['A' + str(i)]
                    dW, db, dA_prev = self.layers[i].back_propagate(dA, Z, A_prev)

                    #
                    dW /= (1/self.batch_size)
                    db /= (1/self.batch_size)

                    self.layers[i].update_weights(dW, db, self.learning_rate)
                    dA = dA_prev

            if epoch % 100 == 0:
                print(f'Cost after {epoch} epoch: {loss_on_epoch[epoch]}')

            #???
            self.loss_on_epoch = loss_on_epoch

    def plot_loss(self):
        epochs = len(self.loss_on_epoch)
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(list(range(epochs)), self.loss_on_epoch)
        plt.show()


def normalize_data(X_train, X_test):
    mi = np.mean(X_train, axis=1, keepdims=True)
    sigma = np.sqrt(np.mean(X_train ** 2, axis=1, keepdims=True))
    assert (mi.shape == (X_train.shape[0], 1))
    assert (sigma.shape == (X_train.shape[0], 1))
    X_train = (X_train - mi) / sigma
    X_test = (X_test - mi) / sigma
    return X_train, X_test

def main():
    pass

if __name__ == "__main__":
    main()