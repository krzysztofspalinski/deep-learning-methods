from neuralnetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
import random
import numpy as np
from matplotlib import pyplot as plt


class NeuralNetworkWrapper:
    def __init__(self,
                 input_dim,
                 neuron_numbers,
                 activation_functions,
                 loss_function,
                 learning_rate,
                 optimizer,
                 batch_size=1,
                 bias=True):
        """
        Wrapper for NeuralNetwork class
        :param input_dim: input size
        :param neuron_numbers: list of hidden layers' size
        :param activation_functions: list of activation functions for each layer
        :param loss_function
        :param learning_rate
        :param batch_size
        :param bias: boolean triggering if bias has to be fitted
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.NN = NeuralNetwork(input_dim,
                                neuron_numbers,
                                activation_functions,
                                loss_function,
                                learning_rate,
                                optimizer,
                                bias)
        self.loss_on_epoch = []

    def create_batches(self, n_obs, batch_size):
        idxs = [i for i in range(n_obs)]
        batches = []
        if batch_size == -1:
            return idxs
        while len(idxs) > batch_size:
            batch = random.sample(idxs, batch_size)
            batches.append(batch)
            for i in batch:
                idxs.remove(i)
        if len(idxs) != 0:
            batches.append(idxs)
        return batches

    def train(self, X, y, epochs, validation_split=0.1, verbosity=True):
        self.validation_split = validation_split
        if validation_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split)
            y_test = np.reshape(y_test, (y_test.shape[0], -1))
            self.loss_on_epoch_valid = []
        else:
            X_train = X
            y_train = y

        y_train = np.reshape(y_train, (y_train.shape[0], -1))

        batches = self.create_batches(X_train.shape[0], self.batch_size)

        for epoch in range(epochs):
            for batch_num in range(len(batches)):
                X_train_batch = X_train[batches[batch_num], :]
                y_train_batch = y_train[batches[batch_num], :]
                self.NN.train(X_train_batch, y_train_batch, 1)

            self.loss_on_epoch.append(self.NN.loss_function(self.NN.predict(X_train), y_train.T, y_train.shape[0]))

            if validation_split > 0:
                self.loss_on_epoch_valid.append(self.NN.loss_function(self.NN.predict(X_test), y_test.T, y_test.shape[0]))
            if verbosity: print(f'Loss after {epoch + 1} epochs: {self.loss_on_epoch[-1]:^.3f}', end="\n")
        print(f'Final loss: {self.loss_on_epoch[-1]:^.3f}', end="\n")
        return

    def plot_loss(self):
        epochs = len(self.loss_on_epoch)
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(list(range(epochs)), self.loss_on_epoch, label='Training Loss')
        if self.validation_split > 0:
            ax1.plot(list(range(epochs)), self.loss_on_epoch_valid, label='Validation Loss')
        ax1.legend(loc="upper right")
        plt.show()

    def predict(self, X):
        return self.NN.predict(X)