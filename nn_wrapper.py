from neuralnetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
import random
import numpy as np
import nn_basic_functions as func
from matplotlib import pyplot as plt


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()


class NeuralNetworkWrapper:
    def __init__(self, input_dim, neuron_numbers, activation_functions, learning_rate, batch_size=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.NN = NeuralNetwork(input_dim, neuron_numbers, activation_functions, learning_rate)
        self.loss_on_batch = []

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
        if validation_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split)
        else:
            X_train = X
            y_train = y

        y_train = np.reshape(y_train, (y_train.shape[0], -1))

        batches = self.create_batches(X_train.shape[0], self.batch_size)

        for epoch in range(epochs):
            loss = None
            for batch_num in range(len(batches)):
                X_train_batch = X_train[batches[batch_num], :]
                y_train_batch = y_train[batches[batch_num], :]
                loss = self.NN.train(X_train_batch, y_train_batch, 1)[0]
                self.loss_on_batch.append(loss)
            if verbosity and epoch % 100 == 0 :
                print(f'Loss after {epoch + 1} epochs: {loss}', end="\n")
        return

    def plot_loss(self):
        epochs = len(self.loss_on_batch)
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(list(range(epochs)), self.loss_on_batch)
        plt.show()