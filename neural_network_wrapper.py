from neural_network_core import NeuralNetworkCore
from sklearn.model_selection import train_test_split
import random
import numpy as np
from matplotlib import pyplot as plt
import optimizers


class NeuralNetworkWrapper:
    def __init__(self,
                 input_dim,
                 neuron_numbers,
                 activation_functions,
                 loss_function,
                 learning_rate,
                 optimizer=optimizers.Optimizer(),
                 batch_size=128,
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
        self.NN = NeuralNetworkCore(input_dim,
                                    neuron_numbers,
                                    activation_functions,
                                    loss_function,
                                    learning_rate,
                                    optimizer,
                                    bias)
        self.loss_on_epoch = []
        self.validation_split = None
        self.loss_on_epoch_valid = None
        self.cache_weights_on_epoch = None

    @staticmethod
    def create_batches(n_obs, batch_size):
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

    def train(self,
              X,
              y,
              epochs,
              validation_split=0.1,
              verbosity=True,
              cache_weights_on_epoch=False,
              cache_accuracy=False,
              test_accuracy=None):

        # caching test set accuracy on epoch end
        if test_accuracy is not None:
            self.test_accuracy = []
            X_test, y_test = test_accuracy

        # caching train & validation accuracy on epoch end
        if cache_accuracy:
            self.accuracy = []
            self.accuracy_valid = []

        # cached network weights on epoch end
        if cache_weights_on_epoch:
            self.cache_weights_on_epoch = []
        self.validation_split = validation_split
        if validation_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split)
            y_test = np.reshape(y_test, (y_test.shape[0], -1))
            self.loss_on_epoch_valid = []
        else:
            X_train = X
            y_train = y
            X_test = None
            y_test = None

        y_train = np.reshape(y_train, (y_train.shape[0], -1))

        batches = self.create_batches(X_train.shape[0], self.batch_size)

        for epoch in range(epochs):
            for batch_num in range(len(batches)):
                X_train_batch = X_train[batches[batch_num], :]
                y_train_batch = y_train[batches[batch_num], :]
                self.NN.train(X_train_batch, y_train_batch, 1)

            self.loss_on_epoch.append(self.NN.loss_function(self.NN.predict(X_train).T, y_train.T, y_train.shape[0]))

            if validation_split > 0:
                self.loss_on_epoch_valid.append(self.NN.loss_function(self.NN.predict(X_test).T,
                                                                      y_test.T, y_test.shape[0]))
            if verbosity:
                print(f'Loss after {epoch + 1} epochs: {self.loss_on_epoch[-1]:^.3f}', end="\n")

            if cache_weights_on_epoch:
                self.cache_weights_on_epoch.append(self.NN.get_weights())

            if cache_accuracy:
                self.accuracy.append(self.eval_accuracy(y_train, self.predict_classes(X_train)))
                self.accuracy_valid.append(self.eval_accuracy(y_test, self.predict_classes(X_test)))

            if test_accuracy:
                self.test_accuracy.append(self.eval_accuracy(y_test, self.predict_classes(X_test)))

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


    def predict_classes(self, X):
        return self.NN.predict_classes(X)

    def eval_accuracy(self, y_true, y_pred):
        return np.sum(np.all(np.equal(y_true, y_pred), axis=1)) / y_true.shape[0]