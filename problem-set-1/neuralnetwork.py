import nn_basic_functions as func
import numpy as np
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class Layer:
    """

    """
    def __init__(self, n_of_neurons: int, n_of_neurons_prev, activation_function_and_derivative):
        self.activation_function, self.activation_function_derivative = func.af_text2func(activation_function_and_derivative)
        self.weights = func.initialize_weights(n_of_neurons_prev, n_of_neurons)
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
    """

    """
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

    def _predict_probability(self, X):
        #TODO: for prediction outside class we need transposition
        """
        predict
        :param X:
        :return:
        """
        for i in range(len(self.layers)):
            X, Z = self.layers[i].propagate(X)
        return Z


    def train(self, X, y, epochs, validation_split=0.1, verbosity=True):
        """

        :param X:
        :param y:
        :param epochs:
        :param validation_split:
        :param verbosity:
        :return:
        """

        if validation_split > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split)

            X_train = np.reshape(X_train, (X_train.shape[0], -1)).T
            X_test = np.reshape(X_test, (X_test.shape[0], -1)).T
            y_train = np.reshape(y_train, (1, y_train.shape[0]))
            y_test = np.reshape(y_test, (1, y_test.shape[0]))
        else:
            X_train = np.reshape(X, (X.shape[0], -1)).T
            y_train = np.reshape(y, (1, y.shape[0]))

        loss_on_epoch = [0 for i in range(epochs)]


        def _create_batches(n_obs, batch_size):
            idxs = [i for i in range(n_obs)]
            batches = []
            while len(idxs) > batch_size:
                batch = random.sample(idxs, batch_size)
                batches.append(batch)
                for i in batch: idxs.remove(i)
            if len(idxs) != 0: batches.append(idxs)
            return batches

        batches = _create_batches(X_train.shape[0], self.batch_size)

        for epoch in range(epochs):

            # TODO: we need functionality of using any metrics
            accuracy_score = 0

            progress_bar = range(len(batches))
            if verbosity: progress_bar = tqdm(progress_bar)

            for batch_num in progress_bar:

                X_train_batch = X_train[:, batches[batch_num]]
                y_train_batch = y_train[:, batches[batch_num]]
                # propagation

                self.cache['A0'] = X_train_batch
                A = X_train_batch
                for i in range(len(self.layers)):
                    A, Z = self.layers[i].propagate(A)
                    self.cache['Z' + str(i + 1)] = Z
                    self.cache['A' + str(i + 1)] = A

                loss_on_epoch[epoch] += func.loss_function(A, y_train_batch, self.batch_size)

                #prediction

                train_pred = (self._predict_probability(X_train_batch) >= 0.5)
                # updating accuracy score
                # weighted average of present and new batch accuracy
                # TODO: we need more scalability here

                accuracy_score_new = func.accuracy_score(y_train_batch, train_pred)

                accuracy_score = accuracy_score_new * self.batch_size + accuracy_score * self.batch_size * batch_num
                accuracy_score /= self.batch_size * (1 + batch_num) #dividing by weights

                progress_bar.set_description(f"Epoch {epoch+1} -- Train accuracy: {accuracy_score:.3f}")


                # TODO: it should work for regression tasks also;
                dA = - (np.divide(y_train_batch, A) - np.divide(1 - y_train_batch, 1 - A))

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

            # if epoch % 100 == 0:
            #     print(f'Cost after {epoch} epoch: {loss_on_epoch[epoch]}')
            test_pred = (self._predict_probability(X_test) >= 0.5)
            print(f"Validation accuracy:{func.accuracy_score(y_test, test_pred):.3f}")

            #???
            self.loss_on_epoch = loss_on_epoch

    def plot_loss(self):
        epochs = len(self.loss_on_epoch)
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(111)
        ax1.plot(list(range(epochs)), self.loss_on_epoch)
        plt.show()




def main():
    pass

if __name__ == "__main__":
    main()