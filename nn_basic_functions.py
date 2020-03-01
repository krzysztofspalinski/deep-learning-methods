import numpy as np


def sigmoid(Z):
	Z = 1 / (1 + np.exp(-Z))
	return Z


def sigmoid_derivative(Z):
	Z = sigmoid(Z) * sigmoid(1 - Z)
	return Z


def relu(Z):
	Z = np.maximum(Z, 0)
	return Z


def relu_derivative(Z):
	derivative = np.zeros(Z.shape)
	derivative[Z > 0] = 1
	return derivative


def leaky_relu(Z):
	Z = ((Z > 0) * Z) + ((Z < 0) * Z * 0.01)
	return Z


def leaky_relu_derivative(Z):
	derivative = np.zeros(Z.shape)
	derivative[Z > 0] = 1
	derivative[Z < 0] = 0.01
	return derivative


def tanh(Z):
	exp2z = np.exp(2*Z)
	Z = (exp2z - 1)/(exp2z + 1)
	return Z


def tanh_derivative(Z):
	Z = 1 - tanh(Z)**2
	return Z


def logistic_loss(y_hat, y, m):
	loss = (-1 / m) * (np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
	loss = np.squeeze(loss)
	return loss


def max_likelihood_loss(y_hat, y, m):
	loss = (-1 / m) * np.sum(y * np.log(y_hat))
	loss = np.squeeze(loss)
	return loss


def initialize_weights(n_of_neurons: int, n_of_neurons_prev: int):
	W = np.random.randn(n_of_neurons, n_of_neurons_prev) * np.sqrt(2 / n_of_neurons_prev)
	b = np.random.randn(n_of_neurons, 1)
	weights = {
		'W': W,
		'b': b}
	return weights


def loss_text2func(name):
	if name == "logistic_loss":
		return logistic_loss
	elif name == "max_likelihood_loss":
		return max_likelihood_loss
	else:
		raise Exception('Wrong name of loss function.')


def af_text2func(name: str):
	if name == 'sigmoid':
		return sigmoid, sigmoid_derivative
	elif name == 'relu': 
		return relu, relu_derivative
	elif name == 'leaky_relu':
		return leaky_relu, leaky_relu_derivative
	elif name == 'tanh':
		return tanh, tanh_derivative
	else:
		raise Exception('Wrong name of activation function.')


def accuracy_score(y_true, y_pred):
	return np.sum(y_true == y_pred) / y_true.shape[1]


def normalize_data(X_train, X_test):
	mi = np.mean(X_train, axis=1, keepdims=True)
	sigma = np.sqrt(np.mean(X_train ** 2, axis=1, keepdims=True))
	assert (mi.shape == (X_train.shape[0], 1))
	assert (sigma.shape == (X_train.shape[0], 1))
	X_train = (X_train - mi) / sigma
	X_test = (X_test - mi) / sigma
	return X_train, X_test
