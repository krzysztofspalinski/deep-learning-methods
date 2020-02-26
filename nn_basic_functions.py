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
	Z = ((Z>0) * Z) + ((Z<0) * Z * 0.01)
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


def loss_function(y_hat, y, m):
	loss = (-1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
	loss = np.squeeze(loss)
	return loss


def initialize_weights(n_of_neurons: int, n_of_neurons_prev: int):
	W = np.random.randn(n_of_neurons, n_of_neurons_prev) * np.sqrt(2 / n_of_neurons_prev)
	b = np.random.randn(n_of_neurons, 1)
	weights = {
		'W': W,
		'b': b}
	return weights


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