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


def softmax(Z):
	Z_exp = np.exp(Z)
	return Z_exp / np.sum(Z_exp, axis=0)


def softmax_derivative(Z):
	"""
	Backpropagation for softmax is done in easier way (dL/dZ = A - y),
	so it is not needed.
	"""
	return None


def linear(Z):
	return Z


def linear_derivative(Z):
	return np.ones_like(Z)


def text2func(name: str):
	if name == 'sigmoid':
		return sigmoid, sigmoid_derivative
	elif name == 'relu': 
		return relu, relu_derivative
	elif name == 'leaky_relu':
		return leaky_relu, leaky_relu_derivative
	elif name == 'tanh':
		return tanh, tanh_derivative
	elif name == 'softmax':
		return softmax, softmax_derivative
	elif name == 'linear':
		return linear, linear_derivative
	else:
		raise NameError('Wrong name of activation function.')



