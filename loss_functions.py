import numpy as np


def logistic_loss(y_hat, y, m):
    loss = (-1 / m) * (np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
    loss = np.squeeze(loss)
    return loss


def logistic_loss_derivative(y_hat, y):
    return np.divide(1 - y, 1 - y_hat) - np.divide(y, y_hat)


def max_likelihood_loss(y_hat, y, m):
    loss = (-1 / m) * np.sum(y * np.log(y_hat))
    loss = np.squeeze(loss)
    return loss


def max_likelihood_loss_derivative(y_hat, y):
    return -np.divide(y, y_hat)


def root_mean_squared_error(y_hat, y, m):
    return (1 / m) * np.sum(np.square(y - y_hat))


def root_mean_squared_error_derivative(y_hat, y):
    return (-2) * (y - y_hat)


def text2func(name):
    if name == "logistic_loss":
        return logistic_loss, logistic_loss_derivative
    elif name == "max_likelihood_loss":
        return max_likelihood_loss, max_likelihood_loss_derivative
    elif name == "root_mean_squared_error":
        return root_mean_squared_error, root_mean_squared_error_derivative
    else:
        raise NameError('Wrong name of loss function.')
