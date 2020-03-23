import numpy as np

epsilon = 1e-10


def logistic_loss(y_hat, y, m):
    loss = (-1 / m) * (np.sum(y * np.log(y_hat + epsilon) + (1 - y) * np.log(1 - y_hat + epsilon)))
    loss = np.squeeze(loss)
    return loss


def logistic_loss_derivative(y_hat, y):
    return np.divide(1 - y, 1 - y_hat + epsilon) - np.divide(y, y_hat + epsilon)


def max_likelihood_loss(y_hat, y, m):
    loss = (-1 / m) * np.sum(y * np.log(y_hat + epsilon))
    loss = np.squeeze(loss)
    return loss


def max_likelihood_loss_derivative(y_hat, y):
    return -np.divide(y, y_hat + epsilon)


def mean_squared_error(y_hat, y, m):
    return (1 / m) * np.sum(np.square(y - y_hat))


def mean_squared_error_derivative(y_hat, y):
    return (-2) * (y - y_hat)


def mean_squared_log_error(y_hat, y, m):
    return (1 / m) * np.sum(np.square(np.log((y + 1) / (y_hat + 1))))


def mean_squared_log_error_derivative(y_hat, y):
    return (-2) * (np.log((y + 1) / (y_hat + 1))) / (y_hat + 1)


def text2func(name):
    if name == "logistic_loss":
        return logistic_loss, logistic_loss_derivative
    elif name == "max_likelihood_loss":
        return max_likelihood_loss, max_likelihood_loss_derivative
    elif name == "mean_squared_error":
        return mean_squared_error, mean_squared_error_derivative
    elif name == "mean_squared_log_error":
        return mean_squared_log_error, mean_squared_log_error_derivative
    else:
        raise NameError('Wrong name of loss function.')
