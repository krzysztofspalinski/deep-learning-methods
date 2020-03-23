import numpy as np


def one_hot_encode(y):
    y = y - 1
    y = y.flatten()
    y_ohc = np.zeros((y.size, int(np.max(y)) + 1))
    y_ohc[np.arange(y.size), y.astype(np.int)] = 1
    return y_ohc


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X_train):
        self.min = np.min(X_train, axis=0)
        self.max = np.max(X_train, axis=0)

    def transform(self, X):
        X = X - self.min
        X = X / (self.max - self.min)
        return X


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std_dev = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std_dev = np.var(X, axis=0) ** 0.5

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std_dev = np.var(X, axis=0) ** 0.5
        return (X - self.mean) / self.std_dev

    def transform(self, X):
        return (X - self.mean) / self.std_dev
