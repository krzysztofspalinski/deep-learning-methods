import numpy as np


def one_hot_encode(y):
    y = y - 1
    y = y.flatten()
    y_ohc = np.zeros((y.size, int(np.max(y)) + 1))
    y_ohc[np.arange(y.size), y.astype(np.int)] = 1
    return y_ohc


def train_test_split(X, y, test_size, seed=42):
    np.random.seed(seed)
    assert(X.shape[0] == y.shape[0])
    dataset_size = X.shape[0]

    indices = np.random.permutation(dataset_size)
    last_training_index = int(dataset_size * (1 - test_size))
    training_idx, test_idx = indices[:last_training_index], indices[last_training_index:]

    X_train, X_test, y_train, y_test = X[training_idx, :], X[test_idx, :], y[training_idx, :], y[test_idx, :]
    return X_train, X_test, y_train, y_test


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
        self.std_dev[self.std_dev == 0] = 1

        return (X - self.mean) / self.std_dev

    def transform(self, X):
        return (X - self.mean) / self.std_dev
