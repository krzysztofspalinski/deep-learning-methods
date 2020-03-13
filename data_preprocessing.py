import numpy as np


def one_hot_encode(y):
    y = y - 1
    y = y.flatten()
    y_ohc = np.zeros((y.size, int(np.max(y)) + 1))
    y_ohc[np.arange(y.size), y.astype(np.int)] = 1
    return y_ohc
