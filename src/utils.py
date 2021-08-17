import numpy as np


def angdiff(th1, th2):
    d = th1 - th2
    d = np.mod(d + np.pi, 2 * np.pi) - np.pi
    return -d


def wraptopi(x):
    pi = np.pi
    x = x - np.floor(x / (2 * pi)) * 2 * pi
    if x >= pi:
        return x - 2 * pi
    return x
