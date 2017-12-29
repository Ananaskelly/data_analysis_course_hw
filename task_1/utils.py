import scipy.io as sio
import numpy as np
from scipy import linalg


def get_x_matrix(x_init, k):
    n = x_init.shape[0]
    x_b = np.zeros((n, k))
    for i in range(k):
        x_b[:, i] = x_init ** i
    return x_b


def ridge(A, b, alphas):
    U, s, Vt = linalg.svd(A, full_matrices=False)
    d = s / (s[:, np.newaxis].T ** 2 + alphas[:, np.newaxis])
    return np.dot(d * U.T.dot(b), Vt).T


def HuberLoss(w, X, y):
    a = y - np.dot(X, w)
    delta = 10

    norm_a = np.linalg.norm(a)
    if norm_a <= delta:
        loss = 1/2 * a ** 2
    else:
        loss = delta * (norm_a - 1/2 * delta)

    return loss


def lasso(w, alpha, X, y):
    a = y - np.dot(X, w)
    return np.linalg.norm(a) + np.sum(np.abs(w)) * alpha
