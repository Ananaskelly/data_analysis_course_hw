import scipy.io as sio
import numpy as np
from scipy import linalg


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