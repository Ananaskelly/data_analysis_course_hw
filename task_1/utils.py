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


def HuberLoss(w, X, y, alpha=0):
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


def mse(w, X, y):
    prediction = np.dot(X, w)
    return np.mean((y - prediction)**2)


def k_fold_validation(x_val, y_val, alphas, k, get_weights, get_loss):

    losses = []
    batch = int(x_val.shape[0]/k)
    for alpha in alphas:
        mean_loss = 0
        for idx in range(k):
            test_set_x = x_val[idx*batch:(idx+1)*batch]
            test_set_y = y_val[idx*batch:(idx+1)*batch]
            train_set_x = np.concatenate((x_val[0:idx * batch], x_val[(idx + 1) * batch:]), axis=0)
            train_set_y = np.concatenate((y_val[0:idx * batch], y_val[(idx + 1) * batch:]), axis=0)

            weights = get_weights(train_set_x, train_set_y, alpha)
            mean_loss += get_loss(weights, test_set_x, test_set_y)

        print(mean_loss/k)
        losses.append(mean_loss/k)

    return np.argmin(losses)