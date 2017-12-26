import numpy as np
import math

from task_2.sub_task_1 import utils

np.set_printoptions(threshold=math.nan)
EPOCH = 10
REG_STEP = 1e-3


class SimpleSVM:

    def __init__(self, X, y, class_num, dim):
        self.X = X
        self.y = y
        self.dim = dim
        self.k = class_num
        self.W = np.random.randn(self.dim, self.k)*0.01

    def loss(self, y, pred):

        if y == pred:
            return 0
        else:
            return 1

    def train(self, reg_step=REG_STEP):
        objective = lambda w: self.risk_emp(w, self.X, self.y, self.k, reg_step)
        w_init = self.W.reshape(self.dim*self.k)
        w_opt = self.fmin_gd(objective, w_init, step=0.05, max_iter=300)
        self.W = w_opt.reshape(self.dim, self.k)

    def loss_augmented_decoding(self, x, W, y):

        N = x.shape[0]
        scores = np.dot(x, W)

        idx = np.argsort(scores, axis=1)
        idx_max1 = idx[:, -1]
        idx_max2 = idx[:, -2]

        d1 = scores[range(N), idx_max1] - scores[range(N), y]
        d2 = scores[range(N), y] - scores[range(N), idx_max2]

        ind = (d1 > 0) * idx_max1 + (d1 == 0) * (d2 > 1) * y + (d1 == 0) * (d2 < 1) * idx_max2

        return ind

    def fmin_gd(self, objective, w_init, step=0.04, max_iter=1000):
        w = w_init
        for i in range(max_iter):
            _, gradient = objective(w)
            w -= step * gradient
        return w

    def risk_emp(self, w, X, y, K, lmbda):
        N, dim = X.shape[0], X.shape[1]
        W = w.reshape(dim, K)

        y_hat = self.loss_augmented_decoding(X, W, y)

        scores = np.dot(X, W)

        loss = np.zeros((N,))
        for i in range(N):
            delta = y[i] != y_hat[i]
            loss[i] = -scores[i, y[i]] + scores[i, y_hat[i]] + delta

        risk = 1 / N * np.sum(loss) + 0.5 * lmbda * np.sum(W ** 2)
        grad_W = lmbda * W
        idx = loss > 0
        for k in range(K):
            idx_k_hat = np.logical_and(idx, y_hat == k)
            idx_k = np.logical_and(idx, y == k)
            grad_W[:, k] = grad_W[:, k] + 1 / N * (np.sum(X[idx_k_hat, :], axis=0) - np.sum(X[idx_k, :], axis=0)).T

        return risk, grad_W.flatten()

    def get_prediction(self, x):
        return np.argmax(np.dot(x, self.W))
