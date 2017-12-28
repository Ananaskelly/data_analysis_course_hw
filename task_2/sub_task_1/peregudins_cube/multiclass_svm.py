import numpy as np
import math

from task_2.sub_task_1.peregudins_cube.base_svm import BaseSvm

np.set_printoptions(threshold=math.nan)
EPOCH = 10
REG_STEP = 1e-3


class MultiClassSVM(BaseSvm):

    def __init__(self, X, y, class_num, dim):
        self.X = X
        self.y = y
        self.dim = dim
        self.k = class_num
        self.W = np.random.randn(self.dim, self.k)*0.01
        self.b = np.random.randn(self.k)*0.01

    def train(self, rate=0.1, reg_step=REG_STEP, batch_size=128):
        objective = lambda w: self.risk_emp(w, self.X, self.y, self.k, reg_step)
        w_init = np.concatenate((self.W.reshape(self.dim*self.k), self.b), axis=0)

        w_opt = self.fmin_gd(objective, w_init, step=0.05, max_iter=300)

        self.W = w_opt[:self.dim*self.k].reshape(self.dim, self.k)
        self.b = w_opt[self.dim*self.k:]

    def loss(self, y, pred):

        if y == pred:
            return 0
        else:
            return 1

    def loss_augmented_decoding(self, x, W, b, y):

        N = x.shape[0]
        scores = np.dot(x, W) + b

        scores[range(N), y] -= 1
        y_hat = np.argmax(scores, axis=1)

        return y_hat

    def fmin_gd(self, objective, w_init, step=0.04, max_iter=1000):
        w = w_init
        for i in range(max_iter):
            _, gradient = objective(w)
            w -= step * gradient
        return w

    def risk_emp(self, w, X, y, K, lmbda):

        N, dim = X.shape[0], X.shape[1]

        W = w[:self.dim*self.k].reshape(dim, K)
        b = w[self.dim*self.k:]

        y_hat = self.loss_augmented_decoding(X, W, b, y)

        scores = np.dot(X, W) + b

        loss = np.zeros((N,))

        for i in range(N):
            delta = y[i] != y_hat[i]
            loss[i] = -scores[i, y[i]] + scores[i, y_hat[i]] + delta

        risk = 1 / N * np.sum(loss) + 0.5 * lmbda * np.sum(W ** 2)
        grad_W = lmbda * W
        grad_b = lmbda * b
        idx = loss > 0
        for k in range(K):
            idx_k_hat = np.logical_and(idx, y_hat == k)
            idx_k = np.logical_and(idx, y == k)
            grad_W[:, k] = grad_W[:, k] + 1 / N * (np.sum(X[idx_k_hat, :], axis=0) - np.sum(X[idx_k, :], axis=0)).T
            grad_b[k] = grad_b[k] + 1 / N * (np.sum(X[idx_k_hat, :]) - np.sum(X[idx_k, :])).T

        return risk, np.concatenate((grad_W.flatten(), grad_b), axis=0)

    def get_prediction(self, x):
        return np.argmax(np.dot(x, self.W) + self.b)
