import numpy as np


EPOCH = 5
LEARNING_RATE = 0.1


class OneVsAllSVM:

    def __init__(self, data_handler):
        self.dataset = data_handler
        self.W = np.zeros((self.dataset.CLASS_NUM, self.dataset.DIM))
        self.b = np.zeros(self.dataset.CLASS_NUM)

    def train(self, use_bounds=False, start=0, end=0):

        X = self.dataset.get_X_lst()
        if use_bounds:
            X = X[start:end]
        X = np.array(X)

        n, dim = X.shape
        for i in range(self.dataset.CLASS_NUM):
            y = self.dataset.get_one_against_all_array(i)
            if use_bounds:
                y = y[start:end]

            y = np.array(y)
            w_init = np.random.randn(dim)
            b_init = np.random.randn()
            w, b_ = self.train_one_vs_all(X, w_init, b_init, y)
            self.W[i] = w
            self.b[i] = b_

        return self.W, self.b

    def train_one_vs_all(self, X, w, b, y, regularization_param=0.001):
        n = X.shape[0]

        for i in range(EPOCH):
            # shuffle
            perm = np.arange(n)
            np.random.shuffle(perm)
            X = X[perm]
            y = y[perm]
            for (idx, x) in enumerate(X):
                pred = np.matmul(x, w) + b
                curr_loss = self._hinge_loss(y[idx], pred)
                if curr_loss == 0:
                    w -= regularization_param * LEARNING_RATE * w
                else:
                    w -= LEARNING_RATE*(regularization_param * w - y[idx] * x)

        return w, b

    def _hinge_loss(self, y_true, pred):
        return max(0, 1 - y_true*pred)

    def get_prediction(self, x_test):
        return np.argmax(np.matmul(x_test, self.W.T) + self.b)
