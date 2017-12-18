import numpy as np
from pystruct.inference import inference_dispatch, compute_energy
from pystruct.utils import make_grid_edges

EPOCH = 1
LEARNING_RATE = 0.1


class StructSVM:

    def __init__(self, data_handler):
        self.dataset = data_handler
        self.k = data_handler.CLASS_NUM
        self.W = np.random.randn(self.dataset.DIM, self.dataset.CLASS_NUM)
        self.A = np.random.randn(self.dataset.CLASS_NUM, self.dataset.CLASS_NUM)
        self.b = np.random.randn(self.dataset.CLASS_NUM)

    def zero_one_loss(self, y, pred):
        for idx, value in enumerate(y):
            if value != pred[idx]:
                return 1

        return 0

    def hamming_loss(self, y, pred):
        score = 0
        for idx, values in enumerate(y):
            for idx2, value in enumerate(values):
                if value != pred[idx][idx2]:
                    score += 1
                    break
        return score

    def train(self, reg_step=0.01):

        X_set, y_set = self.dataset.get_train_set_one_hot()

        for i in range(EPOCH):
            for idx, samples in enumerate(X_set):
                x = samples
                y = y_set[idx]
                y_ = self.loss_augmented_decoding(self.W, self.A, self.b, x, y, self.k)
                y_ = self.dataset.batch_to_one_hot(y_)
                loss = self.hamming_loss(y, y_)

                phi_y = self.get_phi_matrix(y, x)
                phi_y_ = self.get_phi_matrix(y_, x)

                grad = phi_y_ - phi_y

                flatten_w = np.hstack((self.W.flatten(), self.A.flatten(), self.b))

                flatten_w -= LEARNING_RATE*(reg_step*flatten_w + grad)

                br = self.dataset.CLASS_NUM*self.dataset.DIM
                self.W = np.reshape(flatten_w[:br], newshape=(self.dataset.DIM, self.dataset.CLASS_NUM))
                br1 = self.dataset.CLASS_NUM*self.dataset.CLASS_NUM
                self.A = np.reshape(flatten_w[br:br+br1], newshape=(self.dataset.CLASS_NUM, self.dataset.CLASS_NUM))
                self.b = flatten_w[br+br1:]

    def get_phi_matrix(self, y, x):
        n, y_d = y.shape
        _, x_d = x.shape
        m1 = np.zeros((y_d, x_d))
        m2 = np.zeros((y_d, y_d))
        m3 = np.zeros(y_d)
        for idx, v in enumerate(y):
            m1 += np.matmul(v[:, np.newaxis], x[idx][np.newaxis, :])
            if idx != n-1:
                m2 += np.matmul(v[:, np.newaxis], y[idx+1][np.newaxis, :])
            m3 += v

        return np.hstack((m1.flatten(), m2.flatten(), m3))

    def decoding(self, W, A, b, x, k):

        n, dim = x.shape[0], x.shape[1]

        S = np.dot(x, W) + b

        edges = make_grid_edges(S.reshape(1, n, k))
        pairwise = A
        unaries = S

        # decoding
        y = inference_dispatch(unaries, pairwise, edges, inference_method='ad3')

        return y

    def loss_augmented_decoding(self, W, A, b, x, y, k):

        n, dim = x.shape[0], x.shape[1]
        S = np.dot(x, W) + b

        y_ = self.decoding(W, A, b, x, k)
        # y_ = self.dataset.batch_to_one_hot(y_)
        # loss = self.hamming_loss(y, y_)
        s_m = np.ones(shape=(n, k))
        s_m -= y
        S_ = S + s_m

        edges = make_grid_edges(S_.reshape(1, n, k))
        pairwise = A
        unaries = S_

        # decoding
        y = inference_dispatch(unaries, pairwise, edges, inference_method='ad3')

        return y
