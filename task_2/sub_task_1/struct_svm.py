import numpy as np
import math
from pystruct.inference import inference_dispatch, compute_energy
from pystruct.utils import make_grid_edges

np.set_printoptions(threshold=math.nan)
EPOCH = 5
LEARNING_RATE = 0.01


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

    def hamming_loss_v(self, y, pred):
        score = np.zeros(self.dataset.CLASS_NUM)
        for idx, values in enumerate(y):
            for idx2, value in enumerate(values):
                if value != pred[idx][idx2]:
                    score[idx] = 1
                    break
        return score

    def train(self, reg_step=0.001):

        X_set, y_set = self.dataset.get_train_set_one_hot()
        X_set, y_set_s = self.dataset.get_train_set()
        for i in range(EPOCH):
            for idx, samples in enumerate(X_set):
                x = samples
                y = y_set[idx]
                y_ = self.loss_augmented_decoding(self.W, self.A, self.b, x, y, self.k)
                y_s = y_
                y_ = self.dataset.batch_to_one_hot(y_)
                loss = self.hamming_loss(y, y_)
                print("length: {}, loss {}".format(y_.shape[0], loss))
                S = np.dot(x, self.W) + self.b
                n, dim = x.shape[0], x.shape[1]
                edges = make_grid_edges(S.reshape(1, n, self.k))
                pairwise = self.A
                unaries = S
                # phi_y = self.get_phi_matrix(y, x)
                # phi_y_ = self.get_phi_matrix(y_, x)
                phi_y = compute_energy(unaries, pairwise, edges, y_set_s[idx])
                phi_y_ = compute_energy(unaries, pairwise, edges, y_s)
                grad = phi_y_ - phi_y
                print(grad)
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

    def get_prediction(self, x):
        return self.decoding(self.W, self.A, self.b, x, self.k)

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
        y_ = self.dataset.batch_to_one_hot(y_)
        loss = self.hamming_loss_v(y, y_)
        S_ = S + loss

        edges = make_grid_edges(S_.reshape(1, n, k))
        pairwise = A
        unaries = S_

        # decoding
        y = inference_dispatch(unaries, pairwise, edges, inference_method='ad3')

        return y
