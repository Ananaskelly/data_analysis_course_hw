import numpy as np
import math

from task_2.sub_task_1 import utils
from pystruct.inference import inference_dispatch, compute_energy
from pystruct.utils import make_grid_edges

np.set_printoptions(threshold=math.nan)
EPOCH = 10
# LEARNING_RATE = 0.1
REG_STEP = 1e-6


class StructSVM:

    def __init__(self, X, y, y_one_hot, class_num, dim):
        self.X = X
        self.y = y
        self.y_one_hot = y_one_hot
        self.dim = dim
        self.k = class_num
        self.W = np.random.randn(self.dim, self.k)*0.01
        self.A = np.random.randn(self.k, self.k)*0.01
        self.b = np.zeros(self.k)

    @property
    def A_matrix(self):
        return self.A

    def zero_one_loss(self, y, pred):
        for idx, value in enumerate(y):
            if value != pred[idx]:
                return 1

        return 0

    def hamming_loss(self, y, pred):
        score = 0
        y_length = y.shape[0]
        for idx, values in enumerate(y):
            for idx2, value in enumerate(values):
                if value != pred[idx][idx2]:
                    score += 1
                    break
        return float(score)/y_length

    def hamming_loss_(self, y, pred):
        score = 0
        y_length = y.shape[0]
        for idx, value in enumerate(y):
            if value != pred[idx]:
                score += 1
        return float(score)/y_length

    def train(self, rate, batch_size=128):

        for i in range(EPOCH):
            print("---------- Epoch {} ----------".format(i))
            current_loss = 0
            acc_grad = 0
            for idx, samples in enumerate(self.X):
                x = samples
                y = self.y_one_hot[idx]
                y_ = self.loss_augmented_decoding(self.W, self.A, self.b, x, y, self.k)
                y_ = utils.batch_to_one_hot(y_, self.k)
                loss = self.hamming_loss(y, y_)
                current_loss += loss
                if loss > 0 or (loss == 0 and idx % batch_size == 0):
                    phi_y = self.get_phi_matrix(y, x)
                    phi_y_ = self.get_phi_matrix(y_, x)

                    grad = phi_y_ - phi_y
                    acc_grad += grad
                    if idx % batch_size == 0:
                        flatten_w = np.hstack((self.W.T.flatten(), self.A.T.flatten(), self.b))
                        flatten_w -= rate*acc_grad - REG_STEP*np.linalg.norm(flatten_w)
                        br = self.k*self.dim
                        self.W = np.reshape(flatten_w[:br], newshape=(self.k, self.dim)).T
                        br1 = self.k*self.k
                        self.A = np.reshape(flatten_w[br:br+br1], newshape=(self.k, self.k)).T
                        self.b = flatten_w[br+br1:]

                        acc_grad = 0
            print(self.A)
            print("Current loss: {}".format(current_loss/self.X.shape[0]))
            num_ex = len(self.X)
            perm = np.arange(num_ex)
            np.random.shuffle(perm)
            self.X = self.X[perm]
            self.y = self.y[perm]
            self.y_one_hot = self.y_one_hot[perm]
            if i % 3 == 0 and rate > 1e-4:
                rate *= 0.5

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

        y = inference_dispatch(unaries, pairwise, edges, inference_method='ad3')

        return y

    def loss_augmented_decoding(self, W, A, b, x, y, k):

        n, dim = x.shape[0], x.shape[1]
        S = np.dot(x, W) + b

        s_m = np.ones(shape=(n, k))
        s_m -= y
        S_ = S + s_m

        edges = make_grid_edges(S.reshape(1, n, k))
        pairwise = A
        unaries = S_

        # decoding
        ans = inference_dispatch(unaries, pairwise, edges, inference_method='ad3')

        return ans

    def get_prediction(self, x):
        return self.decoding(self.W, self.A, self.b, x, self.k)
