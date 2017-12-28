import numpy as np
import random
import math

from task_2.sub_task_1 import utils
from pystruct.inference import inference_dispatch, compute_energy
from pystruct.inference.inference_methods import inference_max_product
from pystruct.utils import make_grid_edges
from scipy.linalg import norm


np.set_printoptions(threshold=math.nan)
EPOCH = 10
LEARNING_RATE = 1
REG_STEP = 1e-6


class StructSVM:

    def __init__(self, X=None, y=None, y_one_hot=None, class_num=0, dim=0):
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
        # return score
        return float(score)/y_length

    def hamming_loss_(self, y, pred):
        score = 0
        y_length = y.shape[0]
        for idx, value in enumerate(y):
            if value != pred[idx]:
                score += 1
        #return score
        return float(score)/y_length

    def hamming_loss_base(self, y, pred):
        score = 0
        y_length = y.shape[0]
        for idx, value in enumerate(y):
            if value != pred[idx]:
                score += 1
        return score

    def loss(self, y, pred):
        score = 0
        y_length = y.shape[0]
        for idx, value in enumerate(y):
            if value != pred[idx]:
                score += 1
        #return score
        return float(score)/y_length

    def grad(self, y_true, y_pred, x):
        phi_y = self.get_phi_matrix(y_true, x)
        phi_y_ = self.get_phi_matrix(y_pred, x)

        grad = phi_y_ - phi_y

        return grad

    def risk(self, flatten_w, x, y_true, y_true_labels, dim=0, k=0):
        n, _ = x.shape[0], x.shape[1]

        if dim == 0 and k == 0:
            dim = self.dim
            k = self.k

        W = flatten_w[:dim*k].reshape(k, dim).T
        A = flatten_w[dim*k:dim*k+k*k].reshape(k, k).T
        b = flatten_w[dim*k+k*k:]

        y_pred = self.loss_augmented_decoding(W, A, b, x, y_true, k)
        S = np.dot(x, W) + b

        edges = make_grid_edges(S.reshape(1, n, k))
        pairwise = A
        unaries = S
        loss = self.hamming_loss_base(y_true_labels, y_pred)

        score_y = compute_energy(unaries, pairwise, edges, y_true_labels)
        score_y_pred = compute_energy(unaries, pairwise, edges, y_pred)

        return loss - score_y + score_y_pred

    def numerical_grad(self, x_samples, y_samples, y_one_hot_samples, W, A, b, dim_, k):
        eps = 1e-6
        n = x_samples.shape[0]
        flatten_w = np.hstack((W.T.flatten(), A.T.flatten(), b))
        dim = flatten_w.shape[0]
        eps_vec = np.zeros(dim)
        grad = np.zeros(dim)

        flatten_w_norm = 0.5 * REG_STEP * np.linalg.norm(flatten_w)
        for idx in range(dim):
            eps_vec[idx] = eps
            risk_vec_w = [self.risk(flatten_w, x_samples[i], y_one_hot_samples[i], y_samples[i], dim_, k)
                          for i in range(n)]
            risk_vec_eps_w = [self.risk(flatten_w + eps_vec, x_samples[i], y_one_hot_samples[i], y_samples[i], dim_, k)
                              for i in range(n)]

            risk_w = flatten_w_norm + 1 / n * np.sum(risk_vec_w)
            risk_eps_w = 0.5 * REG_STEP * np.linalg.norm(flatten_w + eps_vec) + 1 / n * np.sum(risk_vec_eps_w)

            diff = risk_eps_w - risk_w

            grad[idx] = diff/eps

            eps_vec[idx] = 0

        return grad

    def analytical_grad(self, x_samples, y_samples, y_one_hot_samples, W, A, b, dim, k):
        n = x_samples.shape[0]

        flatten_w = np.hstack((W.T.flatten(), A.T.flatten(), b))
        y_pred = [self.loss_augmented_decoding(W, A, b, x_samples[idx], y_one_hot_samples[idx], k)
                  for idx in range(n)]

        grad = np.zeros(dim*k+k*k+k)
        for i in range(n):
            grad += self.grad(y_one_hot_samples[i], utils.batch_to_one_hot(y_pred[i], k), x_samples[i])

        return 1/n*grad + flatten_w*REG_STEP

    def check_grad(self, dim, k):
        n = 50
        # x_samples = self.X[:n]
        # y_samples = self.y[:n]
        # y_one_hot_samples = self.y_one_hot[:n]
        x_samples = []
        y_samples = []
        y_one_hot_samples = []

        W = np.random.randn(dim, k)*0.01
        A = np.random.randn(k, k)*0.01
        b = np.random.randn(k)*0.01

        for i in range(n):
            am = random.randint(3, 7)
            x_sample = np.random.randn(am, dim)
            y_sample = np.int32(np.random.rand(am, ) * k)
            y_one_hot_sample = utils.batch_to_one_hot(y_sample, k)
            x_samples.append(x_sample)
            y_samples.append(y_sample)
            y_one_hot_samples.append(y_one_hot_sample)

        x_samples = np.array(x_samples)
        y_samples = np.array(y_samples)
        y_one_hot_samples = np.array(y_one_hot_samples)
        n_grad = self.numerical_grad(x_samples, y_samples, y_one_hot_samples, W, A, b, dim, k)
        a_grad = self.analytical_grad(x_samples, y_samples, y_one_hot_samples, W, A, b, dim, k)
        diff = np.linalg.norm(n_grad - a_grad) / np.linalg.norm(n_grad + a_grad)
        return diff, n_grad, a_grad

    def train(self, rate=0.1, reg_step=REG_STEP, batch_size=128):

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

                    acc_grad += self.grad(y, y_, x)

                    if idx % batch_size == 0:
                        flatten_w = np.hstack((self.W.T.flatten(), self.A.T.flatten(), self.b))
                        flatten_w -= (rate*acc_grad + reg_step*flatten_w)
                        br = self.k*self.dim
                        self.W = np.reshape(flatten_w[:br], newshape=(self.k, self.dim)).T
                        br1 = self.k*self.k
                        self.A = np.reshape(flatten_w[br:br+br1], newshape=(self.k, self.k)).T
                        self.b = flatten_w[br+br1:]

                        acc_grad = 0
            print("Current loss: {}".format(current_loss/self.X.shape[0]))
            num_ex = len(self.X)
            perm = np.arange(num_ex)
            np.random.shuffle(perm)
            self.X = self.X[perm]
            self.y = self.y[perm]
            self.y_one_hot = self.y_one_hot[perm]
            if i % 3 == 0 and rate > 1e-5:
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
        return np.concatenate((m1.ravel(), m2.ravel(), m3), axis=0)

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
        ans = inference_max_product(unaries, pairwise, edges)

        return ans

    def get_prediction(self, x):
        return self.decoding(self.W, self.A, self.b, x, self.k)
