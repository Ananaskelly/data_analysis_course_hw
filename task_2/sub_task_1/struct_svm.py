import numpy as np
from pystruct.inference import inference_dispatch, compute_energy
from pystruct.utils import make_grid_edges

EPOCH = 1

class StructSVM:

    def __init__(self, data_handler):
        self.dataset = data_handler
        self.k = data_handler.CLASS_NUM
        self.W = np.random.randn((self.dataset.DIM, self.dataset.CLASS_NUM))
        self.A = np.random.randn((self.dataset.CLASS_NUM, self.dataset.CLASS_NUM))
        self.b = np.random.randn(self.dataset.CLASS_NUM)

    def zero_one_loss(self, y, pred):
        for idx, value in enumerate(y):
            if value != pred[idx]:
                return 1

        return 0

    def hamming_loss(self, y, pred):
        score = 0
        for idx, value in enumerate(y):
            if value != pred[idx]:
                score += 1

        return score

    def train(self):

        X_set, y_set = self.dataset.get_train_set_one_hot()

        for i in range(EPOCH):
            for idx, samples in enumerate(X_set):
                x = samples
                y = y_set[idx]
                y_ = self.loss_augmented_decoding(self.W, self.A, self.b, x, y, self.k)
                loss = self.hamming_loss(y, y_)


    def get_phi_matrix(self, y, x):

        
        for idx, v in enumerate(y):


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
        loss = self.hamming_loss(y, y_)

        S_ = S + int(loss != 0)

        edges = make_grid_edges(S_.reshape(1, n, k))
        pairwise = A
        unaries = S

        # decoding
        y = inference_dispatch(unaries, pairwise, edges, inference_method='ad3')

        return y
