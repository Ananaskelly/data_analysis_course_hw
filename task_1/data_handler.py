import numpy as np
import scipy.io as sio


class DataHandler:

    def __init__(self):
        mat_contents = sio.loadmat('data/data_simple.mat')

        self.X_simple = np.array(mat_contents['x']).squeeze()
        self.Y_simple = np.array(mat_contents['y']).squeeze()

    def get_simple_train_dataset(self):
        return self.X_simple[:40], self.Y_simple[:40]

    def get_simple_test_dataset(self):
        return self.X_simple[40:], self.Y_simple[40:]