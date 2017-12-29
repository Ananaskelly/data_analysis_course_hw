import numpy as np
import scipy.io as sio
import os


class DataHandler:

    def __init__(self):
        mat_contents = sio.loadmat('task_1/data/data_simple.mat')

        self.X_simple = np.array(mat_contents['x']).squeeze()
        self.Y_simple = np.array(mat_contents['y']).squeeze()

        mat_contents = sio.loadmat('task_1/data/data_robust.mat')
        self.X_robust = np.array(mat_contents['x']).squeeze()
        self.Y_robust = np.array(mat_contents['y']).squeeze()

        red_wine = np.genfromtxt('task_1/data/winequality-red.csv', delimiter=';')
        red_wine = red_wine[1:]
        self.X_red_wine = red_wine[:, :11]
        self.Y_red_wine = red_wine[:, 11]

    def get_simple_train_dataset(self):
        return self.X_simple[:40], self.Y_simple[:40]

    def get_simple_test_dataset(self):
        return self.X_simple[40:], self.Y_simple[40:]

    def get_robust_train_dataset(self):
        return self.X_robust[:40], self.Y_robust[:40]

    def get_robust_test_dataset(self):
        return self.X_robust[40:], self.Y_robust[40:]

    def get_wine_train_dataset(self):
        return self.X_red_wine[:478], self.Y_red_wine[:478]

    def get_wine_test_dataset(self):
        return self.X_red_wine[478:], self.Y_red_wine[478:]
