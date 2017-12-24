import numpy as np
import random
from pystruct.datasets import load_letters


class DataHandler:

    CLASS_NUM = 26

    def __init__(self):
        letters = load_letters()
        X, y, folds = letters['data'], letters['labels'], letters['folds']
        self.dict = "abcdefghijklmnopqrstuvwxyz"

        X, y = np.array(X), np.array(y)
        self.X_train, self.X_test = X[folds == 1], X[folds != 1]
        self.y_train, self.y_test = y[folds == 1], y[folds != 1]

        # shuffle
        num_ex = len(self.X_train)
        perm = np.arange(num_ex)
        np.random.shuffle(perm)
        self.X_train = self.X_train[perm]
        self.y_train = self.y_train[perm]

        self.DIM = self.X_train[0].shape[1]
        self.test_num = len(self.X_test)
        self.train_num = len(self.X_train)

        self.y_one_hot_train = self.to_one_hot(self.y_train)
        self.y_one_hot_test = self.to_one_hot(self.y_test)

        self.X_lst_train, self.y_lst_train = self._get_lst(self.X_train, self.y_train)
        self.X_lst_test, self.y_lst_test = self._get_lst(self.X_test, self.y_test)

    @property
    def dictionary(self):
        return self.dict

    @property
    def dim(self):
        return self.DIM

    @property
    def class_num(self):
        return self.CLASS_NUM

    def to_one_hot(self, y_set):
        one_hot_set = []
        for samples in y_set:
            current = np.zeros((len(samples), self.CLASS_NUM))
            for idx, sample in enumerate(samples):
                current.itemset((idx, sample), 1)
            one_hot_set.append(current)
        return np.array(one_hot_set)

    def _get_lst(self, x_set, y_set):
        X_lst = []
        y_lst = []
        for samples in x_set:
            for sample in samples:
                X_lst.append(sample)

        for samples in y_set:
            for sample in samples:
                y_lst.append(sample)
        return X_lst, y_lst

    def get_X_lst(self):
        return self.X_lst_train

    def get_train_set(self):
        return self.X_train, self.y_train

    def get_train_set_one_hot(self):
        return self.X_train, self.y_one_hot_train

    def get_test_set(self):
        return self.X_test, self.y_test

    def get_test_set_one_hot(self):
        return self.X_test, self.y_one_hot_test

    def get_lst_train(self):
        return self.X_lst_train, self.y_lst_train

    def get_lst_test(self):
        return self.X_lst_test, self.y_lst_test

    def get_random_test_example(self):
        idx = random.randint(0, self.test_num)
        return self.X_lst_test[idx], self.y_lst_test[idx]

    def get_one_against_all_array(self, class_no):

        y_binary = []
        for (idx, sample) in enumerate(self.y_lst_train):
            if sample == class_no:
                y_binary.append(1)
            else:
                y_binary.append(-1)
        return y_binary
