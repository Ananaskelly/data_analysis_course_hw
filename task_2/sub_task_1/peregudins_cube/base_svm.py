from abc import ABCMeta, abstractmethod


class BaseSvm:

    @abstractmethod
    def train(self, rate=0.1, reg_step=0.1, batch_size=128):
        return

    @abstractmethod
    def loss(self, true_label, predicted_label):
        return

    @abstractmethod
    def get_prediction(self, x):
        return
