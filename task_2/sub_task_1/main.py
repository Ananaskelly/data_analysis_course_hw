import matplotlib.pyplot as plt
import numpy as np

from task_2.sub_task_1 import data_handler
from task_2.sub_task_1 import utils
from task_2.sub_task_1.peregudins_cube import one_vs_all_svm
from task_2.sub_task_1.peregudins_cube import struct_svm
from task_2.sub_task_1.peregudins_cube import simple_svm


def run_one_vs_all():
    dh = data_handler.DataHandler()
    simple_classifier = one_vs_all_svm.OneVsAllSVM(dh)
    simple_classifier.train()

    X_test, y_test = dh.get_lst_test()
    n = len(X_test)
    score = 0
    for idx,x in enumerate(X_test):
        pred = simple_classifier.get_prediction(x)

        if pred == y_test[idx]:
            score += 1

    print('Accuracy: {}'.format(score/n))


def run_struct():
    dh = data_handler.DataHandler()

    X, y = dh.get_train_set()
    X, y_one_hot = dh.get_train_set_one_hot()
    dim, class_num = dh.dim, dh.class_num

    #################################################################
    # get optimal learning rate with k-fold cross-validation

    # alphas = [10**(-i) for i in range(1, 6)]
    # classifier = lambda X, y, y_: struct_svm.StructSVM(X, y, y_, dh.CLASS_NUM, dh.DIM)
    # opt_alpha_idx = utils.k_fold_validation(X, y, alphas, 3, classifier, y_one_hot, True)
    # print('Optimal regularization coefficient: {0}'.format(alphas[opt_alpha_idx]))

    #################################################################

    ssvm = struct_svm.StructSVM(X, y, y_one_hot, class_num, dim)
    diff = ssvm.check_grad(3, 10)
    print(diff)
    ssvm.train(reg_step=1e-5)

    A_matrix = ssvm.A_matrix
    plt.matshow(A_matrix)
    plt.colorbar()
    plt.title("A_matrix visualization")
    plt.xticks(np.arange(25), dh.dictionary)
    plt.yticks(np.arange(25), dh.dictionary)
    plt.show()

    x_test, y_test = dh.get_test_set()

    mean_loss = 0
    for idx, sample in enumerate(x_test):
        y_pred = ssvm.get_prediction(sample)
        mean_loss += ssvm.hamming_loss_(y_test[idx], y_pred)

    print("Mean loss: {}".format(mean_loss/x_test.shape[0]))


def run_simple():
    dh = data_handler.DataHandler()
    X_train, y_train = dh.get_lst_train()
    X_train = np.array(X_train, dtype='int32')
    y_train = np.array(y_train, dtype='int32')
    simple_classifier = simple_svm.SimpleSVM(X_train, y_train, dh.CLASS_NUM, dh.DIM)

    #################################################################
    # get optimal learning rate with k-fold cross-validation

    alphas = [10**(-i) for i in range(5)]
    classifier = lambda X, y: simple_svm.SimpleSVM(X, y, dh.CLASS_NUM, dh.DIM)
    opt_alpha_idx = utils.k_fold_validation(X_train, y_train, alphas, 3, classifier)
    print('Optimal regularization coefficient: {0}'.format(alphas[opt_alpha_idx]))

    #################################################################

    simple_classifier.train()

    X_test, y_test = dh.get_lst_test()

    n = len(X_test)
    score = 0
    for idx, x in enumerate(X_test):
        pred = simple_classifier.get_prediction(x)
        if pred == y_test[idx]:
            score += 1

    print('Accuracy: {}'.format(score/n))


# run_simple()
run_struct()
