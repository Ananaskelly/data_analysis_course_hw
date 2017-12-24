import matplotlib.pyplot as plt
import numpy as np
from task_2.sub_task_1 import one_vs_all_svm
from task_2.sub_task_1 import struct_svm
from task_2.sub_task_1 import data_handler
from task_2.sub_task_1 import utils


def run_simple():
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
    # alphas = [10**i for i in range(3)]
    # opt_alpha_idx = utils.k_fold_validation(X, y, y_one_hot, alphas, 3, class_num, dim)
    #################################################################

    ssvm = struct_svm.StructSVM(X, y, y_one_hot, class_num, dim)
    ssvm.train(0.1)

    x_test, y_test = dh.get_test_set()

    mean_loss = 0
    for idx, sample in enumerate(x_test):
        y_pred = ssvm.get_prediction(sample)
        print(y)
        print(y_pred)
        mean_loss += ssvm.hamming_loss_(y_test[idx], y_pred)

    print("Mean loss: {}".format(mean_loss/x_test.shape[0]))

    A_matrix = ssvm.A_matrix
    plt.matshow(A_matrix, vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("A_matrix visualization")
    plt.xticks(np.arange(25), dh.dictionary)
    plt.yticks(np.arange(25), dh.dictionary)
    plt.show()

# run_simple()
run_struct()
