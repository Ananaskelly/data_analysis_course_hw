from task_2.sub_task_1 import one_vs_all_svm
from task_2.sub_task_1 import struct_svm
from task_2.sub_task_1 import data_handler


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
    simple_classifier = struct_svm.StructSVM(dh)
    simple_classifier.train()

    """X_test, y_test = dh.get_lst_test()
    n = len(X_test)
    score = 0
    for idx, x in enumerate(X_test):
        pred = simple_classifier.get_prediction(x)

        if pred == y_test[idx]:
            score += 1

    print('Accuracy: {}'.format(score / n))"""

# run_simple()
run_struct()