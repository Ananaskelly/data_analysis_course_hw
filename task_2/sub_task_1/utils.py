import numpy as np


def batch_to_one_hot(y_batch, class_num):
    current = np.zeros((len(y_batch), class_num))
    for idx, sample in enumerate(y_batch):
        current.itemset((idx, sample), 1)
    return current


def k_fold_validation(x_val, y_val, alphas, k, classifier, y_one_hot_val=None, one_hot=False):

    losses = []
    batch = int(x_val.shape[0]/k)
    for alpha in alphas:
        mean_loss = 0
        for idx in range(k):
            test_set_x = x_val[idx*batch:(idx+1)*batch]
            test_set_y = y_val[idx*batch:(idx+1)*batch]
            train_set_x = np.concatenate((x_val[0:idx * batch], x_val[(idx + 1) * batch:]), axis=0)
            train_set_y = np.concatenate((y_val[0:idx * batch], y_val[(idx + 1) * batch:]), axis=0)

            if one_hot:
                train_set_y_one_hot = np.concatenate((y_one_hot_val[0:idx * batch], y_one_hot_val[(idx + 1) * batch:]),
                                                     axis=0)
                ssvm = classifier(train_set_x, train_set_y, train_set_y_one_hot)
            else:
                ssvm = classifier(train_set_x, train_set_y)
            ssvm.train(reg_step=alpha)

            loss = 0
            for idx, sample in enumerate(test_set_x):
                y_pred = ssvm.get_prediction(sample)
                loss += ssvm.loss(test_set_y[idx], y_pred)
            mean_loss += loss/test_set_x.shape[0]

        print(mean_loss/k)
        losses.append(mean_loss/k)

    return np.argmin(losses)


