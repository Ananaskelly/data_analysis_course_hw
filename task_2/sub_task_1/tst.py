"""
================================
Sequence classifcation benchmark
================================
This is a stripped-down version of the "plot_letters.py" example
targetted to benchmark inference and learning algorithms on chains.
"""
import numpy as np

from pystruct.datasets import load_letters

abc = "abcdefghijklmnopqrstuvwxyz"

letters = load_letters()
X, y, folds = letters['data'], letters['labels'], letters['folds']
# we convert the lists to object arrays, as that makes slicing much more
# convenient
X, y = np.array(X), np.array(y)
X_train, X_test = X[folds == 1], X[folds != 1]
y_train, y_test = y[folds == 1], y[folds != 1]

