import numpy as np


def euclidean_dst(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.shape != b.shape:
        raise AssertionError('Arrays shapes must be equal')
    return np.sqrt(np.sum((a-b)**2))
