import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

np.set_printoptions(threshold=np.nan)


def conv2d(img, k):
    return convolve2d(img, k, mode='same')


def get_min_matrix(initial_matrix):
    result = np.zeros(shape=initial_matrix.shape)
    result[0, :] = initial_matrix[0, :]
    result[:, 0] = initial_matrix[:, 0]
    for i in range(1, initial_matrix.shape[0] - 1):
        for j in range(1, initial_matrix.shape[1] - 1):
            result[i, j] = min(initial_matrix[i-1, j-1], initial_matrix[i-1, j], initial_matrix[i-1, j+1])

    return result


def get_min_cost_path(matrix):
    path = []
    current_point = [matrix.shape[0] - 1, np.argmin(matrix[-1, :])]
    path.append(current_point)
    for i in range(matrix.shape[0] - 2, -1, -1):
        x = current_point[1]
        arr = [matrix[i, max(x-1, 0)], matrix[i, x], matrix[i, min(x+1, matrix.shape[1]-1)]]
        next_point = [i, max(x + np.argmin(arr) - 1, 0)]
        path.append(next_point)
        current_point = next_point

    return path


def load_and_process(path, it=1):
    img = cv2.imread(path, 0)
    img_shape = img.shape
    kernel = np.array([1, -1, -1, 1]).reshape(2, 2)

    current_img = np.copy(img)

    for i in range(it):
        e_img = conv2d(current_img, kernel)
        min_matrix = get_min_matrix(e_img)
        min_path = get_min_cost_path(min_matrix)

        mask = np.ones(current_img.shape, dtype=bool)
        for j in range(len(min_path)):
            mask[min_path[j][0], min_path[j][1]] = False
        current_img = np.reshape(current_img[mask], (img_shape[0], img_shape[1] - i - 1))

    return img, current_img

"""path = os.path.join(os.getcwd(), 'sample', 'pineapple.jpg')
orig, res_img = load_and_process(path, 50)

plt.figure(1)
plt.subplot(211)
plt.imshow(res_img, cmap='gray')
plt.subplot(212)
plt.imshow(orig, cmap='gray')
plt.show()"""
