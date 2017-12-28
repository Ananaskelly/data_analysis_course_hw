import numpy as np
import scipy.io as sio

mat_contents = sio.loadmat('data/data_simple.mat')

X = np.array(mat_contents['x']).squeeze()
Y = np.array(mat_contents['y']).squeeze()
