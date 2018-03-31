import numpy as np

a = np.zeros(shape=(10, 10))

for i in range(10):
    a[i] = np.random.rand(10)

b = np.random.rand(10)

for i in range(10):
    b = np.vstack((b, np.random.rand(10)))

print(b.shape)