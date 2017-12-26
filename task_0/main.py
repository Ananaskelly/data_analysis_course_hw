from task_0 import resize_util
import matplotlib.pyplot as plt
import os
import time


start_time = time.time()
path = os.path.join(os.getcwd(), 'sample', 'pineapple.jpg')
orig, res_img = resize_util.load_and_process(path, 50)

plt.figure(1)
plt.title("Resized")
plt.imshow(res_img, cmap='gray')
plt.figure(2)
plt.title("Original")
plt.imshow(orig, cmap='gray')
plt.show()

ex_time = time.time() - start_time
print('Time: ' + '{:f}'.format(ex_time))