import numpy as np
from sklearn.datasets import load_sample_images
import tensorflow as tf
import matplotlib.pyplot as plt

dataset = np.array(load_sample_images().images, dtype=np.float32)
print(dataset.shape)

plt.imshow(load_sample_images().images[0])

temple = load_sample_images().images[0]

red = temple[:, :, 0]

temp_r = np.zeros([427, 640, 3])
temp_r[:, :, 0] = red
plt.imshow(temp_r)
plt.show()

green = temple[:, :, 1]
temp_g = np.zeros([427, 640, 3])
temp_g[:, :, 1] = green
plt.imshow(temp_g)
plt.show()

blue = temple[:, :, 2]
temp_b = np.zeros([427, 640, 3])
temp_b[:, :, 2] = blue
plt.imshow(temp_b)

plt.show()
