import numpy as np
import tensorflow as tf
from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt

dataset = np.array(load_sample_images().images, dtype=np.float32)

batch_size, height, width, channels = dataset.shape

print(batch_size, height, width, channels)

plt.imshow(load_sample_images().images[0])
plt.show()

plt.imshow(load_sample_images().images[1])
plt.show()

# 创建3 * 3 的10个卷积盒
fitler_test = np.zeros(shape=(3, 3, channels, 10), dtype=np.float32)
fitler_test[:, 2, :, 0] = 1     # 垂直
fitler_test[2, :, :, 1] = 1

# filter参数是一个filters的集合
X = tf.placeholder(dtype=tf.float32, shape=(None, height, width, channels))

convolution = tf.nn.conv2d(X, filter=fitler_test, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})
    print(output.shape)

plt.imshow(load_sample_images().images[0])
plt.show()

plt.imshow(output[0, :, :, 0])
plt.show()

plt.imshow(output[0, :, :, 1])
plt.show()

plt.imshow(load_sample_images().images[1])
plt.show()

plt.imshow(output[1, :, :, 0])
plt.show()

plt.imshow(output[1, :, :, 1])
plt.show()