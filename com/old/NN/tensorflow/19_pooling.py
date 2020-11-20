import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_images

dataset = np.array(load_sample_images().images, dtype=np.float32)

batch_size, height, width, channel = dataset.shape
print(batch_size, height, width, channel)

X = tf.placeholder(dtype=tf.float32, shape=(None, height, width, channel), name='X')

# TensorFlow不支持池化多个实例，所以ksize的第一个batch size是1
# TensorFlow不支持池化同时发生的长宽高，所以必须有一个是1，这里channels就是depth维度为1
avg_pool = tf.nn.avg_pool(X, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')

with tf.Session() as sess:
    output = sess.run(avg_pool, feed_dict={X: dataset})
    print(output.shape)

plt.imshow(load_sample_images().images[0])
plt.show()

plt.imshow(output[0, :, :, 0])
plt.show()

plt.imshow(output[0, :, :, 1])

plt.imshow(load_sample_images().images[1])
plt.show()

plt.imshow(output[1, :, :, 0])
plt.show()

plt.imshow(output[1, :, :, 1])
plt.show()