import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

__author__ = 'zhanghaibin'

learning_rate = 0.01

my_mnist = input_data.read_data_sets('./MNIST_data_bak/', one_hot=True)

first = np.reshape(my_mnist.train.images[1], (28, 28))

plt.imshow(first)
plt.show()

# The MNIST data is split into three parts:
# 55,000 data points of training data (mnist.train)
# 10,000 points of test data (mnist.test), and
# 5,000 points of validation data (mnist.validation).

# Each image is 28 pixels by 28 pixels
X = tf.placeholder(dtype=tf.float32, shape=(None, 784))

# labels是每张图片都对应一个one-hot的10个值的向量
y = tf.placeholder(dtype=tf.float32, shape=(None, 10))

# 初始化都是0，二维矩阵784乘以10个W值
W = tf.Variable(np.zeros([784, 10]), dtype=tf.float32)
b = tf.Variable(np.zeros([10]), dtype=tf.float32)
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

# 训练

# 定义损失函数，交叉熵损失函数
# 对于多分类问题，通常使用交叉熵损失函数
# reduction_indices等价于axis，指明按照每行加，还是按照每列加
cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# 评估
# tf.argmax()是一个从tensor中寻找最大值的序号，tf.argmax就是求各个预测的数字中概率最大的那一个
correct_predict = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

# 用tf.cast将之前correct_prediction输出的bool值转换为float32，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for _ in range(10000):
        batch_xs, batch_ys = my_mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={
            X: batch_xs,
            y: batch_ys
        })
        print('TrainSet batch acc:', accuracy.eval(feed_dict={X: batch_xs, y: batch_ys}))
        print('ValidSet batch acc:', accuracy.eval(feed_dict={X: my_mnist.validation.images, y: my_mnist.validation.labels}))
    print('TestSet batch acc:', accuracy.eval(feed_dict={X: my_mnist.validation.images, y: my_mnist.validation.labels}))