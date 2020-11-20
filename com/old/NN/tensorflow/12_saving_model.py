# 有时候需要把模型保存起来，有时候需要做一些checkpoint在训练中
# 以致于如果计算机宕机，我们还可以从之前checkpoint的位置去继续
# TensorFlow使得我们去保存和加载模型非常方便，仅需要去创建Saver节点在构建阶段最后
# 然后在计算阶段去调用save()方法

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01

my_mnist = input_data.read_data_sets('./MNIST_data_bak/', one_hot=True)

X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')
W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32, name='W')
b = tf.Variable(tf.zeros([10]), dtype=tf.float32, name='b')
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

cross_entropy = tf.reduce_mean(- tf.reduce_sum(y * tf.log(y_pred), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_epochs = 100000

with tf.Session() as sess:
    # init.run()
    ckpt = tf.train.get_checkpoint_state('./ckpt/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restore model ===> ', sess.run(b))
    else:
        sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 1000 == 0:
            print('%s bias %s' % (epoch, sess.run(b)))
            saver.save(sess, './ckpt/my_model.ckpt', global_step=epoch)
        batch_xs, batch_ys = my_mnist.train.next_batch(1000)
        sess.run(train_step, feed_dict={
            X: batch_xs,
            y: batch_ys
        })
    saver.save(sess, './ckpt/my_model_final.ckpt')
