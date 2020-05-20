import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.01

my_mnist = input_data.read_data_sets('./MNIST_data_bak/', one_hot=True)

X = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y')
W = tf.Variable(tf.zeros([784, 10]), dtype=tf.float32, name='W')
b = tf.Variable(tf.zeros([10]), dtype=tf.float32, name='b')
y_pred = tf.nn.softmax(tf.matmul(X, W) + b)

correct_predict = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./ckpt/')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    acc = sess.run(accuracy, feed_dict={
        X: my_mnist.test.images,
        y: my_mnist.test.labels
    })
    print(acc)
