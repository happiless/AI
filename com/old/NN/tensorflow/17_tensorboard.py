import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_steps = 1000
learning_rate = 0.01
dropout = 0.9
data_dir = './MNIST_data_bak/'
log_dir = './logs/mnist_with_summaries'

mnist = input_data.read_data_sets('./MNIST_data_bak', one_hot=True)

with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=(None, 784), name='x-input')
    y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='y-input')

with tf.name_scope('input-reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 设计一个MLP多层神经网络来训练数据
# 在每一层中都对模型数据进行汇总
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('bias'):
            bias = bias_variable([output_dim])
            variable_summaries(bias)
        with tf.name_scope('Wx_plus_b'):
            pre_actives = tf.matmul(input_tensor, weights) + bias
            tf.summary.histogram('pre_actives', pre_actives)
        activations = act(pre_actives, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


hidden1 = nn_layer(x, 784, 500, 'layer1')


with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(dtype=tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    # train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    # 因为我们之前定义了太多的tf.summary汇总操作，逐一执行这些操作太麻烦，
    # 使用tf.summary.merge_all()直接获取所有汇总操作，以便后面执行
    merged = tf.summary.merge_all()
    # 定义两个tf.summary.FileWriter文件记录器再不同的子目录，分别用来存储训练和测试的日志数据
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    tf.global_variables_initializer().run()

    def feed_dict(train):
        if train:
            xs, ys = mnist.train.next_batch(100)
            k = dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    saver = tf.train.Saver()
    for i in range(max_steps):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            if i % 100 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, 1)
                saver.save(sess, log_dir + 'model.ckpt', i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()
