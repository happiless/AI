import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_hidden3 = 50
n_outputs = 10

mnist = input_data.read_data_sets('./MNIST_data_bak')

X = tf.placeholder(dtype=tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(dtype=tf.int32, shape=(None,), name='y')

with tf.name_scope('dnn'):
    with tf.contrib.framework.arg_scope(
            [fully_connected],
            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01)):
        hidden1 = fully_connected(X, n_hidden1, scope='hidden1')
        hidden2 = fully_connected(hidden1, n_hidden2, scope='hidden2')
        hidden3 = fully_connected(hidden2, n_hidden3, scope='hidden3')
        logits = fully_connected(hidden3, n_outputs, scope='outputs', activation_fn=None)

with tf.name_scope('loss'):
    # 定义交叉熵损失函数，并且求个样本平均
    # 函数等价于先使用softmax损失函数，再接着计算交叉熵，并且更有效率
    # 类似的softmax_cross_entropy_with_logits只会给one-hot编码，我们使用的会给0-9分类号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    # reg_losses = tf.get_collection('reg_losses')
    reg_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    loss = tf.reduce_mean(cross_entropy, name='loss')
    total_loss = tf.add(loss, reg_losses)

learning_rate = 0.01

with tf.name_scope('train'):
    training_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

with tf.name_scope('eval'):
    # 获取logits里面最大的那1位和y比较类别好是否相同，返回True或者False一组值
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

n_epochs = 400
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for _ in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run([training_op], feed_dict={
                X: X_batch,
                y: y_batch
            })
        acc_train = accuracy.eval(feed_dict={
            X: X_batch,
            y: y_batch
        })
        acc_test = accuracy.eval(feed_dict={
            X: mnist.test.images,
            y: mnist.test.labels
        })
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    saver.save(sess, './ckpt/my_dnn_model_final.ckpt')
