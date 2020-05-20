import tensorflow as tf

n_features = 3


def relu(X, num_nodes=1):
    print(X.get_shape())
    w_shape = (int(X.get_shape()[1]), num_nodes)
    w = tf.Variable(tf.random_uniform(w_shape), name='w')
    b = tf.Variable(0., name='b')
    z = tf.add(tf.matmul(X, w), b)
    return tf.maximum(z, 0., name='relu')


X = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='X')

relus = [relu(X) for _ in range(5)]
output = tf.add_n(relus)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(output.eval(feed_dict={X: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))
