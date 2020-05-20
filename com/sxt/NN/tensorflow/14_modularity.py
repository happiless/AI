import tensorflow as tf

n_features = 3
X = tf.placeholder(dtype=tf.float32, shape=(None, n_features), name='X')

w1 = tf.Variable(tf.random_uniform((n_features, 1)), 'w1')
w2 = tf.Variable(tf.random_uniform((n_features, 1)), 'w2')
b1 = tf.Variable(0.0, name='b1')
b2 = tf.Variable(0.0, name='b2')

z1 = tf.add(tf.matmul(X, w1), b1, name='z1')
z2 = tf.add(tf.matmul(X, w2), b2, name='z2')

relu1 = tf.maximum(z1, 0., name='relu1')
relu2 = tf.maximum(z2, 0., name='relu2')

output = tf.add(relu1, relu2, name='output')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(output.eval(feed_dict={X: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))
    print('w1', w1.eval(), 'b1', b1.eval())
    print('w2', w2.eval(), 'b2', b2.eval())
