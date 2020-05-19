import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

n_epochs = 10000
learning_rate = 0.001

housing = fetch_california_housing(data_home=None, download_if_missing=True)
m, n = housing.data.shape

scaler = StandardScaler().fit(housing.data)
scaled_housing = scaler.transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing]

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

theta = tf.Variable(tf.random_uniform([n+1, 1], -1., 1.), name='theta')
y_pred = tf.matmul(X, theta, name='predictors')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')

# grandients = 2 / m * tf.matmul(tf.transpose(X), error, name='grandients')
grandients = tf.gradients(mse, theta)

training_op = tf.assign(theta, theta - learning_rate * grandients[0])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch:', epoch, 'mse', mse.eval())
        print(sess.run(training_op))
    best_theta = sess.run(theta)
    print(best_theta)
