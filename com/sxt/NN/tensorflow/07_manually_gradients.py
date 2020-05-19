import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

n_epochs = 36500
learning_rate = 0.001

housing = fetch_california_housing(data_home=None, download_if_missing=True)
m, n = housing.data.shape
print(m, n)

housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# 可以使用TensorFlow或者Numpy或者sklearn的StandardScaler去进行归一化
# StandardScaler默认就做了方差归一化，和均值归一化，这两个归一化的目的都是为了更快的进行梯度下降
# 你如何构建你的训练集，你训练除了的模型，就具备什么样的功能！
scaler = StandardScaler().fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
theta = tf.Variable(tf.random_uniform([n+1, 1], -1., 1.), name='theta')

y_pred = tf.matmul(X, theta, name='predictors')
error = y_pred - y
rmse = tf.sqrt(tf.reduce_mean(tf.square(error), name='rmse'))

# 梯度的公式：(y_pred - y) * xj
grandients = 2/m * tf.matmul(tf.transpose(X), error, name='grandients')

# 赋值函数对于BGD来说就是 theta_new = theta - (learning_rate * gradients)
training_op = tf.assign(theta, theta - learning_rate * grandients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print('Epoch:', epoch, 'rmse', rmse.eval())
        sess.run(training_op)

    best_theta = sess.run(theta)    # theta.eval()
    print(best_theta)
