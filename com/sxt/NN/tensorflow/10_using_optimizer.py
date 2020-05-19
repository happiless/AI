import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

n_epochs = 1000
learning_rate = 0.01
batch_size = 2000

housing = fetch_california_housing(data_home=None, download_if_missing=True)
m, n = housing.data.shape

X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target)

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_train = np.c_[np.ones((len(X_train), 1)), X_train]
X_test = scaler.transform(X_test)
X_test = np.c_[np.ones((len(X_test), 1)), X_test]

# 下面部分X，Y最后用placeholder可以改成使用Mini BGD
# 构建计算的图
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
X = tf.placeholder(dtype=tf.float32, name='X')
y = tf.placeholder(dtype=tf.float32, name='y')

theta = tf.Variable(tf.random_uniform([n+1, 1], -1., 1.), name='theta')
y_pred = tf.matmul(X, theta, name='predictors')
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name='mse')

# 梯度的公式：(y_pred - y) * xj
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# gradients = tf.gradients(mse, [theta])[0]
# 赋值函数对于BGD来说就是 theta_new = theta - (learning_rate * gradients)
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# MomentumOptimizer收敛会比梯度下降更快
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mse)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batch = int(len(X_train) / batch_size)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            temp_theta = theta.eval()
            print(temp_theta)
            print('Train => Epoch:', epoch, 'mse:', mse.eval(feed_dict={X: X_train, y: y_train}))
            print('Test => Epoch:', epoch, 'mse:', mse.eval(feed_dict={X: X_test, y: y_test}))

        arr = np.arange(len(X_train))
        np.random.shuffle(arr)
        X_train = X_train[arr]
        y_train = y_train[arr]

        for i in range(n_batch):
            sess.run(optimizer, feed_dict={
                X: X_train[i * batch_size: i * batch_size + batch_size],
                y: y_train[i * batch_size: i * batch_size + batch_size]
            })
    best_theta = theta.eval()
    print(best_theta)
