import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from mpl_toolkits.mplot3d import Axes3D

data = load_breast_cancer()
print(data)
X = data['data'][:, :2]
y = data['target']
print(X)

X = scale(X)
print(X)

lr = LogisticRegression(C=1000, fit_intercept=False)
lr.fit(X, y)
print(lr.coef_)

w1 = lr.coef_[0, 0]
w2 = lr.coef_[0, 1]


def h_theta_function(features, theta1, theta2):
    z = features[0] * theta1 + features[1] * theta2
    return 1/(1+np.exp(-z))


def loss_function(sample_features, sample_labels, w1, w2):
    """

    :param sample_features: 所有样本的X1，X2
    :param sample_labels:  所有样本的y
    :param w1:
    :param w2:
    :return:
    """
    result = 0
    for features, label in zip(sample_features, sample_labels):
        y_hat = h_theta_function(features, w1, w2)
        # 逻辑回归的损失函数
        loss = -1 * (label * np.log(y_hat) + (1-label) * np.log(1 - y_hat))
        result += loss
    return result


w1_space = np.linspace(w1-0.6, w2+0.6, 100)
w2_space = np.linspace(w1-0.6, w2+0.6, 100)

result1 = np.array([loss_function(X, y, i, w2) for i in w1_space])
result2 = np.array([loss_function(X, y, w1, i) for i in w2_space])

fig = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(w1_space, result1)
plt.subplot(2, 2, 2)
plt.plot(w2_space, result2)
plt.subplot(2, 2, 3)
w1_grid, w2_grid = np.meshgrid(w1_space, w2_space)
loss_grid = loss_function(X, y, w1_grid, w2_grid)
plt.contour(w1_grid, w2_grid, loss_grid)
plt.subplot(2, 2, 4)
w1_grid, w2_grid = np.meshgrid(w1_space, w2_space)
loss_grid = loss_function(X, y, w1_grid, w2_grid)
plt.contour(w1_grid, w2_grid, loss_grid, 20, alpha=0.6, cmap=plt.cm.hot)

fig2 = plt.figure()
ax = Axes3D(fig2)
ax.plot_surface(w1_grid, w2_grid, loss_grid)

plt.show()
