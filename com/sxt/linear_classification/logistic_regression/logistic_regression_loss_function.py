from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

data = load_breast_cancer()
print(data)
X, y = scale(data['data'][:, :2]), data['target']
print(X, y)

lr = LogisticRegression(fit_intercept=False)
lr.fit(X, y)
theta1 = lr.coef_[0, 0]
theta2 = lr.coef_[0, 1]
print(theta1, theta2)


def p_theta_function(features, w1, w2):
    z = w1 * features[0] + w2 * features[1]
    return 1 / (1 + np.exp(-z))


def loss_function(sample_features, sample_labels, w1, w2):
    result = 0
    for features, label in zip(sample_features, sample_labels):
        p_result = p_theta_function(features, w1, w2)
        loss_result = -1 * label * np.log(p_result) - (1 - label) * np.log(1 - p_result)
        result += loss_result
    return result


theta1_space = np.linspace(theta1 - 0.6, theta1 + 0.6, 50)
theta2_space = np.linspace(theta2 - 0.6, theta2 + 0.6, 50)

fig1 = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(np.array([loss_function(X, y, i, theta2) for i in theta1_space]))

plt.subplot(2, 2, 2)
plt.plot(np.array([loss_function(X, y, theta1, i) for i in theta2_space]))

plt.subplot(2, 2, 3)
theta1_grid, theta2_grid = np.meshgrid(theta1_space, theta2_space)
loss_grid = loss_function(X, y, theta1_grid, theta2_grid)
plt.contour(theta1_grid, theta2_grid, loss_grid)

plt.subplot(2, 2, 4)
theta1_grid, theta2_grid = np.meshgrid(theta1_space, theta2_space)
loss_grid = loss_function(X, y, theta1_grid, theta2_grid)
plt.contour(theta1_grid, theta2_grid, loss_grid, 30)

fig2 = plt.figure()
ax = Axes3D(fig2)
ax.plot_surface(theta1_grid, theta2_grid, loss_grid)
plt.show()
