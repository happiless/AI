import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X1 = 2 * np.random.rand(100, 1)
X2 = 3 * np.random.rand(100, 1)

y = 5 + 4 * X1 + 3 * X2 + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X1, X2]
# X_b = np.c_[X1, X2, np.ones((100, 1))]

print(X_b)

# 解析解公式求解theta
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta)

# 预测
X_new = np.array([[0, 0], [2, 3]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
print(X_new_b)
y_predict = X_new_b.dot(theta)
print(y_predict)

plt.plot(X_new[:, 1], y_predict)
plt.plot(X2, y, 'b.')

plt.show()
