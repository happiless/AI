import numpy as np

X = np.random.rand(100, 1)


def p_theta_function(features, w1, w2):
    z = w1 * features[0] + w2 * features[1]
    return 1 / (1 + np.exp(-z))


# 创建超参数
n_iterations = 10000

t0, t1 = 5, 500


# 定义一个函数来调整学习率
def learning_rate_schedule(t):
    return t0 / (t + t1)


theta1 = np.random.randn(2, 1)
theta2 = np.random.randn(2, 1)

for i in range(n_iterations):
    gradients = X.T.dot(X.dot(theta1) + X.dot(theta2) - p_theta_function(X, theta1, theta2))
    theta1 = theta1 - learning_rate_schedule(i) * gradients
    theta2 = theta1 - learning_rate_schedule(i) * gradients

print(theta1)
print(theta2)

