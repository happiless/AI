import numpy as np

np.random.seed(1)
X = np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X]

n_epochs = 10000


m = 100

batch_size = 10

num_batches = int(m / batch_size)

t0, t1 = 5, 1000


def learning_rate_schedule(t):
    return t0 / (t + t1)


# 1,初始化θ, W0...Wn，标准正太分布创建W
theta = np.random.randn(2, 1)

# 4,判断是否收敛，一般不会去设定阈值，而是直接采用设置相对大的迭代次数保证可以收敛
for epoch in range(n_epochs):
    arr = np.arange(len(X_b))
    np.random.shuffle(arr)
    X_b = X_b[arr]
    y = y[arr]
    for i in range(num_batches):
        # 2,求梯度，计算gradient
        x_batch = X_b[i * batch_size:i * batch_size + batch_size]
        y_batch = y[i * batch_size:i * batch_size + batch_size]
        gradients = x_batch.T.dot(x_batch.dot(theta) - y_batch)
        # 3,应用梯度下降法的公式去调整θ值 θt+1=θt-η*gradient
        theta = theta - learning_rate_schedule(epoch + m + i) * gradients

print(theta)
