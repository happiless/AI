import numpy as np
np.random.seed(0)

x = np.random.randint(0, 100, 10)
print(x)
y = np.array((x[3], x[7], x[2]))
print(y)
y = np.array(([2, 3], [3, 4]))
print(x[y])

x = np.arange(16).reshape(4, 4)
print(x)
row = np.array([1, 2, 3])
col = np.array([2, 1, 3])
print(x[row, col])

row1 = np.array([[1, 2], [2, 3]])
col1 = np.array([[1, 2], [2, 3]])
print(x[row1, col1])
