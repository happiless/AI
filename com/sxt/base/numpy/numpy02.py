import numpy as np
x = np.arange(1, 4)
y = np.arange(4, 7)
print(x, y)
print(np.concatenate((x, y)))
print(np.hstack((x, y)))
print(np.vstack((x, y)))

x_y = np.concatenate((x, y))
print(np.split(x_y, (3, 5)))

x = np.arange(0, 16).reshape(4, 4)
print(x)
low, high = np.vsplit(x, [1])
print(low, high)
print(np.hsplit(x, [3]))