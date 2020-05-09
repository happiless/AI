import numpy as np
array = np.array([[1, 2, 3], [4, 5, 6]])
print(array)
print(array.ndim)
print(array.shape)
print(array.size)
print(array.dtype)
print(array.itemsize)
print(array.nbytes)

# 创建矩阵
print(np.zeros((3, 5)))
print(np.ones((3, 5)))
print(np.full((3, 5), fill_value=2*3.14))
print(np.arange(1, 10, 2))
print(np.random.randn(3, 3))

print(np.linspace(0, 1, 20))
print(np.random.normal(loc=3, scale=4, size=(3, 2, 3)))
print(np.random.randint(0, 10, (3, 3)))
print(np.eye(10))
print(np.empty((2, 2)))

x2 = np.arange(1, 13).reshape((3, 4))
print(x2)
print(x2[2][2])
print(x2[2, 2])
print(x2[(1, 2), (1, 2)])
print(x2[0:2:1, 1:3:1])
x2_sub = x2[:2,:2]
print(x2_sub)
