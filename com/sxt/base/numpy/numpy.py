import numpy as np

a = np.array([1, 2, 3, 4, 5])

b = [1/i for i in a]
print(b)

print(1/a)
print(a/a)
print(a ** 2)

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 2, out=y)
print(y)

y = np.zeros(10)
np.power(2, x, out=y[::2])
print(y)

x = np.random.randint(0, 100, (3, 4))
y = np.random.randint(0, 100, (3, 4))
print(x, y)
print(x[x<50])

