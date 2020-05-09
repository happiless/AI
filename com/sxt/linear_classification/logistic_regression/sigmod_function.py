import numpy as np
import math
import matplotlib.pyplot as plt


def sigmod(x):
    a = []
    for i in x:
        a.append(1 / (1 + math.exp(-i)))
    return a


x = np.arange(-10, 10, 0.1)
plt.plot(x, sigmod(x))
plt.show()
