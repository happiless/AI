import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X1 = 2 * np.random.rand(100, 1)
X2 = 2 * np.random.rand(100, 1)

y = 5 + 4 * X1 + 3 * X2 + np.random.rand(100, 1)

X = np.c_[X1, X2]
reg = LinearRegression()
reg.fit(X, y)

print(reg.intercept_, reg.coef_)

X_new = np.array([[0, 0],
                  [1, 2],
                  [2, 4]])

y_predict = reg.predict(X_new)

plt.plot(X_new[:, 0], y_predict, 'r-')
plt.plot(X1, y, 'b.')
plt.axis([0, 2, 0, 30])
plt.show()
