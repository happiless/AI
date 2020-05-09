import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X ** 2 + X + np.random.randn(100, 1)

plt.plot(X, y, 'b.')

X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

d = {1: 'g-', 2: 'r+', 10: 'y*'}

for i in d:
    polynomial = PolynomialFeatures(degree=i, include_bias=True)
    X_poly_train = polynomial.fit_transform(X_train)
    X_poly_test = polynomial.fit_transform(X_test)
    print(X_train[0])
    print(X_poly_train[0])
    print(X_test[0])
    print(X_poly_test[0])
    print(X_poly_test.shape)
    lr = LinearRegression()
    lr.fit(X_poly_train, y_train)
    y_predict = lr.predict(X_poly_test)
    print(mean_squared_error(y_true=y_train, y_pred=lr.predict(X_poly_train)))
    print(mean_squared_error(y_true=y_test, y_pred=y_predict))
    plt.plot(X_poly_train[:, 1], lr.predict(X_poly_train), d[i])
    plt.plot(X_poly_test[:, 1], y_predict, d[i])

plt.show()
