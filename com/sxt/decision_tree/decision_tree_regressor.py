import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

N = 100
x = np.random.rand(N) * 6 - 3
y = np.sin(x) + np.random.rand(N) * 0.05

print(y)
print(x.shape)
x = x.reshape(-1, 1)
print(x.shape)

dtr = DecisionTreeRegressor(criterion='mse', max_depth=3)
dtr.fit(x, y)

x_test = np.linspace(-3, 3, 100).reshape(-1, 1)
y_predict = dtr.predict(x_test)

plt.plot(x, y, 'y*', label='actual')
plt.plot(x_test, y_predict, 'b-', lw=2, label='predict')
plt.show()

depth = [2, 4, 6, 8, 10]
color = 'rgbmy'

dt_reg = DecisionTreeRegressor()
x_test = np.linspace(-3, 3, 50).reshape(-1, 1)

plt.plot(x, y, 'ko', label='actual')
for d, c in zip(depth, color):
    dt_reg.set_params(max_depth=d)
    dt_reg.fit(x, y)
    y_hat = dt_reg.predict(x_test)
    plt.plot(x_test, y_hat, color=c, lw=2, label='depth=%d' % d)
plt.legend('upper left')
plt.grid(True)
plt.show()
