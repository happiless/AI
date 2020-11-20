import numpy as np
from sklearn.linear_model import Ridge, SGDRegressor

np.random.seed(1)
X = np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

ridge_reg = Ridge(alpha=0.04, solver='sag')
ridge_reg.fit(X, y)

print(ridge_reg.predict([[1.5]]))
print(ridge_reg.intercept_, ridge_reg.coef_)

sgd_reg = SGDRegressor(penalty='l2', alpha=0.001, max_iter=1000)
sgd_reg.fit(X, y.reshape(-1,))
print(sgd_reg.predict([[1.5]]))
print(sgd_reg.intercept_, sgd_reg.coef_)
