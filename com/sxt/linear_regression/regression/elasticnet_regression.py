import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

elasticnet_reg = ElasticNet(alpha=0.04, l1_ratio=0.15)
elasticnet_reg.fit(X, y)
print(elasticnet_reg.predict([[1.5]]))
print(elasticnet_reg.intercept_, elasticnet_reg.coef_)

sgd_reg = SGDRegressor(penalty='elasticnet')
sgd_reg.fit(X, y)
print(sgd_reg.predict([[1.5]]))
print(sgd_reg.intercept_, sgd_reg.coef_)
