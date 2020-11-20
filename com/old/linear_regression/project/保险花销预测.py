import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('./data/insurance.csv')
print(data.head(n=5))

# EDA数据探索

# plt.hist(data.charges)
plt.hist(np.log(data.charges))
plt.show()

# 特征工程
data = pd.get_dummies(data)
print(data.head(n=5))
data.fillna(0, inplace=True)
x = data.drop('charges', axis=1)
y = data['charges']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# 归一化
scaler = StandardScaler(with_mean=True, with_std=True).fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_train_scaled)
# 升维
polynomial = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = polynomial.fit_transform(x_train_scaled)
x_test_poly = polynomial.fit_transform(x_test_scaled)
# 模型训练
# lr
lr = LinearRegression()
lr.fit(x_train_poly, np.log1p(y_train))
y_train_predict = lr.predict(x_train_poly)
y_test_predict = lr.predict(x_test_poly)

# Ridge
ridge = Ridge(alpha=0.4)
ridge.fit(x_train_poly, np.log1p(y_train))
y_train_predict_ridge = ridge.predict(x_train_poly)
y_test_predict_ridge = ridge.predict(x_test_poly)

# GradientBoostingRegressor
boost = GradientBoostingRegressor()
boost.fit(x_train_poly, np.log1p(y_train))
y_train_predict_boost = boost.predict(x_train_poly)
y_test_predict_boost = boost.predict(x_test_poly)

# 模型评估
# lr
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=y_train_predict))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_test_predict))
exp_rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict)))
exp_rmes_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_test_predict)))
print('lr:', log_rmse_train,log_rmse_test,exp_rmse_train,exp_rmes_test)

log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=y_train_predict_ridge))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_test_predict_ridge))
exp_rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict_ridge)))
exp_rmes_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_test_predict_ridge)))
print('ridge:', log_rmse_train,log_rmse_test,exp_rmse_train,exp_rmes_test)

log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=y_train_predict_boost))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_test_predict_boost))
exp_rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict_boost)))
exp_rmes_test = np.sqrt(mean_squared_error(y_true=y_test, y_pred=np.exp(y_test_predict_boost)))
print('boost:', log_rmse_train,log_rmse_test,exp_rmse_train,exp_rmes_test)
