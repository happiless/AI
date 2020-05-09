import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

# 读取数据
data = pd.read_csv("./data/insurance.csv")

print(data.head(n=5))

# EDA数据探索
# plt.hist(np.log(data['charges']))
sns.kdeplot(data.loc[data.sex == 'male', 'charges'], shade=True, label='male')
sns.kdeplot(data.loc[data.sex == 'female', 'charges'], shade=True, label='female')

sns.kdeplot(data.loc[data.smoker == 'yes', 'charges'], shade=True, label='smoker yes')
sns.kdeplot(data.loc[data.smoker == 'no', 'charges'], shade=True, label='smoker no')

sns.kdeplot(data.loc[data.region == 'southwest', 'charges'], shade=True, label='southwest')
sns.kdeplot(data.loc[data.region == 'southeast', 'charges'], shade=True, label='southeast')
sns.kdeplot(data.loc[data.region == 'northwest', 'charges'], shade=True, label='northwest')
sns.kdeplot(data.loc[data.region == 'northeast', 'charges'], shade=True, label='northeast')

sns.kdeplot(data.loc[data.children == 0, 'charges'], shade=True, label='0')
sns.kdeplot(data.loc[data.children == 1, 'charges'], shade=True, label='1')
sns.kdeplot(data.loc[data.children == 2, 'charges'], shade=True, label='2')
sns.kdeplot(data.loc[data.children == 3, 'charges'], shade=True, label='3')
sns.kdeplot(data.loc[data.children == 4, 'charges'], shade=True, label='4')
sns.kdeplot(data.loc[data.children == 5, 'charges'], shade=True, label='5')

plt.show()


# 特征工程
def greater(df, bmi, num_child):
    df['bmi'] = 'over' if df['bmi'] >= bmi else 'under'
    df['children'] = 'no' if df['children'] == num_child else 'yes'
    return df


# 降噪
data = data.drop(['sex', 'region'], axis=1)
data = data.apply(greater, axis=1, args=(30, 0))
print(data.head())
# OneHot编码
data = pd.get_dummies(data)
data.fillna(0, inplace=True)
print(data)
x = data.drop(['charges'], axis=1)
y = data['charges']
# 模型训练
# 归一化
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
scaler = StandardScaler(with_mean=True, with_std=True).fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
# 升维
poly_features  = PolynomialFeatures()
x_train_poly = poly_features.fit_transform(x_train_scaled)
x_test_poly = poly_features.fit_transform(x_test_scaled)

# lr
lr = LinearRegression()
lr.fit(x_train_poly, np.log1p(y_train))
y_train_predict = lr.predict(x_train_poly)
y_test_predict = lr.predict(x_test_poly)

# Ridge
ridge = Ridge()
ridge.fit(x_train_poly, np.log1p(y_train))
y_train_predict_ridge = ridge.predict(x_train_poly)
y_test_predict_ridge = ridge.predict(x_test_poly)

# GradientBoostingRegressor
boost = GradientBoostingRegressor()
boost.fit(x_train_poly, np.log1p(y_train))
y_train_predict_boost = boost.predict(x_train_poly)
y_test_predict_boost = boost.predict(x_test_poly)

# 模型评估
log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=y_train_predict))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_test_predict))
exp_rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict)))
exp_rmse_test = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict)))
print('lr:', log_rmse_train, log_rmse_test, exp_rmse_train, exp_rmse_test)

log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=y_train_predict_ridge))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_test_predict_ridge))
exp_rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict_ridge)))
exp_rmse_test = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict_ridge)))
print('ridge:', log_rmse_train, log_rmse_test, exp_rmse_train, exp_rmse_test)

log_rmse_train = np.sqrt(mean_squared_error(y_true=np.log1p(y_train), y_pred=y_train_predict_boost))
log_rmse_test = np.sqrt(mean_squared_error(y_true=np.log1p(y_test), y_pred=y_test_predict_boost))
exp_rmse_train = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict_boost)))
exp_rmse_test = np.sqrt(mean_squared_error(y_true=y_train, y_pred=np.exp(y_train_predict_boost)))
print('boost:', log_rmse_train, log_rmse_test, exp_rmse_train, exp_rmse_test)
