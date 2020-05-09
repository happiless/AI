import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

temp = np.array([1, 2, 3, 5, 5]).reshape(-1, 1)
# 最大最小值归一化
scaler = MinMaxScaler()
scaler.fit(temp)

scaler_temp = scaler.transform(temp)
print(scaler_temp)

# 标准差归一化
scaler = StandardScaler()
scaler.fit(temp)
scaler_temp = scaler.transform(temp)
print(scaler_temp)
print(scaler.mean_)
print(scaler.var_)