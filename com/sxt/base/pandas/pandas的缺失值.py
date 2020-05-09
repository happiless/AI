import numpy as np
import pandas as pd

# 缺失值处理
vals1 = np.array([1, None, 3, 4])
# print(np.sum(vals1))
print(vals1.dtype)
# print(np.nansum(vals1))

vals2 = np.array([1, np.nan, 3, 4])
print(np.nansum(vals2))
# print(np.sum(vals2))
print(np.nanmax(vals2))

# 检测缺失值
data = pd.Series([1, np.nan, True, None])
print(data.isnull())
print(data.notnull())
print(data.isna())

# 剔除缺失值
data = data.dropna()
print(data)

df = pd.DataFrame([[1, np.nan, 2], [2, 3, 5], [np.nan, np.nan, 6]])

# 默认情况下，df对象会以行为单位drop掉有缺失值得数据
# df = df.dropna()
print(df)
# 填充缺失值的方法
print(df.dropna(axis='columns'))
print(df.dropna(how='all'))  # 全都是缺失值才drop
print(df.dropna(how='any'))
print(df.dropna(thresh=1))  # 缺失值少于3个得时候就drop

df.fillna(method='ffill')
data.fillna(method='bfill').fillna(0)

df.fillna(method='ffill', axis=1)

for i in df.columns:
    df[i] = df[i].fillna(df[i].mean())
print(df)