import numpy as np
import pandas as pd
import csv
import datetime
import os
import scipy as sp
import xgboost as xgb
import itertools
import operator
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import TransformerMixin
from sklearn import model_selection
from matplotlib import pylab as plb
import matplotlib.pyplot as plt

plot = True
goal = 'Sales'
myId = 'Id'


# store = pd.read_csv('./data/store.csv')
# train = pd.read_csv('./data/train.csv')
# print('store:', store.head(), 'train:', train.head())
#
# plt.hist(train['Sales'], bins=20)
# plt.hist(train[train['Sales'] > 0]['Sales'], bins=20)
#
# print(train[['Sales', 'Open']])
# print(train['Sales'].value_counts())
# print(train['Open'].value_counts())


def to_weight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    print(ind, y)
    w[ind] = 1. / (y[ind] ** 2)
    return w


def rmspe(yhat, y):
    print(y.values)
    w = to_weight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return rmspe


def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = to_weight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat) ** 2))
    return 'rmspe', rmspe


def load_data():
    store = pd.read_csv('./data/store.csv')
    train_org = pd.read_csv('./data/train.csv', dtype={'StateHoliday': pd.np.string_})
    test_org = pd.read_csv('./data/test.csv', dtype={'StateHoliday': pd.np.string_})
    train = pd.merge(train_org, store, on='Store', how='left')
    test = pd.merge(test_org, store, on='Store', how='left')
    features = test.columns.tolist()
    print(features)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    features_numeric = test.select_dtypes(include=numerics).columns.tolist()
    features_non_numeric = [feature for feature in features if feature not in features_numeric]
    print(features_numeric)
    print(features_non_numeric)
    return train, test, features, features_non_numeric


def process_data(train, test, features, features_non_numeric):
    train = train[train['Sales'] > 0]
    for data in [train, test]:
        data['year'] = data.Date.apply(lambda x: x.split('-')[0])
        data['year'] = data['year'].astype(float)
        data['month'] = data.Date.apply(lambda x: x.split('-')[1])
        data['month'] = data['month'].astype(float)
        data['day'] = data.Date.apply(lambda x: x.split('-')[2])
        data['day'] = data['day'].astype(float)

        data['promojan'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jan" in x else 0)
        data['promofeb'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Feb" in x else 0)
        data['promomar'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Mar" in x else 0)
        data['promoapr'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Apr" in x else 0)
        data['promomay'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "May" in x else 0)
        data['promojun'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jun" in x else 0)
        data['promojul'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Jul" in x else 0)
        data['promoaug'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Aug" in x else 0)
        data['promosep'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Sep" in x else 0)
        data['promooct'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Oct" in x else 0)
        data['promonov'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Nov" in x else 0)
        data['promodec'] = data.PromoInterval.apply(lambda x: 0 if isinstance(x, float) else 1 if "Dec" in x else 0)

    # feature set
    noisy_features = [myId, 'Date']
    features = [f for f in features if f not in noisy_features]
    features_non_numeric = [f for f in features_non_numeric if f not in noisy_features]
    features.extend(['year', 'month', 'day'])

    # fill NA
    class DataFrameImputer(TransformerMixin):
        def __init__(self):
            """

            """

        def fit(self, X, y=None):
            self.fill = pd.Series([X[c].value_counts().index[0]  # mode
                                   if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],  # mean
                                  index=X.columns)
            return self

        def transform(self, X, y=None):
            return X.fillna(self.fill)

    train = DataFrameImputer().fit_transform(train)
    test = DataFrameImputer().fit_transform(test)

    # Pre-processing non-numberic values
    le = LabelEncoder()
    for col in features_non_numeric:
        le.fit(list(train[col]) + list(test[col]))
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

    # LR和神经网络这种模型都对输入数据的幅度极度敏感，请先做归一化操作
    scaler = StandardScaler()
    for col in set(features) - set(features_non_numeric) - set([]):
        scaler.fit(train[col].values.reshape(-1, 1))
        train[col] = scaler.transform(train[col].values.reshape(-1, 1))
        test[col] = scaler.transform(test[col].values.reshape(-1, 1))

    return train, test, features, features_non_numeric


def XGB_native(train, test, features, features_non_numeric):
    depth = 13
    eta = 0.01
    ntrees = 8000
    mcw = 3
    params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'eta': eta,
        'max_depth': depth,
        'min_child_weight': mcw,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'silent': 1
    }
    print("Running with params: " + str(params))
    print("Running with ntrees: " + str(ntrees))
    print("Running with features: " + str(features))

    # Train model with local split
    tsize = 0.05
    X_train, X_test = model_selection.train_test_split(train, test_size=tsize)

    d_train = xgb.DMatrix(X_train[features], np.log(X_train[goal] + 1))
    d_valid = xgb.DMatrix(X_test[features], np.log(X_test[goal] + 1))
    watch_list = [(d_valid, 'eval'), (d_train, 'train')]
    gbm = xgb.train(params, d_train, ntrees, evals=watch_list, early_stopping_rounds=100, feval=rmspe_xg,
                    verbose_eval=True)
    train_probs = gbm.predict(xgb.DMatrix(X_test[features]))
    indices = train_probs < 0
    train_probs[indices] = 0
    error = rmspe(np.exp(train_probs) - 1, X_test[goal].values)
    print(error)

    # Predict and Export
    test_probs = gbm.predict(xgb.DMatrix(test[features]))
    indices = test_probs < 0
    test_probs[indices] = 0
    submission = pd.DataFrame({myId: test[myId], goal: np.exp(test_probs) - 1})
    if not os.path.exists('./data/result'):
        os.makedirs('./data/result')
    submission.to_csv('./data/result/dat-xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.csv' %
                      (str(depth), str(eta), str(ntrees), str(mcw), str(tsize)))

    if plot:
        outfile = open('xgb.fmap', 'w')
        i = 0
        for feat in features:
            outfile.write('{0}\t{1}\tq\n'.format(i, feat))
            i = i + 1
        outfile.close()
        importance = gbm.get_fscore(fmap='xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()

        # plot
        plt.figure()
        df.plot()
        df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(25, 15))
        plt.xlabel('relative importance')
        plt.gcf().savefig('Feature_Importance_xgb_d%s_eta%s_ntree%s_mcw%s_tsize%s.png' %
                          (str(depth), str(eta), str(ntrees), str(mcw), str(tsize)))


print('=> 载入数据...')
train, test, features, features_non_numeric = load_data()
print('=> 处理数据与特征工程...')
train, test, features, features_non_numeric = process_data(train, test, features, features_non_numeric)
print('=> 使用XGBoost建模...')
XGB_native(train, test, features, features_non_numeric)
