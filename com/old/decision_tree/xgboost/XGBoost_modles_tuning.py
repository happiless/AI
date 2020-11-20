import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from sklearn import model_selection, metrics
from sklearn.model_selection import GridSearchCV
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

# 数据预处理
# 1. City因为类别太多丢掉
# 2. DOB生成Age字段，然后丢掉原字段
# 3. EMI_Loan_Submitted_Missing 为1(EMI_Loan_Submitted) 为0(EMI_Loan_Submitted缺省) EMI_Loan_Submitted丢掉
# 4. EmployerName丢掉
# 5. Existing_EMI对缺省值用均值填充
# 6. Interest_Rate_Missing同 EMI_Loan_Submitted
# 7. Lead_Creation_Date丢掉
# 8. Loan_Amount_Applied, Loan_Tenure_Applied 均值填充
# 9. Loan_Amount_Submitted_Missing 同 EMI_Loan_Submitted
# 10. Loan_Tenure_Submitted_Missing 同 EMI_Loan_Submitted
# 11. LoggedIn, Salary_Account 丢掉
# 12. Processing_Fee_Missing 同 EMI_Loan_Submitted
# 13. Source - top 2 kept as is and all others combined into different category
# 14. Numerical变化 和 One-Hot编码
train = pd.read_csv('./data/train_modified.csv')
test = pd.read_csv('./data/test_modified.csv')

print(train.shape, test.shape)

target = 'Disbursed'
myId = 'ID'
print(train['Disbursed'].value_counts())


# 写一个大的函数完成以下的功能
# 1. 数据建模
# 2. 求训练准确率
# 3. 求训练集AUC
# 4. 根据xgboost交叉验证更新n_estimators
# 5. 画出特征的重要度

# 建模与交叉验证
def model_fit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_params = alg.get_xgb_params()
        xg_train = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xg_test = xgb.DMatrix(dtest[predictors].values)
        cv_result = xgb.cv(xgb_params, xg_train, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                           early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        print(cv_result.shape[0])
        alg.set_params(n_estimators=cv_result.shape[0])
    # 建模
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    # 对训练集进行预测
    dtrain_predict = alg.predict(dtrain[predictors])
    dtrain_predict_proba = alg.predict_proba(dtrain[predictors])[:, 1]
    print('\n关于现在这个模型')
    print('准确率: %.4g' % metrics.accuracy_score(dtrain[target].values, dtrain_predict))
    print('AUC 得分 (训练集): %f' % metrics.roc_auc_score(dtrain[target], dtrain_predict_proba))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importances Score')


# 第1步- 对于高的学习率找到最合适的estimators个数
predictors = [x for x in train.columns if x not in [target, myId]]

xgb1 = XGBClassifier(learning_rate=0.1,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=1,
                     objective='binary:logistic',
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     nthread=4,
                     scale_pos_weight=1,
                     seed=27)

model_fit(xgb1, train, test, predictors)

param_test1 = {
    'max_depth': range(3, 10, 2),
    'min_child_weight': range(1, 6, 2)
}

g_search1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                 min_child_weight=1, gamma=0, subsample=0.8,
                                                 colsample_bytree=0.8, objective='binary:logistic',
                                                 nthread=4, scale_pos_weight=1, seed=27),
                         param_grid=param_test1, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
g_search1.fit(train[predictors], train[target])
print(g_search1.best_params_, g_search1.best_score_)

# 对于max_depth和min_child_weight查找最好的参数
param_test2 = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [4, 5, 6]
}
g_search2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                 min_child_weight=1, gamma=0, subsample=0.8,
                                                 colsample_bytree=0.8, objective='binary:logistic',
                                                 nthread=4, scale_pos_weight=1, seed=27),
                         param_grid=param_test2, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
g_search2.fit(train[predictors], train[target])
print(g_search2.best_params_, g_search2.best_score_)

# 交叉验证对min_child_weight寻找最合适的参数
param_test2b = {
    'min_child_weight': [6, 8, 10, 12]
}

g_search2b = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                  min_child_weight=1, gamma=0, subsample=0.8,
                                                  colsample_bytree=0.8, objective='binary:logistic',
                                                  nthread=4, scale_pos_weight=1, seed=27),
                          param_grid=param_test2b, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
g_search2b.fit(train[predictors], train[target])
print(g_search2b.grid_scores_, g_search2b.best_params_, g_search2b.best_score_)

# Grid seach选择合适的gamma
param_test3 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
g_search3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                 min_child_weight=1, subsample=0.8, gamma=0,
                                                 colsample_bytree=0.8, objective='binary:logistic',
                                                 nthread=4, scale_pos_weight=1, seed=27),
                         param_grid=param_test3, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
g_search3.fit(train[predictors], train[target])
print(g_search3.grid_scores_, g_search3.best_params_, g_search3.best_score_)

xgb2 = XGBClassifier(learning_rate=0.1, n_estimators=1000, max_depth=4,
                     min_child_weight=6, gamma=0, subsample=0.8,
                     colsample_bytree=0.8, objective='binary:logistic',
                     nthread=4, scale_pos_weight=1, seed=27)
model_fit(xgb2, train, test, predictors)

param_test4 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}
g_search4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=5,
                                                 min_child_weight=1, gamma=0, subsample=0.8,
                                                 colsample_bytree=0.8, objective='binary:logistic',
                                                 nthread=4, scale_pos_weight=1, seed=27),
                         param_grid=param_test4, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
g_search4.fit(train[predictors], train[target])
print(g_search4.grid_scores_, g_search4.best_params_, g_search4.best_score_)

# 对正则化做交叉验证
# 对reg_alpha用grid search寻找最合适的参数
param_test6 = {
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
}
g_search6 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=5,
                                                 min_child_weight=1, gamma=0, subsample=0.8,
                                                 colsample_bytree=0.8, objective='binary:logistic',
                                                 nthread=4, scale_pos_weight=1, seed=27),
                         param_grid=param_test6, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
g_search6.fit(train[predictors], train[target])
print(g_search4.grid_scores_, g_search4.best_params_, g_search4.best_score_)

param_test7 = {
    'reg_alpha': [0, 0.1, 0.01, 0.005, 0.05]
}
g_search7 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=5,
                                                 min_child_weight=1, gamma=0, subsample=0.8,
                                                 colsample_bytree=0.8, objective='binary:logistic',
                                                 nthread=4, scale_pos_weight=1, seed=27),
                         param_grid=param_test7, scoring='roc_auc', n_jobs=1, iid=False, cv=5)
g_search6.fit(train[predictors], train[target])
print(g_search4.grid_scores_, g_search4.best_params_, g_search4.best_score_)

xgb3 = XGBClassifier(learning_rate=0.1,
                     n_estimators=1000,
                     max_depth=5,
                     min_child_weight=1,
                     objective='binary:logistic',
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha=0.005,
                     nthread=4,
                     scale_pos_weight=1,
                     seed=27)
model_fit(xgb3, train, test, predictors)

xgb3 = XGBClassifier(learning_rate=0.1,
                     n_estimators=5000,
                     max_depth=5,
                     min_child_weight=1,
                     objective='binary:logistic',
                     gamma=0,
                     subsample=0.8,
                     colsample_bytree=0.8,
                     reg_alpha=0.005,
                     nthread=4,
                     scale_pos_weight=1,
                     seed=27)
model_fit(xgb3, train, test, predictors)
