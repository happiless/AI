import numpy as np
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X, y = iris['data'][:, 3:], iris['target']
print(X, y)

lr = LogisticRegression(solver='sag', max_iter=1000, multi_class='multinomial')
lr.fit(X, y)
y_predict = lr.predict(X)
print(y_predict)


X_new = np.linspace(0, 3, 1000).reshape(-1, 1)

y_proba = lr.predict_proba(X_new)
print(y_proba)
y_predict_new = lr.predict(X_new)
print(y_predict_new)
