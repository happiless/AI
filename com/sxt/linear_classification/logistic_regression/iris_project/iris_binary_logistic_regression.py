import numpy as np
import sklearn.datasets as datasets
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
print(iris.keys())
X, y = iris['data'][:, 3:], (iris['target'] == 2).astype(np.int)

print(X, y)

lr = LogisticRegression(solver='sag', max_iter=1000)
lr.fit(X, y)

y_predict = lr.predict(X)
print(y_predict)

y_proba = lr.predict_proba(X)
print(y_proba)