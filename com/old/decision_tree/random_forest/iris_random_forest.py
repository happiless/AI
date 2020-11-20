from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
print(X, y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rfc = RandomForestClassifier(n_estimators=5, max_leaf_nodes=16, n_jobs=1, oob_score=True)
rfc.fit(x_train, y_train)

y_test_hat = rfc.predict(x_test)
print(accuracy_score(y_true=y_test, y_pred=y_test_hat))
# print(rfc.feature_importances_)
print(rfc.oob_score_)

rfc = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rfc.fit(iris.data, iris.target)
for name, score in zip(iris.feature_names, rfc.feature_importances_):
    print(name, ":", score)
