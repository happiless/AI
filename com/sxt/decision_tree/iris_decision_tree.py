import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib as mpl

iris = load_iris()
data = pd.DataFrame(iris.data)
print(iris.feature_names)
data.columns = iris.feature_names
data['Species'] = iris.target
print(data)

x = data.iloc[:, 0:4]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

tree_clf = DecisionTreeClassifier(max_depth=8, criterion='gini')
tree_clf.fit(x, y)

print(x_train.shape)

export_graphviz(
    tree_clf,
    out_file='./iris_tree.dot',
    feature_names=iris.feature_names[:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

y_test_hat = tree_clf.predict(x_test)

print('acc score:', accuracy_score(y_true=y_test, y_pred=y_test_hat))

print(tree_clf.feature_importances_)

# print(tree_clf.predict_proba([[5, 1.5]]))
# print(tree_clf.predict([[5, 1.5]]))

depth = np.arange(1, 15)

err_list = []
for d in depth:
    print(d)
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=d)
    dtc.fit(x_train, y_train)
    y_test_hat = dtc.predict(x_test)
    result = (y_test_hat == y_test)
    if d == 1:
        print(result)
    err = 1 - np.mean(result)
    print('error:', err * 100, '%')
    err_list.append(err)

mpl.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(facecolor='w')
plt.plot(depth, err_list, 'ro-', lw=2)
plt.xlabel('决策树深度', fontsize=15)
plt.ylabel('错误率', fontsize=15)
plt.title('决策树深度和过拟合', fontsize=18)
plt.grid(True)
plt.show()
