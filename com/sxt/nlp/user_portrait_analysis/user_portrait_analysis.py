# 用户画像-案例
# 基于用户搜索关键词数据为用户打上标签（年龄，性别，学历）

# 整体流程
# （一）数据预处理
# 编码方式转换
# 对数据搜索内容进行分词
# 词性过滤
# 数据检查

# （二）特征选择
# 建立word2vec词向量模型
# 对所有搜索数据求平均向量

# （三）建模预测
# 不同机器学习模型对比
# 堆叠模型

# 将原始数据转换成utf-8编码，防止后续出现各种编码问题
# 由于原始数据比较大，在分词与过滤阶段会比较慢，这里我们选择了原始数据中的1W个

import csv
import pandas as pd
# import jieba
# import jieba.posseg
import os
import sys
import time
import itertools
from gensim.models import word2vec
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
'''
# （一）数据预处理
# 编码方式转换
data_path = './data/user_tag_query.10W.TRAIN'
csv_file = open(data_path + '-1w.csv', 'w')
writer = csv.writer(csv_file)
writer.writerow(['ID', 'Age', 'Gender', 'Education', 'QueryList'])
# 转换成utf-8编码格式
with open(data_path, 'r', encoding='gbk', errors='ignore') as f:
    lines = f.readlines()
    print(len(lines))
    for line in lines[:10000]:
        try:
            data = line.strip().split('\t')
            write_data = [data[0], data[1], data[2], data[3]]
            querystr = ''
            data[-1] = data[-1][:-1]
            for d in data[4:]:
                try:
                    cur_str = d.encode('utf8')
                    cur_str = cur_str.decode('utf8')
                    querystr += cur_str + '\t'
                    print(querystr)
                except:
                    continue
            querystr = querystr[:-1]
            write_data.append(querystr)
            writer.writerow(write_data)
        except:
            continue

# 编码转换完成的数据，取的是1W的子集
train_name = data_path + '-1w.csv'
data = pd.read_csv(train_name, encoding='gbk')
print(data.info())

data.Age.to_csv('./data/train_age.csv', index=False)
data.Gender.to_csv('./data/train_gender.csv', index=False)
data.Education.to_csv('./data/train_education.csv', index=False)
data.QueryList.to_csv('./data/train_querylist.csv', index=False)


# 对数据搜索内容进行分词
def input(train_name):
    train_data = []
    with open(train_name, 'rb') as f:
        line = f.readline()
        count = 0
        while line:
            try:
                train_data.append(line)
                count += 1
            except:
                print('error', count, line)
            line = f.readline()
    return train_data


start = time.clock()
filepath = './data/train_querylist.csv'
query_list = input(filepath)

write_path = './data/train_querylist_writefile-1w.csv'
csv_file = open(write_path, 'w')
# part-of-speech tagging 词性标注
POS = {}
for i in range(len(query_list)):
    if i % 2000 == 0 and i >= 1000:
        print(i, 'finished')
    s = []
    str = ""
    words = jieba.posseg.cut(query_list[i]) # 带有词性的精确分词模式
    allowPOS = ['n', 'v', 'j']
    for word, flag in words:
        print(word, flag)
        # 婚外 j 女人 n 不 d 爱 v 你 r 的 uj 表现 v
        POS[flag] = POS.get(flag, 0) + 1
        if (flag[0] in allowPOS) and len(word) >= 2:
            str += word + " "

    cur_str = str.encode('utf8')
    cur_str = cur_str.decode('utf8')
    s.append(cur_str)
    csv_file.write(" ".join(s) + "\n")
csv_file.close()
end = time.clock()
print("total time: %f s" % (end - start))
print(POS)
'''
# （二）特征选择

# 建立word2vec词向量模型
# 使用Gensim库建立word2vec词向量模型
# 参数定义：
# sentences：可以是一个list
# sg： 用于设置训练算法，默认为0，对应CBOW算法；sg=1则采用skip-gram算法。
# size：是指特征向量的维度，默认为100。大的size需要更多的训练数据,但是效果会更好. 推荐值为几十到几百。
# window：表示当前词与预测词在一个句子中的最大距离是多少
# alpha: 是学习速率
# seed：用于随机数发生器。与初始化词向量有关。
# min_count: 可以对字典做截断. 词频少于min_count次数的单词会被丢弃掉, 默认值为5
# max_vocab_size: 设置词向量构建期间的RAM限制。如果所有独立单词个数超过这个，则就消除掉其中最不频繁的一个。每一千万个单词需要大约1GB的RAM。设置成None则没有限制。
# workers参数控制训练的并行数。
# hs: 如果为1则会采用hierarchica·softmax技巧。如果设置为0（default），则negative sampling会被使用。
# negative: 如果>0,则会采用negativesampling，用于设置多少个noise words
# iter： 迭代次数，默认为5

# 将数据变换成list of list格式
# pip install --upgrade smart_open
train_path = './data/train_querylist_writefile-1w.csv'
save_path = "1w_word2vec_300.model"
with open(train_path, 'r') as f:
    my_list = []
    lines = f.readlines()
    for line in lines:
        cur_list = []
        data = line.strip().split(' ')
        for d in data:
            cur_list.append(d)
        my_list.append(cur_list)

    model = word2vec.Word2Vec(my_list, size=300, window=10, workers=4)
    model.save(save_path)

model = Word2Vec.load(save_path)
print(model.most_similar('大哥'))
print(model.most_similar('清华'))

# 对所有搜索数据求平均向量
# 加载训练好的word2vec模型，求用户搜索结果的平均向量
with open(train_path, 'r') as f:
    cur_index = 0
    lines = f.readlines()
    doc_cev = np.zeros((len(lines), 300))
    for line in lines:
        word_cev = np.zeros((1, 300))
        words = line.strip().split(" ")
        word_num = 0
        # 求模型的平均向量
        for word in words:
            if word in model:
                word_num += 1
                word_cev += np.array([model[word]])
        doc_cev[cur_index] = word_cev / float(word_num)
        cur_index += 1

print(doc_cev.shape)
print(doc_cev[0])

gender_label = np.loadtxt('./data/train_gender.csv', int)

education_label = np.loadtxt('./data/train_education.csv', int)

age_label = np.loadtxt('./data/train_age.csv', int)


def remove_zero(x, y):
    """
    把标签列Y为0的去除掉，对应Y为0的X矩阵的行也相应去掉
    :param x: 列表包含一个个用户搜素词的平均向量
    :param y: 用户性别标签列/年龄标签列/教育标签列
    :return: 返回去除标签列为0的记录X和y
    """
    nonzero = np.nonzero(y)
    y = y[nonzero]
    x = np.array(x)
    x = x[nonzero]
    return x, y


gender_train, gender_label = remove_zero(doc_cev, gender_label)
education_train, education_label = remove_zero(doc_cev, education_label)
age_train, age_label = remove_zero(doc_cev, age_label)
print(gender_train.shape, gender_label.shape)
print(education_train.shape, education_label.shape)
print(age_train.shape, age_label.shape)


# 定义一个函数去绘制混淆矩阵，为了评估看着方便
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # 加上图里面的颜色渐近条
    plt.colorbar()
    # 分别给横纵坐标在0和1的位置写上数字0和1
    tick_marks = np.arange(len(classes))
    plt.xticks(ticks=tick_marks, labels=classes, rotation=0)
    plt.yticks(ticks=tick_marks, labels=classes)
    thresh = cm.max() / 2.
    # itertools.product(a, b)       两个元组进行笛卡尔积
    # 在混淆矩阵图形四象限的格子里面写上数值，如果底色深就用白色，如果底色浅就用
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')


# （三）建模预测
# 不同机器学习模型对比
# 建立一个基础预测模型
X_train, X_test, y_train, y_test = train_test_split(gender_train, gender_label, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print(accuracy_score(y_pred=y_pred, y_true=y_test))
cnf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Test Recall metric => ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
print('Test accuracy metric => ', cnf_matrix[1, 1] / (cnf_matrix[0, 0] + cnf_matrix[0, 1]
                                                      + cnf_matrix[1, 0] + cnf_matrix[1, 1]))
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Gender-Confusion matrix')
plt.show()

rfc = RandomForestClassifier(n_estimators=100, min_samples_split=5, max_depth=10)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(accuracy_score(y_true=y_test, y_pred=y_pred))

cnf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
print('Test Recall metric => ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))
print('Test accuracy metric => ', cnf_matrix[1, 1] / (cnf_matrix[0, 0] + cnf_matrix[0, 1]))

plt.figure()
plot_confusion_matrix(cnf_matrix,
                      classes=class_names,
                      title='Gender-Confusion Matrix')
plt.show()

# 堆叠模型
clf1 = RandomForestClassifier(n_estimators=100, min_samples_split=5, max_depth=10)
clf2 = SVC()
clf3 = LogisticRegression()

base_model = [
    ['rf', clf1],
    ['svm', clf2],
    ['lr', clf3]
]

models = base_model
# 把第一阶段模型预测的结果，存在S_train和S_test中，供给第二阶段去训练
S_train = np.zeros(X_train.shape[0], len(models))
S_test = np.zeros(X_test.shape[0], len(models))

X_train, X_test, y_train, y_test = train_test_split(gender_train, gender_label, test_size=0.2, random_state=0)
folds = KFold(n_splits=5, random_state=0)
for i, bm in models:
    clf = bm[1]
    for train_idx, valid_idx in folds.split(X_train):
        X_train_cv = X_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_valid = X_train[valid_idx]
        clf.fit(X_train_cv, y_train_cv)
        y_valid = clf.predict(X_valid)[:]
        S_train[train_idx, i] = y_valid
    y_pred = clf.predict(X_test)
    S_test[:, i] = y_test
    print(accuracy_score(y_true=y_test, y_pred=y_pred))

# 第二阶段算法随便选择一个，这里选择了随机森林
final_rfc = RandomForestClassifier(n_estimators=100)
final_rfc.fit(S_train, y_train)
S_pred = final_rfc.predict(S_test)
print(S_pred)
print(final_rfc.score(S_test, y_test))

