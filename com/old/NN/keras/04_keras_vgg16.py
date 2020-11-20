# 使用迁移学习的思想，以VGG16作为模板搭建模型，训练识别手写字体
# 引入VGG16模块
from keras.applications.vgg16 import VGG16

from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist

import cv2
import h5py
import numpy as np

model_vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(48, 48, 3))
for layer in model_vgg16.layers:
    layer.trainable = False
model = Flatten(name='flatten')(model_vgg16.output)
model = Dense(4096, activation='relu', name='fc1')(model)
model = Dense(4096, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation='softmax')(model)
model_vgg_mnist = Model(inputs=model_vgg16.input, outputs=model, name='vgg16')
model_vgg_mnist.summary()

# 新的模型不需要训练原有卷积结构里面的1471万个参数，但是注意参数还是来自于最后输出层前的两个
# 全连接层，一共有1.2亿个参数需要训练
sgd = SGD(lr=0.05, decay=1e-5)
model_vgg_mnist.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

# 因为VGG16对网络输入层的要求，我们用OpenCV把图像从32*32变成224*224，把黑白图像转成RGB图像
# 并把训练数据转化成张量形式，供keras输入
(X_train, y_train), (X_test, y_test) = \
    mnist.load_data('/Users/zhanghaibin/PycharmProjects/AI/com/old/NN/keras/data/keras_mnist_data')
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in X_train]
# 下面concatenate做的事情是把每个样本按照行堆叠在一起，因为是np下面的方法，所以返回的是ndarray
# np.newaxis它本质是None，arr是(48,48,3)，arr[None]是(1,48,48,3)
X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32')

X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in X_test]
X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')

print(X_train.shape)
print(X_test.shape)

X_train /= 255
X_test /= 255


def train_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe


y_train_ohe = np.array([train_y(y_train[i]) for i in range(len(y_train))])
y_test_ohe = np.array([train_y(y_test[i]) for i in range(len(y_test))])

model_vgg_mnist.fit(X_train, y_train_ohe, validation_data=(X_test, y_test_ohe),
                    epochs=100, batch_size=100)
