from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy as np

# 先读入数据
(x_train, y_train), (x_test, y_test) = \
    mnist.load_data('/Users/zhanghaibin/PycharmProjects/AI/com/old/NN/keras/data/keras_mnist_data')
print(x_train, y_train, x_test, y_test)
print(x_train.shape)
print(x_train[0].shape)
print(x_train[0])

# 下面把训练集中的手写黑白字体变成标准的四维张量形式，即（样本数量，高，宽，1）
# 并把像素值变成浮点格式
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype(np.float32)

# 由于每个像素值都介于0到255，所以这里统一除以255，把像素值控制在0-1范围
x_train /= 255
x_test /= 255


# one-hot编码
def train_y(y):
    y_ohe = np.zeros(10)
    y_ohe[y] = 1
    return y_ohe


y_train_ohe = np.array([train_y(y_train[i]) for i in range(len(y_train))])
y_test_ohe = np.array([train_y(y_test[i]) for i in range(len(y_test))])

# 搭建卷积神经网络
model = Sequential()
# 添加一层卷积层，构造64个过滤器，每个过滤器覆盖范围是3*3*1
# 过滤器步长为1，图像四周补一圈0，并用relu进行非线性变化
model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                 input_shape=(28, 28, 1), activation='relu'))
# 添加一层最大池化层
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# 设立Dropout层，Dropout的概率为0.5
model.add(Dropout(0.5))
# 重复构造，搭建深度网络
model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                 activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                 activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))

# 把当前层节点展平, 准备全连接
model.add(Flatten())

# 构造全连接层神经网络层
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 定义损失函数，一般来说分类问题的损失函数都选择采用交叉熵
model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train_ohe, validation_data=(x_test, y_test_ohe),
          epochs=20, batch_size=128)

# 在测试集上评价模型的准确率
# verbose : 进度表示方式。0表示不显示数据，1表示显示进度条
