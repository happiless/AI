from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import RMSprop


'''
首先我们来看一下全连接层的缺点：

在AlexNet及其之前的大抵上所有的基于神经网络的机器学习算法都要在卷积层之后添加上
全连接层来进行特征的向量化，此外出于神经网络黑盒子的考虑，有时设计几个全连接网络还可以
提升卷积神经网络的分类性能，一度成为神经网络使用的标配。
但是，我们同时也注意到，全连接层有一个非常致命的弱点就是参数量过大，
特别是与最后一个卷积层相连的全连接层。一方面增加了Training以及testing的计算量，
降低了速度；另外一方面参数量过大容易过拟合。虽然使用了类似dropout等手段去处理，
但是毕竟dropout是hyper-parameter， 不够优美也不好实践。

那么我们有没有办法将其替代呢？当然有，就是GAP(Global Average Pooling)。
我们要明确以下，全连接层将卷积层展开成向量之后不还是要针对每个feature map进行分类吗，
GAP的思路就是将上述两个过程合二为一，一起做了
'''


def create_inception_v3(classes=2):
    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = True

    # decay 每次更新后学习率的衰减值
    model.compile(optimizer=RMSprop(learning_rate=0.001, decay=0.9, epsilon=0.1),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # model.save('./model/InceptionV3.h5')
    return model
