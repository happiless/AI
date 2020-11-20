import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import Generator
import InceptionV3
from keras.callbacks import TensorBoard


def train(steps_per_epoch=326, validation_steps=67, epochs=50):
    print('=======> training start')
    train_generator = Generator.train_data()
    valid_generator = Generator.valid_data()
    visualization = TensorBoard(log_dir='./logs', write_graph=True)

    model = InceptionV3.create_inception_v3()

    # 运行机器学习算法时，很多人一开始都会有意无意将数据集默认直接装进显卡显存中，
    # 如果处理大型数据集（例如图片尺寸很大）或是网络很深且隐藏层很宽，也可能造成显存不足。
    # 这个情况随着工作的深入会经常碰到，解决方法其实很多人知道，就是分块装入。
    # 以keras为例，默认情况下用fit方法载数据，就是全部载入。
    # 换用fit_generator方法就会以自己手写的方法用yield逐块装入。
    model.fit_generator(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=valid_generator,
                        validation_steps=validation_steps,
                        verbose=1,
                        callbacks=[visualization])
    model.save('./model/InceptionV3.h5')


train()

