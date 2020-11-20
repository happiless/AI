# coding=utf-8
from keras.preprocessing.image import ImageDataGenerator

batch_size = 5
width, height = 299, 299


def train_data(train_data_dir='./data/train'):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=15,
                                       shear_range=0.5,
                                       zoom_range=0.2,
                                       width_shift_range=0.3,
                                       height_shift_range=0.3,
                                       horizontal_flip=True,
                                       vertical_flip=True)
    # .flow_from_directory(directory)
    # 方法实例化一个针对图像batch的生成器，这些生成器可以被用作keras模型相关方法的输入，
    # 如fit_generator，evaluate_generator和predict_generator
    # class_mode: 值为"categorical", "binary".用于计算分类正确率或调用
    # predict_classes方法.
    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size=(width, height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    return train_generator


def valid_data(valid_data_dir='./data/validation'):
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_generator = valid_datagen.flow_from_directory(valid_data_dir,
                                                        target_size=(width, height),
                                                        batch_size=batch_size,
                                                        class_mode='categorical')
    return valid_generator

