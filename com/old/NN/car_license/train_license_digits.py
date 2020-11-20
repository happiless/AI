import sys
import os
import time
import random

import numpy as np
import tensorflow as tf

from PIL import Image

SIZE = 1280
WIDTH = 32
HEIGHT = 40
NUM_CLASSES = 34
iterations = 1000

SAVER_DIR = 'train-saver/digits/'

LETTERS_DIGITS = (
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G",
    "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z")
license_num = ""

x = tf.placeholder(dtype=tf.float32, shape=(None, SIZE))
y_ = tf.placeholder(dtype=tf.float32, shape=(None, NUM_CLASSES))

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])


# 定义卷积层函数
def conv_layer(inputs, w, b, conv_strides, kernel_size, pool_strides, padding):
    l1_conv = tf.nn.conv2d(inputs, w, strides=conv_strides, padding=padding)
    l2_conv = tf.nn.relu(l1_conv + b)
    return tf.nn.max_pool(l2_conv, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 定义全连接层
def full_connect(inputs, w, b):
    return tf.nn.relu(tf.matmul(inputs, w) + b)


if __name__ == '__main__' and sys.argv[1] == 'train':
    pass

if __name__ == '__main__' and sys.argv[1] == 'predict':
    pass
