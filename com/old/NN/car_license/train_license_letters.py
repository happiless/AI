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
NUM_CLASSES = 26
iterations = 500

SAVER_DIR = './train_saver/letters/'

LETTER_DIGITS = (
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "I", "O")

license_num = ""

time_begin = time.time()

# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(dtype=tf.float32, shape=(None, SIZE))
y = tf.placeholder(dtype=tf.float32, shape=(None, NUM_CLASSES))
x_image_ = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])


def conv_layer(inputs, w, b, conv_strides, kernel_size, pool_strides, padding):
    l1_conv = tf.nn.conv2d(inputs, w, strides=conv_strides, padding=padding)
    l1_relu = tf.nn.relu(l1_conv + b)
    return tf.nn.max_pool(l1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


def full_connect(inputs, w, b):
    return tf.nn.relu(tf.matmul(inputs, w) + b)


if __name__ == '__main__' and sys.argv[1] == 'train':
    pass

if __name__ == '__main__' and sys.argv[1] == 'predict':
    pass
