"""
车牌框的识别 剪切保存
"""
# 使用的是HyperLPR已经训练好了的分类器
import os
import cv2
from PIL import Image
import time
import numpy as np
import tensorflow as tf
from pip._vendor.distlib._backport import shutil


def find_car_num_brod():
    watch_cascade = cv2.CascadeClassifier('./cascade.xml')
    # 读取图片
    image = cv2.imread("./car_image/su.jpg")
    resize_h = 1000
    height = image.shape[0]
    scale = float(image.shape[1] / height)
    image = cv2.resize(image, (int(scale * resize_h), resize_h))
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    watches = watch_cascade.detectMultiScale(image_gray, 1.2,
                                             minNeighbors=4,
                                             minSize=(36, 9),
                                             maxSize=(106 * 40, 59 * 40))
    print("检测到车牌数", len(watches))
    if len(watches) == 0:
        return False
    for (x, y, w, h) in watches:
        print(x, y, w, h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cut_img = image[y - 5:y + 5 + h, x + 8:x - 8 + w]  # 裁剪坐标为[y0:y1, x0:x1]
        cut_gray = cv2.cvtColor(cut_img, cv2.COLOR_RGB2GRAY)
        cv2.imshow('rectangle', cut_gray)
        cv2.waitKey(0)
        cv2.imwrite('./num_for_car.jpg', cut_gray)
        im = Image.open('./num_for_car.jpg')
        size = 720, 180
        mmm = im.resize(size, Image.ANTIALIAS)
        mmm.save('./num_for_car.jpg', 'JPEG', quality=95)

    return True


'''
剪切后车牌的字符单个拆分保存处理
'''


def cut_car_num_for_chart():
    # 1、读取图像，并把图像转换为灰度图像并显示
    img = cv2.imread('./num_for_car.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('gray', img_gray)  # 显示图片
    # 2、将灰度图像二值化，设定阈值是100   转换后 白底黑字 ---》 目标黑底白字

    pass


if __name__ == '__main__':
    find_car_num_brod()
