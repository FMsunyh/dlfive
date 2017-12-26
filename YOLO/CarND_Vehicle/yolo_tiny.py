#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/21/2017 12:03 PM 
# @Author : sunyonghai 
# @File : yolo_tiny.py 
# @Software: BG_AI
# =========================================================
from PIL import Image

from GPU_setting import set_gpu

set_gpu()

from keras import Sequential
from keras.layers import Convolution2D, LeakyReLU, MaxPooling2D, Flatten, Dense, Permute
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from utils.utils import load_weights, yolo_net_out_to_car_boxes, draw_box, draw_label, non_max_suppression, \
    decode_netout
import matplotlib.pyplot as plt
import  cv2

img_rows, img_cols = 448, 448
input_shape = (img_rows, img_cols, 3)

def yolo_tiny():
    model = Sequential()

    model.add(Convolution2D(filters=16, kernel_size=(3, 3), input_shape=input_shape, padding='same', name='conv_1', strides=(1, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', name='conv_2'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', name='conv_3'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same', name='conv_4'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same', name='conv_5'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    model.add(Convolution2D(filters=512, kernel_size=(3, 3), padding='same', name='conv_6'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

    model.add(Convolution2D(filters=1024, kernel_size=(3, 3), padding='same', name='conv_7'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Convolution2D(filters=1024, kernel_size=(3, 3), padding='same', name='conv_8'))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Convolution2D(filters=1024, kernel_size=(3, 3), padding='same', name='conv_9'))
    model.add(LeakyReLU(alpha=0.1))

    # model.add(Permute((2, 3, 1), input_shape=(7, 7, 1024)))
    model.add(Flatten())
    model.add(Dense(units=256))
    model.add(Dense(units=4096))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(units=1470))

    return model

def preprocess(im_path):
    try:
        img = image.load_img(im_path, grayscale=False, target_size=(448, 448))
    except Exception as ex:
        print(ex)
    else:
        img_arr = image.img_to_array(img)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = preprocess_input(img_arr)
    return img_arr


def preprocess_cv2(im_path):
    in_image = cv2.imread(im_path)
    input_image = cv2.resize(in_image, (448, 448))
    input_image = input_image / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    return input_image


if __name__ == '__main__':
    set_gpu()
    model = yolo_tiny()
    model.summary()

    # load_weights(model, 'weights/yolo-tiny.weights')
    #
    # batch = preprocess('test_images/test1.jpg')
    # # batch = preprocess_cv2('test_images/test1.jpg')
    # # batch = pre_image('test_images/test1.jpg')
    # netout = model.predict(batch)
    # boxes = yolo_net_out_to_car_boxes(netout[0], threshold=0.17)
    #
    # l_boxes = [box.get_rectangle_c() for box in boxes]
    #
    # dets = np.array(l_boxes)
    # thresh = 0.3
    # boxes = non_max_suppression(dets, thresh)
    #
    # bg_image = Image.open('test_images/test1.jpg')
    # for box in boxes:
    #     bg_image = draw_box(bg_image, box, outline=(255, 0, 0))
    #     bg_image = draw_label(bg_image, box, label='car', fill=(255, 0, 0))
    # bg_image.save('test_images/net_test1.jpg')

    # print(out[0])

