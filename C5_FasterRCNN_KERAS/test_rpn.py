#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 1/3/2018 5:22 PM 
# @Author : sunyonghai 
# @File : test_rpn.py
# @Software: BG_AI
# =========================================================
from GPU_setting import set_gpu

set_gpu()

from keras import backend as K
from PIL import ImageDraw
from keras.layers import Input
from keras.models import Model
from keras_frcnn import resnet as nn, roi_helpers, config
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

def main():
    model_path = 'model/model_rpn.hdf5'
    file_dir = '/home/syh/dl/dlfive/C5_FasterRCNN_KERAS/data/VOC2012/JPEGImages/'
    # file_dir = '/home/syh/dl/dlfive/C5_FasterRCNN_KERAS/data/VOC2012/JPEGImages/2007_000027.jpg'

    input_shape_img = (None, None, 3)
    img_input = Input(shape=input_shape_img)
    num_anchors = 9

    share_layers = nn.nn_base(input_tensor=img_input)
    rpn = nn.rpn(share_layers, num_anchors)

    model = Model(img_input, rpn)

    model.load_weights(model_path)

    def read_img(path):
        img = load_img(path)
        x = img_to_array(img)
        x = np.expand_dims(img, axis=0)
        return img, x

    _, x = read_img(file_dir + '2007_000027.jpg')

    [Y1, Y2, F] = model.predict(x)

    def draw_rectangle(src_img, coordinates, num_rect):
        draw = ImageDraw.Draw(src_img)
        for i in range(num_rect):
            draw.rectangle(list(coordinates[i]), outline=(0, 0, 255))
        return src_img

    C = config.Config()

    ori_img, x = read_img(file_dir + '2007_000027.jpg')
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.005)
    coordinates = R * 16
    max_rect = len(R)

    result = draw_rectangle(ori_img, coordinates, max_rect)
    result.save("data/test/test_rpn.png", "png")