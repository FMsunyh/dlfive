#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/28/2017 6:08 PM 
# @Author : sunyonghai 
# @File : ResNet50.py 
# @Software: BG_AI
# =========================================================
import glob
from keras.utils import Sequence
import math
import numpy as np
import cv2
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input
from keras import optimizers
from keras.applications.resnet50 import ResNet50


def my_generator(path, batch_size, target_size=(256, 256)):
    a_filenames = glob.glob(path + 'a/*.jpg')
    a_filenames.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    p_filenames = glob.glob(path + 'p/*.jpg')
    p_filenames.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    n_filenames = glob.glob(path + 'n/*.jpg')
    n_filenames.sort(key=lambda x: int(x.split('/')[-1][:-4]))
    num_imgs = len(glob.glob(path + 'a/*.jpg'))
    num_batches = math.ceil(num_imgs / batch_size)

    while True:
        for idx in range(num_batches):
            batch_a = a_filenames[idx * batch_size: (idx + 1) * batch_size]
            batch_p = p_filenames[idx * batch_size: (idx + 1) * batch_size]
            batch_n = n_filenames[idx * batch_size: (idx + 1) * batch_size]

            a_arrays = np.array([cv2.resize(cv2.imread(filename), target_size) for filename in batch_a])
            p_arrays = np.array([cv2.resize(cv2.imread(filename), target_size) for filename in batch_p])
            n_arrays = np.array([cv2.resize(cv2.imread(filename), target_size) for filename in batch_n])
            batch_y = np.zeros((batch_size, batch_size))

            yield [a_arrays, p_arrays, n_arrays], batch_y

def dummy_gen():

    while True:
        for i in range(100):
            a = np.random.random((1, 224, 224, 3))
            b = np.random.random((1, 224, 224, 3))
            c = np.random.random((1, 224, 224, 3))
            y = np.zeros((10, 10))

            yield [a, b, c], y

# inp = Input((256, 256, 3))
# inp1 = Input((256, 256, 3))
# inp2 = Input((256, 256, 3))
# inp3 = Input((256, 256, 3))

# x = Conv2D(128, 3, input_shape=(256, 256, 3))(inp)
# x = Conv2D(256, 3)(x)
# x = Conv2D(256, 3)(x)
# base_model = Model(inp, x)
base_model = ResNet50(weights='imagenet')
base_model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy')

# out1 = base_model(inp1)
# out2 = base_model(inp2)
# out3 = base_model(inp3)

# model = Model([inp1, inp2, inp3], [out1, out2, out3])


# my_gen = my_generator(path, batch_size=1, target_size=(256, 256))
# my_gen = dummy_gen()
# model.predict_generator(my_gen, steps=10, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

x = np.random.random((1, 224, 224, 3))
X = np.concatenate([x for i in range(1000)])
y = np.random.random((1000, 1000))
base_model.fit(X, y, batch_size=10, epochs=10, verbose=1)