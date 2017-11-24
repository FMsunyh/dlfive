#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 11/13/2017 10:34 AM 
# @Author : sunyonghai 
# @File : AlexNet.py
# @Software: BG_AI
# =========================================================

# endregion

# region import package

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# endregion

# In[1]:
# region Environment GPU setting

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)

# endregion

# In[2]:
# region Hyper Parameters

learning_rate = 0.01
training_epochs = 5000
batch_size = 128
display_step = 100
keep_rate = 0.8
# network parameter
num_classes = 10
# endregion

# In[3]:
# region Training Data

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

# endregion


# In[4]:
# region Prepare for Training

# tf Graph input.
X = tf.placeholder("float", [None, 28*28])
Y = tf.placeholder("float", [None, num_classes])

# set model weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial =tf.constant(0.1, shape=shape)
    return initial

# Convolution
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

# Pooling
def max_pool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

weights = {
    # 11 x 11 convolution, 1 input image, 96 outputs
    'W_conv1':weight_variable([11,11,3,96]),

    'W_conv2':weight_variable([5,5,96,256]),


    'W_conv3':weight_variable([3,3,256,384]),

    'W_conv4':weight_variable([3,3,384,384]),

    'W_conv5':weight_variable([3,3,384,256]),


    # fully connected
    'W_fc1':weight_variable([4096, 4096]),

    'W_fc2':weight_variable([4096, 4096]),

    # 4096 inputs, 10 outputs (class prediction)
    'out':weight_variable([4096, num_classes])
}

biases = {
    'b_conv1': bias_variable([96]),
    'b_conv2': bias_variable([256]),
    'b_conv3': bias_variable([384]),
    'b_conv4': bias_variable([384]),
    'b_conv5': bias_variable([256]),
    'b_fc1': bias_variable([4096]),
    'b_fc2': bias_variable([4096]),
    'out': bias_variable([num_classes]),
}

# In[5]:
# region create model.

def neural_net(x):

    #reshape x to 4D tensor [-1, width, height, color_channel]
    x = tf.reshape(x, shape=[-1,28,28,1])

    # First Convolution Layer
    conv1 = tf.nn.relu(tf.add(conv2d(x, weights['W_conv1']),biases['b_conv1']))
    conv1 = max_pool2d(conv1) # reduce the image to size 28*28 -> 14*14

    # Second Convolution Layer
    conv2 = tf.nn.relu(tf.add(conv2d(conv1, weights['W_conv2']), biases['b_conv2']))
    conv2 = max_pool2d(conv2) # reduce the image to size 14*14 -> 7*7

    # Densely Connected Layer
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.add(tf.matmul(fc, weights['W_fc1']), biases['b_fc1']))

    fc = tf.nn.dropout(fc, keep_prob=keep_rate)
    out_layer = tf.add(tf.matmul(fc, weights['out']), biases['out'])
    return out_layer

logits = neural_net(X)

# endregion