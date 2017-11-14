#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 11/14/2017 9:49 AM 
# @Author : sunyonghai 
# @File : excise4_1_1.py 
# @Software: BG_AI
# =========================================================

import tensorflow as tf
import os

import cifar10_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate = 0.001
training_epochs = 1
batch_size = 100
display_step = 10

image_size = 24
image_channel = 3
n_classes = 10

datast_dir = '/home/syh/dl/dlfive/data/cifar/'


def get_distorted_train_batch(data_dir, batch_size):
    #data_dir='data/cifar/'

    if not data_dir:
        raise ValueError('Please supply a data_dir')

    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
    return images, labels

def get_undistorted_eval_batch(eval_data, data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')

    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    images, labels = cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir, batch_size=batch_size)
    return images, labels



def get_undistorted_train_batch(eval_data, data_dir, batch_size):
    pass


def WeightsVariable(shape, name_str, stddev = 0.1):
    initial = tf.truncated_normal(shape=shape,stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial,dtype=tf.float32, name=name_str)

def BiasesVariable(shape, name_str, init_value=0.0):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

def Conv2(x, W , b, stride=1, padding='SAME', activation=tf.nn.relu, act_name='relu'):
    with tf.name_scope('conv2_bias'):
        y = tf.nn.conv2d(x,W, strides=[1,stride ,stride,1],padding=padding)
        y = tf.nn.bias_add(y,b)

    with tf.name_scope(act_name):
        y = activation(y)
    return  y

def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2, padding='SAME'):
    return  pool(x, ksize=[1,k,k,1], strides=[1,stride, stride, 1], padding='SAME')

def FullyConnected(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x,W)
        y = tf.add(y,b)

    with tf.name_scope(act_name):
        y = activate(y)

    return y

def Inference(images_holder):
    with tf.name_scope('Conv2d_1'):
        conv1_kernel_num = 64
        weights = WeightsVariable(shape=[5,5, image_channel, conv1_kernel_num], name_str='weights', stddev=5e-2)
        biases = BiasesVariable(shape=[conv1_kernel_num], name_str='baises', init_value=0.0)
        conv1_out = Conv2(images_holder, weights, biases, stride=1, padding='SAME')

    with tf.name_scope('Pool2d_1'):
        pool1_out = Pool2d(conv1_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME')


    with tf.name_scope('Conv2d_2'):
        conv2_kernel_num = 64
        weights = WeightsVariable(shape=[3,3, conv1_kernel_num, conv2_kernel_num], name_str='weights', stddev=5e-2)
        biases = BiasesVariable(shape=[conv2_kernel_num], name_str='baises', init_value=0.0)
        conv2_out = Conv2(pool1_out, weights, biases, stride=1, padding='SAME')

    with tf.name_scope('Pool2d_2'):
        pool2_out = Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME')

    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool2_out, [batch_size, -1])
        feats_dim = features.get_shape()[1].value

    with tf.name_scope('FC1_nolinear'):
        fc1_units_num = 384
        weights = WeightsVariable(shape=[feats_dim, fc1_units_num], name_str='weights', stddev=4e-2)
        biases = BiasesVariable(shape=[fc1_units_num], name_str='biases', init_value=0.1)

        fc1_out = FullyConnected(features, weights,biases,activate=tf.nn.relu, act_name='relu')


    with tf.name_scope('FC2_nolinear'):
        fc2_units_num = 192
        weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num], name_str='weights', stddev=4e-2)
        biases = BiasesVariable(shape=[fc2_units_num], name_str='biases', init_value=0.1)

        fc2_out = FullyConnected(fc1_out, weights, biases, activate=tf.nn.relu, act_name='relu')

    with tf.name_scope('FC3_linear'):
        fc3_units_num = n_classes
        weights = WeightsVariable(shape=[fc2_units_num, fc3_units_num], name_str='weights', stddev=1.0/fc3_units_num)
        biases = BiasesVariable(shape=[fc3_units_num], name_str='biases', init_value=0.0)

        logits = FullyConnected(fc2_out, weights, biases, activate=tf.identity, act_name='linear')

    return logits

with tf.Graph().as_default():
    with tf.name_scope('Inputs'):
        images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel],name='images')
        lables_holder = tf.placeholder(tf.int32, [batch_size], name='labels')

    with tf.name_scope('Inference'):
        logits = Inference(images_holder)

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=lables_holder, logits=logits)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        total_loss  = cross_entropy_mean

    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        globel_step = tf.Variable(0, name='global_step', trainable=False,dtype=tf.int64)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=globel_step)

    with tf.name_scope('Evaluate'):
        top_K_op = tf.nn.in_top_k(predictions=logits, targets=lables_holder, k=1)

    with tf.name_scope('GetTrainBatch'):
        image_train, labels_train = get_distorted_train_batch(data_dir=datast_dir, batch_size=batch_size)

    with tf.name_scope('GetTestBatch'):
        image_test, labels_test = get_undistorted_eval_batch(eval_data=True,data_dir=datast_dir, batch_size=batch_size)


    init_op = tf.global_variables_initializer()

    print('save the graph')
    graph_writer = tf.summary.FileWriter(logdir='graphs/excise412', graph=tf.get_default_graph())

    graph_writer.close()