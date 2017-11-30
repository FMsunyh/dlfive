#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 11/29/2017 10:47 AM 
# @Author : sunyonghai 
# @File : alexnet_graph.py 
# @Software: BG_AI
# =========================================================
import tensorflow as tf
import  os

# region Environment GPU setting

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)

# endregion

# region Hyper Parameters

learning_rate_init = 0.001
training_epochs = 1
batch_size = 32
display_step = 10
keep_rate = 0.8
conv1_kernel_num = 96
conv2_kernel_num = 256
conv3_kernel_num = 384
conv4_kernel_num = 384
conv5_kernel_num = 256
fc1_units_num = 4096
fc2_units_num = 4096

image_size = 224
image_channel = 3
n_classes = 1000


def WeightsVariable(shape, name_str, stddev=0.1):
    initial = tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)


def BiasesVariable(shape, name_str, init_value=0.0):
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name_str)

def Conv2(x, W, b, stride=1, padding='SAME', activation=tf.nn.relu, act_name='relu'):
    with tf.name_scope('conv2_bias'):
        y = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
        y = tf.nn.bias_add(y, b)

    with tf.name_scope(act_name):
        y = activation(y)
    return y

def Pool2d(x, pool=tf.nn.max_pool, k=2, stride=2, padding='SAME'):
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding=padding)


def FullyConnected(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)

    with tf.name_scope(act_name):
        y = activate(y)

    return y

def Inference(images_holder):
    with tf.name_scope('Conv2d_1'):
        weights = WeightsVariable(shape=[11, 11, image_channel, conv1_kernel_num], name_str='weights', stddev=5e-2)
        biases = BiasesVariable(shape=[conv1_kernel_num], name_str='baises', init_value=0.0)
        conv1_out = Conv2(images_holder, weights, biases, stride=4, padding='SAME')

        with tf.name_scope('Pool2d_1'):
            pool1_out = Pool2d(conv1_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')

        with tf.name_scope('Conv2d_2'):
            weights = WeightsVariable(shape=[5, 5, conv1_kernel_num, conv2_kernel_num], name_str='weights', stddev=5e-2)
            biases = BiasesVariable(shape=[conv2_kernel_num], name_str='baises', init_value=0.0)
            conv2_out = Conv2(pool1_out, weights, biases, stride=1, padding='SAME')

        with tf.name_scope('Pool2d_2'):
            pool2_out = Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')

        with tf.name_scope('Conv2d_3'):
            weights = WeightsVariable(shape=[3, 3, conv2_kernel_num, conv3_kernel_num], name_str='weights', stddev=5e-2)
            biases = BiasesVariable(shape=[conv3_kernel_num], name_str='baises', init_value=0.0)
            conv3_out = Conv2(pool2_out, weights, biases, stride=1, padding='SAME')

        with tf.name_scope('Conv2d_4'):
            weights = WeightsVariable(shape=[3, 3, conv3_kernel_num, conv4_kernel_num], name_str='weights', stddev=5e-2)
            biases = BiasesVariable(shape=[conv4_kernel_num], name_str='baises', init_value=0.0)
            conv4_out = Conv2(conv3_out, weights, biases, stride=1, padding='SAME')

        with tf.name_scope('Conv2d_5'):
            weights = WeightsVariable(shape=[3, 3, conv4_kernel_num, conv5_kernel_num], name_str='weights', stddev=5e-2)
            biases = BiasesVariable(shape=[conv5_kernel_num], name_str='baises', init_value=0.0)
            conv5_out = Conv2(conv4_out, weights, biases, stride=1, padding='SAME')

        with tf.name_scope('Pool2d_5'):
            pool5_out = Pool2d(conv5_out, pool=tf.nn.max_pool, k=3, stride=2, padding='VALID')

        with tf.name_scope('FeatsReshape'):
            features = tf.reshape(pool5_out, [batch_size, -1])
            feats_dim = features.get_shape()[1].value

        with tf.name_scope('FC1_nolinear'):
            weights = WeightsVariable(shape=[feats_dim, fc1_units_num], name_str='weights', stddev=4e-2)
            biases = BiasesVariable(shape=[fc1_units_num], name_str='biases', init_value=0.1)

            fc1_out = FullyConnected(features, weights, biases, activate=tf.nn.relu, act_name='relu')

        with tf.name_scope('FC2_nolinear'):
            weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num], name_str='weights', stddev=4e-2)
            biases = BiasesVariable(shape=[fc2_units_num], name_str='biases', init_value=0.1)

            fc2_out = FullyConnected(fc1_out, weights, biases, activate=tf.nn.relu, act_name='relu')

        with tf.name_scope('FC3_linear'):
            fc3_units_num = n_classes
            weights = WeightsVariable(shape=[fc2_units_num, fc3_units_num], name_str='weights', stddev=1.0 / fc2_units_num)
            biases = BiasesVariable(shape=[fc3_units_num], name_str='biases', init_value=0.1)

            logits = FullyConnected(fc2_out, weights, biases, activate=tf.identity, act_name='linear')

        return logits

def GraphModel():
    with tf.Graph().as_default():
        with tf.name_scope('Inputs'):
            images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel],
                                           name='images')
            labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')

        with tf.name_scope('Inference'):
            logits = Inference(images_holder)

        with tf.name_scope('Loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder, logits=logits)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

            total_loss_op = cross_entropy_mean

        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            globel_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

            train_op = optimizer.minimize(total_loss_op, global_step=globel_step)

        with tf.name_scope('Evaluate'):
            top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)

        init_op = tf.global_variables_initializer()

        print('save the graph')
        summary_writer = tf.summary.FileWriter(logdir='logs/alexnet_1')
        summary_writer.add_graph(graph=tf.get_default_graph())
        summary_writer.flush()
        summary_writer.close()

GraphModel()