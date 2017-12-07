#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/1/2017 5:34 PM 
# @Author : sunyonghai 
# @File : vggnet11_session.py 
# @Software: BG_AI
# =========================================================
import csv

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

learning_rate_init = 0.001
training_epochs = 5
batch_size = 32
display_step = 2
keep_prob = 0.8
# conv1_kernel_num = 96
# conv2_kernel_num = 256
# conv3_kernel_num = 384
# conv4_kernel_num = 384
# conv5_kernel_num = 256
# fc1_units_num = 4096
# fc2_units_num = 4096

image_size = 224
image_channel = 3
n_classes = 1000

num_examples_per_epoch_for_train = 10000
num_examples_per_epoch_for_eval = 1000

def get_faked_train_batch(batch_size):
    images = tf.Variable(tf.random_normal(shape=[batch_size, image_size, image_size, image_channel], mean=0.0,stddev=1.0, dtype=tf.float32))
    labels = tf.Variable(tf.random_uniform(shape=[batch_size], minval=0, maxval=n_classes, dtype=tf.int32))

    return images, labels

def get_faked_test_batch(batch_size):
    images = tf.Variable(tf.random_normal(shape=[batch_size, image_size, image_size, image_channel], mean=0.0,stddev=1.0, dtype=tf.float32))
    labels = tf.Variable(tf.random_uniform(shape=[batch_size], minval=0, maxval=n_classes, dtype=tf.int32))

    return images, labels

def Conv2d_Op(input_op, name, kh, kw, n_out, dh, dw, activation_func=tf.nn.relu, activation_name='relu'):
    with tf.name_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        kernels = tf.get_variable(scope+'weight', shape=[kh, kw, n_in, n_out], dtype=tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())

        conv = tf.nn.conv2d(input_op, kernels, strides=(1,dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='bias')
        z = tf.nn.bias_add(conv, biases)
        activation = activation_func(z, name=activation_name)

        return activation

def Pool2d_Op(input_op, name, kh=2, kw=2, dh=2, dw=2, padding='SMAE', pool_func=tf.nn.max_pool):
    with tf.name_scope(name) as  scope:
        return pool_func(input_op, ksize=[1, kh,kw,1], strides=[1, dw,dh,1], padding=padding, name=name)

def FullyConnected_Op(input_op, name, n_out, activation_func=tf.nn.relu, activation_name='relu'):
    with tf.name_scope(name) as scope:
        n_in = input_op.get_shape()[-1].value
        kernels = tf.get_variable(scope+'weight', shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        bias_init_val = tf.constant(0.1, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='bias')
        z = tf.add(tf.matmul(input_op, kernels), biases)

        activation = activation_func(z, name=activation_name)

        return  activation

def print_activation(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def AddActivationSummary(x):
    tf.summary.histogram('/activations', x)
    tf.summary.scalar('/sparsity', tf.nn.zero_fraction(x))

def AddLossesSummary(losses):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses)

    for loss in losses:
        tf.summary.scalar(loss.op.name + '(raw)', loss)
        tf.summary.scalar(loss.op.name + 'avg', loss_averages.average(loss))

    return loss_averages_op

def Inference(images_hodler, keep_prob=keep_prob):
    conv1_1 = Conv2d_Op(images_hodler, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1)
    pool1   = Pool2d_Op(conv1_1, name='pool1', kh=2, kw=2, dh=2, dw=2, padding='SAME')

    AddActivationSummary(conv1_1)
    print_activation(pool1)


    conv2_1 = Conv2d_Op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1)
    pool2   = Pool2d_Op(conv2_1, name='pool2', kh=2, kw=2, dh=2, dw=2, padding='SAME')

    AddActivationSummary(conv2_1)
    print_activation(pool2)


    conv3_1 = Conv2d_Op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_2 = Conv2d_Op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1)
    pool3 = Pool2d_Op(conv3_2, name='pool3', kh=2, kw=2, dh=2, dw=2, padding='SAME')

    AddActivationSummary(conv3_2)
    print_activation(pool3)

    conv4_1 = Conv2d_Op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_2 = Conv2d_Op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool4 = Pool2d_Op(conv4_2, name='pool4', kh=2, kw=2, dh=2, dw=2, padding='SAME')

    AddActivationSummary(conv4_2)
    print_activation(pool4)

    conv5_1 = Conv2d_Op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_2 = Conv2d_Op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool5 = Pool2d_Op(conv5_2, name='pool5', kh=2, kw=2, dh=2, dw=2, padding='SAME')

    AddActivationSummary(conv5_2)
    print_activation(pool5)

    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool5, [batch_size, -1])
        # feats_dim = features.get_shape()[1].value

    fc1_out = FullyConnected_Op(features, name='fc1', n_out=4096)
    print_activation(fc1_out)


    with tf.name_scope('dropout_1'):
        fc1_dropout = tf.nn.dropout(fc1_out, keep_prob=keep_prob)

    fc2_out = FullyConnected_Op(fc1_dropout, name='fc2', n_out=4096)
    print_activation(fc2_out)

    with tf.name_scope('dropout_2'):
        fc2_dropout = tf.nn.dropout(fc2_out, keep_prob=keep_prob)

    logits = FullyConnected_Op(fc2_dropout, name='fc3', n_out=n_classes, activation_func=tf.identity, activation_name='identify')

    print_activation(logits)

    return logits


def train():
    with tf.Graph().as_default():
        with tf.name_scope('Inputs'):
            images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel],
                                           name='images')
            labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')

        with tf.name_scope('Inference'):
            keep_prob_holder = tf.placeholder(tf.float32, name='KeepProb')
            logits = Inference(images_holder, keep_prob_holder)

        with tf.name_scope('Loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder, logits=logits)

            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            total_loss_op = cross_entropy_mean

            average_losses_op = AddLossesSummary([total_loss_op])

            with tf.name_scope('Train'):
                learning_rate = tf.placeholder(tf.float32, name='LearningRate')
                global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
                optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(total_loss_op, global_step=global_step)

            with tf.name_scope('Evaluate'):
                top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)

            with tf.name_scope('GetTrainBatch'):
                images_train, labels_train = get_faked_train_batch(batch_size=batch_size)

            with tf.name_scope('GetTestBatch'):
                images_test, labels_test = get_faked_test_batch(batch_size=batch_size)

            init_op = tf.global_variables_initializer()

            print('save the graph')

            summary_writer = tf.summary.FileWriter(logdir='logs/vggnet11')
            summary_writer.add_graph(graph=tf.get_default_graph())
            summary_writer.flush()
            summary_writer.close()

            results_list = list()
            results_list.append(['learning_rate', learning_rate_init,
                                 'training_epochs', training_epochs,
                                 'batch_size', batch_size,
                                 'display_step', display_step])

            results_list.append(['train_step', 'train_loss', 'train_step', 'train_accuracy'])



            results_file = open('evaluate_results.csv', 'w')
            csv_writer = csv.writer(results_file, dialect='excel')
            for row in results_list:
                csv_writer.writerow(row)


def main(argv=None):
    # maybe_download_and_extract(data_dir=datast_dir)

    train_dir = 'logs/'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    train()


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)
