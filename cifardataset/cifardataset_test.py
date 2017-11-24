#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 11/15/2017 5:57 PM 
# @Author : sunyonghai 
# @File : test.py 
# @Software: BG_AI
# =========================================================

import tensorflow as tf
import os
import numpy as np
import csv
import cifar_input
import tarfile
import urllib
import sys

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

learning_rate_value = 0.001
training_epochs = 1
batch_size = 100
display_step = 10
keep_rate = 0.8
conv1_kernel_num = 64
conv2_kernel_num = 64
fc1_units_num = 1024
fc2_units_num = 512

# network parameter
dataset_dir_cifar10 = '../data/cifar/cifar-10-batches-bin/'
dataset_dir_cifar100 = '../data/cifar/cifar-100-binary/'
num_examples_per_epoch_for_train = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_examples_per_epoch_for_eval = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
image_size = cifar_input.IMAGE_SIZE
image_channel = cifar_input.IMAGE_DEPTH

cifar10or20or100 = 20
if cifar10or20or100 == 10:
    n_classes = cifar_input.NUM_CLASSES_CIFAR10
    dataset_dir = dataset_dir_cifar10
elif cifar10or20or100 == 20:
    n_classes = cifar_input.NUM_CLASSES_CIFAR20
    dataset_dir = dataset_dir_cifar100
elif cifar10or20or100 == 100:
    n_classes = cifar_input.NUM_CLASSES_CIFAR100
    dataset_dir = dataset_dir_cifar100

# endregion

# region Get the data
def get_distorted_train_batch(data_dir, batch_size):
    # data_dir='data/cifar/'

    if not data_dir:
        raise ValueError('Please supply a data_dir')

    images, labels = cifar_input.distorted_inputs(data_dir=data_dir, cifar10or20or100=n_classes, batch_size=batch_size)
    return images, labels


def get_undistorted_eval_batch(eval_data, data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')

    images, labels = cifar_input.inputs(eval_data=eval_data, data_dir=data_dir, cifar10or20or100=n_classes, batch_size=batch_size)
    return images, labels

# endregion

def maybe_download_and_extract(data_dir):
    """download from Alex's website."""
    dest_directory = data_dir
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>>Downloading %s %.1f%%' % (filename,
                                                            float(count * batch_size) / float(total_size) * 100.0))
            tf.sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    tarfile.open(filepath, 'r:gz').extractall(dest_directory)

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
    return pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')


def FullyConnected(x, W, b, activate=tf.nn.relu, act_name='relu'):
    with tf.name_scope('Wx_b'):
        y = tf.matmul(x, W)
        y = tf.add(y, b)

    with tf.name_scope(act_name):
        y = activate(y)

    return y


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


def Inference(images_holder):
    with tf.name_scope('Conv2d_1'):
        weights = WeightsVariable(shape=[5, 5, image_channel, conv1_kernel_num], name_str='weights', stddev=5e-2)
        biases = BiasesVariable(shape=[conv1_kernel_num], name_str='baises', init_value=0.0)
        conv1_out = Conv2(images_holder, weights, biases, stride=1, padding='SAME')

        AddActivationSummary(conv1_out)

    with tf.name_scope('Pool2d_1'):
        pool1_out = Pool2d(conv1_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME')

    with tf.name_scope('Conv2d_2'):
        weights = WeightsVariable(shape=[3, 3, conv1_kernel_num, conv2_kernel_num], name_str='weights', stddev=5e-2)
        biases = BiasesVariable(shape=[conv2_kernel_num], name_str='baises', init_value=0.0)
        conv2_out = Conv2(pool1_out, weights, biases, stride=1, padding='SAME')

        AddActivationSummary(conv2_out)

    with tf.name_scope('Pool2d_2'):
        pool2_out = Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2, padding='SAME')

    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool2_out, [batch_size, -1])
        feats_dim = features.get_shape()[1].value

    with tf.name_scope('FC1_nolinear'):
        weights = WeightsVariable(shape=[feats_dim, fc1_units_num], name_str='weights', stddev=4e-2)
        biases = BiasesVariable(shape=[fc1_units_num], name_str='biases', init_value=0.1)

        fc1_out = FullyConnected(features, weights, biases, activate=tf.nn.relu, act_name='relu')

        AddActivationSummary(fc1_out)

    with tf.name_scope('FC2_linear'):
        weights = WeightsVariable(shape=[fc1_units_num, fc2_units_num], name_str='weights', stddev=4e-2)
        biases = BiasesVariable(shape=[fc2_units_num], name_str='biases', init_value=0.1)

        logits = FullyConnected(fc1_out, weights, biases, activate=tf.identity, act_name='linear')

        AddActivationSummary(logits)

    # with tf.name_scope('FC3_linear'):
    #     fc3_units_num = n_classes
    #     weights = WeightsVariable(shape=[fc2_units_num, fc3_units_num], name_str='weights', stddev=1.0 / fc3_units_num)
    #     biases = BiasesVariable(shape=[fc3_units_num], name_str='biases', init_value=0.0)
    #
    #     logits = FullyConnected(fc2_out, weights, biases, activate=tf.identity, act_name='linear')
    #
    #     AddActivationSummary(logits)

    return logits


def TrainModel():
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

            total_loss = cross_entropy_mean

            average_losses = AddLossesSummary([total_loss])

        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            globel_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)

            train_op = optimizer.minimize(total_loss, global_step=globel_step)

        with tf.name_scope('Evaluate'):
            top_K_op = tf.nn.in_top_k(predictions=logits, targets=labels_holder, k=1)

        with tf.name_scope('GetTrainBatch'):
            images_train, labels_train = get_distorted_train_batch(data_dir=dataset_dir, batch_size=batch_size)

        with tf.name_scope('GetTestBatch'):
            images_test, labels_test = get_undistorted_eval_batch(eval_data=True, data_dir=dataset_dir,
                                                                  batch_size=batch_size)

        merged_summaries = tf.summary.merge_all()

        init_op = tf.global_variables_initializer()

        print('save the graph')
        summary_writer = tf.summary.FileWriter(logdir='logs/cifardataset')
        summary_writer.add_graph(graph=tf.get_default_graph())
        summary_writer.flush()

        results_list = list()
        results_list.append(['learning_rate', learning_rate,
                             'training_epochs', training_epochs,
                             'batch_size', batch_size,
                             'display_step', display_step,
                             'conv1_kernel_num', conv1_kernel_num,
                             'conv2_kernel_num', conv2_kernel_num,
                             'fc1_units_num', fc1_units_num,
                             'fc2_units_num', fc2_units_num])

        results_list.append(['train_step', 'train_loss', 'train_step', 'train_accuracy'])

        sess = get_session()
        sess.run(init_op)

        print('==>>>>>==start training==<<<<<==')
        total_batches = int(num_examples_per_epoch_for_train / batch_size)

        print('Per batch Size:', batch_size)
        print('Train sample Count:', num_examples_per_epoch_for_train)
        print('Total batch Count:', total_batches)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # tf.train.start_queue_runners(sess=sess)
        training_step = 0

        for epoch in range(training_epochs):
            for batch_idx in range(total_batches):
                images_batch, labels_batch = sess.run([images_train, labels_train])

                _, loss_value = sess.run([train_op, total_loss], feed_dict={images_holder: images_batch,
                                                                            labels_holder: labels_batch,
                                                                            learning_rate: learning_rate_value})

                training_step = sess.run(globel_step)

                if training_step % display_step == 0:
                    predictions = sess.run([top_K_op],
                                           feed_dict={images_holder: images_batch,
                                                      labels_holder: labels_batch}
                                           )

                    batch_acc = np.sum(predictions) * 1.0 / batch_size
                    print('training epoch:' + str(epoch)
                          + ',Training step:' + str(training_step)
                          + ', Training Loss=' + '{:.6f}'.format(loss_value)
                          + ', Training Accuracy=' + '{:.5f}'.format(batch_acc)
                          )

                    results_list.append([training_step, loss_value, training_step, batch_acc])
                    summary_str = sess.run(merged_summaries,
                                           feed_dict={images_holder: images_batch, labels_holder: labels_batch})
                    summary_writer.add_summary(summary=summary_str, global_step=training_step)
                    summary_writer.flush()

            summary_writer.close()
        print('finish training......')

        print('==>>>>>==start testing==<<<<<==')
        total_batches = int(num_examples_per_epoch_for_eval / batch_size)
        total_exmples = total_batches * batch_size

        print('Per batch Size:', batch_size)
        print('Test sample Count:', total_exmples)
        print('Total batch Count:', total_batches)

        correct_predicted = 0
        for test_step in range(total_batches):
            images_batch, labels_batch = sess.run([images_test, labels_test])
            predictions = sess.run([top_K_op],
                                   feed_dict={images_holder: images_batch,
                                              labels_holder: labels_batch}
                                   )
            correct_predicted += np.sum(predictions)
        accuracy_score = correct_predicted * 1.0 / total_exmples
        print("Accuracy on Test Exmples:", accuracy_score)
        results_list.append(['Accuracy on Test Examples: ', accuracy_score])

        print('finish testing......')

        results_file = open('evaluate_results.csv', 'w')
        csv_writer = csv.writer(results_file, dialect='excel')
        for row in results_list:
            csv_writer.writerow(row)

        coord.request_stop()
        coord.join(threads)
        sess.close()


def main(argv=None):
    # maybe_download_and_extract(data_dir=datast_dir)

    train_dir = 'logs/'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    TrainModel()


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)