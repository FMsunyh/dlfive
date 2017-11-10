#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 11/9/2017 4:24 PM
# @Author : sunyonghai
# @File : mnist_visual.ipynb
# @Software: BG_AI
# =========================================================

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)


learning_rate = 0.01
training_epochs = 1000
batch_size = 100
display_step = 4
keep_rate = 0.8

# network parameter
n_input = 28 * 28
num_classes = 10

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, num_classes])


# set model weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return initial


# Convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Pooling
def max_pool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


weights = {
    # 5 x 5 convolution, 1 input image, 32 outputs
    'W_conv1': weight_variable([5, 5, 1, 32]),

    # 5x5 conv, 32 inputs, 64 outputs
    'W_conv2': weight_variable([5, 5, 32, 64]),

    # fully connected, 7*7*64 inputs, 1024 outputs
    'W_fc1': weight_variable([7 * 7 * 64, 1024]),

    # 1024 inputs, 10 outputs (class prediction)
    'out': weight_variable([1024, num_classes])
}

biases = {
    'b_conv1': bias_variable([32]),
    'b_conv2': bias_variable([64]),
    'b_fc1': bias_variable([1024]),
    'out': bias_variable([num_classes]),
}

print('Network ready')


def neural_net(x, weights, biases):
    # reshape x to 4D tensor [-1, width, height, color_channel]
    _x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # First Convolution Layer
    _conv1_2d = conv2d(_x, weights['W_conv1'])
    _conv1_bais = tf.add(_conv1_2d, biases['b_conv1'])
    _conv1_relu = tf.nn.relu(_conv1_bais)
    _pool1 = max_pool2d(_conv1_relu)  # reduce the image to size 28*28 -> 14*14

    # Second Convolution Layer
    _conv2_2d = conv2d(_pool1, weights['W_conv2'])
    _conv2_bais = tf.add(_conv2_2d, biases['b_conv2'])
    _conv2_relu = tf.nn.relu(_conv2_bais)
    _pool2 = max_pool2d(_conv2_relu)  # reduce the image to size 14*14 -> 7*7

    # Densely Connected Layer
    _fc1 = tf.reshape(_pool2, [-1, 7 * 7 * 64])
    _fc1_bais = tf.add(tf.matmul(_fc1, weights['W_fc1']), biases['b_fc1'])

    _fc1_relu = tf.nn.relu(_fc1_bais)

    fc = tf.nn.dropout(_fc1_relu, keep_prob=keep_rate)
    _logit = tf.add(tf.matmul(fc, weights['out']), biases['out'])

    out_layer = {
        'x': _x,
        'conv1_2d': _conv1_2d,
        'conv1_bais': _conv1_bais,
        'conv1_relu': _conv1_relu,
        'pool1': _pool1,

        'conv2_2d': _conv2_2d,
        'conv2_bais': _conv2_bais,
        'conv2_relu': _conv2_relu,
        'pool2': _pool2,

        'fc1': _fc1,
        'fc1_bais': _fc1_bais,
        'fc1_relu': _fc1_relu,
        'logit': _logit
    }

    return out_layer


cnn_out = neural_net(X, weights, biases)

logits = cnn_out['logit']
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

print('function ready')

savedir = 'mnist_visual/'
saver = tf.train.Saver(max_to_keep=3)
save_step = 100
if not os.path.exists(savedir):
    os.makedirs(savedir)
print('save ready')


def train_neural_network(sess):
    sess.run(init)
    avg_loss = 0

    for epoch in range(1, training_epochs + 1):
        total_batchs = int(mnist.train.num_examples / batch_size)

        for i in range(total_batchs):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            feeds = {X: batch_x, Y: batch_y}
            sess.run(train_op, feed_dict=feeds)

            avg_loss += sess.run(loss_op, feed_dict=feeds)
        avg_loss = avg_loss / total_batchs

        # Display logs per epoch step
        if epoch % display_step == 0 or epoch == 1:
            # Calculate batch loss and accuracy，计算每一批数据的误差及准确度
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_loss))
            feeds = {X: batch_x, Y: batch_y}
            train_acc = sess.run(accuracy, feeds)
            print("Train accuracy: %.3f" % train_acc)

            feeds = {X: mnist.test.images, Y: mnist.test.labels}
            test_acc = sess.run(accuracy, feeds)
            print("Test accuracy: %.3f" % test_acc)

        if epoch % save_step == 0:
            savename = savedir + 'net-' + str(epoch) + '.ckpt'
            saver.save(sess, savename)
            print('[%s] saved.' % savename)

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images，计算测试数据上的准确度
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

# sess = get_session()
# train_neural_network(sess)
# sess.close()

do_restore = 0
if do_restore == 1:
    sess = get_session()
    epoch = 8
    savename = savedir+"net-"+str(epoch)+".ckpt"
    saver.restore(sess, savename)
    print ("NETWORK RESTORED")
    sess.close()
else:
    print ("DO NOTHING")

def predict(image):
    sess = get_session()
    epoch = 200
    savename = savedir + "net-" + str(epoch) + ".ckpt"
    saver.restore(sess, savename)
    print("NETWORK RESTORED")
    prediction = tf.argmax(logits, 1)
    out = sess.run(prediction, feed_dict={X:image})
    print(out)
    sess.close()

if __name__ == '__main__':
    trainimg = mnist.train.images
    x = trainimg[0:1, :]
    predict(x)