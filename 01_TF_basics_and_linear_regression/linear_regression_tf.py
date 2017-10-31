# # coding: utf-8
#
# # region tarin step
#
# # 1  Environment, 环境
# # 2  Hyper Parameters, 超参数
# # 3  Training Data, 训练数据
# # 4  Prepare for Training, 训练准备
# #   4.1  mx Graph Input, mxnet图输入
# #   4.2  Construct a linear model, 构造线性模型
# #   4.3  Mean squared error, 损失函数：均方差
# # 5  Start training, 开始训练
# # 6  Regression result, 回归结果
#
# # endregion
#
#
# # region import package
#
# import tensorflow as tf
# import numpy
# import os
# rng = numpy.random
#
# # endregion
# # Creates a graph.
# with tf.device('/cpu:0'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# # Creates a session with log_device_placement set to True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # Runs the op.
# print(sess.run(c))

#
# import sys
# import numpy as np
# import tensorflow as tf
# from datetime import datetime
#
# device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
# shape = (int(sys.argv[2]), int(sys.argv[2]))
# if device_name == "gpu":
#     device_name = "/gpu:0"
# else:
#     device_name = "/cpu:0"
#
# with tf.device(device_name):
#     random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
#     dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
#     sum_operation = tf.reduce_sum(dot_operation)
#
#
# startTime = datetime.now()
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
#         result = session.run(sum_operation)
#         print(result)
#
# # It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
# print("\n" * 5)
# print("Shape:", shape, "Device:", device_name)
# print("Time taken:", datetime.now() - startTime)
#
# print("\n" * 5)


# import tensorflow as tf
#
# # 新建一个 graph.
# with tf.device('/gpu:1'):
#   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#   c = tf.matmul(a, b)
# # 新建 session with log_device_placement 并设置为 True.
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # 运行这个 op.
# print( sess.run(c))


# coding: utf-8

# region tarin step

# 1  Environment, 环境
# 2  Hyper Parameters, 超参数
# 3  Training Data, 训练数据
# 4  Prepare for Training, 训练准备
#   4.1  mx Graph Input, mxnet图输入
#   4.2  Construct a linear model, 构造线性模型
#   4.3  Mean squared error, 损失函数：均方差
# 5  Start training, 开始训练
# 6  Regression result, 回归结果

# endregion


# region import package

import tensorflow as tf
import numpy
rng = numpy.random

# endregion

# In[1]:
# region Environment GPU setting

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)

# endregion


# In[2]:
# region Hyper Parameters

learning_rate = 0.01
training_epochs = 2000
display_step = 50

# endregion

# In[3]:
# region Training Data

train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167, 7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221, 2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

# endregion

# In[4]:
# region Prepare for Training

X = tf.placeholder("float")
Y = tf.placeholder("float")

# set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# endregion

# In[5]:
# region Construct a linear model.

pred = tf.add( tf.multiply(X, W), b)

# endregion

# In[6]:
# region loss function, mean squared error

cost = tf.reduce_sum( tf.pow(pred-Y,2))/(2*n_samples )

# endregion

# In[7]:
# region Optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# endregion

# In[8]:
# region Initialize the variables

init = tf.global_variables_initializer()

# endregion

# In[9]:

# region Starting training

sess = get_session()

sess.run(init)

for epoch in range(training_epochs):
    for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X:x, Y:y})

    # Display logs per epoch step
    if (epoch + 1) % display_step == 0:
        c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

sess.close()

# endregion

# In[10]:
# region Training result

# endregion