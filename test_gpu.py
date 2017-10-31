import tensorflow as tf
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
for i in range(10000):
    print(sess.run(c))


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
