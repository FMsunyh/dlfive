#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 3/12/2018 9:29 AM 
# @Author : sunyonghai 
# @File : TFQueue.py 
# @Software: ZJ_AI
# =========================================================


import tensorflow as tf
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

q = tf.FIFOQueue(3, 'int32')
init = q.enqueue_many(([0,10,20], ))
x = q.dequeue()
y = x+1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ =sess.run([x, q_inc])
        print(v)

    for i in range(0, 3):
        print(sess.run(q.dequeue()))