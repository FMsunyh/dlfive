#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/19/2017 10:43 AM 
# @Author : sunyonghai 
# @File : server.py 
# @Software: BG_AI
# =========================================================
import resnet_model
from resnet_main import FLAGS, train, evaluate
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(_):
    # 设备选择
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    # 执行模式
    if FLAGS.mode == 'train':
        batch_size = 128
    elif FLAGS.mode == 'eval':
        batch_size = 100

    # 数据集类别数量
    if FLAGS.dataset == 'cifar10':
        num_classes = 10
    elif FLAGS.dataset == 'cifar100':
        num_classes = 100

    # 残差网络模型参数
    hps = resnet_model.HParams(batch_size=batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom')
    # 执行训练或测试
    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(hps)
        elif FLAGS.mode == 'eval':
            evaluate(hps)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
