#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 11/30/2017 10:38 AM 
# @Author : sunyonghai 
# @File : GPU_setting.py 
# @Software: BG_AI
# =========================================================

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_session():
    cfg = tf.ConfigProto()
    cfg.gpu_options.allocator_type = 'BFC'
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.90
    cfg.gpu_options.allow_growth = True
    return tf.Session(config=cfg)