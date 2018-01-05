#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 1/3/2018 6:33 PM 
# @Author : sunyonghai 
# @File : config_rpn.py
# @Software: BG_AI
# =========================================================
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_path', 'data/', 'Path to training data.')

tf.app.flags.DEFINE_integer('num_rois', 32, 'Number of ROIs per iteration. Higher means more memory use.')


tf.app.flags.DEFINE_bool('horizontal_flips', False, 'Augment with horizontal flips in training. (Default=true).')
tf.app.flags.DEFINE_bool('vertical_flips', False, 'Augment with vertical flips in training. (Default=true).')
tf.app.flags.DEFINE_bool('rot_90', False, 'Augment with 90 degree rotations in training. (Default=false).')

tf.app.flags.DEFINE_integer('num_epochs', 5000, 'Number of epochs.')

tf.app.flags.DEFINE_string('config_filename', 'config/config.pickle',
                           'Location to store all the metadata related to the training (to be used when testing).')

tf.app.flags.DEFINE_string('output_weight_path', 'model/model_rpn.hdf5', 'Output path for weights.')

tf.app.flags.DEFINE_string('input_weight_path', 'model/model_rpn.hdf5',
                           'Input path for weights. If not specified, will try to load default weights provided by keras.')

