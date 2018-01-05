#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 1/3/2018 4:48 PM 
# @Author : sunyonghai 
# @File : train_rpn.py
# @Software: BG_AI
# =========================================================
from GPU_setting import set_gpu

set_gpu()

import random
import pprint
import sys
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
from keras_frcnn import resnet as nn
from keras_frcnn.pascal_voc_parser import get_data
from keras.utils import generic_utils
import os
from config_rpn import FLAGS

sys.setrecursionlimit(40000)

# training parameters from command line
# parser = OptionParser()
# parser.add_option("-p", "--path", dest="train_path", help="Path to training data.", default="data/")
# parser.add_option("-n", "--num_rois", dest="num_rois",
#                   help="Number of ROIs per iteration. Higher means more memory use.", default=32)
# parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=true).",
#                   action="store_true", default=False)
# parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).",
#                   action="store_true", default=False)
# parser.add_option("--rot", "--rot_90", dest="rot_90",
#                   help="Augment with 90 degree rotations in training. (Default=false).", action="store_true",
#                   default=False)
# parser.add_option("--num_epochs", dest="num_epochs", help="Number of epochs.", default=2000)
# parser.add_option("--config_filename", dest="config_filename",
#                   help="Location to store all the metadata related to the training (to be used when testing).",
#                   default="config/config.pickle")
# parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
#                   default='model/model_frcnn.hdf5')
# parser.add_option("--input_weight_path", dest="input_weight_path",
#                   help="Input path for weights. If not specified, will try to load default weights provided by keras.")
# (options, args) = parser.parse_args()




# read training data
all_imgs, classes_count, class_mapping = get_data(FLAGS.train_path)

C = config.Config()
def save_config():
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    # get settings from command line, and save them into Config
    C.num_rois = int(FLAGS.num_rois)
    C.use_horizontal_flips = bool(FLAGS.horizontal_flips)
    C.use_vertical_flips = bool(FLAGS.vertical_flips)
    C.rot_90 = bool(FLAGS.rot_90)
    C.model_path = FLAGS.output_weight_path
    if FLAGS.input_weight_path:
        C.base_net_weights = FLAGS.input_weight_path
    C.class_mapping = class_mapping

    # print info for training data
    print('Training images per class:')
    pprint.pprint(classes_count)
    print('Num classes (including bg) = {}'.format(len(classes_count)))

    # generate Config file for test phase
    config_output_filename = FLAGS.config_filename
    with open(config_output_filename, 'wb') as config_f:
        pickle.dump(C, config_f)
        print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
            config_output_filename))

save_config()

# shuffle data randomlyl, and split them into two parts: train, val
random.shuffle(all_imgs)
train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']
print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

# generate positive and negative rpn anchors for groungtruth bbox
# K.image_dim_ordering() is to get the backend name string ('tf' for tensorflow)
data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
# data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

# set input format according to the backend
input_shape_img = (None, None, 3)
# input_shape_img = (256, 256, 3)

# set up two inputs
img_input = Input(shape=input_shape_img)

# define the base network - resnet
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)


# create rpn and fast R-CNN models
model_rpn = Model(img_input, rpn[:2])

model_rpn.summary()

# load basenet (shared convolutions) weights from pre-trained model
try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    ))

# build optimizer for both models
optimizer = Adam(lr=1e-4)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

# some settings for training
epoch_length = 1000
num_epochs = int(FLAGS.num_epochs)
iter_num = 0
losses = np.zeros((epoch_length, 2))
# rpn_accuracy_rpn_monitor = []
# rpn_accuracy_for_epoch = []
start_time = time.time()
best_loss = np.Inf

# start training epoch by epoch
print('Starting training')
for epoch_num in range(num_epochs):
    # progress bar
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
    # start
    while True:
        try:
            # get batch data
            X, Y, img_data = next(data_gen_train)

            """rpn training"""
            # train rpn model on batch data
            loss_rpn = model_rpn.train_on_batch(X, Y)
            # predict rpn model on batch data
            # P_rpn = model_rpn.predict_on_batch(X)
            # organise all losses
            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            iter_num += 1

            # update progress bar
            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1]))])

            # operations if one epoch is done
            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])

                if C.verbose:
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))

                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_rpn.save_weights(C.model_path)
                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')
