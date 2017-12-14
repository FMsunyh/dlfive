#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/12/2017 2:58 PM 
# @Author : sunyonghai 
# @File : server.py 
# @Software: BG_AI
# =========================================================

import tensorflow as tf

from vggnet16 import train

def main(argv=None):
    # maybe_download_and_extract(data_dir=datast_dir)

    train_dir = 'logs/'
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)


    train()


if __name__ == '__main__':
    tf.app.run(main=main, argv=None)