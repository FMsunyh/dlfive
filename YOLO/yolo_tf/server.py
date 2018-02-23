#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/26/2017 11:51 AM 
# @Author : sunyonghai 
# @File : server.py 
# @Software: BG_AI
# =========================================================
import os
import argparse
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.pascal_voc import pascal_voc

from train import update_config_paths, Solver

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    # parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--data_dir', default="/mnt/hdd/zip_data/syh/data/yolo_tf/data/", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    # if args.data_dir != cfg.DATA_PATH:
    update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"


    yolo = YOLONet()
    pascal = pascal_voc('train')

    solver = Solver(yolo, pascal)

    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    train()