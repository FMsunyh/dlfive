#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/21/2017 2:21 PM 
# @Author : sunyonghai 
# @File : utils.py
# @Software: BG_AI
# =========================================================

import numpy as np
import math
from PIL import Image, ImageDraw

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4

def load_weights(model, weight_path):
    weight_reader = WeightReader(weight_path)
    weight_reader.reset()

    nb_conv = 9
    for i in range(1, nb_conv + 1):
        conv_layer = model.get_layer('conv_' + str(i))
        if len(conv_layer.get_weights()) > 1:
            bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])


class Box:
    def __init__(self, x=0.0, y=0.0, w=0.0, h=0.0, c=0.0, prob=0.0):
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.c = c
        self.prob = prob

    def get_left(self):
        left = self.x - self.w / 2.0
        return max(left, 0)

    def get_top(self):
        top = self.y - self.h / 2.0
        return max(top, 0)

    def get_right(self):
        right = self.x + self.w / 2.0
        return right

    def get_bottom(self):
        bottom = self.y + self.h / 2.0
        return bottom

    def get_rectangle(self):
        return self.get_left(), self.get_top(), self.get_right(), self.get_bottom()

    def get_rectangle_c(self):
        return self.get_left(), self.get_top(), self.get_right(), self.get_bottom(), self.c



def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

def decode_netout(netout, obj_threshold, anchors):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4] = sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + sigmoid(y)) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = Box(x, y, w, h, confidence, classes)

                    boxes.append(box)
    return boxes

def yolo_net_out_to_car_boxes(net_out, threshold=0.2, sqrt=1.8, C=20, B=2, S=7):
    class_num = 6
    boxes = []
    SS = S * S  # number of grid cells
    prob_size = SS * C  # class probabilities
    conf_size = SS * B  # confidences for each grid cell

    probs = net_out[0: prob_size]
    confs = net_out[prob_size: (prob_size + conf_size)]
    cords = net_out[(prob_size + conf_size):]
    probs = probs.reshape([SS, C])
    confs = confs.reshape([SS, B])
    cords = cords.reshape([SS, B, 4])

    for grid in range(SS):
        for b in range(B):
            bx = Box()
            bx.c = confs[grid, b]
            bx.x = (cords[grid, b, 0] + grid % S) / S
            bx.y = (cords[grid, b, 1] + grid // S) / S
            bx.w = cords[grid, b, 2] ** sqrt
            bx.h = cords[grid, b, 3] ** sqrt
            p = probs[grid, :] * bx.c

            if p[class_num] >= threshold:
                bx.prob = p[class_num]
                boxes.append(bx)
    return boxes


def draw_box(bg_image, box=None, outline=None):

    if not isinstance(bg_image, Image.Image):
        return bg_image

    draw = ImageDraw.Draw(bg_image)
    if box is not None:
        # (left, top, width , height)
        # left = box.x - box.w / 2.0
        # top = box.y - box.h / 2.0
        # right = box.x + box.w / 2.0
        # bottom = box.y + box.h / 2.0
        rect = box.get_rectangle()
        draw.rectangle(xy=rect, fill=None, outline=outline)

    return bg_image

def draw_label(bg_image, box=None, label='', fill=None):
    if not isinstance(bg_image, Image.Image):
        return bg_image
    draw = ImageDraw.Draw(bg_image)

    print(repr(label))
    (w, h) = draw.textsize(label.encode())

    left = box.x - box.w / 2.0
    top = box.y - box.h / 2.0 - h
    right = left + w
    bottom = top + h
    draw.rectangle(xy=(max(left, 0), max(top, 0), right, bottom), fill=fill)
    draw.text((max(left, 0), max(top, 0)), label.encode(), fill=(255, 255, 255))

    return bg_image

def box_to_list(boxes):
    if len(boxes) < 0:
        return []

    l_boxes = []
    for box in boxes:

        l_boxes.append([])

def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


if __name__ == '__main__':
    image = Image.open('../test_images/test1.jpg')

    image = draw_box(None, image)
    image.save('../test_images/rect_test1.jpg')
