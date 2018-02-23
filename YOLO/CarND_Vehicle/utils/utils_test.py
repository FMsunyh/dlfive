#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 12/22/2017 10:15 AM
# @Author : sunyonghai
# @File : util_test.py.py
# @Software: BG_AI
# =========================================================
from PIL import Image
import utils
import numpy as np
import unittest

class MyTestCase(unittest.TestCase):
    def setUp(self):
        # Do something to initiate the test environment here.
        pass

    def tearDown(self):
        # Do something to clear the test environment here.
        pass

    def test_draw_box(self):
        image = Image.open('../test_images/test1.jpg')
        box = utils.Box()
        box.x = 200
        box.y = 200
        box.w = 200
        box.h = 200
        image = utils.draw_box(image, box, outline=(255, 0, 0))
        image.save('../test_images/rect_test1.jpg')
        # self.assertEqual(True, False)

    def test_draw_label(self):
        image = Image.open('../test_images/test1.jpg')
        box = utils.Box()
        box.x = 200
        box.y = 200
        box.w = 200
        box.h = 200
        image = utils.draw_box(image, box, outline=(255, 0, 0))
        image = utils.draw_label(image, box, label='car', fill=(255, 0, 0))
        image.save('../test_images/rect_label_test1.jpg')
        # self.assertEqual(True, False)

    def test_non_max_suppression(self):
        dets = np.array([
            [204, 102, 358, 250, 0.5],
            [257, 118, 380, 250, 0.7],
            [280, 135, 400, 250, 0.6],
            [255, 118, 360, 235, 0.7]])

        thresh = 0.3

        boxes = utils.non_max_suppression(dets, thresh)

        print("array test: ", boxes)

    def test_nms(self):
        dets = np.array([
            [204, 102, 358, 250, 0.5],
            [257, 118, 380, 250, 0.7],
            [280, 135, 400, 250, 0.6],
            [255, 118, 360, 235, 0.7]])

        thresh = 0.5

        boxes = utils.nms(dets, thresh)

        print("nms test: ", boxes)

        self.assertEqual(3, boxes[0])

    def test_non_max_suppression_box(self):

        boxes = list()
        boxes.append(utils.Box(204 + (358-204) / 2.0, 102 + (250-102) / 2.0, 358-204, 250-102, 0.5))
        boxes.append(utils.Box(257 + (380-257) / 2.0, 118 + (250-118) / 2.0, 380-257, 250-118, 0.7))
        boxes.append(utils.Box(280 + (400-280) / 2.0, 135 + (250-135) / 2.0, 400-280, 250-135, 0.6))
        boxes.append(utils.Box(255 + (360-255) / 2.0, 118 + (235-118) / 2.0, 360-255, 235-118, 0.7))

        l_boxes = [box.get_rectangle_c() for box in boxes]

        dets = np.array(l_boxes)
        thresh = 0.3
        boxes = utils.non_max_suppression(dets, thresh)

        print("box object test: ", boxes)


if __name__ == '__main__':
    unittest.main()
