#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/27/2017 10:53 AM 
# @Author : sunyonghai 
# @File : read_image.py 
# @Software: BG_AI
# =========================================================
import cv2
import numpy as np

def image_read(imname, flipped=False):
    image = cv2.imread(imname)
    # image = cv2.resize(image, (self.image_size, self.image_size))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # image = (image / 255.0) * 2.0 - 1.0
    if flipped:
        image = image[:, ::-1, :]
    return image

def main():
    flip_im = image_read('cat.jpg', True)
    cv2.imwrite('flip_cat.jpg', flip_im)


if __name__ == '__main__':
    main()