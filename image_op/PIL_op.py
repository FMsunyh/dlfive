#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/21/2017 7:54 PM 
# @Author : sunyonghai 
# @File : PIL_op.py
# @Software: BG_AI
# =========================================================

from PIL import Image


def PIL_op(im_path):
    im = Image.open(im_path)
    w, h = im.size
    print(w, h)
    im.thumbnail((w//2, h//2))
    im.save('image/PIL_test.jpg', 'jpeg')



if __name__ == '__main__':
    im_path = 'image/test.jpg'
    PIL_op(im_path)
