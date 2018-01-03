#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 12/28/2017 5:17 PM 
# @Author : sunyonghai 
# @File : scratch.py 
# @Software: BG_AI
# =========================================================
import cv2

img = cv2.imread('data/test/2012_003869.jpg', cv2.C)
print(img.shape)
# cv2.imshow('img', img)
# cv2.waitKey(0)