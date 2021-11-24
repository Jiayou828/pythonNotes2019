# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import cv2 as cv
import matplotlib.pyplot as mp
original = cv.imread('../../data/table.jpg')
cv.imshow('Original', original)
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
star = cv.xfeatures2d.StarDetector_create()
keypoints = star.detect(gray)
sift = cv.xfeatures2d.SIFT_create()
_, desc = sift.compute(gray, keypoints)
print(desc.shape)
mp.matshow(desc, cmap='jet', fignum='Description')
mp.title('Description', fontsize=20)
mp.xlabel('Feature', fontsize=14)
mp.ylabel('Sample', fontsize=14)
mp.tick_params(which='both', top=False,
               labeltop=False, labelbottom=True,
               labelsize=10)
mp.show()
