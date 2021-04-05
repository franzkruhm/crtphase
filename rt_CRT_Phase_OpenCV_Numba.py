# CRT phase lines effect 
# this version 7/3/2021
# dependencies: opencv, numpy, numba (numba not absolutely necessary, used to get a bit more performance)
# Copyright 2018-2021 Franz KRUHM
#
# License: GPL 3
# This is inspired heavily by Polygon1993 iconic CRT effect
#
# This is a webcam version; I'm considering several performance improvements but this should give you
# the basic idea.
#
# If you use this code in/for your project please let me know/give credit and consider a donation at
# paypal.me/fkruhm or via cryptocurrencies like bitcoin (see franzkruhm.com for addresses)
# thanks.
#
# f k r u h m   at  gmail    com


import cv2
import time
import sys
import os
import numpy as np
from numba import jit

@jit(nopython=True,fastmath=True)
def phaser(img, height, width, density, phaseStart):
    y = 0
    phase = 0
    xStart = width - 1
    xEnd = -1
    xDir = -1
    while y < height - 1:
        y += 1
        phase = 1 - phaseStart
        x = xStart
        while x != xEnd:
            x += xDir
            phase += img[y, x]/255*density
            if phase >= 1:
                phase -= 1
                img[y, x] = 255
            else:
                img[y, x] = 0

@jit(forceobj=True, fastmath=True, parallel=True)
def phaseCRT(input_image, phaseStart, density, var_ghost):
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    height = img.shape[0]
    width = img.shape[1]
    phaser(img, height, width, density, phaseStart)
    img_blur = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.addWeighted(img, 1, img_blur, 3, 0)
    return cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)

if __name__ == '__main__':
    video_cap = cv2.VideoCapture(0)
    while True:
        ret, frame = video_cap.read()
        cv2.imshow('Video', phaseCRT(frame, 0.25, 0.25, 255))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_cap.release()
    cv2.destroyAllWindows()
