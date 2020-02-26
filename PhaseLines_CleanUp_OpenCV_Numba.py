# CRT phase lines effect 
# this version 26/02/2020
# dependencies: opencv, numpy, numba (numba not absolutely necessary, used to try to get a bit more performance)
# Copyright 2018-2020 Franz KRUHM
#
# This is a briefly optimized version; I'm considering several performance improvements but this should give you
# the basic idea.
#
# If you use this code in/for your project please let me know/give credit and consider a donation (paypal)
# thanks.
#
# f k r u h m   at  gmail    com

import cv2 
import numpy as np 
from numba import jit 
from multiprocessing import Process, Pool, active_children 

max_threads = 24

@jit(nopython=True, fastmath=True)
def phaseLines(img):
    y = 0
    height, width = img.shape
    phaseStart = 0.25
    density = 0.25
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
            on = 0
            phase += img[y, x]/255*density
            if phase >= 1:
                on = 1
                phase -= 1
                img[y, x] = 255
            else:
                img[y, x] = 0

def phaseCRT(input_image):
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img) # test footage was black on white background so you may not need this depending on your input
    phaseLines(img)
    img_blur = cv2.GaussianBlur(img, (11, 11), 0) 
    img = cv2.addWeighted(img, 1, img_blur, 3, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    return img

# fix filepath
def thread_do(name, frame):
    cv2.imwrite("d:\\ballet\\frm"+str(name).zfill(4)+".png", phaseCRT(frame)) # edit to output to png frames

def wait_threads(max):
    while len(active_children()) >= max:
        time.sleep(0.01)

if __name__ == '__main__':
    invid = cv2.VideoCapture("d:\\ballet_white.mp4") # source video path
    i = 0
    threads = list()
    while (True):
        ret, frame = invid.read()
        if ret == True:
            wait_threads(max_threads)
            threads.append(Process(target=thread_do, args=(i, frame,)))
            threads[-1].start()
            i += 1
        else:
            print("done.")
            break
    invid.release()
    cv2.waitKey(0)
