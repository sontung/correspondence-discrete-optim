import cv2
import numpy as np
import sys
from utils import read_correspondence, read_correspondence_from_dump
from math_utils import compute_epip_line
import tkinter
import os
from PIL import Image, ImageTk
from sys import argv
import matplotlib.pyplot as plt
from pathlib import Path

img1 = cv2.imread("data/im1.png")
img2 = cv2.imread("data/im2.png")

im_mask1 = cv2.imread("data/im1masked.png")
im_mask2 = cv2.imread("data/im2masked.png")

print("done reading")


img3_ori = np.hstack([img1, img2])
x_sel, y_sel = (895, 320)


def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, mouseX2, mouseY2
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img3, (x, y), 5, (255, 0, 0), 5)
        if mouseX > 0:
            mouseX2, mouseY2 = x*2-img3.shape[1], y*2
        else:
            mouseX, mouseY = x*2, y*2


CORRESPONDENCES_PRECOMPUTED = read_correspondence_from_dump("data/corr-exact.txt")
RES_DICT = {}
for pair in CORRESPONDENCES_PRECOMPUTED:
    x, y, x2, y2 = map(int, pair[:4])
    RES_DICT[(x, y)] = (x2, y2)

p1, p2 = read_correspondence()
F_MAT, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)
mouseX = -1
mouseY = -1
mouseX2 = -1
mouseY2 = -1
cv2.namedWindow('image')
down_scale = 2
cv2.setMouseCallback('image', draw_circle)
img3_ori = cv2.resize(img3_ori, (img3_ori.shape[1] // down_scale, img3_ori.shape[0] // down_scale))
im_mask12 = cv2.resize(im_mask1, (im_mask1.shape[1] // down_scale, im_mask1.shape[0] // down_scale))
RESULT = None
GLOBAL_RESULT = None
COEFF_SIFT = None
MODE = None
FLOW_FIELD = None
img3 = img3_ori.copy()

while True:

    if MODE == '1':
        if RESULT is not None:
            img3 = img3_ori.copy()
            x, y, x_corr, y_corr, color, coeff = RESULT
            cv2.circle(img3, (x, y), 5, (255, 0, 0), 5)
            cv2.circle(img3, tuple([y_corr + int(img3.shape[1] / 2), x_corr]), 5, color, 2)
            cv2.circle(img3, tuple([x, y]), 5, color, 2)
            cv2.line(img3, (x, y), (y_corr + int(img3.shape[1] / 2), x_corr), color, 2)

            # epip line
            cv2.line(img3,
                     (int((coeff[1] * 1 + coeff[2]/2) / -coeff[0]) + img3.shape[1] // down_scale, 1),
                     (int((coeff[1] * 1000 + coeff[2]/2) / -coeff[0]) + img3.shape[1] // down_scale, 1000),
                     color, 2)
    elif MODE == '3':
        if RESULT is not None:
            img3 = img3_ori.copy()
            x, y, x2z, y2z, x2c, y2c = RESULT
            cv2.circle(img3, (x, y), 5, (255, 255, 0), 10)
            cv2.circle(img3, tuple([y2z + int(img3.shape[1] / 2), x2z]), 5, (255, 255, 0), 5)
            cv2.circle(img3, tuple([y2c + int(img3.shape[1] / 2), x2c]), 5, (255, 128, 0), 5)
    if MODE != "4":
        cv2.imshow("image", img3)
    k = cv2.waitKey(20) & 0xFF

    if k == 27:
        break
    elif k == ord('a'):
        print(mouseX*2, mouseY*2)

    elif k == ord('e'):
        if mouseX > 0:
            MODE = '1'
            color = tuple(np.random.randint(0, 255, 3).tolist())
            print("finding correspondence for %d %d" % (mouseX, mouseY))
            if (mouseY, mouseX) in RES_DICT:

                x_corr, y_corr = RES_DICT[(mouseY, mouseX)]
                x, y = mouseX, mouseY
                color = tuple(np.random.randint(0, 255, 3).tolist())
                coeff = compute_epip_line(
                    F_MAT, np.array([x, y, 1]).astype(np.float32).reshape((1, 3))
                ).reshape((3, 1))

                x, y, x_corr, y_corr = x//down_scale, y//down_scale, x_corr//down_scale, y_corr//down_scale

                RESULT = x, y, x_corr, y_corr, color, coeff
                mouseX = -1
                mouseY = -1
            else:
                print("not found")

    elif k == ord('u'):
        MODE = '1'
        mouseX, mouseY = x_sel, y_sel
        color = tuple(np.random.randint(0, 255, 3).tolist())
        print("finding correspondence for %d %d" % (mouseX, mouseY))
        if (mouseY, mouseX) in RES_DICT:

            x_corr, y_corr = RES_DICT[(mouseY, mouseX)]
            x, y = mouseX, mouseY
            color = tuple(np.random.randint(0, 255, 3).tolist())
            coeff = compute_epip_line(
                F_MAT, np.array([x, y, 1]).astype(np.float32).reshape((1, 3))
            ).reshape((3, 1))

            x, y, x_corr, y_corr = x // down_scale, y // down_scale, x_corr // down_scale, y_corr // down_scale

            RESULT = x, y, x_corr, y_corr, color, coeff
            mouseX = -1
            mouseY = -1
        else:
            print("not found")

    elif k == ord('d'):
        img3 = img3_ori.copy()
        mouseX = -1
        mouseY = -1
        mouseX2 = -1
        mouseY2 = -1
        MODE = None
        RESULT = None
