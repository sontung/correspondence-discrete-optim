from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
import sys
import os
import pickle
import math
import random
import subprocess


def h_stack():
    for c in range(3, 100, 3):

        im1 = Image.open("samples/s2/0-%d.png" % c)
        im2 = Image.open("samples/s2/1-%d.png" % c)
        res = np.hstack([im1, im2])
        print(res.shape, im1.size)
        Image.fromarray(res).save("stacks/res-%d.png" % c)


def read_correspondence(js_dir="data/res-3.json", im_size=3840):
    with open(js_dir) as json_file:
        data = json.load(json_file)
    points1 = []
    points2 = []
    for line in data["shapes"]:
        points1.append(tuple(line["points"][0]))

        point2 = line["points"][1]
        point2[0] -= im_size
        assert im_size >= point2[0] > -1
        points2.append(tuple(point2))

    return points1, points2


def read_correspondence_from_dump(txt_dir="data/corr-3.txt"):
    sys.stdin = open(txt_dir, "r")
    lines = sys.stdin.readlines()
    pairs = [tuple(map(float, line[:-1].split(" ")))[:-1] for line in lines]
    return pairs


def extract_frame():
    # videos = ["/home/sontung/Desktop/D/asilla/asilla/2018 Nissan Kicks/CEN1807_FPS500_OBA02_SHOULDER.mp4",
    #           "/home/sontung/Desktop/D/asilla/asilla/2018 Nissan Kicks/CEN1807_FPS500_OBA03_DRV.mp4"]
    videos = ["1.mp4",
              "2.mp4"]

    for c, v in enumerate(videos):
        cap = cv2.VideoCapture(v)
        count = 0

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                count += 1

                # Display the resulting frame
                if count % 3 == 0:
                    cv2.imwrite('samples/s2/%d-%d.png' % (c, count), frame)
                if count > 50 or not ret:
                    break
    return count+1


def compute_zncc(x, y, x2, y2, f, g, window_size, f_, g_, using_global_mean=True):
    # du1 = 0
    # du2 = 0
    # du3 = 0
    # f_ = [np.mean(f[:, :, c]) for c in range(3)]
    # g_ = [np.mean(g[:, :, c]) for c in range(3)]
    # for delta_i in range(-window_size, window_size+1, 1):
    #     for delta_j in range(-window_size, window_size + 1, 1):
    #         du1 += (f[x+delta_i, y+delta_j] - f_) * (g[x2+delta_i, y2+delta_j] - g_)
    #         du2 += (f[x+delta_i, y+delta_j] - f_) ** 2
    #         du3 += (g[x2+delta_i, y2+delta_j] - g_) ** 2
    # s2 = np.mean(du1/(np.sqrt(du2)*np.sqrt(du3)))
    # f_ = [np.mean(f[:, :, c]) for c in range(3)]
    # g_ = [np.mean(g[:, :, c]) for c in range(3)]

    f = f[x-window_size: x+window_size+1, y-window_size: y+window_size+1]
    g = g[x2-window_size: x2+window_size+1, y2-window_size: y2+window_size+1]
    if not using_global_mean:
        f_ = [np.mean(f[:, :, c]) for c in range(3)]
        g_ = [np.mean(g[:, :, c]) for c in range(3)]
    du1 = np.multiply(f-f_, g-g_)
    du2 = np.multiply(f-f_, f-f_)
    du3 = np.multiply(g-g_, g-g_)
    s2 = np.sum(du1) / (np.sqrt(np.sum(du2)) * np.sqrt(np.sum(du3)) + 0.00001)
    return s2


def check_inside(nvert, vertx, verty, testx, testy):
    i = 0
    j = nvert - 1
    c = False
    while i < nvert:
        if ((verty[i] > testy) != (verty[j] > testy)) and \
                (testx < (vertx[j]-vertx[i]) * (testy-verty[i]) / (verty[j]-verty[i]) + vertx[i]):
            c = not c
        j = i
        i += 1
    return c


def non_linear_distance(m, l):
    return abs(int(m[1]*l[0]+m[0]*l[1]+l[2])) / math.sqrt(l[0]**2+l[1]**2)


def compute_neighbor_deviation(x, y, x2, y2, res_dict):
    dev_x = 0
    dev_y = 0
    if (x-1, y) in res_dict:
        x_n, y_n = res_dict[(x-1, y)]
        dev_x += abs(x_n-x2)
        dev_y += abs(y_n-y2)
    if (x+1, y) in res_dict:
        x_n, y_n = res_dict[(x+1, y)]
        dev_x += abs(x_n-x2)
        dev_y += abs(y_n-y2)
    if (x, y-1) in res_dict:
        x_n, y_n = res_dict[(x, y-1)]
        dev_x += abs(x_n-x2)
        dev_y += abs(y_n-y2)
    if (x, y+1) in res_dict:
        x_n, y_n = res_dict[(x, y+1)]
        dev_x += abs(x_n-x2)
        dev_y += abs(y_n-y2)
    dev = dev_x/8.0+dev_y/8.0
    return dev


def ssd_algo_dense(img1, img2, mask1, mask2, fundamental_mat, skipped=1):
    res_dict = {}
    for _ in range(3):
        correspondences = []
        for count in range(0, len(mask2), skipped):
            x, y = mask2[count]
            coeff = cv2.computeCorrespondEpilines(np.expand_dims([y, x, 1], axis=0), 2, fundamental_mat).squeeze()

            res = []
            for (x2, y2) in mask1:
                if abs(int(coeff[1] * x2 + coeff[2] + coeff[0]*y2)) <= 1:  # lie on the epip line
                    if compute_neighbor_deviation(x, y, x2, y2, res_dict) <= 7:
                        score = 0
                        wss = [5, 7, 9, 15, 17, 19]
                        for ws in wss:
                            feature1 = compute_pixel_feature(img1, x2, y2, ws)
                            feature2 = compute_pixel_feature(img2, x, y, ws)

                            feature1_b = cv2.edgePreservingFilter(feature1, cv2.RECURS_FILTER)
                            feature2_b = cv2.edgePreservingFilter(feature2, cv2.RECURS_FILTER)

                            score += math.sqrt(np.sum(np.abs(feature2 - feature1))*np.sum(np.abs(feature2_b - feature1_b)))
                        res.append([x2, y2, score/len(wss)])

            if len(res) == 0:
                continue
            x_corr, y_corr, score = min(res, key=lambda du: du[2])
            res_dict[(x, y)] = (x_corr, y_corr)

            correspondences.append([x, y, x_corr, y_corr, score])

    return correspondences


if __name__ == '__main__':
    extract_frame()
    # h_stack()
    # p1, p2 = read_correspondence()
    # print(p1.shape, p2.shape)
    # feature_matcher()
    # extract_frame()
    # h_stack()
    # validate_f_mat()
    # reverse_epip_line()
    # evaluate()
    # make_demo_video()
    # feature_matcher()
    # check_results()
    # vis_results()
    # vis_color_encoded()