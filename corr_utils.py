from math_utils import compute_zncc, compute_epip_line
from utils import read_correspondence, read_correspondence_from_dump
import numpy as np
import cv2


def evaluate_corr_pairs(pairs, im1, im2, f_mat):
    cost = 0
    epip_cost = 0
    smooth_cost = 0
    point1 = []
    point2 = []
    for pair in pairs:
        x, y, x2, y2 = pair
        point1.append([y, x, 1])
        point2.append([y2, x2, 1])
    point1 = np.array(point1)
    point2 = np.array(point2)
    epip_fitness = np.abs(np.sum((point1 @ f_mat) * point2, axis=1))

    for index, pair in enumerate(pairs):
        x, y, x2, y2 = pair
        x, y, x2, y2 = map(int, (x, y, x2, y2))

        zncc_score = compute_zncc(x, y, x2, y2, im1, im2, 19)[0]
        epip_cost += epip_fitness[index]
        cost += zncc_score

    return cost, epip_cost, smooth_cost


if __name__ == '__main__':
    PAIRS = read_correspondence_from_dump()
    p1, p2 = read_correspondence()
    F_MAT, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)

    photo, epip = evaluate_corr_pairs(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), F_MAT)

    print(photo, epip)