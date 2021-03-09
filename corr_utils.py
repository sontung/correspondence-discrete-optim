from math_utils import compute_zncc, compute_epip_line
from utils import read_correspondence, read_correspondence_from_dump
from PIL import Image
import numpy as np
import cv2
import flow_vis


def compute_deviation(x, y, uv_map):
    ux = 0
    vx = 0
    uy = 0
    vy = 0
    u, v = uv_map[x, y]
    if (x + 1, y) in uv_map:
        u2, v2 = uv_map[(x + 1, y)]
        ux += abs(u2 - u)
        vx += abs(v2 - v)
    if (x - 1, y) in uv_map:
        u2, v2 = uv_map[(x - 1, y)]
        ux += abs(u2 - u)
        vx += abs(v2 - v)
    if (x, y + 1) in uv_map:
        u2, v2 = uv_map[(x, y + 1)]
        uy += abs(u2 - u)
        vy += abs(v2 - v)
    if (x, y - 1) in uv_map:
        u2, v2 = uv_map[(x, y - 1)]
        uy += abs(u2 - u)
        vy += abs(v2 - v)

    return (ux+uy+vx+vy)/4


def evaluate_corr_pairs(pairs, im1, im2, f_mat):
    cost = 0
    epip_cost = 0
    smooth_cost = 0
    point1 = []
    point2 = []
    uv_map = {}
    for pair in pairs:
        x, y, x2, y2 = pair
        point1.append([y, x, 1])
        point2.append([y2, x2, 1])
        uv_map[(x, y)] = (x2-x, y2-y)
    point1 = np.array(point1)
    point2 = np.array(point2)
    epip_fitness = np.abs(np.sum((point1 @ f_mat) * point2, axis=1))

    for index, pair in enumerate(pairs):
        x, y, x2, y2 = pair
        x, y, x2, y2 = map(int, (x, y, x2, y2))

        zncc_score = compute_zncc(x, y, x2, y2, im1, im2, 19)[0]
        epip_cost += epip_fitness[index]
        cost += zncc_score
        smooth_cost += compute_deviation(x, y, uv_map)

    return cost/len(pairs), epip_cost/len(pairs), smooth_cost/len(pairs)


def visualize_flow(pairs, img1, name="ssd"):
    flow = np.zeros_like(img1)[:, :, :2]
    for line in pairs:
        x, y, x_corr, y_corr = map(int, line)
        flow[x_corr, y_corr] = [x_corr - x, y_corr - y]
        flow[x, y] = [x_corr - x, y_corr - y]
    flow_rgb = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    Image.fromarray(flow_rgb).save("saved/%s.png" % name)


if __name__ == '__main__':
    PAIRS = read_correspondence_from_dump("data/corr-ssd.txt")
    p1, p2 = read_correspondence()
    F_MAT, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)

    photo, epip, smooth = evaluate_corr_pairs(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), F_MAT)
    visualize_flow(PAIRS, cv2.imread("data/im1.png"), "ssd")
    print("ssd solution", photo, epip, smooth)

    PAIRS = read_correspondence_from_dump("data/corr-dm.txt")
    photo, epip, smooth = evaluate_corr_pairs(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), F_MAT)
    visualize_flow(PAIRS, cv2.imread("data/im1.png"), "dm")
    print("DM solution", photo, epip, smooth)
