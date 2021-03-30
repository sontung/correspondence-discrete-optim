from math_utils import compute_zncc, compute_epip_line
from utils import read_correspondence, read_correspondence_from_dump
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys
import pickle
import flow_vis
sys.path.append("Ambrosio-Tortorelli-Minimizer")
from AmbrosioTortorelliMinimizer import AmbrosioTortorelliMinimizer


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
        x, y, x2, y2 = pair[:4]
        point1.append([y, x, 1])
        point2.append([y2, x2, 1])
        uv_map[(x, y)] = (x2-x, y2-y)
    point1 = np.array(point1)
    point2 = np.array(point2)
    epip_fitness = np.abs(np.sum((point1 @ f_mat) * point2, axis=1))

    for index, pair in enumerate(pairs):
        x, y, x2, y2 = pair[:4]
        x, y, x2, y2 = map(int, (x, y, x2, y2))

        zncc_score = compute_zncc(x, y, x2, y2, im1, im2, 19)[0]
        epip_cost += epip_fitness[index]
        cost += zncc_score
        smooth_cost += compute_deviation(x, y, uv_map)

    return cost/len(pairs), epip_cost/len(pairs), smooth_cost/len(pairs)


def visualize_flow(pairs, img1, img2, name="ssd"):
    flow1 = np.zeros_like(img1)[:, :, :2].astype(np.float64)
    flow2 = np.zeros_like(img1)[:, :, :2].astype(np.float64)

    for line in pairs:
        x, y, x_corr, y_corr = map(int, line[:4])

        flow2[x_corr, y_corr] = [x_corr - x, y_corr - y]
        flow1[x, y] = [x_corr - x, y_corr - y]

    flow_rgb1 = flow_vis.flow_to_color(flow1, convert_to_bgr=False)
    flow_rgb2 = flow_vis.flow_to_color(flow2, convert_to_bgr=False)

    Image.fromarray(np.hstack([flow_rgb1, flow_rgb2])).save("saved/%s.png" % name)

    cv2.imwrite("saved/%s-blend.png" % name, np.hstack([cv2.addWeighted(img1, 0.3, flow_rgb1, 0.7, 0),
                                                        cv2.addWeighted(img2, 0.3, flow_rgb2, 0.7, 0)]))

    return flow_rgb1


def analyze_flow():
    with open('post_analysis.pickle', 'rb') as handle:
        [dict1, dict2] = pickle.load(handle)
    print(dict1)
    # sys.exit()
    inliers = dict2[max(list(dict1.keys()), key=lambda du: dict1[du])]
    outliers = dict2[min(list(dict1.keys()), key=lambda du: dict1[du])]
    pairs = read_correspondence_from_dump("data/corr-exact.txt")
    dict3 = {}
    for pair in pairs:
        x, y, x2, y2 = map(int, pair[:4])
        dict3[(x, y)] = (x2, y2)
    u_avg = []
    v_avg = []
    deg1 = []
    x_arr = []
    for x, y in inliers:
        x_arr.append(x)
        x2, y2 = dict3[(x, y)]
        u_avg.append(x2-x)
        v_avg.append(y2-y)
        deg1.append(np.arctan2(y2-y, x2-x))
    print(np.mean(u_avg), np.var(u_avg), np.mean(v_avg), np.var(v_avg))

    # u_avg = []
    # v_avg = []
    deg2 = []
    for x, y in outliers:
        x_arr.append(x)
        x2, y2 = dict3[(x, y)]
        u_avg.append(x2-x)
        v_avg.append(y2-y)
        deg2.append(np.arctan2(y2-y, x2-x))
    plt.subplot("511")
    plt.plot(deg1, "b.")
    plt.plot(deg2, "r.")
    plt.subplot("512")
    plt.plot(sorted(u_avg), "b.")
    plt.subplot("513")
    plt.plot(sorted(v_avg), "r.")
    plt.subplot("514")
    plt.plot(x_arr, "r.")
    plt.show()
    print(np.mean(u_avg), np.var(u_avg), np.mean(v_avg), np.var(v_avg))


def post_analysis(y_trans, y_ori):
    dict3 = {}
    for i in range(y_ori.shape[0]):
        x, y = y_ori[i]
        x2, y2 = y_trans[i]
        dict3[(x, y)] = (x2, y2)
    u_avg = []
    v_avg = []
    deg1 = []
    for x, y in dict3:
        x2, y2 = dict3[(x, y)]
        u_avg.append(x2-x)
        v_avg.append(y2-y)
        deg1.append(np.arctan2(y2-y, x2-x))
    print(np.mean(u_avg), np.var(u_avg), np.mean(v_avg), np.var(v_avg))

    plt.subplot("311")
    plt.plot(deg1, "b.")
    plt.subplot("312")
    plt.plot(u_avg, "b.")
    plt.subplot("313")
    plt.plot(v_avg, "r.")

    plt.show()
    print(np.mean(u_avg), np.var(u_avg), np.mean(v_avg), np.var(v_avg))


def find_edges():

    def helper(img, mask):
        result, edges = [], []
        for c in cv2.split(img):
            solver = AmbrosioTortorelliMinimizer(c, alpha=1000, beta=0.01,
                                                 epsilon=0.01)

            f, v = solver.minimize()
            result.append(f)
            edges.append(v)

        edges = np.maximum(*edges) * mask[:, :, 0]
        _, edges = cv2.threshold(edges, 180, 255, cv2.THRESH_BINARY)
        return edges

    im1 = cv2.imread("data/im1.png")
    im2 = cv2.imread("data/im2.png")
    im1m = cv2.imread("data/im1masked.png")
    im2m = cv2.imread("data/im2masked.png")

    edge1 = helper(im1, im1m)
    edge2 = helper(im2, im2m)

    cv2.imwrite("data/im1edges.png", edge1)
    cv2.imwrite("data/im2edges.png", edge2)


if __name__ == '__main__':
    # find_edges()

    # analyze_flow()

    p1, p2 = read_correspondence()
    F_MAT, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)
    # #

    # PAIRS = read_correspondence_from_dump("data/corr-ssd.txt")
    # # photo, epip, smooth = evaluate_corr_pairs(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), F_MAT)
    # # visualize_flow(PAIRS, cv2.imread("data/im1.png"), "ssd")
    # # print("ssd solution", photo, epip, smooth)
    # #
    # # PAIRS = read_correspondence_from_dump("data/corr-dm.txt")
    # # photo, epip, smooth = evaluate_corr_pairs(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), F_MAT)
    # # visualize_flow(PAIRS, cv2.imread("data/im1.png"), "dm")
    # # print("DM solution", photo, epip, smooth)
    #
    PAIRS = read_correspondence_from_dump("data/corr-exact.txt")
    photo, epip, smooth = evaluate_corr_pairs(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), F_MAT)
    print("exact solution", photo, epip, smooth)
    rgb = visualize_flow(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), "exact")
    #
    PAIRS = read_correspondence_from_dump("data/corr-ssd.txt")
    photo, epip, smooth = evaluate_corr_pairs(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), F_MAT)
    print("ssd solution", photo, epip, smooth)
    visualize_flow(PAIRS, cv2.imread("data/im1.png"), cv2.imread("data/im2.png"), "ssd")
