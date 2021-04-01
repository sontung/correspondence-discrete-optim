from functools import partial
import matplotlib.pyplot as plt
from em_registration import AffineRegistration
import numpy as np
import cv2
import sys
import pickle
import random
from pathlib import Path
from corr_utils import evaluate_corr_pairs, visualize_flow

from math_utils import compute_zncc_min_version
from utils import read_correspondence



def write():
    IM1 = cv2.imread("data/im1.png")
    IM2 = cv2.imread("data/im2.png")
    IM1_masked = cv2.imread("data/im1masked.png")
    IM2_masked = cv2.imread("data/im2masked.png")

    my_file = Path("data/target.txt")

    if my_file.is_file():
        return

    colors = []
    sys.stdout = open("data/source.txt", "w")
    for i in range(IM1_masked.shape[0]):
        for j in range(IM1_masked.shape[1]):
            if IM1_masked[i, j, 0] > 0:
                print(i, j)
                colors.append(IM1[i, j])
    colors = np.array(colors)
    np.save('data/source_colors.npy', colors)
    colors = []
    sys.stdout = open("data/target.txt", "w")
    for i in range(IM2_masked.shape[0]):
        for j in range(IM2_masked.shape[1]):
            if IM2_masked[i, j, 0] > 0:
                print(i, j)
                colors.append(IM2[i, j])
    colors = np.array(colors)
    np.save('data/target_colors.npy', colors)
    sys.stdout = sys.__stdout__


def read_prior_probab(txt_file="data/corr-ssd.txt",
                      edge_im=None,
                      edge_only=False,
                      min_score=-1.0,
                      randomize=False):
    sys.stdin = open(txt_file, "r")

    if edge_only:
        lines = sys.stdin.readlines()
        a_dict = {}
        for line in lines:
            x, y, x2, y2, score = map(float, line[:-1].split(" "))
            x, y, x2, y2 = map(int, [x, y, x2, y2])

            if edge_im[x, y] > 0:
                a_dict[(x, y)] = (x2, y2)
        assert len(a_dict) > 0
        return a_dict

    if randomize:
        lines = sys.stdin.readlines()
        a_dict = {}

        ind1 = []
        ind2 = []
        ind3 = []
        prob = []
        for count, line in enumerate(lines):
            x, y, x2, y2, score = map(float, line[:-1].split(" "))
            x, y, x2, y2 = map(int, [x, y, x2, y2])
            ind1.append((x, y))
            ind2.append((x2, y2))
            ind3.append(count)
            prob.append(score)
            assert score > 0
        chosen = random.choices(ind3, k=len(ind3)*60//100)
        for ind in chosen:
            a_dict[ind1[ind]] = ind2[ind]
    else:
        lines = sys.stdin.readlines()
        a_dict = {}
        for line in lines:
            x, y, x2, y2, score = map(float, line[:-1].split(" "))
            x, y, x2, y2 = map(int, [x, y, x2, y2])
            if score >= min_score:
                a_dict[(x, y)] = (x2, y2)
    return a_dict


def solve_procedure(prior, x_full, y_full, im, x_color, y_color):
    x_solve = []
    y_solve = []
    for (x, y) in prior:
        y_solve.append([x, y])
        if prior[(x, y)] not in x_solve:
            x_solve.append(prior[(x, y)])
    x_solve = np.array(x_solve)
    y_solve = np.array(y_solve)

    print("solve", np.mean(x_solve, axis=0), np.mean(y_solve, axis=0))

    zncc_mat = np.zeros((y_solve.shape[0], x_solve.shape[0]))
    x2y22id = {tuple(x_solve[i]): i for i in range(x_solve.shape[0])}
    xy2id = {tuple(y_solve[i]): i for i in range(y_solve.shape[0])}

    print("there are %d priors" % len(prior))
    for (x, y) in prior:
        i = x2y22id[prior[(x, y)]]
        j = xy2id[(x, y)]
        zncc_mat[j, i] = 1

    reg = AffineRegistration(**{'X': x_solve, 'Y': y_solve,
                                "X_full": x_full, "Y_full": y_full,
                                "image": im,
                                'X_color': x_color,
                                'Y_color': y_color,
                                "zncc": zncc_mat})
    print("registering")
    reg.register()
    return reg


def visualize_transformation(x, y, x_color, y_color, y_transformed):

    plt.subplot(311)
    plt.scatter(y_transformed[0:-1:1, 0],  y_transformed[0:-1:1, 1], color=y_color)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(312)
    plt.scatter(y[0:-1:1, 0],  y[0:-1:1, 1], color=y_color)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.subplot(313)
    plt.scatter(x[0:-1:1, 0],  x[0:-1:1, 1], color=x_color)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


def visualize_all_results(results, colors):
    for count, res in enumerate(results):
        plt.subplot("%d1%d" % (len(results), count+1))
        plt.scatter(res[0:-1:1, 0],  res[0:-1:1, 1], color=colors[count])
        plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def solve_partial():
    im = cv2.imread("data/im2.png")
    x_full = np.loadtxt('data/target.txt')
    y_full = np.loadtxt('data/source.txt')
    x_color = np.load("data/target_colors.npy")/255
    y_color = np.load("data/source_colors.npy")/255

    prior = read_prior_probab(min_score=0.5)

    model = solve_procedure(prior, x_full, y_full, im, x_color, y_color)
    y_trans = model.transform_point_cloud(y_full)
    return y_trans


def solve_simple(corr, x_full, y_full, min_score=0.5):
    a_dict = {}
    for x, y, x2, y2, score in corr:
        x, y, x2, y2 = map(int, [x, y, x2, y2])
        if score >= min_score:
            a_dict[(x, y)] = (x2, y2)
    model = solve_procedure(a_dict, x_full, y_full, None, None, None)
    y_trans = model.transform_point_cloud(y_full)
    return y_trans


def solve_ransac(nb_epc=10):
    im = cv2.imread("data/im2.png")
    x_full = np.loadtxt('data/target.txt')
    y_full = np.loadtxt('data/source.txt')
    x_color = np.load("data/target_colors.npy")/255
    y_color = np.load("data/source_colors.npy")/255

    y_trans = np.zeros_like(y_full)
    for _ in range(nb_epc):
        prior = read_prior_probab(randomize=True)
        model = solve_procedure(prior, x_full, y_full, im, x_color, y_color)
        y_trans += model.transform_point_cloud(y_full)
    y_trans /= nb_epc
    # visualize_transformation(x_full, y_full, x_color, y_color, y_trans)

    return y_trans


def solve_edge_only():
    im = cv2.imread("data/im1.png")
    img = cv2.imread("data/im1.png", 1)
    x_full = np.loadtxt('data/target.txt')
    y_full = np.loadtxt('data/source.txt')
    x_color = np.load("data/target_colors.npy")/255
    y_color = np.load("data/source_colors.npy")/255
    print("full", np.mean(x_full, axis=0), np.mean(y_full, axis=0))

    result, edges = [], []
    for c in cv2.split(img):
        solver = AmbrosioTortorelliMinimizer(c, alpha=1000, beta=0.01,
                                             epsilon=0.01)

        f, v = solver.minimize()
        result.append(f)
        edges.append(v)

    edges = np.maximum(*edges)*cv2.imread("data/im1masked.png")[:, :, 0]
    _, edges = cv2.threshold(edges, 180, 255, cv2.THRESH_BINARY)
    cv2.imwrite("edges.png", edges)

    prior = read_prior_probab(randomize=True, edge_im=edges, edge_only=True)
    model = solve_procedure(prior, x_full, y_full, im, x_color, y_color)
    y_trans = model.transform_point_cloud(y_full)
    # visualize_transformation(x_full, y_full, x_color, y_color, y_trans)
    return y_trans


def solve_outliers(pairs, img):
    im = cv2.imread("data/im1.png")
    x_full = np.loadtxt('data/target.txt')
    y_full = np.loadtxt('data/source.txt')
    x_color = np.load("data/target_colors.npy")/255
    y_color = np.load("data/source_colors.npy")/255

    rgb = visualize_flow(pairs, img)
    rgb2 = rgb.reshape((-1, 3))
    rgb2 = np.float32(rgb2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(rgb2, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape(img.shape)

    dict1 = {}
    dict2 = {}
    for du1 in range(res.shape[0]):
        for du2 in range(res.shape[1]):
            if not (res[du1, du2] == np.array([255, 255, 255])).all():
                k = tuple(res[du1, du2])
                if k not in dict1:
                    dict1[k] = 0
                    dict2[k] = [(du1, du2)]
                else:
                    dict1[k] += 1
                    dict2[k].append((du1, du2))
    with open('post_analysis.pickle', 'wb') as handle:
        pickle.dump([dict1, dict2], handle, protocol=pickle.HIGHEST_PROTOCOL)

    inliers = dict2[max(list(dict1.keys()), key=lambda du: dict1[du])]
    outliers = dict2[min(list(dict1.keys()), key=lambda du: dict1[du])]
    print("outliers", len(outliers), "inliers", len(inliers))
    prior = {}
    for x, y, x_corr, y_corr in pairs:
        if (x, y) in outliers:
            prior[(x, y)] = (x_corr, y_corr)
    model = solve_procedure(prior, x_full, y_full, im, x_color, y_color)
    y_trans = model.transform_point_cloud(y_full)
    return y_trans


def output_correspondence(y_ori, y_trans, target):
    # from corr_utils import post_analysis
    # post_analysis(y_trans, y_ori)
    # post_analysis(np.round(y_trans), y_ori)
    # y_trans2 = []
    # for i in range(y_trans.shape[0]):
    #     x2, y2 = map(int, y_trans[i])
    #     y_trans2.append([x2, y2])
    # post_analysis(y_trans2, y_ori)

    img1 = cv2.imread("data/im1.png")
    img2 = cv2.imread("data/im2.png")
    p1, p2 = read_correspondence()
    fundamental_mat, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)

    def helper(y_trans):
        ind1 = np.repeat(y_trans, target.shape[0], axis=0)
        ind2 = np.tile(target, (y_trans.shape[0], 1))
        diff = (ind1-ind2)**2
        diff = np.sum(diff, axis=1)
        diff = np.sqrt(diff).reshape((y_trans.shape[0], target.shape[0]))
        diff = np.argmin(diff, axis=1)
        corr = []
        for i in range(y_trans.shape[0]):
            x, y = y_ori[i]
            x2, y2 = target[diff[i]]
            corr.append([x, y, x2, y2])

        with open("data/corr-exact.txt", "w") as f:
            for x, y, x_corr, y_corr in corr:
                print(x, y, x_corr, y_corr, file=f)

        p, e, s = evaluate_corr_pairs(corr, img1, img2, fundamental_mat)
        print("ssd results", p, e, s)
        return corr

    # def helper2(y_trans):
    #     corr = []
    #     for i in range(y_trans.shape[0]):
    #         x, y = y_ori[i]
    #         x2, y2 = map(float, y_trans[i])
    #         corr.append([x, y, x2, y2])
    #     with open("data/corr-exact.txt", "w") as f:
    #         for x, y, x_corr, y_corr in corr:
    #             print(x, y, x_corr, y_corr, file=f)
    #
    #     p, e, s = evaluate_corr_pairs(corr, img1, img2, fundamental_mat)
    #     print("ssd results", p, e, s)
    #
    #     return corr

    corr = helper(y_trans)

    # y_trans = solve_outliers(corr, img1)
    # corr = helper2(y_trans)


def final_process(y_ori, y_trans, target, img1, img2, fundamental_mat):

    ind1 = np.repeat(y_trans, target.shape[0], axis=0)
    ind2 = np.tile(target, (y_trans.shape[0], 1))
    diff = (ind1-ind2)**2
    diff = np.sum(diff, axis=1)
    diff = np.sqrt(diff).reshape((y_trans.shape[0], target.shape[0]))
    diff = np.argmin(diff, axis=1)
    corr = []
    for i in range(y_trans.shape[0]):
        x, y = y_ori[i]
        x2, y2 = target[diff[i]]
        corr.append([x, y, x2, y2])

    p, e, s = evaluate_corr_pairs(corr, img1, img2, fundamental_mat)
    print("ssd results", p, e, s)
    return corr


if __name__ == '__main__':
    # y0 = np.loadtxt('data/source.txt')
    # y2 = solve_ransac()
    # y1 = solve_partial()
    # y3 = solve_edge_only()
    # visualize_all_results([np.loadtxt('data/target.txt'), y0, y1, y2, y3],
    #                       [np.load("data/target_colors.npy")/255,
    #                        np.load("data/source_colors.npy")/255,
    #                        np.load("data/source_colors.npy")/255,
    #                        np.load("data/source_colors.npy")/255,
    #                        np.load("data/source_colors.npy")/255]
    #                       )
    # y_true = np.loadtxt('data/target.txt')
    # print(
    #     np.mean(np.abs(y1 - y_true)),
    #     np.mean(np.abs(y2 - y_true)),
    #     np.mean(np.abs(y3 - y_true)),
    # )

    output_correspondence(np.loadtxt('data/source.txt'), solve_partial(), np.loadtxt('data/target.txt'))
