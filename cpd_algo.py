from functools import partial
import matplotlib.pyplot as plt
from em_registration import AffineRegistration
import numpy as np
import cv2
import sys
import random
from pathlib import Path
from math_utils import compute_zncc_min_version
sys.path.append("Ambrosio-Tortorelli-Minimizer")
sys.path.append("fit_ellipse")
from AmbrosioTortorelliMinimizer import AmbrosioTortorelliMinimizer


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


def visualize(iteration, error, X, Y, X_color, Y_color, ax):
    plt.cla()
    ax.scatter(X[0:-1:1, 0],  X[0:-1:1, 1], color=X_color, label='Target')
    ax.scatter(Y[0:-1:1, 0],  Y[0:-1:1, 1], color=Y_color, label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


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
        plt.scatter(res[0:-1:1, 0],  res[0:-1:1, 1], color=colors)
        plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


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

            # for u in range(edge_im.shape[0]):
            #     for v in range(edge_im.shape[1]):
            #         if edge_im[u, v] > 0:
            #             print(u, v)
            # print(x, y, x2, y2)

            if edge_im[x2, y2] > 0:
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


def main_full():
    IM1 = cv2.imread("data/im1.png")
    IM2 = cv2.imread("data/im2.png")

    X = np.loadtxt('data/source.txt')
    Y = np.loadtxt('data/target.txt')
    X_color = np.load("data/source_colors.npy")/255
    Y_color = np.load("data/target_colors.npy")/255

    prior = read_prior_probab()

    zncc_mat = np.zeros((Y.shape[0], X.shape[0]))

    xy2id = {tuple(X[i]): i for i in range(X.shape[0])}
    x2y22id = {tuple(Y[i]): i for i in range(Y.shape[0])}

    print("there are %d priors" % len(prior))
    for (x, y) in prior:
        i = x2y22id[prior[(x, y)]]
        j = xy2id[(x, y)]
        zncc_mat[i, j] = 1

    # X_color = np.array([(0, 0.0, 0) for _ in range(X.shape[0])])
    # Y_color = np.array([(1, 1.0, 1) for _ in range(Y.shape[0])])

    reg = AffineRegistration(**{'X': X, 'Y': Y,
                                "X_full": X, "Y_full": Y,
                                "image": IM2,
                                'X_color': X_color,
                                'Y_color': Y_color, "zncc": zncc_mat})

    reg.register()
    Y_trans = reg.transform_point_cloud(Y)
    visualize_transformation(X, Y, X_color, Y_color, Y_trans)


def solve_partial():
    IM1 = cv2.imread("data/im1.png")
    IM2 = cv2.imread("data/im2.png")

    X = np.loadtxt('data/source.txt')
    Y = np.loadtxt('data/target.txt')
    X_color = np.load("data/source_colors.npy")/255
    Y_color = np.load("data/target_colors.npy")/255

    prior = read_prior_probab(min_score=0.5)
    model = solve_procedure(prior, X, Y, IM2, X_color, Y_color)
    Y_trans = model.transform_point_cloud(Y)
    # visualize_transformation(X, Y, X_color, Y_color, Y_trans)
    return Y_trans


def solve_procedure(prior, x_mat, y_mat, im, x_color, y_color):
    X_solve = []
    Y_solve = []
    for (x, y) in prior:
        X_solve.append([x, y])
        if prior[(x, y)] not in Y_solve:
            Y_solve.append(prior[(x, y)])
    X_solve = np.array(X_solve)
    Y_solve = np.array(Y_solve)

    zncc_mat = np.zeros((Y_solve.shape[0], X_solve.shape[0]))
    print("zncc mat shape", zncc_mat.shape)
    xy2id = {tuple(X_solve[i]): i for i in range(X_solve.shape[0])}
    x2y22id = {tuple(Y_solve[i]): i for i in range(Y_solve.shape[0])}

    print("there are %d priors" % len(prior))
    for (x, y) in prior:
        i = x2y22id[prior[(x, y)]]
        j = xy2id[(x, y)]
        zncc_mat[i, j] = 1

    reg = AffineRegistration(**{'X': X_solve, 'Y': Y_solve,
                                "X_full": x_mat, "Y_full": y_mat,
                                "image": im,
                                'X_color': x_color,
                                'Y_color': y_color, "zncc": zncc_mat})

    reg.register()
    return reg


def solve_edge_only():
    IM2 = cv2.imread("data/im2.png")
    img = cv2.imread("data/im2.png", 1)
    X = np.loadtxt('data/source.txt')
    Y = np.loadtxt('data/target.txt')
    X_color = np.load("data/source_colors.npy")/255
    Y_color = np.load("data/target_colors.npy")/255

    result, edges = [], []
    for c in cv2.split(img):
        solver = AmbrosioTortorelliMinimizer(c, alpha=1000, beta=0.01,
                                             epsilon=0.01)

        f, v = solver.minimize()
        result.append(f)
        edges.append(v)

    edges = np.maximum(*edges)*cv2.imread("data/im2masked.png")[:, :, 0]
    _, edges = cv2.threshold(edges, 180, 255, cv2.THRESH_BINARY)
    prior = read_prior_probab(randomize=True, edge_im=edges, edge_only=True)
    model = solve_procedure(prior, X, Y, IM2, X_color, Y_color)
    Y_trans = model.transform_point_cloud(Y)
    return Y_trans


def solve_ransac(nb_epc=10):
    IM2 = cv2.imread("data/im2.png")

    X = np.loadtxt('data/source.txt')
    Y = np.loadtxt('data/target.txt')
    X_color = np.load("data/source_colors.npy")/255
    Y_color = np.load("data/target_colors.npy")/255

    Y_trans = np.zeros_like(Y)
    for _ in range(nb_epc):
        prior = read_prior_probab(randomize=True)
        model = solve_procedure(prior, X, Y, IM2, X_color, Y_color)
        Y_trans += model.transform_point_cloud(Y)
    Y_trans /= nb_epc
    # visualize_transformation(X, Y, X_color, Y_color, Y_trans)
    return Y_trans


if __name__ == '__main__':
    write()
    y3 = solve_edge_only()
    y1 = solve_ransac()
    y2 = solve_partial()
    print(np.mean(np.abs(y2-y1)))
    visualize_all_results([y1, y2, y3], np.load("data/target_colors.npy")/255)