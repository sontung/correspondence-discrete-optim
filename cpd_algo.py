from functools import partial
import matplotlib.pyplot as plt
from em_registration import AffineRegistration
import numpy as np
import cv2
import sys
from pathlib import Path
from math_utils import compute_zncc_min_version


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


def visualize_transformation(reg):
    X = np.loadtxt('data/target.txt')
    Y = np.loadtxt('data/source.txt')
    X_color = np.load("data/target_colors.npy")
    Y_color = np.load("data/source_colors.npy")

    Y_transformed = reg.transform_point_cloud(Y)
    plt.subplot(311)
    plt.scatter(Y_transformed[0:-1:1, 0],  Y_transformed[0:-1:1, 1], color=Y_color/255.0)
    plt.subplot(312)
    plt.scatter(Y[0:-1:1, 0],  Y[0:-1:1, 1], color=Y_color/255.0)
    plt.subplot(313)
    plt.scatter(X[0:-1:1, 0],  X[0:-1:1, 1], color=X_color/255.0)
    plt.show()


def read_prior_probab(txt_file="data/corr-ssd.txt"):
    sys.stdin = open(txt_file, "r")
    lines = sys.stdin.readlines()
    a_dict = {}
    for line in lines:
        x, y, x2, y2 = map(int, line[:-1].split(" "))
        a_dict[(x, y)] = (x2, y2)
    return a_dict


def main():

    prior = read_prior_probab()

    X = []
    Y = []
    for (x2, y2) in prior:
        X.append(prior[(x2, y2)])
        Y.append((x2, y2))
    X = np.array(X)
    Y = np.array(Y)

    zncc_mat = np.zeros((Y.shape[0], X.shape[0]))

    xy2id = {tuple(X[i]): i for i in range(X.shape[0])}
    x2y22id = {tuple(Y[i]): i for i in range(Y.shape[0])}

    X = []
    Y = []
    for (x2, y2) in prior:
        i = xy2id[prior[(x2, y2)]]
        j = x2y22id[(x2, y2)]
        zncc_mat[j, i] = 1
        X.append(prior[(x2, y2)])
        Y.append((x2, y2))
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape, zncc_mat.shape)

    reg = AffineRegistration(**{'X': X, 'Y': Y,
                                'X_color': np.array([(0.5, 0.5, 0) for _ in range(X.shape[0])]),
                                'Y_color': np.array([(1, 0.5, 1) for _ in range(Y.shape[0])]), "zncc": zncc_mat})

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])
    reg.register(callback)
    plt.show()
    visualize_transformation(reg)


def main_full():
    IM2 = cv2.imread("data/im2.png")

    X = np.loadtxt('data/target.txt')
    Y = np.loadtxt('data/source.txt')
    X_color = np.load("data/target_colors.npy")/255
    Y_color = np.load("data/source_colors.npy")/255

    prior = read_prior_probab()

    zncc_mat = np.zeros((Y.shape[0], X.shape[0]))

    xy2id = {tuple(X[i]): i for i in range(X.shape[0])}
    x2y22id = {tuple(Y[i]): i for i in range(Y.shape[0])}

    print("there are %d priors" % len(prior))
    for (x, y) in prior:
        i = x2y22id[prior[(x, y)]]
        j = xy2id[(x, y)]
        zncc_mat[i, j] = 1

    print(X.shape, Y.shape, zncc_mat.shape)

    # X_color = np.array([(0, 0.0, 0) for _ in range(X.shape[0])])
    # Y_color = np.array([(1, 1.0, 1) for _ in range(Y.shape[0])])

    Y_s = Y[0:-1:10, :]
    zncc_mat = zncc_mat[0:-1:10, :]

    reg = AffineRegistration(**{'X': X, 'Y': Y_s, "X_full": X, "Y_full": Y, "image": IM2,
                                'X_color': X_color,
                                'Y_color': Y_color, "zncc": zncc_mat})

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])
    reg.register(callback)
    plt.show()
    visualize_transformation(reg)


if __name__ == '__main__':
    write()
    main_full()
