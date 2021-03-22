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
    sys.stdout = open("data/target.txt", "w")
    for i in range(IM1_masked.shape[0]):
        for j in range(IM1_masked.shape[1]):
            if IM1_masked[i, j, 0] > 0:
                print(i, j)
                colors.append(IM1[i, j])
    colors = np.array(colors)
    np.save('data/target_colors.npy', colors)
    colors = []
    sys.stdout = open("data/source.txt", "w")
    for i in range(IM2_masked.shape[0]):
        for j in range(IM2_masked.shape[1]):
            if IM2_masked[i, j, 0] > 0:
                print(i, j)
                colors.append(IM2[i, j])
    colors = np.array(colors)
    np.save('data/source_colors.npy', colors)
    sys.stdout = sys.__stdout__


def visualize(iteration, error, X, Y, X_color, Y_color, ax):
    plt.cla()
    ax.scatter(X[0:-1:1, 0],  X[0:-1:1, 1], color=X_color/255.0, label='Target')
    ax.scatter(Y[0:-1:1, 0],  Y[0:-1:1, 1], color="white", label='Source')
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


def main():
    IM1 = cv2.imread("data/im1.png")
    IM2 = cv2.imread("data/im2.png")
    X = np.loadtxt('data/target.txt')
    Y = np.loadtxt('data/source.txt')
    X_color = np.load("data/target_colors.npy")
    Y_color = np.load("data/source_colors.npy")

    Y = Y[0:-1:100, :]
    Y_color = Y_color[0:-1:100, :]

    # my_file = Path("saved/zncc_mat.npy")
    # if my_file.exists():
    #     zncc_mat = np.load("saved/zncc_mat.npy")
    # else:
    #     zncc_mat = np.zeros((Y.shape[0], X.shape[0]))
    #     for i in range(zncc_mat.shape[0]):
    #         for j in range(zncc_mat.shape[1]):
    #             x, y = map(int, X[j])
    #             x2, y2 = map(int, Y[j])
    #             zncc = compute_zncc_min_version(x, y, x2, y2, IM1, IM2, 5, debug=False)
    #             zncc_mat[i, j] = zncc[0]
    #     np.save('saved/zncc_mat.npy', zncc_mat)

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = AffineRegistration(**{'X': X, 'Y': Y, 'X_color': X_color, 'Y_color': Y_color})
    reg.register(callback)
    plt.show()
    # visualize_transformation(reg)

    Y = np.loadtxt('data/source.txt')
    Y_color = np.load("data/source_colors.npy")
    Y = Y[0:-1:50, :]
    Y_color = Y_color[0:-1:50, :]
    reg.reload_inputs(X, Y, X_color, Y_color)
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])
    reg.register(callback)
    plt.show()
    visualize_transformation(reg)



if __name__ == '__main__':
    write()
    # sys.exit()
    main()
