from functools import partial
import matplotlib.pyplot as plt
from pycpd import AffineRegistration
import numpy as np
import cv2
import sys
from pathlib import Path
from math_utils import compute_zncc_min_version


def write():
    IM1_masked = cv2.imread("data/im1maskeds.png")
    IM2_masked = cv2.imread("data/im2maskeds.png")

    my_file = Path("data/target.txt")

    if my_file.is_file():
        return

    sys.stdout = open("data/target.txt", "w")
    for i in range(IM1_masked.shape[0]):
        for j in range(IM1_masked.shape[1]):
            if IM1_masked[i, j, 0] > 0:
                print(i, j)
    sys.stdout = open("data/source.txt", "w")
    for i in range(IM2_masked.shape[0]):
        for j in range(IM2_masked.shape[1]):
            if IM2_masked[i, j, 0] > 0:
                print(i, j)
    sys.stdout = sys.__stdout__


def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[0:-1:1, 0],  X[0:-1:1, 1], color='red', label='Target')
    ax.scatter(Y[0:-1:1, 0],  Y[0:-1:1, 1], color='blue', label='Source')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
        iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)


def main():
    IM1 = cv2.imread("data/im1.png")
    IM2 = cv2.imread("data/im2.png")
    X = np.loadtxt('data/target.txt')
    Y = np.loadtxt('data/source.txt')

    my_file = Path("saved/zncc_mat.npy")

    if my_file.exists():
        zncc_mat = np.load("saved/zncc_mat.npy")
    else:
        zncc_mat = np.zeros((Y.shape[0], X.shape[0]))
        for i in range(zncc_mat.shape[0]):
            for j in range(zncc_mat.shape[1]):
                x, y = map(int, X[j])
                x2, y2 = map(int, Y[j])
                zncc = compute_zncc_min_version(x, y, x2, y2, IM1, IM2, 5, debug=False)
                zncc_mat[i, j] = zncc[0]
        np.save('saved/zncc_mat.npy', zncc_mat)

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg = AffineRegistration(**{'X': X, 'Y': Y, "zncc_mat": zncc_mat})
    reg.register(callback)
    plt.show()


if __name__ == '__main__':
    write()
    main()
