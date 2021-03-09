from __future__ import division
import numpy as np
import cv2
import math
from matplotlib import pyplot
from scipy.optimize import leastsq


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C


def find_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    a = (y2-y1)/(x2-x1)*1.0
    c = y1-x1*a
    b = -1.0
    return a, b, c


def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return int(x), int(y)
    else:
        return False


def shoelace(x, y):
    S1 = np.sum(x*np.roll(y,-1))
    S2 = np.sum(y*np.roll(x,-1))
    area = .5*np.absolute(S1 - S2)
    return area


def shoelace_slow(vertices):
    (x1, y1), (x2, y2), (x3, y3) = vertices
    ref_res = shoelace(np.array([x1, x2, x3]), np.array([y1, y2, y3]))
    return ref_res


def check_inside_tri(bary):
    return abs(np.sum(bary) - 1) <= 0.01


def cartesian2bary(point, vertices):
    tri = []
    x, y = point
    for v in vertices:
        temp_list = list(vertices)
        temp_list.remove(v)
        tri.append([v, temp_list])
    total = shoelace_slow(vertices)
    color = [shoelace_slow([(x, y), du[1][0], du[1][1]]) / total for du in tri]
    assert total > 0, vertices
    return np.array(color), check_inside_tri(color)


def bary_quad(vertices=((818-300, 383-300), (874-300, 315-300), (878-300, 384-300))):
    image = np.zeros((900-300, 400-300, 3))
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            b, r = cartesian2bary((x, y), vertices)
            if r:
                image[x, y] = b*255
            else:
                image[x, y] = 0

    cv2.imwrite("test.png", image)


def compute_rot_mat(x_ax=0, y_ax=0, z_ax=0):
    x_ax = math.radians(x_ax)
    y_ax = math.radians(y_ax)
    z_ax = math.radians(z_ax)
    r_x = np.array([[1, 0, 0, 0],
                    [0, math.cos(x_ax), -math.sin(x_ax), 0],
                    [0, math.sin(x_ax), math.cos(x_ax), 0],
                    [0, 0, 0, 1]])
    r_y = np.array([[math.cos(y_ax), 0, math.sin(y_ax), 0],
                    [0, 1, 0, 0],
                    [-math.sin(y_ax), 0, math.cos(y_ax), 0],
                    [0, 0, 0, 1]])
    r_z = np.array([[math.cos(z_ax), -math.sin(z_ax), 0, 0],
                    [math.sin(z_ax), math.cos(z_ax), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])

    res = r_x@r_y@r_z
    return res


def plane_fitting(xyz):
    p0 = [0.506645455682, -0.185724560275, -1.43998120646, 1.37626378129]

    def f_min(X,p):
        plane_xyz = p[0:3]
        distance = (plane_xyz*X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)

    def residuals(params, signal, X):
        return f_min(X, params)

    sol = leastsq(residuals, p0, args=(None, xyz))[0]

    return sol


def compute_zncc(x, y, x2, y2, f, g, window_size):
    f = f[x-window_size: x+window_size+1, y-window_size: y+window_size+1]
    g = g[x2-window_size: x2+window_size+1, y2-window_size: y2+window_size+1]
    f__ = f.copy()
    g__ = g.copy()
    f_ = np.mean(f.reshape((-1, 3)), axis=0)
    g_ = np.mean(g.reshape((-1, 3)), axis=0)
    f = f-f_
    g = g-g_

    du1 = np.sum(np.multiply(f, g))
    du2 = np.sum(np.multiply(f, f))
    du3 = np.sum(np.multiply(g, g))
    s2 = du1 / (math.sqrt(du2 * du3) + 0.00001)
    if s2 > 1 or s2 < -1:
        print(s2)
    return s2, f__, g__


def compute_zncc_min_version(x, y, x2, y2, f, g, window_size, debug=False):
    f = f[x-window_size: x+window_size+1, y-window_size: y+window_size+1]
    g = g[x2-window_size: x2+window_size+1, y2-window_size: y2+window_size+1]

    f__, g__ = None, None
    if debug:
        f__ = f.copy()
        g__ = g.copy()
    f_ = np.mean(f.reshape((-1, 3)), axis=0)
    g_ = np.mean(g.reshape((-1, 3)), axis=0)
    f = f-f_
    g = g-g_

    du1 = np.sum(np.multiply(f, g))
    du2 = np.sum(np.multiply(f, f))
    du3 = np.sum(np.multiply(g, g))
    s2 = du1 / (math.sqrt(du2 * du3) + 0.00001)
    return 1-s2, f__, g__


def compute_epip_line(f_mat, yx1):
    f_mat = f_mat[:3, :]
    return cv2.computeCorrespondEpilines(yx1, 2, f_mat)


def compute_epip_fitness(z, coeff):
    return abs(np.dot(z, coeff))


if __name__ == '__main__':
    bary_quad()
