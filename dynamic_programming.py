import numpy as np
import cv2
import sys
import pickle
from math_utils import compute_zncc
from corr_utils import evaluate_corr_pairs, compute_deviation
from utils import read_correspondence
from PIL import Image
from pathlib import Path


class Graph:
    def __init__(self, data_, mask1, mask2, im1, im2, epip_matrix):
        all_paths = []
        data_index = 0
        best_bound = 0
        best_path_index = -1

        for i in range(len(mask1)):
            for j in range(len(mask2)):
                print(i, j)
                x, y = mask1[i]
                x2, y2 = mask2[j]
                a_pair = Pair(im1, im2, x, y, x2, y2, (i, j), epip_matrix, mask2)
                if a_pair.return_epip() < 0.5:
                    a_path = GraphPath(mask1)
                    a_path.add(a_pair)

    def explore_path(self, path):
        

    @staticmethod
    def find_children(i, mask1, mask2, im1, im2, epip_matrix):
        children = []
        if i+1 < len(mask1):
            x, y = mask1[i+1]
            for j in range(len(mask2)):
                x2, y2 = mask2[j]
                children.append(Pair(im1, im2, x, y, x2, y2, (i, j), epip_matrix, mask2))
        return children


class GraphPath:
    def __init__(self, mask1):
        self.mask_size = len(mask1)
        self.bound = -len(mask1)*1
        self.path = []
        self.uv_map = {}
        self.total_epip = 0
        self.total_photo = 0

    def add(self, pair):
        x, y, x2, y2 = pair.return_coord()
        assert (x, y) not in self.uv_map
        self.uv_map[(x, y)] = (x2-x, y2-y)

        smooth = 0
        for (x, y) in self.uv_map:
            smooth += compute_deviation(x, y, self.uv_map)
        self.total_epip += pair.return_epip()
        self.total_photo += pair.return_photo()
        self.path.append(pair)
        self.recompute_bound(smooth)

    def recompute_bound(self, smooth):
        self.bound = -self.total_photo + self.total_epip + smooth - (self.mask_size-len(self.path))

    def return_bound(self):
        return self.bound


class Pair:
    def __init__(self, im1, im2, x, y, x2, y2, pair_index, epip_matrix, mask2):
        self.photo = compute_zncc(x, y, x2, y2, im1, im2, 19)[0]
        self.epip = epip_matrix[pair_index[0]*len(mask2)+pair_index[1]]
        self.point1 = (x, y)
        self.point2 = (x2, y2)

    def return_photo(self):
        return self.photo

    def return_epip(self):
        return self.epip

    def return_coord(self):
        return self.point1[0], self.point1[1], self.point2[0], self.point2[1]



if __name__ == '__main__':
    IM1 = cv2.imread("data/im1.png")
    IM2 = cv2.imread("data/im2.png")
    IM1_masked = cv2.imread("data/im1masked.png")
    IM2_masked = cv2.imread("data/im2masked.png")

    my_file = Path("saved/mask.pkl")

    if my_file.is_file():
        with open("saved/mask.pkl", "rb") as f:
            [MASK1, MASK2] = pickle.load(f)
    else:
        im_mask11 = np.zeros_like(IM1)
        MASK1 = []
        mask1_x = []
        mask1_y = []
        for x in range(IM1.shape[0]):
            for y in range(IM1.shape[1]):
                if IM1_masked[x, y, 0] > 0:
                    mask1_x.append(x)
                    mask1_y.append(y)
        center_x = int(np.mean(mask1_x))
        center_y = int(np.mean(mask1_y))
        print(center_x, center_y)
        for du1 in range(center_x - 5, center_x + 5):
            for du2 in range(center_y - 5, center_y + 5):
                MASK1.append((du1, du2))
                im_mask11[du2, du1] = IM1[du2, du1]

        im_mask22 = np.zeros_like(IM1)
        MASK2 = []
        mask2_x = []
        mask2_y = []
        for x in range(IM2.shape[0]):
            for y in range(IM2.shape[1]):
                if IM2_masked[x, y, 0] > 0:
                    mask2_x.append(x)
                    mask2_y.append(y)

        center_x = int(np.mean(mask2_x))
        center_y = int(np.mean(mask2_y))
        print(center_x, center_y)
        for du1 in range(center_x - 50, center_x + 50):
            for du2 in range(center_y - 50, center_y + 50):
                MASK2.append((du1, du2))
                im_mask22[du1, du2, :] = IM2[du1, du2, :]
        with open("saved/mask.pkl", "wb") as f:
            pickle.dump([MASK1, MASK2], f)

    p1, p2 = read_correspondence()
    F_MAT, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)
    pixel_p = []
    pixel_q = []
    data = []
    for p_index, (x, y) in enumerate(MASK1):
        for q_index, (x2, y2) in enumerate(MASK2):
            data.append([x, y, x2, y2, p_index, q_index])
            pixel_p.append([y, x, 1])
            pixel_q.append([y2, x2, 1])
    pixel_p = np.array(pixel_p)
    pixel_q = np.array(pixel_q)
    epip_mat = np.abs(np.sum((pixel_p @ F_MAT) * pixel_q, axis=1))
    g = Graph(data, MASK1, MASK2, IM1, IM2, epip_mat)