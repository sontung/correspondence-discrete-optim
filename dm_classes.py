import numpy as np
import cv2
import sys
import pickle
import gc
import pybnb
import heapq
from math_utils import compute_zncc_min_version as compute_zncc
# from math_utils import compute_zncc

from corr_utils import evaluate_corr_pairs, compute_deviation
from utils import read_correspondence
from PIL import Image
from pathlib import Path
from copy import deepcopy
from multiprocessing import Queue, Process, Value, Lock


class Graph:
    def __init__(self, seed, mask1, mask2):
        min_cost_per_depth_to_be_stored = 0.5
        all_paths = []
        vertices = sorted(seed.keys())
        local_cost_mem = {}
        print(vertices)
        while True:
            print()

    @staticmethod
    def refine_stored_path(all_paths, path):
        new_all_path = []
        common_list = []
        for path2 in all_paths:
            common = len(list(set(path.return_path()).intersection(path2[1].return_path()))) / path2[1].return_depth()
            if common < 0.8:
                new_all_path.append(path2)
                common_list.append(common + 0.0001)
        new_all_path_ind = sorted(range(len(new_all_path)), key=lambda du3: new_all_path[du3][0] / common_list[du3])
        new_all_path = [new_all_path[ind] for ind in new_all_path_ind]
        return new_all_path


class GraphPath:
    def __init__(self):
        self.bound = 0
        self.path = []
        self.uv_map = {}
        self.total_epip = 0
        self.total_photo = 0
        self.current_level = 0
        self.fully_expanded = False

    def compare(self, another_path):
        return len(list(set(self.return_path()).intersection(another_path.return_path()))) / self.return_depth()

    def terminate(self):
        return self.fully_expanded

    def return_depth(self):
        return len(self.path)

    def clone(self):
        new_path = GraphPath()
        new_path.bound = self.bound
        new_path.path = self.path[:]
        new_path.uv_map = self.uv_map.copy()
        new_path.total_epip = self.total_epip
        new_path.total_photo = self.total_photo
        new_path.current_level = self.current_level
        new_path.fully_expanded = self.fully_expanded
        return new_path

    def expand(self, seed_maps, im1, im2, mask1, mask2, epip_matrix, zncc_dict):
        i = self.current_level + 1
        new_graphs = []
        if i in seed_maps:
            j = seed_maps[i][0]
            x, y = mask1[i]
            x2, y2 = mask2[j]

            for j in seed_maps[i][1:]:
                new_graph = self.clone()
                x2, y2 = mask2[j]
                new_pair = Pair(im1, im2, x, y, x2, y2, (i, j), epip_matrix, mask2, zncc_dict)
                new_graph.add(new_pair)
                new_graphs.append(new_graph)

            new_pair = Pair(im1, im2, x, y, x2, y2, (i, j), epip_matrix, mask2, zncc_dict)
            self.add(new_pair)
        else:
            self.fully_expanded = True
        return new_graphs

    def add(self, pair, compute_smooth=True):
        x, y, x2, y2 = pair.return_coord()
        self.uv_map[(x, y)] = (x2 - x, y2 - y)

        smooth = 0
        if compute_smooth:
            for (x, y) in self.uv_map:
                smooth += compute_deviation(x, y, self.uv_map)
        self.total_epip += pair.return_epip()
        self.total_photo += pair.return_photo()
        self.path.append(pair)
        old_bound = self.bound
        self.recompute_bound(smooth)
        assert old_bound < self.bound

    def recompute_bound_full(self):
        smooth = 0
        for (x, y) in self.uv_map:
            smooth += compute_deviation(x, y, self.uv_map)
        self.bound = self.total_photo + self.total_epip + smooth

    def recompute_bound(self, smooth):
        self.bound = self.total_photo + self.total_epip + smooth

    def return_bound(self):
        return self.bound

    def return_path(self):
        return self.path

    def return_bound_depth(self):
        return self.bound / self.return_depth()

    def dump_to_file(self, name="data/corr-dm.txt"):
        sys.stdout = open(name, "w")
        for pair in self.path:
            x, y, x2, y2 = pair.return_coord()
            print(x, y, x2, y2)
        sys.stdout = sys.__stdout__


class Pair:
    def __init__(self, im1, im2, x, y, x2, y2, pair_index, epip_matrix, mask2, zncc_dict):
        if (x, y, x2, y2) in zncc_dict:
            self.photo = zncc_dict[(x, y, x2, y2)]
        else:
            self.photo = compute_zncc(x, y, x2, y2, im1, im2, 19)[0]
            zncc_dict[(x, y, x2, y2)] = self.photo
        self.epip = epip_matrix[pair_index[0] * len(mask2) + pair_index[1]]
        self.point1 = (x, y)
        self.point2 = (x2, y2)
        self.pair_index = pair_index

    def return_index(self):
        return self.pair_index

    def return_photo(self):
        return self.photo

    def return_epip(self):
        return self.epip

    def return_coord(self):
        return self.point1[0], self.point1[1], self.point2[0], self.point2[1]

    def __str__(self):
        return "%d %d" % (self.pair_index[0], self.pair_index[1])

    def __eq__(self, other):
        return self.pair_index == other.pair_index

    def __hash__(self):
        return self.pair_index.__hash__()


class Graph2:
    def __init__(self):
        return


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
    g = Graph(MASK1, MASK2, IM1, IM2, epip_mat)
