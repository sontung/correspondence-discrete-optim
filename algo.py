import numpy as np
import cv2
import sys
import io
import pickle
import cProfile
import pstats
from math_utils import compute_zncc
from corr_utils import evaluate_corr_pairs
from utils import read_correspondence, compute_mask
from PIL import Image
from pathlib import Path
from dynamic_programming import Graph, GraphPath, Pair


def ssd_algo_dense_fast(img1, img2, mask1, mask2, fundamental_mat):
    if fundamental_mat.shape[0] > 3:
        fundamental_mat = fundamental_mat[:3, :]
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    ws = 50
    mask2_mat = []
    for x2, y2 in mask2:
        mask2_mat.append([y2, x2, 1])
    mask2_mat = np.array(mask2_mat)
    correspondences = []
    for count in range(0, len(mask1)):
        x, y = mask1[count]
        coeff = np.array([y, x, 1]) @ fundamental_mat
        epip = np.abs(np.sum(coeff*mask2_mat, axis=1))

        max_score = None
        best_match = None

        upper_limit = (-np.min(epip)+np.max(epip))*0.01 + np.min(epip)
        epip_fitness_test = (epip <= upper_limit).astype(np.int)
        for mask2_idx, data in enumerate(mask2):
            x2, y2 = data
            if epip_fitness_test[mask2_idx] == 1:
                score, f__, g__ = compute_zncc(x, y, x2, y2, img1, img2, ws)
                if best_match is None or score > max_score:
                    max_score = score
                    best_match = (x2, y2, f__, g__)

        x_corr, y_corr, f__, g__ = best_match
        Image.fromarray(np.hstack([f__, g__]).astype(np.uint8)).save("debugs/%d.png" % count)

        correspondences.append([x, y, x_corr, y_corr])

    with open("data/corr-ssd.txt", "w") as f:
        for x, y, x_corr, y_corr in correspondences:
            print(x, y, x_corr, y_corr, file=f)

    p, e, s = evaluate_corr_pairs(correspondences, img1, img2, fundamental_mat)
    print("ssd results", p, e, s)

    return correspondences


def ssd_compute_seeding(img1, img2, mask1, mask2, fundamental_mat):
    if fundamental_mat.shape[0] > 3:
        fundamental_mat = fundamental_mat[:3, :]
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    ws = 19
    mask2_mat = []
    for x2, y2 in mask2:
        mask2_mat.append([y2, x2, 1])
    mask2_mat = np.array(mask2_mat)
    seeds = {}
    for count in range(0, len(mask1)):
        x, y = mask1[count]
        coeff = np.array([y, x, 1]) @ fundamental_mat
        epip = np.abs(np.sum(coeff*mask2_mat, axis=1))

        upper_limit = (-np.min(epip)+np.max(epip))*0.1 + np.min(epip)
        epip_fitness_test = (epip <= upper_limit).astype(np.int)
        zncc_list = []
        for mask2_idx, data in enumerate(mask2):
            x2, y2 = data
            if epip_fitness_test[mask2_idx] == 1:
                score, f__, g__ = compute_zncc(x, y, x2, y2, img1, img2, ws)
                zncc_list.append((mask2_idx, score))
        zncc_list = sorted(zncc_list, key=lambda du: du[-1], reverse=True)
        print(zncc_list[0])
        seeds[count] = [du[0] for du in zncc_list[:50]]

    return seeds


def dynamic_programming_algo(img1, img2, mask1, mask2, fundamental_mat):
    pixel_p = []
    pixel_q = []
    for p_index, (x, y) in enumerate(mask1):
        for q_index, (x2, y2) in enumerate(mask2):
            pixel_p.append([y, x, 1])
            pixel_q.append([y2, x2, 1])
    pixel_p = np.array(pixel_p)
    pixel_q = np.array(pixel_q)
    epip_mat = np.abs(np.sum((pixel_p @ fundamental_mat) * pixel_q, axis=1))

    seed_file = Path("saved/seeds.pkl")
    if seed_file.exists():
        with open("saved/seeds.pkl", "rb") as seed_f:
            seeds = pickle.load(seed_f)
    else:
        seeds = ssd_compute_seeding(img1, img2, mask1, mask2, fundamental_mat)
        with open("saved/seeds.pkl", "wb") as seed_f:
            pickle.dump(seeds, seed_f)
    print("seeding done")

    profiler = cProfile.Profile()
    profiler.enable()
    dm = Graph(mask1, mask2, img1, img2, epip_mat, seeds, 0.5*len(mask1), 1)
    profiler.disable()
    s = io.StringIO()
    sortby = 'time'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    ps.print_stats(100)
    print(s.getvalue())


def run_ssd():
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
    ssd_algo_dense_fast(IM1, IM2, MASK1, MASK2, F_MAT)


def run_dm():
    IM1 = cv2.imread("data/im1.png")
    IM2 = cv2.imread("data/im2.png")
    IM1_masked = cv2.imread("data/im1masked.png")
    IM2_masked = cv2.imread("data/im2masked.png")

    mask_file = Path("saved/mask.pkl")
    if mask_file.exists():
        with open("saved/mask.pkl", "rb") as seed_f:
            [MASK1, MASK2] = pickle.load(seed_f)
    else:
        [MASK1, MASK2] = compute_mask(IM1, IM2, IM1_masked, IM2_masked)
        with open("saved/mask.pkl", "wb") as seed_f:
            pickle.dump([MASK1, MASK2], seed_f)
    p1, p2 = read_correspondence()
    F_MAT, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)
    dynamic_programming_algo(IM1, IM2, MASK1, MASK2, F_MAT)


if __name__ == '__main__':
    run_dm()
