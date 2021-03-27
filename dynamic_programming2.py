from algo import ssd_compute_seeding
from pathlib import Path
from utils import read_correspondence, compute_mask
from dm_classes import Graph
import pickle
import cv2
import numpy as np


im1 = cv2.imread("data/im1.png")
im2 = cv2.imread("data/im2.png")
im1_masked = cv2.imread("data/im1masked.png")
im2_masked = cv2.imread("data/im2masked.png")

mask_file = Path("saved/dm-mask.pkl")
if mask_file.exists():
    with open("saved/dm-mask.pkl", "rb") as a_file:
        [mask1, mask2] = pickle.load(a_file)
else:
    mask1 = []
    for x in range(im1.shape[0]):
        for y in range(im1.shape[1]):
            if im1_masked[x, y, 0] > 0:
                mask1.append((x, y))

    mask2 = []
    for x in range(im2.shape[0]):
        for y in range(im2.shape[1]):
            if im2_masked[x, y, 0] > 0:
                mask2.append((x, y))

    with open("saved/dm-mask.pkl", "wb") as a_file:
        pickle.dump([mask1, mask2], a_file)


seed_file = Path("saved/dm-seed.pkl")
if seed_file.is_file():
    with open("saved/dm-seed.pkl", "rb") as f:
        seed = pickle.load(f)
else:

    p1, p2 = read_correspondence()
    F_MAT, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)

    seed = ssd_compute_seeding(im1, im2, mask1, mask2, F_MAT)
    with open("saved/dm-seed.pkl", "wb") as f:
        pickle.dump(seed, f)

g = Graph(seed)