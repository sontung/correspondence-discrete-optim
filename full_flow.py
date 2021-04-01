import numpy as np
import cv2
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from math_utils import compute_zncc
from exact_algo import solve_simple, final_process
from PIL import Image


def sparse_correspondence(img_left=None, img_right=None):
    if img_left is None:
        img_left = imread("../data_light/images/opencv_frame_0.png")*imread("../data_light/images_mask/opencv_frame_0_bin.png")
        img_right = imread("../data_light/images/opencv_frame_1.png")*imread("../data_light/images_mask/opencv_frame_1_bin.png")
    img_left_gray, img_right_gray = map(rgb2gray, (img_left, img_right))

    descriptor_extractor = ORB()

    descriptor_extractor.detect_and_extract(img_left_gray)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img_right_gray)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left, descriptors_right,
                                cross_check=True)

    model, inliers = ransac((keypoints_left[matches[:, 0]],
                             keypoints_right[matches[:, 1]]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=5000)

    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

    total_zncc = []
    corr = []
    for i in range(inlier_keypoints_left.shape[0]):
        x, y = map(int, inlier_keypoints_left[i])
        x2, y2 = map(int, inlier_keypoints_right[i])
        score, _, _ = compute_zncc(x, y, x2, y2, img_left, img_right, 19)
        total_zncc.append(score)
        corr.append([x, y, x2, y2, score])
    print("avg. zncc", np.mean(total_zncc))
    f_mat, _ = cv2.findFundamentalMat(np.int32(inlier_keypoints_left), np.int32(inlier_keypoints_right), cv2.FM_LMEDS)
    return f_mat, corr


def dense_correspondence(img_left=None, img_right=None):
    f_mat, sparse_corr = sparse_correspondence()

    mask1 = imread("../data_light/images_mask/opencv_frame_0_bin.png")
    mask2 = imread("../data_light/images_mask/opencv_frame_1_bin.png")

    img_left = imread("../data_light/images/opencv_frame_0.png") * mask1
    img_right = imread("../data_light/images/opencv_frame_1.png") * mask2

    x_full = []
    for i in range(mask1.shape[0]):
        for j in range(mask1.shape[1]):
            if mask1[i, j, 0] > 0:
                x_full.append([i, j])

    y_full = []
    for i in range(mask2.shape[0]):
        for j in range(mask2.shape[1]):
            if mask2[i, j, 0] > 0:
                y_full.append([i, j])
    x_full = np.array(x_full)
    y_full = np.array(y_full)

    y_trans = solve_simple(sparse_corr, x_full, y_full)
    print("solving correspondences for %d pairs" % x_full.shape[0])
    final_process(y_full, y_trans, x_full, img_left, img_right, f_mat)


if __name__ == '__main__':
    dense_correspondence()
