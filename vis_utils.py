import cv2
import numpy as np
import open3d as o3d
from utils import read_correspondence, read_correspondence_from_dump
from math_utils import compute_epip_line


def vis_corr_res(im1_dir="data/im1.png", im2_dir="data/im2.png",
                 corr_text_file="data/corr-3.txt", 
                 init_corr_points="data/res-3.json"):
    """
    visualize each corr pair in the corr text
    :return:
    """
    im1 = cv2.imread(im1_dir)
    im2 = cv2.imread(im2_dir)
    p1, p2 = read_correspondence(init_corr_points)
    pairs = read_correspondence_from_dump(corr_text_file)

    point1 = []
    point2 = []
    for pair in pairs:
        x, y, x2, y2, _ = pair
        x, y, x2, y2 = map(int, (x, y, x2, y2))
        point1.append([y, x, 1])
        point2.append([y2, x2, 1])
    point1 = np.array(point1)
    point2 = np.array(point2)
        
    f_mat, _ = cv2.findFundamentalMat(np.int32(p1[:7]), np.int32(p2[:7]), cv2.FM_7POINT)
    epip_fitness = np.abs(np.sum((point1 @ f_mat) * point2, axis=1))

    for pair in pairs:
        x, y, x2, y2, _ = pair
        x, y, x2, y2 = map(int, (x, y, x2, y2))
        im12 = im1.copy()
        im22 = im2.copy()
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(im12, (y, x), 15, color, -1)
        cv2.circle(im22, (y2, x2), 15, color, -1)

        coeff = np.array([y, x, 1.0]) @ f_mat
        cv2.line(im22,
                 (int((coeff[1] * 1 + coeff[2]) / -coeff[0]), 1),
                 (int((coeff[1] * 1000 + coeff[2]) / -coeff[0]), 1000),
                 color, 10)

        im3 = np.hstack([im12, im22])
        cv2.imwrite("debugs/test.png", im3)
        break


def pointcloud_vis(pc_dir="/home/sontung/work/pc-folder"):
    all_shapes = []
    for i in range(75, 100, 3):

        pcd = o3d.io.read_point_cloud("%s/point_cloud-%d.txt" % (pc_dir, i), "xyzrgb")
        all_shapes.append(pcd)
        o3d.visualization.draw_geometries(all_shapes)


if __name__ == '__main__':
    pointcloud_vis()
    # vis_corr_res()