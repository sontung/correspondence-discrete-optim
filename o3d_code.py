import open3d as o3d
import numpy as np
import cv2
import sys
from pathlib import Path
import copy

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])

IM1 = cv2.imread("data/im1.png")
IM2 = cv2.imread("data/im2.png")
IM1_masked = cv2.imread("data/im1masked.png")
IM2_masked = cv2.imread("data/im2masked.png")

my_file = Path("data/p1.txt")

if not my_file.is_file():
    sys.stdout = open("data/p1.txt", "w")
    for i in range(IM1_masked.shape[0]):
        for j in range(IM1_masked.shape[1]):
            if IM1_masked[i, j, 0] > 0:
                print(i, j, 0, IM1[i, j, 0]/255, IM1[i, j, 0]/255, IM1[i, j, 1]/255)

    sys.stdout = open("data/p2.txt", "w")
    for i in range(IM2_masked.shape[0]):
        for j in range(IM2_masked.shape[1]):
            if IM2_masked[i, j, 0] > 0:
                print(i, j, 0, IM2[i, j, 0]/255, IM2[i, j, 0]/255, IM2[i, j, 1]/255)
    sys.stdout = sys.__stdout__

source_down = o3d.io.read_point_cloud("data/p1.txt", "xyzrgb")
target_down = o3d.io.read_point_cloud("data/p2.txt", "xyzrgb")

radius = 100
current_transformation = np.array([
    [1.0860602, -0.18158359, 0, 657.68625465],
    [-0.41378524, 0.68242138, 0, 1392.08818249],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.]
])

current_transformation = np.array([
    [1, 0, 0, 657.68625465],
    [0, 1, 0, 1392.08818249],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.]
])

current_transformation = np.identity(4)

source_down.translate(-source_down.get_center()+target_down.get_center())

source_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
target_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))


result_icp = o3d.pipelines.registration.registration_colored_icp(
    source_down, target_down, radius, current_transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                               relative_rmse=1e-6,
                                                               max_iteration=50))
current_transformation = result_icp.transformation
print(result_icp, current_transformation)
source_down2 = copy.deepcopy(source_down)
source_down.transform(current_transformation)
o3d.visualization.draw_geometries([source_down, target_down])
