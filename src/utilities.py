import cv2
import open3d as o3d
import numpy as np


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def write_point_cloud(file_name, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(file_name, pcd)

def construct_camera_matrix(camera_params):
    K = np.array([
        [camera_params['fx'],  camera_params['s'], camera_params['cx']],
        [                  0, camera_params['fy'], camera_params['cy']],
        [                  0,                   0,                  1],
    ])

    return K 

def back_project_points(K, imagePts):
    '''
    Function to back-project rays in camera frame

    Input:
        K - 3 x 3 - camera intrinsic matrix
        imagePts - N x 2 - image pixels

    Returns:
        points3D - N x 3 - 3D rays in camera frame
    '''

    imageHomogeneousPts = np.hstack((imagePts, np.ones((imagePts.shape[0], 1))))
    invK = np.linalg.inv(K)
    points3D = invK @ imageHomogeneousPts.T
    points3D = points3D.T

    return points3D
