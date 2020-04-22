import cv2
# import open3d as o3d
import numpy as np
import config
import matplotlib.pyplot as plt
# import process_extrinsics


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

def print_camera_params():
    '''
    Function that returns string output to be written in the bundle adjustment file for camera initialization
    '''
    camera_params = config.CAMERA_PARAMS
    content = '%d %d %d\n' % (camera_params['fx'], camera_params['k1'], camera_params['k2'])
    rotation = np.eye(3)
    translation = np.zeros(3)

    for i in range(3):
        rot = '%d %d %d\n' % (rotation[i, 0], rotation[i, 1], rotation[i, 2])
        content = content + rot

    content = content + '0 0 0\n'
    return content

def read_extrinsics_params(file):
    '''
    Function that reads a Camera Extrinsics file with Rodrigous parameters
    and outputs the parameters in a numpy array

    Input:
        file - File name

    Return:
        params - N*9 array of parameters
    '''
    data = np.genfromtxt(file, delimiter=',')
    data = np.delete(data, -1, 1)
    return data

def params_to_transfomation_mtx(params):
    '''
    Function that takes in the input Rodrigous parameters and
    outputs a transformation matrix

    Input:
        params - Rodrigous parameters
    Return:
        transformation - N * 4*4 transfomation matrices
    '''
    transformations = []
    for i in range(len(params)):
        param = params[i]
        rodrigous_rot = param[0:3]
        translation = param[3:6]
        focal_length = param[6:7]
        distortion_coeff = param[7:9]

        rotation_matrix, _ = cv2.Rodrigues(rodrigous_rot)
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3] = translation

        transformations.append(transformation_matrix)

    transformations = np.array(transformations)

    return transformations

def params_to_projection_mtx(params):
    '''
    Function that takes in the input Rodrigous parameters and
    outputs a projection matrix

    Input:
        params - Rodrigous parameters
    Return:
        projections - N * 3*4 projection matrices
    '''
    projections = []
    for i in range(len(params)):
        param = params[i]
        rodrigous_rot = param[0:3]
        translation = param[3:6]
        focal_length = param[6:7]
        distortion_coeff = param[7:9]

        rotation_matrix, _ = cv2.Rodrigues(rodrigous_rot)
        transformation_matrix = np.eye(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, 3] = translation

        K = construct_camera_matrix(camera_params)

        mat = transformation_matrix[:3, :]
        projection_matrix = K @ mat

        projections.append(projection_matrix)

    projections = np.array(projections)

    return projections

def get_transformations(file):
    params = read_extrinsics_params(file)
    transformations = params_to_transfomation_mtx(params)
    return transformations

def get_projections(file):
    params = read_extrinsics_params(file)
    projections = params_to_projection_mtx(params)
    return projections

def custom_draw_geometry_with_camera_trajectory(pcd):
    vis = o3d.visualization.Visualizer()
    vis.add_geometry(pcd)
    depth = vis.capture_depth_float_buffer(True)
    plt.imshow(depth)
    plt.imsave("depth.png", np.asarray(depth), dpi = 1)


def point_cloud_2_depth_map(pcd):
    '''
    Create depth map out of point cloud
    Input:
        pcd - point cloud object
    '''
    points_3D = np.asarray(pcd.points)

    points_3D = points_3D[ points_3D[:,2] > 0, :]
    points_3D = points_3D.T

    min_depth = np.min(points_3D[2, :])
    max_depth = np.max(points_3D[2, :])

    camera_params = config.CAMERA_PARAMS
    transformations = get_transformations(config.EXTRINSIC_FILE)

    K = construct_camera_matrix(camera_params)

    points_3D = np.vstack((points_3D, np.ones((1, points_3D.shape[1]))))

    image_coordinates = K @ (transformations[0][:3,:] @ points_3D)

    image_coordinates = np.int0(image_coordinates / image_coordinates[2, :])

    pixel_depth_val = 255 - ((points_3D[2, :] - min_depth) * 255 / (max_depth - min_depth))

    depth_image = np.zeros((camera_params['cy'] * 2, camera_params['cx'] * 2))

    height_image = int(camera_params['cy'] * 2)
    width_image = int(camera_params['cx'] * 2)

    point_in_view = 0
    for i in range(image_coordinates.shape[1]):
        if image_coordinates[1, i] < depth_image.shape[0] and image_coordinates[0, i] < depth_image.shape[1] and image_coordinates[0, i] >= 0 and image_coordinates[1, i] >= 0:
            depth_image[height_image - image_coordinates[1, i], width_image - image_coordinates[0, i]] = pixel_depth_val[i]
            point_in_view +=1

    plt.imsave(config.SPARSE_DEPTH_MAP, depth_image, cmap='gray')

    return depth_image



def custom_draw_geometry(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    depth = vis.capture_depth_float_buffer(True)
    vis.run()
    plt.imshow(depth)
    plt.show()
    plt.imsave("depthq.png", np.asarray(depth), dpi = 100, cmap='gray')
    vis.destroy_window()


if __name__=='__main__':
    pcd = o3d.io.read_point_cloud("../output/final_point_cloud.ply")
    # custom_draw_geometry(pcd)
    point_cloud_2_depth_map(pcd)
