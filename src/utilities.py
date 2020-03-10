import cv2
import open3d as o3d
import numpy as np
import config
import matplotlib.pyplot as plt


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





def custom_draw_geometry_with_camera_trajectory(pcd):
    vis = o3d.visualization.Visualizer()
    vis.add_geometry(pcd)
    depth = vis.capture_depth_float_buffer(True)
    plt.imshow(depth)
    plt.imsave("depth.png", np.asarray(depth), dpi = 1)

def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
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

    # custom_draw_geometry_with_camera_trajectory.index = -1
    # custom_draw_geometry_with_camera_trajectory.trajectory =\
    #         o3d.io.read_pinhole_camera_trajectory(
    #                 "../../TestData/camera_trajectory.json")
    # custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    # )
    # if not os.path.exists("../../TestData/image/"):
    #     os.makedirs("../../TestData/image/")
    # if not os.path.exists("../../TestData/depth/"):
    #     os.makedirs("../../TestData/depth/")

    # def move_forward(vis):
    #     # This function is called within the o3d.visualization.Visualizer::run() loop
    #     # The run loop calls the function, then re-render
    #     # So the sequence in this function is to:
    #     # 1. Capture frame
    #     # 2. index++, check ending criteria
    #     # 3. Set camera
    #     # 4. (Re-render)
    #     ctr = vis.get_view_control()
    #     glb = custom_draw_geometry_with_camera_trajectory
    #     if glb.index >= 0:
    #         print("Capture image {:05d}".format(glb.index))
    #         depth = vis.capture_depth_float_buffer(False)
    #         image = vis.capture_screen_float_buffer(False)
    #         plt.imsave("../../TestData/depth/{:05d}.png".format(glb.index),\
    #                 np.asarray(depth), dpi = 1)
    #         plt.imsave("../../TestData/image/{:05d}.png".format(glb.index),\
    #                 np.asarray(image), dpi = 1)
    #         #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
    #         #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
    #     glb.index = glb.index + 1
    #     if glb.index < len(glb.trajectory.parameters):
    #         ctr.convert_from_pinhole_camera_parameters(
    #             glb.trajectory.parameters[glb.index])
    #     else:
    #         custom_draw_geometry_with_camera_trajectory.vis.\
    #                 register_animation_callback(None)
    #     return False

    # vis = custom_draw_geometry_with_camera_trajectory.vis
    # vis.create_window()
    # vis.add_geometry(pcd)
    # vis.get_render_option().load_from_json("../../TestData/renderoption.json")
    # vis.register_animation_callback(move_forward)
    # vis.run()
    # vis.destroy_window()

if __name__=='__main__':
    pcd = o3d.io.read_point_cloud("../output/final_point_cloud.ply")
    custom_draw_geometry(pcd)