import numpy as np
import cv2
import open3d as o3d 
import copy  

def read_extrinsics_params(file):
    data = np.genfromtxt(file, delimiter=',')
    return np.delete(data, -1, 1)

def params_to_transfomation_mtx(params):
    transformations = []
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

        K = np.array([
            [focal_length, 0, 0],
            [0, focal_length, 0],
            [0,            0, 1],
        ])
        # print(transformation_matrix[0:3,:])
        mat = transformation_matrix[:3, :]
        projection_matrix = K @ mat
        
        transformations.append(transformation_matrix)
        projections.append(projection_matrix)
    
    transformations = np.array(transformations)
    projections = np.array(projections)

    return transformations, projections

if __name__ == '__main__':
    params = read_extrinsics_params('./extrinsics.txt')
    transformations, projections = params_to_transfomation_mtx(params)
    print(transformations[0])