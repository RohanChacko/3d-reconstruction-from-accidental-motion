import numpy as np
import cv2
import open3d as o3d 
import copy  

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
    
if __name__ == '__main__':
    params = read_extrinsics_params('./extrinsics.txt')
    transformations, projections = params_to_transfomation_mtx(params)
    print(transformations[0])