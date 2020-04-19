import cv2
import matplotlib.pyplot as plt
import numpy as np
from utilities import *
from itertools import compress
import config

class KLT_Tracker:
    
    def __init__(self, images, feature_params, lk_params, camera_params):
        self.reference_image = images[0]
        self.images = images[1:]
        self.feature_params = feature_params
        self.lk_params = lk_params
        self.reference_features = self.get_features()
        self.optical_flow = [[(i.ravel()[0], i.ravel()[1])] for i in self.reference_features]
        self.K = construct_camera_matrix(camera_params)
        self.reference_features_world_points = None
        self.reference_features_textures = None

    def get_features(self):
        return cv2.goodFeaturesToTrack(gray(self.reference_image),
                                mask = None,
                                **self.feature_params)

    def generate_optical_flow(self):


        optical_flow = self.optical_flow
        reference_features = self.reference_features
        for idx, current_image in enumerate(self.images):

            #if (idx+1) % 15 == 0 :
            #    reference_features = cv2.goodFeaturesToTrack(gray(self.reference_image), mask = None, **self.feature_params)
            #    optical_flow = [[(i.ravel()[0], i.ravel()[1])] for i in reference_features]

            current_features, status, error = cv2.calcOpticalFlowPyrLK(gray(self.reference_image), 
                                                                       gray(current_image), 
                                                                       reference_features, 
                                                                       None, 
                                                                       **self.lk_params)

            reference_features = reference_features[status == 1]
            current_features = current_features[status == 1]

            optical_flow = [optical_flow[i] for i, j in enumerate(status) if j == 1]

            for i, feature in enumerate(current_features):
                optical_flow[i].append((feature.ravel()[0], feature.ravel()[1]))

            reference_features = reference_features.reshape((reference_features.shape[0],1,2))

        self.reference_features = reference_features
        self.optical_flow = optical_flow

        return optical_flow
        
    def draw_optical_flow(self):
        image = self.reference_image.copy()
        mask = np.zeros_like(image)
        for feature in self.optical_flow:
            feature = np.array(feature, np.int32).reshape((-1,1,2))
            cv2.polylines(mask, [feature], False, (255,0,0))
        image = cv2.add(image, mask)
        plt.imsave(config.OPTICAL_FLOW_PLOT, image)
    
    def homography_filter(self, threshold = 0.95):
        '''
        Function to remove outliers points from optical flow
        '''
        
        no_of_cams = len(self.optical_flow[0])
        no_of_pts = len(self.optical_flow)

        image_pts = np.zeros((no_of_cams, no_of_pts, 2))

        # creating image_pts with dimensions as camId, pointId, 2
        for i in range(no_of_pts):
            for j in range(no_of_cams):

                image_pts[j, i, 0] = self.optical_flow[i][j][0]
                image_pts[j, i, 1] = self.optical_flow[i][j][1]

        reference_image_pts = image_pts[0, :, :]    

        mask = np.zeros((no_of_pts, 1))
        
        # calculating the number of frames each point is an inlier in
        for j in range(1, no_of_cams):
            
            homography_matrix, inliers = cv2.findHomography(image_pts[j, :, :], reference_image_pts, cv2.RANSAC, 5.0)
            mask = mask + inliers
        
        # mask ensuring points present in cameras below the threshold percentage are removed 

        mask = (mask >= threshold * no_of_cams)
        # print(mask[:,0])
        reference_image_pts = reference_image_pts[mask[:, 0], :]
        # print(reference_image_pts.shape)
        self.optical_flow = list(compress(self.optical_flow, mask))
        self.reference_features = np.reshape(reference_image_pts, (reference_image_pts.shape[0], 1, 2))
        
        return self.optical_flow

    def generate_initial_point_cloud(self, point_cloud_path):
        reference_features = self.reference_features.reshape(self.reference_features.shape[0], 2).astype('uint32')
        reference_features_textures = (self.reference_image[reference_features[:,1], reference_features[:,0], :] ).astype('float64')


        # no_of_cams = len(self.optical_flow[0])
        # no_of_pts = len(self.optical_flow)
        # image_pts = np.zeros((no_of_cams, no_of_pts, 2))

        # # creating image_pts with dimensions as camId, pointId, 2
        # for i in range(no_of_pts):
        #     for j in range(no_of_cams):

        #         image_pts[j, i, 0] = self.optical_flow[i][j][0]
        #         image_pts[j, i, 1] = self.optical_flow[i][j][1]

        # reference_features = image_pts[0, :, :].astype('uint16') 
        # reference_features_textures = (self.reference_image[reference_features[:,0], reference_features[:,1], :]).astype('uint32')
        reference_features_points = np.concatenate((reference_features, np.zeros((reference_features.shape[0], 1))), axis =1)
        # print(reference_features)
        # point_map = np.zeros((0,3))
        # color_map = np.zeros((0,3))
        # print(self.reference_image.shape)

        # for i in range(0,1080,5):
        #     print(i)
        #     for j in range(0,1920,5):
        #         point_map = np.concatenate((point_map, np.array([[i, j, 0]])))
        #         color_map = np.concatenate((color_map, [self.reference_image[i,j,:]]))
        
        # print(point_map.shape)
        # color_map = color_map/255
        # for i in range(len(color_map)):
        #     print(color_map[i])
        # color_map = color_map.astype('uint32')
        points3D = np.hstack((reference_features, np.ones((reference_features.shape[0],1))))

        # points3D = back_project_points(self.K, reference_features)
        points3D = points3D.T
        # print(points3D, 'points3D')
        
        points3D = points3D / points3D[2, :]
        depthVector = np.random.uniform(2, 4, (points3D.shape[1]))
        # points3D[:,2] = points3D[:,2] * 700
        
        
        cam_params = config.CAMERA_PARAMS
        h = cam_params['cy'] * 2
        w = cam_params['cx'] * 2
        points3D[0, :] = w / 2 - points3D[0,:] 
        points3D[1, :] = h / 2 - points3D[1,:] 
        # points3D[:2, :] = points3D[:2, :] / config.CAMERA_PARAMS['fx']
        # points3D[2,:] = points3D[2,:] * 700 #config.CAMERA_PARAMS['fx'] / depthVector

        points3D[2,:] = points3D[2,:] * config.CAMERA_PARAMS['fx'] / depthVector
        points3D = points3D.T
        # reference_features_points = np.concatenate((reference_features, np.random.uniform(2, 4, (reference_features.shape[0], 1))), axis =1)
        # print(points3D, 'points3D')
        self.reference_features_world_points = points3D
        self.reference_features_textures = reference_features_textures

        # Scale the points correctly
        # write_point_cloud(point_cloud_path, reference_features_points, reference_features_textures)
        # write_point_cloud(point_cloud_path, points3D, reference_features_textures)
        # write_point_cloud(point_cloud_path, point_map, color_map)

    def generate_bundle_file(self, bundle_file_path):
        '''
        Function to create bundle file for ceres solver
        '''

        f = open(bundle_file_path, 'w')
        num_of_cam = len(self.optical_flow[0])
        num_of_pts = len(self.optical_flow)
        
        # printing number of cameras and points
        content = '%d %d\n' % (num_of_cam, num_of_pts)
        f.write(content)

        content = print_camera_params()
        file_content = ''

        # printing camera initializations
        for i in range(num_of_cam):
            file_content = file_content + content
        
        f.write(file_content)
        # print(self.reference_features_world_points)
        file_content = ''
        for pt in range(num_of_pts):
            
            point = self.reference_features_world_points[pt, :]
            color = self.reference_features_textures[pt, :]
            content = '%f %f %f\n %d %d %d\n' % (point[0], point[1], point[2], color[0], color[1], color[2])
            
            for cam in range(num_of_cam):
                # print(pt, cam)
                # print(self.optical_flow[pt][cam])
                contentLine = '%d %d %d %d ' % (cam, pt*num_of_cam + cam, config.CAMERA_PARAMS['cx'] - self.optical_flow[pt][cam][0], config.CAMERA_PARAMS['cy'] - self.optical_flow[pt][cam][1])

                # contentLine = '%d %d %d %d ' % (cam, pt*num_of_cam + cam, self.optical_flow[pt][cam][0], self.optical_flow[pt][cam][1])
                content = content + contentLine

            content = content + '\n'     
            file_content = file_content + content

        f.write(file_content)
        f.close()  
        pass
    



