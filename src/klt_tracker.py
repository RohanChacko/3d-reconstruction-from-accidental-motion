import numpy as np
import cv2


class KLT_Tracker:
    
    def __init__(self, images, feature_params, lk_params):
        self.reference_image = images[0]
        self.images = images[1:]
        self.reference_features = None
        self.optical_flow = None
        self.feature_params = feature_params
        self.lk_params = lk_params
        pass

    def get_features(self):

        pass

    def track_features(self):
        pass
    
    def homography_filter(self, threshold):
        '''
        Function to remove outliers points from optical flow
        '''
        no_of_cams = len(self.optical_flow[0])
        no_of_pts = len(self.optical_flow)

        image_pts = np.zeros((no_of_cams, no_of_pts, 2))
        for i in range(no_of_pts):
            for j in range(no_of_cams):
                image_pts[j, i, 0] = self.optical_flow[i][j][0]
                image_pts[j, i, 1] = self.optical_flow[i][j][1]

        reference_image_pts = image_pts[0, :, :]    

        for j in range(1, no_of_cams):
            
            homography_matrix = cv2.findhomography() 



        
        pass
