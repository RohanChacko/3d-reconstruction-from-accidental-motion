import cv2
import matplotlib.pyplot as plt
from glob import glob
from klt_tracker import KLT_Tracker
from bundle_adjuster import BundleAdjuster
import config
from utilities import * 

if __name__ == '__main__':

    # Image Directory
    image_dir = config.IMAGE_DIR

    # Image Files
    image_files = sorted(glob(image_dir+'/*'))

    # Read Images
    images = []
    for image in image_files:
        images.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
        # images.append(cv2.imread(image))

    # Initialize KLT Tracker
    klt_tracker = KLT_Tracker(images, config.feature_params, config.lk_params, config.CAMERA_PARAMS)

    # Generate Optical Flow
    optical_flow = klt_tracker.generate_optical_flow()

    # Filter out Outliers
    optical_flow = klt_tracker.homography_filter()
    
    # back projecting rays in camera reference frame
    # points3D = back_project_points(klt_tracker.K, klt_tracker.reference_features.reshape(klt_tracker.reference_features.shape[0], 2))

    # print(points3D.shape)
    # Generate Bundle file
    

    # Draw Optical Flow
    klt_tracker.draw_optical_flow()

    # Generate Initialization Point Cloud
    klt_tracker.generate_initial_point_cloud(config.INITIAL_POINT_CLOUD)

    # Generate bundle adjustment input file
    klt_tracker.generate_bundle_file('../output/bundle.out')
    
    # Bundle Adjustment
    ceres_params = config.CERES_PARAMS
    bundle_adjuster = BundleAdjuster(config.INITIAL_POINT_CLOUD, 
                                     config.FINAL_POINT_CLOUD,
                                     config.BUNDLE_FILE,
                                     config.CERES_PARAMS)

    bundle_adjuster.bundle_adjust()
