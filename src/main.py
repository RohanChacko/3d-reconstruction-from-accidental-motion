import cv2
import matplotlib.pyplot as plt
from glob import glob
import open3d as o3d
from klt_tracker import KLT_Tracker
#from dense_crf import DenseCRF
from bundle_adjuster import BundleAdjuster
import config
import dense_crf
from utilities import * 
import argparse


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

    # Read the point cloud generated after Bundle Adjustment
    pcd = o3d.io.read_point_cloud(config.FINAL_POINT_CLOUD)

    # Extract depth from the point cloud
    depth_map = point_cloud_2_depth_map(pcd)

    # run plane sweep and CRF optimization

    folder = config.IMAGE_DIR
    parser = argparse.ArgumentParser()
	# General Params
    
    parser.add_argument("--folder", help='sub-directory in dataset dir', default="{}".format(folder.split('/')[-1]), required=False)
    parser.add_argument("--nsamples", help='Number of depth samples', default=64, required=False)
    parser.add_argument("--pc_cost", help='Path to photoconsistency cost array', default=None)
    parser.add_argument("--show_unary", help='Save depth map with just unary (photoconsistency score) potentials', default=False)

	# CRF Params
    parser.add_argument("--iters", help='Number of iters for CRF inference', default=config.CRF_PARAMS['iters'])
    parser.add_argument("--p_std", help='Std. dev of positional term', default=config.CRF_PARAMS['pos_std'])
    parser.add_argument("--c_std", help='Std. dev of color term', default=config.CRF_PARAMS['rgb_std'])
    parser.add_argument("--wt", help='Weight for truncated linear', default=config.CRF_PARAMS['weight'])
    parser.add_argument("--max_p", help='Max Penalty for truncated linear', default=config.CRF_PARAMS['max_penalty'])

    # Plane sweep Params
    parser.add_argument("--max_d", help='Max depth of computed of 3D scene', default=config.PS_PARAMS['max_depth'])
    parser.add_argument("--min_d", help='Min depth of computed of 3D scene', default=config.PS_PARAMS['min_depth'])
    parser.add_argument("--scale", help='Scale of image (downsampling)', default=config.PS_PARAMS['scale'])
    parser.add_argument("--patch_rad", help='Patch radius for photoconsistency', default=config.PS_PARAMS['patch_radius'])

    args = parser.parse_args()

    dense_crf.dense_depth(args)

