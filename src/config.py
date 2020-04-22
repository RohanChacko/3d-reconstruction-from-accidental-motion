import cv2
import open3d as o3d
# Image Directory

IMAGE_DIR = '../datasets/{}_still'
# IMAGE_DIR = '../datasets/stone6_still'

OUTPUT_FOLDER = '../output/'

# Extrinsics File
EXTRINSIC_FILE = '../output/extrinsics.csv'
# EXTRINSIC_FILE = '../output/{}/exp1/extrinsics.csv'
# NUM_IMAGES = 30

# Initial Point Cloud
INITIAL_POINT_CLOUD = '../output/initial_point_cloud.ply'

# FINAL Point Cloud
FINAL_POINT_CLOUD = '../output/final_point_cloud.ply'

# Bundle File
BUNDLE_FILE = '../output/bundle.out'

# Optical Flow Plot
OPTICAL_FLOW_PLOT = '../output/optical_flow.png'

# Sparse Depth Map
SPARSE_DEPTH_MAP = '../output/sparse_depth_map.png'

# Shi-Tomasi parameters
feature_params = dict(maxCorners = 2000,
                      qualityLevel = 0.03,
                      minDistance = 10,
                      blockSize = 15
                      )

# Lucas-Kanade parameters
lk_params = dict(   winSize  = (25,25),
                 	maxLevel = 8,
            		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.3))
# Ceres-Solver parameters
CERES_PARAMS = dict(
                    solver = '../ceres-bin/bin/bundle_adjuster',
                    maxIterations = 1000,
                    input_ply = '../output/initial.ply',
                    output_ply = '../output/final.ply',
                    inner_iterations = 'true',
                    nonmonotonic_steps = 'false'
                    )

CAMERA_PARAMS = dict(fx=1781.0,
                     fy=1781.0,
                     cx=240,
                     cy=424,
                     k1=0,
                     k2=0,
                     s=0,
                    )
PS_PARAMS = dict(max_depth=4,
                min_depth=2,
                scale=2,
                patch_radius=1)

CRF_PARAMS = dict(iters=100,
                pos_std="3,3",
                rgb_std="20,20,20",
                weight=1,
                max_penalty=0.15)
