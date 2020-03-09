import cv2

# Image Directory
IMAGE_DIR = '../datasets/0034_still'

# Initial Point Cloud 
INITIAL_POINT_CLOUD = '../output/initial_point_cloud.ply'

# FINAL Point Cloud
FINAL_POINT_CLOUD = '../output/final_point_cloud.ply'

# Bundle File
BUNDLE_FILE = '../output/bundle.out'

# Shi-Tomasi parameters
feature_params = dict(maxCorners = 800, 
                      qualityLevel = 0.05, 
                      minDistance = 10, 
                      blockSize = 11
                      )

# Lucas-Kanade parameters
lk_params = dict(
                 winSize = (15,15), 
                 maxLevel = 2, 
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                 )

# Ceres-Solver parameters
CERES_PARAMS = dict(
                    solver = '../ceres-bin/bin/bundle_adjuster',
                    maxIterations = 100,
                    input_ply = '',
                    inner_iterations = 'true',
                    nonmonotonic_steps = 'false'
                    )

CAMERA_PARAMS = dict(fx=1781,
                     fy=1781,
                     cx=960,
                     cy=540,
                     k1=0,
                     k2=0,
                     s=0,
                    )
