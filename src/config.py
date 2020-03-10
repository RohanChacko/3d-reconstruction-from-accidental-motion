import cv2

# Image Directory
IMAGE_DIR = '../datasets/stone6_still'

# Initial Point Cloud 
INITIAL_POINT_CLOUD = '../output/initial_point_cloud.ply'

# FINAL Point Cloud
FINAL_POINT_CLOUD = '../output/final_point_cloud.ply'

# Bundle File
BUNDLE_FILE = '../output/bundle.out'

# Shi-Tomasi parameters
feature_params = dict(maxCorners = 5000, 
                      qualityLevel = 0.03, 
                      minDistance = 20, 
                      blockSize = 15
                      )

# Lucas-Kanade parameters
# lk_params = dict(
#                  winSize = (8,8), 
#                  maxLevel = 2, 
#                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#                  )

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

CAMERA_PARAMS = dict(fx=1781,
                     fy=1781,
                     cx=960,
                     cy=540,
                     k1=0,
                     k2=0,
                     s=0,
                    )
