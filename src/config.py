import cv2

# Image Directory
IMAGE_DIR = '../datasets/0016_still'

# Shi-Tomasi parameters
feature_params = dict(maxCorners = 300, 
                      qualityLevel = 0.2, 
                      minDistance = 2, 
                      blockSize = 7
                      )

# Lucas-Kanade parameters
lk_params = dict(winSize = (15,15), 
                 maxLevel = 2, 
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                 )
