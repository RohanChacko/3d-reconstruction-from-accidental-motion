import cv2
import matplotlib.pyplot as plt
from glob import glob
from klt_tracker import KLT_Tracker
import config

if __name__ == '__main__':

    # Image Directory
    image_dir = config.IMAGE_DIR

    # Image Files
    image_files = sorted(glob(image_dir+'/*'))

    # Read Images
    images = []
    for image in image_files:
        # images.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
        images.append(cv2.imread(image))

    # Initialize KLT Tracker
    klt_tracker = KLT_Tracker(images, config.feature_params, config.lk_params)

    # Generate Optical Flow
    optical_flow = klt_tracker.generate_optical_flow()

    # Draw Optical Flow
    klt_tracker.draw_optical_flow()

    klt_tracker.generate_initial_point_cloud()