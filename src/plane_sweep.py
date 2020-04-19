import cv2
import numpy as np
import config
import utilities
import csv
import os

def Sad(img1, img2) :

    # img1, img2 are patches from two different images
    return np.sum(np.abs(img1 - img2))


def HomographyFrom(K, C1, R1, C2, R2, dep):

    H  = dep * K @ R2 @ R1.T * np.linalg.inv(K)
    H[2,:] += K @ R2 @ (C1 - C2)

    return H


def MergeScores(scores):

    score = 0
    valid_ratio = 0.5
    num_valid_scores = int(len(scores) * valid_ratio)

    scores = sorted(scores)
    for i in range(num_valid_scores):
        score += scores[i]

    return score/num_valid_scores

def plane_sweep(min_depth=2, max_depth=4, scale = 2, num_samples=16, patch_radius = 1):

    # Intrinsics, Camera centers, Rotation mtx
    K = utilities.construct_camera_matrix(config.CAMERA_PARAMS)
    C = []
    R = []

    # Get extrinsics
    with open(config.EXTRINSIC_FILE) as ext_file:
        csv_reader = csv.reader(ext_file, delimiter=',')

        for row in csv_reader:

            p = [float(r) for r in row[:-1]]
            rot, _ = cv2.Rodrigues(np.array(p[:3]))
            trans = np.array(p[3:6])
            c = -1 * trans @ np.linalg.inv(rot)

            C.append(c)
            R.append(rot)

    # Get all images
    all_img = []
    for file in os.listdir(config.IMAGE_DIR) :

        if file.endswith('.png') :
            im = cv2.imread(os.path.join(config.IMAGE_DIR, file))
            all_img.append(im)


    # inverse perspetive/depth sampling
    depth_samples = []
    step = 1.0 / (num_samples - 1.0)
    for val in range(num_samples):
        sample = (max_depth * min_depth) / (max_depth - (max_depth - min_depth) * val * step)
        depth_samples.append(sample)


    scaled_gray_images = []
    for img in all_img :
        img = img.astype(np.float32)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for s in range(scale):
            gray_img = cv2.pyrDown(gray_img)

        scaled_gray_images.append(gray_img)

    ref_img = scaled_gray_images[0]
    height, width = ref_img.shape

    num_images = len(all_img)
    cost_volume_arr = np.zeros((num_samples, height, width))

    for idx, depth in enumerate(depth_samples):

        print(f"Sample...{idx+1}")
        homographies = np.zeros((num_images, 3, 3))
        warped_images = []

        for ind in range(num_images) :

            h = HomographyFrom(K, C[0], R[0], C[ind], R[ind], depth)
            actual_scale = 2**scale
            h[:,:2] *= actual_scale
            h[2,:] *= actual_scale
            homographies[ind,:,:] = h

        # Assume 0th image is reference image
        for i in range(1, num_images):
            warp = cv2.warpPerspective(scaled_gray_images[i], homographies[i], ref_img.shape, cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            warped_images.append(warp)

        # len(scores) == len(warped_images)
        for x in range(width - patch_radius):
            for y in range(height - patch_radius):
                p2 = ref_img[y - patch_radius :y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
                scores = []
                for i in range(len(warped_images)):
                    p1 = warped_images[i][y - patch_radius :y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
                    s = Sad(p1, p2)
                    # print(s)
                    scores.append(s)

                cost_volume_arr[idx, y, x] = MergeScores(scores)

    np.save('cost_volume.npz',cost_volume_arr)

    return cost_volume_arr
