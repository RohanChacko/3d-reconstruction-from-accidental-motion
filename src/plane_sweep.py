from numpy.lib.stride_tricks import as_strided
import numpy as np
import utilities
import config
import csv
import cv2
import os

def Sad(warp_patch, ref_patch) :

    # err = np.sum(np.abs(warp_patch - ref_patch[:warp_patch.shape[0], :warp_patch.shape[1]]))
    err = np.sum(np.abs(warp_patch - ref_patch))
    return err


def HomographyFrom(K, C1, R1, C2, R2, dep):

    # C1, R1 : Reference Image
    H  = dep * K @ R2 @ R1.T @ np.linalg.inv(K)
    H[:,2] += K @ R2 @ (C1 - C2)
    return H


def MergeScores(scores, valid_ratio = 0.5):
    '''
    Takes the average of top k values in array. k == valid_scores.
    '''

    score = 0
    num_valid_scores = int(len(scores) * valid_ratio)

    idx = np.argpartition(scores, num_valid_scores)
    scores = np.array(scores).astype('float64')
    score = np.sum(scores[idx[:num_valid_scores]])

    return score/num_valid_scores

def GetMin(values, size):
    '''
    Get smallest two values in array
    '''

    assert(size>1)

    f = 0
    s = 0

    f, s = np.partition(values, 1)[0:2]

    return f, s


def Modulate(cost_volume_arr):

    first = 0
    second = 0
    confidence = 0
    num_samples = cost_volume_arr.shape[0]

    for r in range(cost_volume_arr.shape[1]):
        for c in range(cost_volume_arr.shape[2]):

            values = cost_volume_arr[:, r, c]
            first, second = GetMin(values, num_samples)
            confidence = (second + 1) / (first + 1)
            cost_volume_arr[:, r, c] = values * confidence

    return cost_volume_arr

def plane_sweep(folder, outfile, depth_samples, min_depth, max_depth, scale, patch_radius):

    print(f"Number of depth samples: {depth_samples.shape[0]}")

    # Intrinsics, Camera centers, Rotation mtx
    K = utilities.construct_camera_matrix(config.CAMERA_PARAMS)
    C = []
    R = []

    # Get extrinsics
    with open(config.EXTRINSIC_FILE.format(folder)) as ext_file:
        csv_reader = csv.reader(ext_file, delimiter=',')

        for row in csv_reader:

            p = [float(r) for r in row[:-1]]
            rot, _ = cv2.Rodrigues(np.array(p[:3]))
            trans = np.array(p[3:6])
            c = -1 * np.linalg.inv(rot) @ trans

            C.append(c)
            R.append(rot)

    # Get all images
    all_img = []
    for file in sorted(os.listdir(config.IMAGE_DIR.format(folder))) :

        if file.endswith('.png') :
            im = cv2.imread(os.path.join(config.IMAGE_DIR.format(folder), file))
            all_img.append(im)

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
    cost_volume_arr = np.zeros((depth_samples.shape[0], height, width))

    for idx, depth in enumerate(depth_samples):

        print(f"Depth sample...{idx+1}")
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
            warp = cv2.warpPerspective(scaled_gray_images[i], homographies[i], ref_img.shape[::-1], cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            warped_images.append(warp)


        img1 = as_strided(ref_img, shape=(ref_img.shape[0] - 2, ref_img.shape[1] - 2, 3, 3), strides=ref_img.strides + ref_img.strides, writeable=False)
        h, w, _, _ = img1.shape
        img1 = img1.reshape((img1.shape[0]*img1.shape[1], 9))
        lol = np.zeros((len(warped_images), img1.shape[0], img1.shape[1]))
        for i in range(len(warped_images)):
            x = as_strided(warped_images[i], shape=(warped_images[i].shape[0] - 2, warped_images[i].shape[1] - 2, 3, 3), strides=warped_images[i].strides + warped_images[i].strides, writeable=False)
            x = x.reshape((x.shape[0]*x.shape[1], 9))
            lol[i,:,:] = x

        diff = np.sum(np.abs(lol - img1), axis=2)
        ix = np.argpartition(diff,int(0.5*len(warped_images)), axis=0)
        ix = ix[:int(0.5*len(warped_images)),:]

        srt = np.take_along_axis(diff, ix, axis=0)
        score = np.sum(srt, axis=0) / int(0.5*len(warped_images))

        # TODO : Border pixels take default value cost arr. Fix that
        cost_volume_arr[idx, patch_radius:height-patch_radius, patch_radius:width-patch_radius] = score.reshape((h,w))


    cost_volume_arr = Modulate(cost_volume_arr)
    np.save(outfile, cost_volume_arr)

    return cost_volume_arr.astype('float32')
