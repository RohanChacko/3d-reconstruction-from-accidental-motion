import os
import cv2
import csv
import config
import utilities
import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import as_strided


def Sad(ref_patch, warp_patch) :

    '''
    Calculates L1 Loss between two grayscale image patches
    N : Total number of warp images
    P : Total number of patches per image
    w : Length of one side of patch

    Dimension of patch ndarray : N x P x (w*w)
    Returned array dim : N x P
    '''

    # err = np.sum(np.abs(warp_patch - ref_patch[:warp_patch.shape[0], :warp_patch.shape[1]]))
    err = np.sum(np.abs(warp_patch - ref_patch), axis=2)
    return err


def HomographyFrom(K, C1, R1, C2, R2, dep):

    # C1, R1 : Reference Image
    H  = dep * K @ R2 @ R1.T @ np.linalg.inv(K)
    H[:,2] += K @ R2 @ (C1 - C2)
    return H


def MergeScores(scores, valid_ratio = 0.5):
    '''
    Takes the average of top k values in array. k == valid_scores.
    N : Total number of warp images
    P : Total number of patches per image

    Dimension of scores array: N x P
    Dimension of returned array: (N*valid_ratio) x P
    '''

    num_valid_scores = int(scores.shape[0] * valid_ratio)

    ix = np.argpartition(scores, num_valid_scores, axis=0)
    ix = ix[:num_valid_scores,:]

    srt = np.take_along_axis(scores, ix, axis=0)
    score = np.sum(srt, axis=0) / num_valid_scores

    return score

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

    for idx, depth in enumerate(tqdm(depth_samples)):

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


        ref_img_patches = as_strided(ref_img, shape=(ref_img.shape[0] - 2*patch_radius,
                                    ref_img.shape[1] - 2*patch_radius, 2*patch_radius + 1, 2*patch_radius + 1),
                                    strides=ref_img.strides + ref_img.strides, writeable=False)

        h, w, _, _ = ref_img_patches.shape
        patch_size = 2*patch_radius + 1
        ref_img_patches = ref_img_patches.reshape((ref_img_patches.shape[0]*ref_img_patches.shape[1], patch_size**2))
        warp_patches = np.zeros((len(warped_images), ref_img_patches.shape[0], ref_img_patches.shape[1]))
        for i in range(len(warped_images)):

            x = as_strided(warped_images[i], shape=(warped_images[i].shape[0] - 2*patch_radius,
                            warped_images[i].shape[1] - 2*patch_radius, 2*patch_radius + 1, 2*patch_radius + 1),
                            strides=warped_images[i].strides + warped_images[i].strides, writeable=False)

            x = x.reshape((x.shape[0]*x.shape[1], patch_size**2))
            warp_patches[i,:,:] = x

        L1_diff = Sad(ref_img_patches, warp_patches)
        score = MergeScores(L1_diff, valid_ratio = 0.5)

        # TODO : Border pixels take default value cost arr. Fix that
        cost_volume_arr[idx, patch_radius:height-patch_radius, patch_radius:width-patch_radius] = score.reshape((h,w))


    cost_volume_arr = Modulate(cost_volume_arr)
    np.save(outfile, cost_volume_arr)

    return cost_volume_arr.astype('float32')
