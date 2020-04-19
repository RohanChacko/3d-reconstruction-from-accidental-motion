import cv2
import numpy as np

def Sad(img1, img2) :
    
    # img1, img2 are patches from two different images

    err = 0

    for r in range(img1.shape[0]) :
        for c in range(img1.shape[1]:
                diff = img1[r, c] - img2[r, c]
                err += np.abs(diff)

    return err


def HomographyFrom(cam, ref_cam, dep):
    
    K1 = cam1.K()
    R1 = cam1.R()
    C1 = cam1.C()
    K2 = ref_cam.K()
    R2 = ref_cam.R()
    C2 = ref_cam.C()

    H  = d * K2 @ R2 @ R1.T * np.linalg.inv(K1)
    H[2,:] += K2 @ R2 @ (C1 - C2)
    
    return H


def MergeScores(scores):
    score = 0
    valid_ratio = 0.5
    num_valid_scores = len(scores) * valid_ratio
  
    scores = sort(scores)
    for i in range(num_valid_scores):
        score += scores[i]
  
    score /= num_valid_scores
    return score

def plane_sweep(min_depth=2, max_depth=4, scale = 2, num_samples=64):


    # inverse perspetive/depth sampling
    depth_samples = []
    step = 1.0 / (num_samples - 1.0)
    for i in range(num_samples):
        sample = (max_depth * min_depth) / (max_depth - (max_depth - min_depth) * i *step)
        depth_samples.append(sample)
    

    scaled_gray_images = []
    for i in all_images :
        
        img = np.int32(img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for s in range(scale):
            img = cv2.pyrDown(img)
        scaled_gray_images.append(img)
        
  height, width = ref_image.shape
  num_images = 99
  patch_radius = 1
  
  cost_volume_arr = np.zeros((num_samples, height, width))
  cost_volume = 0

  ref_camera_ext = # Reference camera extrinsics
  ref_image = # ref image

  for d in range(num_samples):

    depth = depth_samples[d]
    homographies = np.zeros((num_images, 3, 3))
    warped_images = []
    
    for i in range(num_images) :
        
        h = HomographyFrom(cam[idx], ref_camera, depth)
        actual_scale = 2**scale
        h[:2,:] *= actual_scale
        h[2,:] *= actual_scale
        homographies[i,:,:] = h

    # Assume 0th image is reference image
    for i in range(1,num_images):
      warp = cv2.warpPerspective(scaled_gray_images[i], homographies[i], ref_image.shape, cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
      warped_images.append(warp)
    
    
    # len(scores) == len(warped_images)
    scores = []
    for x in range(width - patch_radius):
        for y in range(height - patch_radius):
            p2 = ref_image[y - patch_radius :y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
            
            for i in range(len(warped_images)):
                p1 = warped_images[i][y - patch_radius :y + patch_radius + 1, x - patch_radius : x + patch_radius + 1]
                scores.append(Sad(p1, p2))

        cost_volume_arr[d,y,x] = MergeScores(scores)
