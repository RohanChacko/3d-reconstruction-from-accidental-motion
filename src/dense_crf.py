'''
This module creates a dense reconstruction of the 3D Scene from the sparse 3D points
estimated from the bundle adjustment module. It initializes a CRF model based on the
sparse points and the input RGB image.
'''

import cv2
import numpy as np
import config
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral

from plane_sweep import plane_sweep, Modulate

class DenseCRF:

	def __init__(self, rgb, unary, outfile) :

		self.outfile = outfile
		self.unary = unary.reshape((16,-1))
		self.labels = unary.shape[0]
		self.rgb = rgb
		self.d = dcrf.DenseCRF2D(self.rgb.shape[1], self.rgb.shape[0], self.labels)

	def create_model(self) :

		# get unary potentials (neg log probability)
		self.d.setUnaryEnergy(self.unary)

		# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
		self.d.addPairwiseBilateral(sxy=(5, 5), srgb=(20, 20, 20), rgbim=self.rgb, compat=np.array([1.0, self.labels*0.15]), kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		# self.d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=self.rgb, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)


	def inference(self, iters=100) :

		# Run inference steps.
		Q = self.d.inference(iters)
		# Find out the most probable class for each pixel.
		MAP = np.argmax(Q, axis=0)
		print(MAP.dtype)
		# print(MAP[MAP !=0])

unary = plane_sweep(min_depth=2, max_depth=4, scale=2, num_samples=64, patch_radius = 1)
# unary = np.load('cost_volume_64.npy')
unary = Modulate(unary)
# print(unary.dtype)

# for i in range(unary.shape[1]):
# 	for j in range(unary.shape[2]):
# 		print(unary[:2,i,j])

img = cv2.imread('../datasets/stone6_still/stone6_still_0001.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
d = DenseCRF(img, unary, '')
d.inference(200)
