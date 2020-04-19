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

from plane_sweep import plane_sweep

class DenseCRF:

	def __init__(self, rgb, unary, outfile) :

		self.outfile = outfile
		self.unary = unary
		self.labels = unary.shape[0]
		self.rgb = rgb
		self.d = dcrf.DenseCRF2D(self.rgb.shape[1], self.rgb.shape[0], self.labels)

	def create_model(self) :

		# get unary potentials (neg log probability)
		self.d.setUnaryEnergy(self.unary)

		# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
		self.d.addPairwiseBilateral(sxy=(5, 5), srgb=(20, 20, 20), rgbim=self.rgb, compat=np.array([1, self.labels*0.15]), kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)


	def inference(self, iters=100) :

		# Run inference steps.
		Q = self.d.inference(iters)
		print(Q.shape)
		# Find out the most probable class for each pixel.
		MAP = np.argmax(Q, axis=0)

unary = plane_sweep(min_depth=2, max_depth=4, scale = 2, num_samples=64, patch_radius = 1)
print(unary.shape)
img = cv2.imread('../datasets/stone6_still/stone6_still_0001.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
d = DenseCRF(img, unary, '')
d.inference(100)
