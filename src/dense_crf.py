'''
This module creates a dense reconstruction of the 3D Scene from the sparse 3D points
estimated from the bundle adjustment module. It initializes a CRF model based on the 
sparse points and the input RGB image. 
'''

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral


class DenseCRF:

	def __init__(self, rgb, unary, outfile) :

		self.outfile = outfile
                self.unary = unary
                self.labels = labels
                self.rgb = rgb

	def create_model(self) :

		self.d = dcrf.DenseCRF2D(self.rgb.shape[1], self.rgb.shape[0], self.n_labels)

		# get unary potentials (neg log probability)
		self.d.setUnaryEnergy(self.unary)

		# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
		self.d.addPairwiseBilateral(sxy=(5, 5), srgb=(20, 20, 20), rgbim=self.rgb, compat=np.array([1, self.labels*0.15]), kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

		
	def inference(self, iters=200) :

		# Run inference steps.
		Q = self.d.inference(iters)
                print(Q.shape)
		# Find out the most probable class for each pixel.
		MAP = np.argmax(Q, axis=0)

