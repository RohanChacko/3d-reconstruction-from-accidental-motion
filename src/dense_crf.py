'''
This module creates a dense reconstruction of the 3D Scene from the sparse 3D points
estimated from the bundle adjustment module. It initializes a CRF model based on the 
sparse points and the input RGB image. 
'''

import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian


class DenseCRF:

	def __init__(self, depth_map, rgb, outfile) :

		self.depth = depth_map.astype(np.uint32)
		self.rgb = rgb
		self.outfile = outfile

		anno_lbl = self.depth[:,:,0] + (self.depth[:,:,1] << 8) + (self.depth[:,:,2] << 16)
		self.colors, labels = np.unique(anno_lbl, return_inverse=True)

		self.HAS_UNKNOWN_LABEL = 0 in self.colors
		if self.HAS_UNKNOWN_LABEL:
			self.colors = self.colors[1:]

		# And create a mapping back from the labels to 32bit integer self.colors.
		self.colorize = np.empty((len(self.colors), 3), np.uint8)
		self.colorize[:,0] = (self.colors & 0x0000FF)
		self.colorize[:,1] = (self.colors & 0x00FF00) >> 8
		self.colorize[:,2] = (self.colors & 0xFF0000) >> 16

		# Compute the number of classes in the label image.
		# We subtract one because the number shouldn't include the value 0 which stands
		# for "unknown" or "unsure".
		self.n_labels = len(set(labels.flat)) - int(HAS_UNKNOWN_LABEL)
		#print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNKNOWN_LABEL else ""), set(labels.flat))
	

	def create_model(self) :

		self.d = dcrf.DenseCRF2D(self.rgb.shape[1], self.rgb.shape[0], self.n_labels)

		# get unary potentials (neg log probability)
		U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=self.HAS_UNKNOWN)
		self.d.setUnaryEnergy(U)

		# This adds the color-independent term, features are the locations only.
		self.d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

		# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
		self.d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=self.rgb, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

		
	def inference(self, iters) :

		# Run five inference steps.
		Q = self.d.inference(iters)

		# Find out the most probable class for each pixel.
		MAP = np.argmax(Q, axis=0)

		# Convert the MAP (labels) back to the corresponding self.colors and save the image.
		# Note that there is no "unknown" here anymore, no matter what we had at first.
		MAP = self.colorize[MAP,:]
		cv2.imwrite(self.out_file, MAP.reshape(self.rgb.shape))

		# Just randomly manually run inference iterations
		Q, tmp1, tmp2 = self.d.startInference()
		for i in range(iters):
		    #print("KL-divergence at {}: {}".format(i, self.d.klDivergence(Q)))
		    d.stepInference(Q, tmp1, tmp2)
		

i
