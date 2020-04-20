'''
This module creates a dense reconstruction of the 3D Scene from the sparse 3D points
estimated from the bundle adjustment module. It initializes a CRF model based on the
sparse points and the input RGB image.
'''

import cv2
import numpy as np
import config
from scipy.special import softmax
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax
import matplotlib.pyplot as plt
from plane_sweep import plane_sweep, Modulate

class DenseCRF:

	def __init__(self, rgb, unary, outfile) :

		self.depth_samples = []
		self.labels = unary.shape[0]
		max_depth = 4.0
		min_depth = 2.0
		num_samples = self.labels
		step = 1.0 / (num_samples - 1.0)

		# NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
		for val in range(num_samples):
			sample = (max_depth * min_depth) / (max_depth - (max_depth - min_depth) * val * step)
			sample = 1781.0/sample
			sample = (((sample - 445.25) * 255) / (890.5 - 445.25) )
			self.depth_samples.append(sample)

		self.outfile = outfile
		unary = unary.astype('float32')
		m = softmax(unary, axis=2)
		# for i in range(unary.shape[1]):
		# 	for j in range(unary.shape[2]):
		# 		print(np.argmin(unary[:,i,j]),":",np.min(unary[:,i,j]), np.argmax(unary[:,i,j]), ":", np.max(unary[:,i,j]))
		gd = np.argmin(unary, axis=0)

		gd_im = np.zeros((unary.shape[1], unary.shape[2]))
		for i in range(unary.shape[1]):
			for j in range(unary.shape[2]):

				gd_im[i,j] = self.depth_samples[gd[i,j]]
				print(gd_im[i,j])

		# gd_im = ((( gd_im - 2) * 255) / 2)
		print(np.max(gd_im))

		# plt.hist(gd_im, bins=np.arange(255))
		# plt.savefig('hist.png', )

		cv2.imwrite('unary_16.png', 255.0 - gd_im)

		x = unary_from_softmax(unary)
		print(x.shape, m.shape)
		for i in range(unary.shape[1]):
			for j in range(unary.shape[2]):
				unary[:,i,j] /= np.sum(unary[:,i,j])
		self.unary = unary
		self.rgb = rgb
		self.d = dcrf.DenseCRF2D(self.rgb.shape[1], self.rgb.shape[0], self.labels)
		print(self.depth_samples)
	def create_model(self) :

		# get unary potentials (neg log probability)
		self.d.setUnaryEnergy(self.unary)

		# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
		self.d.addPairwiseBilateral(sxy=(5, 5), srgb=(20, 20, 20), rgbim=self.rgb, compat=np.array([1.0, self.labels*0.15]), kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
		# self.d.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=self.rgb, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)


	def inference(self, iters=100) :

		# Run inference steps.
		Q = self.d.inference(iters)

		# Q, tmp1, tmp2 = self.d.startInference()
		# for i in range(50):
		    # print("KL-divergence at {}: {}".format(i, self.d.klDivergence(Q)))
		    # self.d.stepInference(Q, tmp1, tmp2)
		# Find out the most probable class for each pixel.
		# print(Q[])
		MAP = np.argmax(Q, axis=0).reshape((self.rgb.shape[:2]))
		depth_map = np.zeros((MAP.shape[0], MAP.shape[1]))
		for i in range(MAP.shape[0]):
			for j in range(MAP.shape[1]):

				depth_map[i,j] = 255.0*self.depth_samples[MAP[i,j]]/self.depth_samples[0]

		cv2.imwrite('depth_map.png', depth_map)
		print(MAP[MAP!= 0])

# unary = plane_sweep(min_depth=2, max_depth=4, scale=2, num_samples=32, patch_radius = 1)
unary = np.load('cost_volume_16_3.npy')
unary = Modulate(unary)


# for i in range(unary.shape[1]):
# 	for j in range(unary.shape[2]):
# 		print(unary[:2,i,j])

img = cv2.imread('../datasets/stone6_still/stone6_still_0001.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.pyrDown(img)
img = cv2.pyrDown(img)
d = DenseCRF(img, unary, '')
cv2.imwrite('lab.png', img)
d.inference(200)
