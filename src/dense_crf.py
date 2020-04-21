'''
This module creates a dense reconstruction of the 3D Scene from the sparse 3D points
estimated from the bundle adjustment module. It initializes a CRF model based on the
sparse points and the input RGB image.
'''

import os
import cv2
import config
import argparse
import numpy as np
from plane_sweep import plane_sweep
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax

def compute_unary_image(unary, depth_samples, outfile):

	gd = np.argmin(unary, axis=0)
	gd_im = np.zeros((unary.shape[1], unary.shape[2]))
	for i in range(unary.shape[1]):
		for j in range(unary.shape[2]):
			gd_im[i,j] = (((depth_samples[gd[i,j]] - depth_samples[-1]) * 255) / (depth_samples[0] - depth_samples[-1]) )

	cv2.imwrite(outfile, gd_im)

def DenseCRF(unary, img, depth_samples, params, folder, outfile='depth_map.png', show_unary=False):

	labels = unary.shape[0]
	iters = params['iters']
	weight = params['weight']
	pos_std = params['pos_std']
	rgb_std = params['rgb_std']
	max_penalty = params['max_penalty']

	# Get initial crude depth map from photoconsistency
	if show_unary :
		compute_unary_image(unary, depth_samples, outfile=f'../output/{folder}/cost_volume_{depth_samples.shape[0]}_unary.png')

	# Normalize values for each pixel location
	for r in range(unary.shape[1]):
		for c in range(unary.shape[2]):
			if np.sum(unary[:, r, c]) <= 1e-9:
				unary[:, r, c] = 0.0
			else:
				unary[:, r, c] = (unary[:, r, c]/np.sum(unary[:, r, c]))

	# Convert to class probabilities for each pixel location
	unary = unary_from_softmax(unary)
	d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], labels)

	# Add photoconsistency score as uanry potential. 16-size vector
	# for each pixel location
	d.setUnaryEnergy(unary)
	# Add color-dependent term, i.e. features are (x,y,r,g,b)
	d.addPairwiseBilateral(sxy=pos_std, srgb=rgb_std, rgbim=img, compat=np.array([weight, labels*max_penalty]), kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

	# Run inference steps
	Q = d.inference(iters)

	# Extract depth values. Map to [0-255]
	MAP = np.argmax(Q, axis=0).reshape((img.shape[:2]))
	depth_map = np.zeros((MAP.shape[0], MAP.shape[1]))

	for i in range(MAP.shape[0]):
		for j in range(MAP.shape[1]):

			sample = depth_samples[MAP[i,j]]
			sample = (((sample - depth_samples[-1]) * 255) / (depth_samples[0] - depth_samples[-1]) )
			depth_map[i,j] = sample

	for i in range(config.PS_PARAMS['scale']):

		depth_map = cv2.pyrUp(depth_map)

	cv2.imwrite(outfile, depth_map)

def dense_depth(args) :

	folder = args.folder
	num_samples = int(args.nsamples)
	pc_path = args.pc_cost
	show_unary = args.show_unary

	scale = int(args.scale)
	max_depth = float(args.max_d)
	min_depth = float(args.min_d)
	patch_radius = int(args.patch_rad)

	pc_score = 0
	if pc_path is not None :
		pc_score = np.load(pc_path)
		num_samples = pc_score.shape[0]

	# Create depth samples in the specified depth range
	depth_samples = np.zeros(num_samples)
	step = step = 1.0 / (num_samples - 1.0)

	for val in range(num_samples):
		sample = (max_depth * min_depth) / (max_depth - (max_depth - min_depth) * val * step)
		depth_samples[val] = config.CAMERA_PARAMS['fx']/sample
		# depth_samples[val] = sample

	# Get reference image
	file = ''
	for f in sorted(os.listdir(config.IMAGE_DIR.format(folder))):
		if f.endswith('.png'):
			file = f
			break

	ref_img = cv2.imread(os.path.join(config.IMAGE_DIR.format(folder), file))
	ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
	for s in range(scale):
		ref_img = cv2.pyrDown(ref_img)

	if pc_path is None :

		# Perform plane sweep to calculate photo-consistency loss
		outfile = f'../output/{folder}/cost_volume_{depth_samples.shape[0]}'
		print("Calculating photoconsistency score...")
		pc_score = plane_sweep(folder, outfile, depth_samples, min_depth, max_depth, scale, patch_radius)
		print("Finished computing photoconsistency score...")

	outfile = f'../output/{folder}/cost_volume_{depth_samples.shape[0]}_depth_map.png'
	crf_params = dict()
	crf_params['iters'] = int(args.iters)
	crf_params['pos_std'] = tuple(float(x) for x in args.p_std.split(','))
	crf_params['rgb_std'] = tuple(float(x) for x in args.c_std.split(','))
	crf_params['weight'] = float(args.wt)
	crf_params['max_penalty'] = float(args.max_p)
	print(crf_params)
	# Use photoconsistency score as unary potential
	print("Applying Dense CRF to smoothen depth map...")
	depth_map = DenseCRF(pc_score, ref_img, depth_samples, crf_params, folder, outfile, show_unary)
	print("Finished solving CRF...")


if __name__ == '__main__' :

	parser = argparse.ArgumentParser()
	# General Params
	parser.add_argument("--folder", help='sub-directory in dataset dir', default='stone6', required=True)
	parser.add_argument("--nsamples", help='Number of depth samples', default=16, required=True)
	parser.add_argument("--pc_cost", help='Path to photoconsistency cost array', default=None)
	parser.add_argument("--show_unary", help='Save depth map with just unary (photoconsistency score) potentials', default=False)

	# CRF Params
	parser.add_argument("--iters", help='Number of iters for CRF inference', default=config.CRF_PARAMS['iters'])
	parser.add_argument("--p_std", help='Std. dev of positional term', default=config.CRF_PARAMS['pos_std'])
	parser.add_argument("--c_std", help='Std. dev of color term', default=config.CRF_PARAMS['rgb_std'])
	parser.add_argument("--wt", help='Weight for truncated linear', default=config.CRF_PARAMS['weight'])
	parser.add_argument("--max_p", help='Max Penalty for truncated linear', default=config.CRF_PARAMS['max_penalty'])

	# Plane sweep Params
	parser.add_argument("--max_d", help='Max depth of computed of 3D scene', default=config.PS_PARAMS['max_depth'])
	parser.add_argument("--min_d", help='Min depth of computed of 3D scene', default=config.PS_PARAMS['min_depth'])
	parser.add_argument("--scale", help='Scale of image (downsampling)', default=config.PS_PARAMS['scale'])
	parser.add_argument("--patch_rad", help='Patch radius for photoconsistency', default=config.PS_PARAMS['patch_radius'])

	args = parser.parse_args()

	dense_depth(args)
