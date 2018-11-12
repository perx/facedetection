from sklearn.feature_extraction import image as IMG
import numpy as np
import cv2
from utils import integrate_images
from operator import itemgetter

#extract patches from the image for all scales of the image
#return the INTEGREATED images and the coordinates of the patches
def image2patches(scales, image, patch_w = 16, patch_h = 16):
	all_patches = np.zeros((0, patch_h, patch_w))
	all_x1y1x2y2 = []
	for s in scales:
		simage = cv2.resize(image, None, fx = s, fy = s, interpolation = cv2.INTER_CUBIC)
		height, width = simage.shape
		print('Image shape is: %d X %d' % (height, width))
		patches = IMG.extract_patches_2d(simage, (patch_w, patch_h)) # move along the row first

		total_patch = patches.shape[0]
		row_patch = (height - patch_h + 1)
		col_patch = (width - patch_w + 1)
		assert(total_patch == row_patch * col_patch)
		scale_xyxy = []
		for pid in range(total_patch):
			y1 = pid / col_patch
			x1 = pid % col_patch
			y2 = y1 + patch_h - 1
			x2 = x1 + patch_w - 1
			scale_xyxy.append([int(x1 / s), int(y1 / s), int(x2 / s), int(y2 / s)])
		all_patches = np.concatenate((all_patches, patches), axis = 0)
		all_x1y1x2y2 += scale_xyxy
	return integrate_images(normalize(all_patches)), all_x1y1x2y2

#return a vector of prediction (0/1) after nms, same length as scores
#input: [x1, y1, x2, y2, score], threshold used for nms
#output: [x1, y1, x2, y2, score] after nms
def calc_overlap(b1, b2):
	x1 = max(b1[0], b2[0])
	y1 = max(b1[1], b2[1])
	x2 = min(b1[2], b2[2])
	y2 = min(b1[3], b2[3])

	w = max(0, x2 - x1 + 1)
	h = max(0, y2 - y1 + 1)
	area1 = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
	area2 = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)

	area = float(w) * h
	overlap = area / (area1 + area2)

	return overlap

def nms(xyxys, overlap_thresh):
	# sorted_idx = np.argsort(xyxys[:,4])[::-1]
	# highest=sorted_idx[0]
	# for box1_idx in sorted_idx:
	# 	box1 = xyxys[box1_idx]
	# 	temp_xyxys = sorted_idx.copy()
	# 	delete = []
	# 	for i in temp_xyxys:
	# 		box2 = xyxys[i]
	# 		if box1_idx == i:
	# 			continue
	# 		overlap = calc_overlap(box1, box2)
	# 		if overlap > overlap_thresh:
	# 			delete.append(i)
	# 	sorted_idx = np.delete(sorted_idx,delete,axis=0)
	# print(xyxys[highest])
	# print((xyxys[sorted_idx])[0])
	# return xyxys[sorted_idx]
	x1s = xyxys[:, 0]
	y1s = xyxys[:, 1]
	x2s = xyxys[:, 2]
	y2s = xyxys[:, 3]
	scores = xyxys[:, 4]
	area = (x2s - x1s + 1) * (y2s - y1s + 1)
	sorted_idx = np.argsort(scores)
	output_idx = []

	while len(sorted_idx) > 0:
		last = len(sorted_idx) - 1
		greatest_area_idx=sorted_idx[last]
		output_idx.append(greatest_area_idx)

		xx1 = np.maximum(x1s[greatest_area_idx], x1s[sorted_idx[:last]])
		yy1 = np.maximum(y1s[greatest_area_idx], y1s[sorted_idx[:last]])
		xx2 = np.minimum(x2s[greatest_area_idx], x2s[sorted_idx[:last]])
		yy2 = np.minimum(y2s[greatest_area_idx], y2s[sorted_idx[:last]])

		# area of overlaps
		area_of_overlap = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
		overlap = area_of_overlap / (area[sorted_idx[:last]])

		sorted_idx = np.delete(sorted_idx, np.concatenate(([last],np.where(overlap > overlap_thresh)[0])))

	return xyxys[output_idx]

def normalize(images):
	standard = np.std(images)
	images = (images - np.min(images)) / (np.max(images) - np.min(images))
	return images

def main():
	original_img = cv2.imread('Testing_Images/Face_1.jpg', cv2.IMREAD_GRAYSCALE)
	scales = 1 / np.linspace(1, 10, 46)
	patches, patch_xyxy = image2patches(scales, original_img)
	print(patches.shape)
	print(len(patch_xyxy))
if __name__ == '__main__':
	main()
