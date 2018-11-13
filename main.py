import numpy as np
import time
import cv2
from boosting_classifier import Boosting_Classifier
from visualizer import Visualizer
from im_process import normalize
from utils import *

def main():
	#flag for debugging
	flag_subset = False
	boosting_type = 'Ada' #'Real' or 'Ada'
	training_epochs = 100 if not flag_subset else 20
	act_cache_dir = 'wc_activations.npy' if not flag_subset else 'wc_activations_subset.npy'
	chosen_wc_cache_dir = 'chosen_wcs.pkl' if not flag_subset else 'chosen_wcs_subset.pkl'

	#data configurations
	pos_data_dir = 'newface16'
	neg_data_dir = 'nonface16'
	image_w = 16
	image_h = 16
	data, labels = load_data(pos_data_dir, neg_data_dir, image_w, image_h, flag_subset)
	data = integrate_images(normalize(data))

	#number of bins for boosting
	num_bins = 25

	#number of cpus for parallel computing
	num_cores = 32 if not flag_subset else 1 #always use 1 when debugging
	
	#create Haar filters
	filters = generate_Haar_filters(4, 4, 16, 16, image_w, image_h, flag_subset)

	#create visualizer to draw histograms, roc curves and best weak classifier accuracies
	drawer = Visualizer([10, 20, 50, 100], [1, 10, 20, 50, 100])
	
	#create boost classifier with a pool of weak classifier
	boost = Boosting_Classifier(filters, data, labels, training_epochs, num_bins, drawer, num_cores, boosting_type)

	#calculate filter values for all training images
	start = time.clock()
	boost.calculate_training_activations(act_cache_dir, act_cache_dir)
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	boost.train(chosen_wc_cache_dir, chosen_wc_cache_dir)
	#boost.visualizer.display_haar(boost.chosen_wcs,20)
	#boost.visualize()

	# test_img = './Test/audrey.jpeg'
	# original_img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
	# img_rgb = cv2.imread(test_img)
	# result_img = boost.face_detection(original_img, img_rgb)
	# cv2.imwrite('Result_img_%s.png' % boosting_type, result_img)

	#negative training
	negative_patches1 = boost.get_hard_negative_patches(cv2.imread('./Test/Non_face_1.jpg', cv2.IMREAD_GRAYSCALE))[0]
	negative_patches2 = boost.get_hard_negative_patches(cv2.imread('./Test/Non_face_2.jpg', cv2.IMREAD_GRAYSCALE))[0]

	negative_patches = np.concatenate((negative_patches1,negative_patches2))
	nhm_labels = np.ones(negative_patches.shape[0])*-1
	boost.data = np.concatenate((boost.data,negative_patches))
	boost.labels = np.concatenate((boost.labels,nhm_labels))

	# calculate filter values for all training images
	start = time.clock()
	boost.calculate_training_activations("wc_activations_nms.npy")
	end = time.clock()
	print('%f seconds for activation calculation' % (end - start))

	boost.train("chosen_wcs_nhm.pkl")

	test_img = './Test/Face_1.jpg'
	original_img = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)
	img_rgb = cv2.imread(test_img)
	result_img = boost.face_detection(original_img, img_rgb)
	cv2.imwrite('Result_img_%s.png' % boosting_type, result_img)

if __name__ == '__main__':
	main()
