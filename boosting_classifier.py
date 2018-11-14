import os
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import pickle
import copy
import cv2
from weak_classifier import Ada_Weak_Classifier, Real_Weak_Classifier
from im_process import image2patches, nms, normalize
import time


class Boosting_Classifier:
	def __init__(self, haar_filters, data, labels, num_chosen_wc, num_bins, visualizer, num_cores, style):
		self.filters = haar_filters
		self.data = data
		self.labels = labels
		self.num_chosen_wc = num_chosen_wc
		self.num_bins = num_bins
		self.visualizer = visualizer
		self.num_cores = num_cores
		self.style = style
		self.chosen_wcs = None
		if style == 'Ada':
			self.weak_classifiers = [Ada_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
		elif style == 'Real':
			self.weak_classifiers = [Real_Weak_Classifier(i, filt[0], filt[1], self.num_bins)\
									 for i, filt in enumerate(self.filters)]
	
	def calculate_training_activations(self, save_dir = None, load_dir = None):
		print('Calcuate activations for %d weak classifiers, using %d imags.' % (len(self.weak_classifiers), self.data.shape[0]))
		if load_dir is not None and os.path.exists(load_dir):
			print('[Find cached activations, %s loading...]' % load_dir)
			wc_activations = np.load(load_dir)
		else:
			if self.num_cores == 1:
				wc_activations = [wc.apply_filter(self.data) for wc in self.weak_classifiers]
			else:
				wc_activations = Parallel(n_jobs = self.num_cores)(delayed(wc.apply_filter)(self.data) for wc in self.weak_classifiers)
			wc_activations = np.array(wc_activations)
			if save_dir is not None:
				print('Writing results to disk...')
				np.save(save_dir, wc_activations)
				print('[Saved calculated activations to %s]' % save_dir)
		for wc in self.weak_classifiers:
			wc.activations = wc_activations[wc.id, :]
		return wc_activations
	
	#select weak classifiers to form a strong classifier
	#after training, by calling self.sc_function(), a prediction can be made
	#self.chosen_wcs should be assigned a value after self.train() finishes
	#call Weak_Classifier.calc_error() in this function
	#cache training results to self.visualizer for visualization
	def train(self, save_dir = None, load_dir = None):
		if load_dir is not None:
			self.chosen_wcs = pickle.load(open(load_dir, 'rb'))
			print("Loaded trained weak classfiers from ",load_dir)
			return
		######################
		######## TODO ########
		######################
		self.chosen_wcs=[]
		#1. Initialize weights
		weights=np.ones(self.data.shape[0])*(1/self.data.shape[0])

		for epoch in range(self.num_chosen_wc):
		#2. Find a weak classifier with the least error
			wc_outputs = []
			start = time.time()
			wc_outputs=Parallel(n_jobs=-1)(delayed(wc.calc_error)(weights,self.labels) for wc in self.weak_classifiers)

			# for i, wc in enumerate(self.weak_classifiers):
			#  	wc_outputs.append(wc.calc_error(weights=weights,labels=self.labels))

			wc_errors, wc_thresholds, wc_polarity = zip(*wc_outputs)
			end=time.time()
			chosen_wc_err = min(wc_errors)
			chosen_wc_idx = np.argmin(wc_errors)
			chosen_alpha = 0.5*(np.log((1-chosen_wc_err)/chosen_wc_err))
			chosen_wc = copy.deepcopy(self.weak_classifiers[chosen_wc_idx])
			chosen_wc.threshold, chosen_wc.polarity=wc_thresholds[chosen_wc_idx],wc_polarity[chosen_wc_idx]

			self.chosen_wcs.append((chosen_alpha,chosen_wc))

			#self.visualizer.weak_classifier_accuracies[epoch]=(1-chosen_wc_err)*100

			print("WC ",epoch, np.argmin(wc_errors), ", error = ", chosen_wc_err, ", alpha = ", chosen_alpha)
			print("Time taken = ",(end-start))
			#3. Update weights of data points
			new_weights=np.ones(weights.shape)
			for ac_idx,activation in enumerate(chosen_wc.activations):
				#e^alpha * y * ht * -1
				# print("Prediction 1 ",chosen_wc.predict_label(activation))
				# print("Prediction 2 ",chosen_wc.predict_image(self.data[ac_idx]))
				reduction_factor=chosen_alpha * chosen_wc.predict_label(activation) * self.labels[ac_idx] * -1
				new_weights[ac_idx] = weights[ac_idx]*(np.exp(reduction_factor))

			weights = new_weights/new_weights.sum()

			#caching for visualizer
			sc_score = np.array([self.sc_function(im) for im in self.data])
			self.visualizer.strong_classifier_errors.append(1-np.mean(np.sign(sc_score) == self.labels))
			if epoch+1 in (1, 10, 50, 100):
				top_1000_wcs_idx = np.argsort(wc_errors)[:1000]
				top_1000_err = []
				for wc_idx in top_1000_wcs_idx:
					wc = self.weak_classifiers[wc_idx]
					predictions = np.array([wc.predict_label(act,wc_thresholds[wc_idx],wc_polarity[wc_idx]) for act in wc.activations])
					top_1000_err.append(np.mean(predictions == self.labels))
				self.visualizer.weak_classifier_accuracies[epoch+1] = np.array(sorted(top_1000_err)[::-1])
				self.visualizer.strong_classifier_scores[epoch+1] = sc_score
				#self.visualizer.strong_classifier_scores[epoch] = np.array([self.sc_function(im) for im in self.data])

		if save_dir is not None:
			pickle.dump(self.chosen_wcs, open(save_dir, 'wb'))

	def sc_function(self, image):
		if self.style == "Ada":
			return np.sum([np.array([alpha * wc.predict_image(image) for alpha, wc in self.chosen_wcs])])
		elif self.style == "Real":
			return np.sum([np.array([wc.predict_image(image) for wc in self.chosen_wcs])])

	def load_trained_wcs(self, save_dir):
		self.chosen_wcs = pickle.load(open(save_dir, 'rb'))	

	def face_detection(self, img, img_rgb, scale_step = 20):
		
		# this training accuracy should be the same as your training process,
		##################################################################################
		train_predicts = []
		for idx in range(self.data.shape[0]):
			train_predicts.append(self.sc_function(self.data[idx, ...]))
		print('Check training accuracy is: ', np.mean(np.sign(train_predicts) == self.labels))
		##################################################################################

		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Face Detection in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		print(np.mean(np.array(predicts) > 0), np.sum(np.array(predicts) > 0))
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
		if pos_predicts_xyxy.shape[0] == 0:
			return
		xyxy_after_nms,_ = nms(pos_predicts_xyxy, 0.01)
		print('after nms:', xyxy_after_nms.shape[0])
		if img_rgb is not None:
			img = img_rgb
		for idx in range(xyxy_after_nms.shape[0]):
			pred = xyxy_after_nms[idx, :]
			cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2) #gree rectangular with line width 3
		return img

	def get_hard_negative_patches(self, img, scale_step = 10):
		scales = 1 / np.linspace(1, 8, scale_step)
		patches, patch_xyxy = image2patches(scales, img)
		print('Get Hard Negative in Progress ..., total %d patches' % patches.shape[0])
		predicts = [self.sc_function(patch) for patch in tqdm(patches)]
		predicts = np.array(predicts)
		wrong_patches = patches[np.where(predicts > 0), ...]
		pos_predicts_xyxy = np.array([patch_xyxy[idx] + [score] for idx, score in enumerate(predicts) if score > 0])
		patches_after_nms, nms_patches_idx = nms(pos_predicts_xyxy, 0.01)
		return wrong_patches[0], wrong_patches[0][nms_patches_idx]

	def visualize(self):
		self.visualizer.labels = self.labels
		self.visualizer.draw_histograms()
		self.visualizer.draw_rocs()
		self.visualizer.draw_wc_accuracies()
		self.visualizer.plot_sc_train_error()
