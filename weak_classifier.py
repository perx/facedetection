from abc import ABC, abstractmethod
import numpy as np

class Weak_Classifier(ABC):
	#initialize a harr filter with the positive and negative rects
	#rects are in the form of [x1, y1, x2, y2] 0-index
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		self.id = id
		self.plus_rects = plus_rects
		self.minus_rects = minus_rects
		self.num_bins = num_bins
		self.activations = None

	#take in one integrated image and return the value after applying the image
	#integrated_image is a 2D np array
	#return value is the number BEFORE polarity is applied
	def apply_filter2image(self, integrated_image):
		pos = 0
		for rect in self.plus_rects:
			rect = [int(n) for n in rect]
			pos += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		neg = 0
		for rect in self.minus_rects:
			rect = [int(n) for n in rect]
			neg += integrated_image[rect[3], rect[2]]\
				 + (0 if rect[1] == 0 or rect[0] == 0 else integrated_image[rect[1] - 1, rect[0] - 1])\
				 - (0 if rect[1] == 0 else integrated_image[rect[1] - 1, rect[2]])\
				 - (0 if rect[0] == 0 else integrated_image[rect[3], rect[0] - 1])
		return pos - neg


	#take in a list of integrated images and calculate values for each image
	#integrated images are passed in as a 3-D np-array
	#calculate activations for all images BEFORE polarity is applied
	#only need to be called once
	def apply_filter(self, integrated_images):
		values = []
		for idx in range(integrated_images.shape[0]):
			values.append(self.apply_filter2image(integrated_images[idx, ...]))
		if (self.id + 1) % 100 == 0:
			print('Weak Classifier No. %d has finished applying' % (self.id + 1))
		return values

	#using this function to compute the error of
	#applying this weak classifier to the dataset given current weights
	#return the error and potentially other identifier of this weak classifier
	#detailed implementation is up you and depends
	#your implementation of Boosting_Classifier.train()
	@abstractmethod
	def calc_error(self, weights, labels):
		pass

	@abstractmethod
	def predict_image(self, integrated_image):
		pass

class Ada_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.polarity = None
		self.threshold = None

	def calc_error(self,weights,labels):
		def calc_err_at_t(t):

			predictions=np.sign(t - self.activations)
			err=((predictions == labels)*np.array(weights)).sum()
			# err=0
			# for i, activation in enumerate(self.activations):
			# 	classification = 1
			# 	if activation < t:
			# 		classification = -1
			# 	if classification != labels[i]:
			# 		err += weights[i]

			if err > 0.5:
				err = 1 - err
				polarity = -1
			else:
				polarity = 1
			return (err, polarity, t)

		step_size = 25

		steps = np.linspace(np.amin(self.activations), np.amax(self.activations), step_size)
		# error_list = Parallel(n_jobs=-1)(delayed(calc_err_at_t)(t) for t in steps)
		# min_err, self.polarity, self.threshold=min(error_list)
		predictions = np.sign((np.tile(self.activations, (step_size, 1)).transpose() - steps).transpose())
		comparison = (predictions != np.tile(labels,(step_size,1)))
		err_matrix=(comparison*weights).sum(axis=1)
		#err_matrix = np.matmul(comparison, weights)

		polarity_matrix=np.ones(err_matrix.shape[0])
		for i in range(err_matrix.shape[0]):
			if err_matrix[i]>0.5:
				err_matrix[i]=1-err_matrix[i]
				polarity_matrix[i]=-1
		# idx = err_matrix > 0.5
		# err_matrix[idx] = 1 - err_matrix[idx]
		# polarity_matrix[idx] = -1
		#polarity_matrix=np.sign(0.5-err_matrix)
		# polarity_matrix[polarity_matrix==0]=1
		# err_matrix=np.where(err_matrix>0.5,1-err_matrix,err_matrix)
		idx=np.argmin(err_matrix)
		min_err, self.polarity, self.threshold = err_matrix[idx], polarity_matrix[idx], steps[idx]
		return min_err, steps[idx], polarity_matrix[idx]

	def predict_label(self, activation, threshold=None, polarity=None):
		threshold = self.threshold if threshold is None else threshold
		polarity = self.polarity if polarity is None else polarity
		classification = -1
		if activation > threshold:
			classification = 1
		classification *= polarity
		return classification

	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		return self.polarity * np.sign(value - self.threshold)


class Real_Weak_Classifier(Weak_Classifier):
	def __init__(self, id, plus_rects, minus_rects, num_bins):
		super().__init__(id, plus_rects, minus_rects, num_bins)
		self.thresholds = None #this is different from threshold in ada_weak_classifier, think about it
		self.bin_pqs = None
		self.train_assignment = None

	def calc_error(self, weights, labels):
		######################
		######## TODO ########
		######################
		self.bin_pqs = np.ones((2, self.num_bins))*1e-6
		self.train_assignment = np.zeros(len(self.activations))
		self.thresholds = np.linspace(np.amin(self.activations), np.amax(self.activations), self.num_bins)
		self.train_assignment = np.digitize(self.activations, self.thresholds, right=True)
		for i,a in enumerate(self.activations):
			if labels[i] == 1:
				self.bin_pqs[0][self.train_assignment[i]] += weights[i]
			else:
				self.bin_pqs[1][self.train_assignment[i]] += weights[i]
		print("Classifier ",self.id, " p ",self.bin_pqs[0]," q ", self.bin_pqs[1])
		return self.bin_pqs

	def predict_image(self, integrated_image):
		value = self.apply_filter2image(integrated_image)
		bin_idx = np.sum(self.thresholds < value)
		return 0.5 * np.log(self.bin_pqs[0, bin_idx] / self.bin_pqs[1, bin_idx])


def main():
	plus_rects = [(1, 2, 3, 4)]
	minus_rects = [(4, 5, 6, 7)]
	num_bins = 50
	ada_hf = Ada_Weak_Classifier(plus_rects, minus_rects, num_bins)
	real_hf = Real_Weak_Classifier(plus_rects, minus_rects, num_bins)

if __name__ == '__main__':
	main()
