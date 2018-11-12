import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import cv2

class Visualizer:
	def __init__(self, histogram_intervals, top_wc_intervals):
		self.histogram_intervals = histogram_intervals
		self.top_wc_intervals = top_wc_intervals
		self.weak_classifier_accuracies = {}
		self.strong_classifier_scores = {}
		self.labels = None
	
	def draw_histograms(self):
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			pos_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == 1]
			neg_scores = [scores[idx] for idx, label in enumerate(self.labels) if label == -1]

			bins = np.linspace(np.min(scores), np.max(scores), 100)

			plt.figure()
			plt.hist(pos_scores, bins, alpha=0.5, label='Faces')
			plt.hist(neg_scores, bins, alpha=0.5, label='Non-Faces')
			plt.legend(loc='upper right')
			plt.title('Using %d Weak Classifiers' % t)
			plt.savefig('histogram_%d.png' % t)

	def draw_rocs(self):
		plt.figure()
		for t in self.strong_classifier_scores:
			scores = self.strong_classifier_scores[t]
			fpr, tpr, _ = roc_curve(self.labels, scores)
			plt.plot(fpr, tpr, label = 'No. %d Weak Classifiers' % t)
		plt.legend(loc = 'lower right')
		plt.title('ROC Curve')
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('ROC Curve')

	def draw_wc_accuracies(self):
		plt.figure()
		for t in self.weak_classifier_accuracies:
			accuracies = self.weak_classifier_accuracies[t]
			plt.plot(accuracies, label = 'After %d Selection' % t)
		plt.ylabel('Accuracy')
		plt.xlabel('Weak Classifiers')
		plt.title('Top 1000 Weak Classifier Accuracies')
		plt.legend(loc = 'upper right')
		plt.savefig('Weak Classifier Accuracies')

	def display_haar(self, wcs, n):
		white=(255, 255, 255)
		black = (0, 0, 0)
		for i, (a,wc) in enumerate(wcs[:n]):
			img = np.ones((16,16,3), np.uint8)*200
			neg_c,pos_c = (black, white) if wc.polarity == 1 else (white, black)
			for pos in wc.plus_rects:
				x1, y1, x2, y2 = tuple(round(i) for i in pos)
				cv2.rectangle(img, (x1, y1),(x2, y2), pos_c, -1)
			for neg in wc.minus_rects:
				x1, y1, x2, y2 = tuple(round(i) for i in neg)
				cv2.rectangle(img, (x1, y1),(x2, y2), neg_c, -1)
			plt.imsave('Haar Filter {}'.format(i), img, origin='lower')
			#plt.imshow(img, origin='lower')
			#plt.show()
		return

if __name__ == '__main__':
	main()
