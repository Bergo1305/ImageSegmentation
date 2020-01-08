import matplotlib.pyplot as plt
import cv2
import numpy as np
from abstract.abstract import KMEANSClusteringAbstract


class KMEANSClusteringSegmentation(KMEANSClusteringAbstract):
    def __init__(self,
                 image_path,
                 number_of_clusters,
                 attempt=10
                 ):
        super().__init__(image_path)
        self.clusters = number_of_clusters
        self.attempt = attempt
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    def cluster(self):

        vectorized = np.float32(self.image.reshape((-1, 3)))
        result, label, center = cv2.kmeans(vectorized,
                                           self.clusters,
                                           None,
                                           self.criteria,
                                           self.attempt,
                                           cv2.KMEANS_PP_CENTERS
                                           )
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape(self.image.shape)
        return self.image, result_image

    def plot_detected(self, image, result_image):
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1), plt.imshow(image)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(result_image)
        plt.title('Segmented Image when K = %i' % self.clusters), plt.xticks([]), plt.yticks([])
        plt.show()

    def plot(self):
        img, detected = self.cluster()
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1), plt.imshow(img)
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(detected)
        plt.title('Segmented Image when K = %i' % self.clusters), plt.xticks([]), plt.yticks([])
        plt.show()
