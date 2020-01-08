import os
import cv2
from os import listdir
from os.path import isfile, join
from abc import abstractmethod
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from config import LOGGER


class RegionBasedSegmentationAbstract(object):

    def __init__(self, image_path):
        try:
            self.image = plt.imread(image_path)

        except Exception as e:
            LOGGER.error(f"Error with loading image {image_path}")
            LOGGER.error(f"Reason: {e}")

    @abstractmethod
    def plot(self):
        raise NotImplementedError


class EdgeSegmentationAbstract(object):
    def __init__(self, images_path):
        self.images = []
        self.images_directory = images_path
        if os.path.isfile(images_path):
            try:
                self.images.append([rgb2gray(plt.imread(images_path)), images_path])
            except Exception as e:
                LOGGER.error(f"Error with loading {images_path}")
                LOGGER.error(f"Reason: {e}")
        else:
            if images_path[-1] != "/":
                images_path += "/"
            images = [f for f in listdir(images_path) if isfile(join(images_path, f))]
            for image in images:
                try:
                    self.images.append([rgb2gray(plt.imread(f"{images_path}{image}")), image])
                except Exception as e:
                    LOGGER.error(f"Error with loading {image}")
                    LOGGER.error(f"Reason: {e}")

    @abstractmethod
    def _gaussian_kernel(self, size, sigma):
        raise NotImplementedError

    @abstractmethod
    def _sobel_filters(self, img):
        raise NotImplementedError

    @abstractmethod
    def _non_max_suppression(self, img, D):
        raise NotImplementedError

    @abstractmethod
    def _threshold(self, img):
        raise NotImplementedError

    @abstractmethod
    def _hysteresis(self, img):
        raise NotImplementedError


class KMEANSClusteringAbstract(object):
    def __init__(self, image_path):
        original = cv2.imread(image_path)
        if original is None:
            raise Exception(f"Unable to open image. Check your directory {image_path}")
        image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        self.image = image

    def cluster(self):
        raise NotImplementedError

    def plot_detected(self, img1, img2):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError


class SuperPixelSegmentationAbstract(object):
    def __init__(self, image_path):
        try:
            self.image = img_as_float(io.imread(image_path))
        except Exception as e:
            LOGGER.error(f"Unable to open image {image_path}")
            LOGGER.error(f"Reason {e}")

    def segment(self):
        raise NotImplementedError

    def plot(self):
        raise NotImplementedError
