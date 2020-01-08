import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
from abstract.abstract import EdgeSegmentationAbstract
from config import ROOT_DIRECTORY, LOGGER, KERNEL_HORIZONTAL, KERNEL_VERTICAL, LAPLACIAN


class EdgeSegmentation(EdgeSegmentationAbstract):
    def __init__(self,
                 images_directory,
                 output_directory=None,
                 sigma=1,
                 kernel_size=5,
                 weak_pixel=75,
                 strong_pixel=255,
                 low_threshold=0.05,
                 high_threshold=0.15
                 ):
        super().__init__(images_directory)
        self.detected_images = {}
        self.img_smoothed = None
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholdImg = None
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.lowThreshold = low_threshold
        self.highThreshold = high_threshold
        self.output_directory = output_directory

    def _gaussian_kernel(self, size, sigma=2):
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma * sigma)
        return np.exp(-((x * x + y * y) /
                        (2.0 * sigma * sigma))) * normal

    def _gradient(self, img):
        ix = ndimage.filters.convolve(img, KERNEL_HORIZONTAL)
        iy = ndimage.filters.convolve(img, KERNEL_VERTICAL)

        g = np.hypot(ix, iy)
        g = g / g.max() * 255
        theta = np.arctan2(iy, ix)
        return g, theta

    def _non_max_suppression(self, img, d):
        m, n = img.shape
        z = np.zeros((m, n), dtype=np.int32)
        angle = d * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, m - 1):
            for j in range(1, n - 1):
                try:
                    q = 255
                    r = 255

                    if 0 <= angle[i, j] < 22.5 or \
                            157.5 <= angle[i, j] <= 180:
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    elif 22.5 <= angle[i, j] < 67.5:
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    elif 67.5 <= angle[i, j] < 112.5:
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    elif 112.5 <= angle[i, j] < 157.5:
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if img[i, j] >= q and img[i, j] >= r:
                        z[i, j] = img[i, j]
                    else:
                        z[i, j] = 0

                except IndexError as e:
                    LOGGER.error(f"Reason: {e}")

        return z

    def _threshold(self, img):
        high_threshold = img.max() * self.highThreshold
        low_threshold = high_threshold * self.lowThreshold
        res = np.zeros(img.shape, dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= high_threshold)
        weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res

    def _hysteresis(self, img):
        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if img[i, j] == weak:
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or
                                (img[i + 1, j + 1] == strong)or (img[i, j - 1] == strong) or
                                (img[i, j + 1] == strong) or (img[i - 1, j - 1] == strong)
                                or (img[i - 1, j] == strong) or (img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass

        return img

    def detect(self):
        for img in self.images:
            [image, image_name] = [img[0], img[1]]
            self.img_smoothed = convolve(image, self._gaussian_kernel(self.kernel_size, self.sigma))
            self.gradientMat, self.thetaMat = self._gradient(self.img_smoothed)
            self.nonMaxImg = self._non_max_suppression(self.gradientMat, self.thetaMat)
            self.thresholdImg = self._threshold(self.nonMaxImg)
            img_final = self._hysteresis(self.thresholdImg)
            self.detected_images[f"{image_name}"] = img_final

        return self.detected_images

    def plot(self):
        detected = self.detect()
        if len(detected) == 1:
            for name, detected_img in detected.items():
                name = name
                detected_img = detected_img
            image = plt.imread(self.images_directory)
            plt.figure(figsize=(15, 15))
            plt.subplot(1, 2, 1), plt.imshow(image)
            plt.title("Originalna slika"), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 2, 2), plt.imshow(detected_img)
            plt.title("Detekcija ivica"), plt.xticks([]), plt.yticks([])
            plt.show()
        else:
            for name, detected_image in self.detected_images.items():
                image = plt.imread(f"{self.images_directory}/{name}")
                plt.figure(figsize=(15, 15))
                plt.subplot(1, 2, 1), plt.imshow(image)
                plt.title("Originalna slika"), plt.xticks([]), plt.yticks([])
                plt.subplot(1, 2, 2), plt.imshow(detected_image)
                plt.title("Detekcija ivica"), plt.xticks([]), plt.yticks([])
                plt.show()

    def plot_two(self, image_url_1, image_url_2, tekst=None, original_tekst="Original image"):
        image = plt.imread(image_url_1)
        detected = plt.imread(image_url_2)
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1), plt.imshow(image)
        plt.title(original_tekst), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(detected)
        plt.title(tekst), plt.xticks([]), plt.yticks([])
        plt.show()

    def save(self):
        self.detected_images = self.detect()
        if self.output_directory is not None:
            if not os.path.isdir(self.output_directory):
                os.mkdir(self.output_directory)
            for name, detected_image in self.detected_images.items():
                cv2.imwrite(f"{self.output_directory}/{name}", detected_image)
        else:
            LOGGER.warning("Please provide output directory where images will be saved...")

    def detect_horizontal(self, show=False):
        if self.output_directory is None:
            raise LOGGER.warning("Please provide output directory where images will be saved...")
        if not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)
        for img in self.images:
            [image, image_name] = [img[0], img[1]]
            out_h = ndimage.convolve(image, KERNEL_HORIZONTAL, mode='reflect')
            out_h *= 255
            cv2.imwrite(f"{self.output_directory}/{image_name}", out_h)
            if show:
                self.plot_two(f"{ROOT_DIRECTORY}/data/{image_name}",
                              f"{ROOT_DIRECTORY}/detected/{image_name}",
                              tekst="Horizontal line detection")

    def detect_vertical(self, show=False):
        if self.output_directory is None:
            raise LOGGER.warning("Please provide output directory where images will be saved...")
        if not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)
        for img in self.images:
            [image, image_name] = [img[0], img[1]]
            out_h = ndimage.convolve(image, KERNEL_VERTICAL, mode='reflect')
            out_h *= 255
            cv2.imwrite(f"{self.output_directory}/{image_name}", out_h)
            if show:
                self.plot_two(f"{ROOT_DIRECTORY}/data/{image_name}",
                              f"{ROOT_DIRECTORY}/detected/{image_name}",
                              tekst="Vertical line detection")
