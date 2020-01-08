from abstract.abstract import RegionBasedSegmentationAbstract
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from config import LOGGER


class RegionBasedSegmentation(RegionBasedSegmentationAbstract):
    def __init__(self, image_path, threshold=None):
        super().__init__(image_path)
        self.threshold = threshold
        self.gray_image = self._segment()

    def _segment(self):
        LOGGER.info(f"Segment image based on gray scale values, all above threshold will be threaded as object, "
                    f"and all below threshold are background pixels...")
        gray_image = rgb2gray(self.image)
        gray_reshaped = gray_image.reshape(gray_image.shape[0] * gray_image.shape[1])

        self.threshold = gray_reshaped.mean() if self.threshold is None else self.threshold

        for i in range(gray_reshaped.shape[0]):
            if gray_reshaped[i] > gray_reshaped.mean():
                gray_reshaped[i] = 1
            else:
                gray_reshaped[i] = 0

        gray_image = gray_reshaped.reshape(gray_image.shape[0], gray_image.shape[1])
        return gray_image

    def plot(self):
        plt.imshow(self.gray_image, cmap='gray')
        plt.show()
