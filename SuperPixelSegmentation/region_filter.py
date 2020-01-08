from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from abstract.abstract import SuperPixelSegmentationAbstract


class SuperPixelSegmentation(SuperPixelSegmentationAbstract):

    def __init__(self,
                image_path,
                numbers_of_regions=None
                ):
        super().__init__(image_path)
        self.numbers_regions = [100, 200, 300] if numbers_of_regions is None \
            else numbers_of_regions
        self.a = 0

    def segment(self):
        for num_segments in self.numbers_regions:
            segments = slic(self.image, n_segments=num_segments, sigma=5)
            fig = plt.figure("Superpixels -- %d segments" % num_segments)
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(self.image, segments))
            plt.axis("off")
        return plt

    def plot(self):
        segemnet = self.segment()
        segemnet.show()