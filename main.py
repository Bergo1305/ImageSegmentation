from config import ROOT_DIRECTORY
from ClusteringSegmentation.cluster_filter import KMEANSClusteringSegmentation
from EdgeSegmentation.edge_filter import EdgeSegmentation
from SuperPixelSegmentation.region_filter import SuperPixelSegmentation


IMAGE_PATH = ROOT_DIRECTORY + "/data/muscled.jpg"

if __name__ == "__main__":

    edge_segmentation = EdgeSegmentation(IMAGE_PATH)
    clustering_segmentation = KMEANSClusteringSegmentation(IMAGE_PATH, number_of_clusters=3)
    superpixel_segmentation = SuperPixelSegmentation(IMAGE_PATH)

    edge_segmentation.plot()
    clustering_segmentation.plot()
    superpixel_segmentation.plot()
