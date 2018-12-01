import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2


class Segmentation:
    def __init__(self, filepath, number_clusters=3):
        img = Image.open(filepath).convert('RGB')
        self.pixels = np.float32(img).reshape(-1, 3)
        self.k = number_clusters

    def kmeans(self):
        n = self.pixels.shape[0]
        c = self.pixels.shape[1]  # Number of features in the data
        centers = np.random.randn(self.k, c)
        distances = np.zeros((n, self.k))
        maxiter = 1

        for i in range(maxiter):

            # Assign all points to the nearest centroid
            for j, c in enumerate(centers):
                distances[:, j] = np.linalg.norm(self.pixels - c, axis=1)

            # Determine class membership of each point
            # by picking the closest centroid
            classes = np.argmin(distances, axis=1)

            # Update centroid location using the newly
            # assigned data point classes
            for c in range(self.k):
                pix_class = self.pixels[classes == c]
                centers[c] = np.mean(pix_class, 0)

        return classes, centers


if __name__ == '__main__':
    FILE_IN = '../drosophila_kc167_1_images/CPvalid1_48_40x_Tiles_p0003DAPI.TIF'
    K = [2, 5, 7, 9]
    img = cv2.imread(FILE_IN)
    Z = img.reshape((-1, 3))
    for number_clusters in K:
        segm = Segmentation(FILE_IN, number_clusters)
        label, center = segm.kmeans()

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((512, 512, 3))
        plt.imshow(res2)
        plt.show()


