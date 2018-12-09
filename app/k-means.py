import numpy as np
from PIL import Image
import os
import time
from sklearn.cluster import KMeans as KM
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self, filepath, number_clusters=3):
        img = Image.open(filepath).convert('RGB')
        self.pixels = np.float32(img).reshape(-1, 3)
        self.k = number_clusters

    def kmeans(self):
        n = self.pixels.shape[0]
        centers = [self.pixels[0].tolist()]
        for value in self.pixels.tolist():
            if value not in centers:
                centers.append(value)
            if len(centers) == self.k:
                break

        centers = np.array(centers)
        distances = np.zeros((n, self.k))
        maxiter = 5

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
                centers[c] = np.average(pix_class, 0)

        return classes, centers


if __name__ == '__main__':
    dir_from = '../drosophila_kc167_1_images'
    dir_to = '../KMeansResults'
    number_clusters = [i for i in range(2, 70, 2)]  #[2, 4, 6, 8, 10, 12, 14, 16]
    for file_name in os.listdir(dir_from):
        array_time = []
        path_file = dir_from + '/' + file_name
        name_without_extension = file_name[:-4]
        dir_save = dir_to + '/' + name_without_extension + '/'
        # os.mkdir(dir_save)
        data = KMeans(path_file).pixels
        sse = []
        for K in number_clusters:
            path_file = '../drosophila_kc167_1_images/CPvalid1_48_40x_Tiles_p0003DAPI.TIF'
            #label, center = KMeans(path_file, K).kmeans()
            km = KM(K)
            km.fit(data)
            sse.append(km.inertia_)
        # Plot sse against k
        plt.figure(figsize=(6, 6))
        plt.plot(number_clusters, sse, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance');
        plt.show()
        exit()
            # center = np.uint8(center)
            # res = center[label].reshape((512, 512, 3))
            # Image.fromarray(res, 'RGB').save(dir_save + str(K) + '.bmp')
