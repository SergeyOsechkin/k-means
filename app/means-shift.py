import numpy as np
from PIL import Image
import math


def euclid_distance(x, xi):
    return np.sqrt(np.sum((x - xi)**2))


def neighbourhood_points(X, x_centroid, distance=5):
    eligible_X = []
    for x in X:
        distance_between = euclid_distance(x, x_centroid)
        # print('Evaluating: [%s vs %s] yield dist=%.2f' % (x, x_centroid, distance_between))
        if distance_between <= distance:
            eligible_X.append((x, distance_between))
    return eligible_X


def gaussian_kernel(distance, bandwidth):
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((distance / bandwidth))**2)
    return val


def means_shift():
    dir_from = '../drosophila_kc167_1_images'
    dir_to = '../MeanShiftResults'
    path_file = '../drosophila_kc167_1_images/CPvalid1_48_40x_Tiles_p0003DAPI.TIF'
    img = Image.open(path_file).convert('RGB')
    pixels = np.float32(img).reshape(-1, 3)
    n_iterations = 5
    for it in range(n_iterations):
        print("Begin iteration with number ", it)
        for i, x in enumerate(pixels):
            print(i)
            ### Step 1. For each datapoint x ∈ X, find the neighbouring points N(x) of x.
            neighbours = neighbourhood_points(pixels, x)
            print("Calculate neighbours is success")

            ### Step 2. For each datapoint x ∈ X, calculate the mean shift m(x).
            numerator = 0
            denominator = 0
            for neighbour, distance in neighbours:
                weight = gaussian_kernel(distance, 1.0)
                numerator += (weight * neighbour)
                denominator += weight

            new_x = numerator / denominator

            ### Step 3. For each datapoint x ∈ X, update x ← m(x).
            pixels[i] = new_x
            if i == 100:
                res = pixels.reshape((512, 512, 3))
                Image.fromarray(res, 'RGB').show()
                exit()
        print("The end iteration with number ", it)

    res = pixels.reshape((512, 512, 3))
    Image.fromarray(res, 'RGB').show()


if __name__ == '__main__':
    import numpy as np
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.datasets.samples_generator import make_blobs

    # #############################################################################
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

    # #############################################################################
    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt
    from itertools import cycle

    plt.figure(1)
    plt.clf()

    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
        plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
