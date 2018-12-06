import numpy as np
from PIL import Image
from sklearn.cluster import mean_shift


if __name__ == '__main__':
    dir_from = '../drosophila_kc167_1_images'
    dir_to = '../MeanShiftResults'
    path_file = '../drosophila_kc167_1_images/CPvalid1_48_40x_Tiles_p0003DAPI.TIF'
    img = Image.open(path_file).convert('RGB')
    pixels = np.float32(img).reshape(-1, 3)
    center, label = mean_shift(pixels, max_iter=1)
    center = np.uint8(center)
    res = center[label].reshape((512, 512, 3))
    Image.fromarray(res, 'RGB').show()
