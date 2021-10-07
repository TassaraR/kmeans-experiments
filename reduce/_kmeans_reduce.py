import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


class KMeansReduction:
    def __init__(self, im, sample_ratio=0.5):

        self._im = im
        self._arr_im = self._im.reshape(-1, 3)
        if sample_ratio >= 1:
            self._samples = self._arr_im
        else:
            sample_size = int(self._arr_im.shape[0] * sample_ratio)
            self._samples = shuffle(self._arr_im)[:sample_size]

        self._km = None
        self._centers = None
        self._total_colors = None
        self._reduced_colors = None
        self._new_im = None

    def resample(self, size):
        sample_size = int(self._arr_im.shape[0] * size)
        self._samples = shuffle(self._im)[:sample_size]

    def fit(self, n_colors):

        self._reduced_colors = n_colors
        self._km = KMeans(n_clusters=n_colors)
        self._km.fit(self._samples)
        self._centers = self._km.cluster_centers_

        recolor_idx = self._km.predict(self._arr_im)
        recolor_arr = self._centers[recolor_idx]
        self._new_im = recolor_arr.reshape(self._im.shape)

    def get_reduced(self):
        return self._new_im

    def fit_transform(self, n_colors):
        self.fit(n_colors)
        return self._new_im

    def reduction_info(self):
        self._total_colors = np.unique(self._arr_im,
                                       axis=1).shape[0]
        diff = 1 - (self._reduced_colors / self._total_colors)
        info = dict(total_colors=self._total_colors,
                    reduced_colors=self._reduced_colors,
                    reduced_pct=diff)
        return info

    def transfer_reduced(self, im):
        arr_im = im.reshape(-1, 3)
        new_colors = self._km.predict(arr_im)
        new_im = self._centers[new_colors].reshape(im.shape)
        return new_im
