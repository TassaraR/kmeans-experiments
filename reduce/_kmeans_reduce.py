import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


class KMeansReduction:
    """
    Compresses images by reducing the amount of colors
    using KMeans Algorithm.
    Can also performed reduced color transfer from image
    to image

    Parameters
    ----------
    im : np.ndarray
        Input image

    sample_ratio : float, default=0.5
        Percentage of the image that will be used to
        create the clusters
    """
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
        """
        Changes the percentage of the image to
        be used to create clusters

        Parameters
        ----------
        size : float
            Input percentage to be used.
        """
        sample_size = int(self._arr_im.shape[0] * size)
        self._samples = shuffle(self._im)[:sample_size]

    def fit(self, n_colors):
        """
        Applies KMeans algorithm and reduces colors

        Parameters
        ----------
        n_colors : int
            Amount of colors of the output image
        """
        self._reduced_colors = n_colors
        self._km = KMeans(n_clusters=n_colors)
        self._km.fit(self._samples)
        self._centers = self._km.cluster_centers_

        recolor_idx = self._km.predict(self._arr_im)
        recolor_arr = self._centers[recolor_idx]
        self._new_im = recolor_arr.reshape(self._im.shape)

    def get_reduced(self):
        """
        Returns the color-reduced image

        Returns
        -------
        new_img : np.ndarray or None
            color-reduced image
        """
        return self._new_im

    def fit_transform(self, n_colors):
        """
        Runs fit method and returns the color-reduced image

        Parameters
        ----------
        n_colors : int
            Amount of colors the input image will be reduced

        Returns
        -------
        new_img : np.ndarray
            color-reduced image
        """
        self.fit(n_colors)
        return self._new_im

    def reduction_info(self):
        """
        Displays color data about the reduction process

        Returns
        -------
        info: dict
            Returns the the images original amount of colors,
            reduced colors and reduced percentage

        """
        if self._total_colors is None:
            self._total_colors = np.unique(self._arr_im,
                                           axis=1).shape[0]
        diff = 1 - (self._reduced_colors / self._total_colors)
        info = dict(total_colors=self._total_colors,
                    reduced_colors=self._reduced_colors,
                    reduced_pct=diff)
        return info

    def transfer_reduced(self, im):
        """
        transfers the reduced colors of the original image
        onto another image

        Returns
        -------
        new_img : np.ndarray
            color-transfered image
        """
        arr_im = im.reshape(-1, 3)
        new_colors = self._km.predict(arr_im)
        new_im = self._centers[new_colors].reshape(im.shape)
        return new_im
