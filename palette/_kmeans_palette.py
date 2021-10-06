import numpy as np
import math
from collections.abc import Iterable
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansPalette:
    """
    Extracts most representative colors from an image using
    K-Means clustering Algorithm

    Parameters
    ----------
    im : np.ndarray
        Input image

    im_resize : int, default=200 or smallest image side
        The original image will be resized as extracting
        palettes from the whole image is expensive.
        A small image may be less precise.
    """
    def __init__(self, im, im_resize=200):

        if im.shape[2] != 3:
            raise ValueError('Images must have 3 channels')

        self._im = im

        h, w, d = im.shape
        small_side = min(h, w)
        if small_side > im_resize:
            # Resize changes images from 0-255 to 0.0-1.0
            self._im_small = resize(im, (im_resize, im_resize))
        else:
            self._im_small = resize(im, (small_side, small_side))

        self._n_colors = None
        self._colors = None

    def fit(self, n_colors):
        """
        Computes clusters by applying K-Means

        Parameters
        ----------
        n_colors : int
            Number of colors / clusters of the palette
        """
        im_reshaped = self._im_small.reshape(-1, 3)
        km = KMeans(n_clusters=n_colors)
        km.fit(im_reshaped)

        centers = km.cluster_centers_
        labels = np.bincount(km.labels_)
        relevance = np.argsort(labels)
        self._n_colors = n_colors
        self._colors = centers[relevance]

    def fit_best(self, test_range, apply=True, get_scores=False, verbose=True):
        """
        Searches and / or returns the best number of colors for the palette
        given an array using silhouette score

        Parameters
        ----------
        test_range : int or iterable
            number of clusters / colors to iterate over

        apply : bool, default=True
            Fits the model with K-Means using the number of colors with the
            highest silhouette score

        get_scores: bool, default=False
            Returns silhouette scores

        verbose : bool, default=True
            Prints silhouette scores on screen
        """
        if isinstance(test_range, int):
            n_clusters = range(2, test_range + 1)
        elif isinstance(test_range, Iterable):
            n_clusters = sorted(list(test_range))
        else:
            raise ValueError("test_range must be an integer or iterable")

        im_reshaped = self._im_small.reshape(-1, 3)

        sil_scores = []
        for centers in n_clusters:
            km = KMeans(n_clusters=centers)
            pred = km.fit_predict(im_reshaped)
            sil = silhouette_score(im_reshaped, pred)
            sil_scores.append((centers, sil))

        best_idx = np.argmax([x[1] for x in sil_scores])
        best_clusters = sil_scores[best_idx][0]

        if apply:
            self.fit(n_colors=best_clusters)

        if verbose:
            print(f'Best n_clusters: {best_clusters}')

        if get_scores:
            return sil_scores

    def get_palette(self, as_float=True):
        """
        Returns palette if trained

        parameters
        ----------
        as_float : bool, default=True
            Returns the palette either in a 0.0-1.0 or 0-255 range

        Returns
        -------
        colors : np.ndarray of shape (n_colors, RGB Channels)
            Palette of colors
        """
        if as_float:
            return self._colors
        return (self._colors * 255).round().clip(0, 255).astype(np.uint8)

    def plot_palette(self, imsize=7, ncols=None, include_img=True):
        """
        Plots the palette with or without the original image

        parameters
        ----------
        imsize : int, default=7
            size of the matplotlib figure

        ncols : int, default=None
            Number of columns for the palette matrix plot

        include_img : bool, default=True
            Whether to include or not the original image
        """
        if self._colors is None:
            raise ValueError("Model must be fitted in order to plot results")

        block_size = 20
        offset = 5
        total_colors = self._n_colors

        if include_img:
            fig, ax = plt.subplots(2, 1, facecolor='white')
            ax[0].imshow(self._im, aspect='equal')
            ax[0].axis('off')
            width = ax[0].bbox.width
        else:
            fig, ax = plt.subplots()

        if not ncols:
            if include_img:
                ncols = int(width // block_size)
            else:
                ncols = 10
        nrows = math.ceil(total_colors / ncols)

        num_color = 0
        for i in range(nrows, 0, -1):
            for j in range(ncols):
                row = i * (block_size + offset)
                col = j * (block_size + offset)
                rectangle = plt.Rectangle((col, row),
                                          block_size, block_size, fc=self._colors[num_color])

                if include_img:
                    ax[1].add_patch(rectangle)
                else:
                    ax.add_patch(rectangle)

                if num_color + 1 == total_colors:
                    break
                num_color += 1

        if include_img:
            ax[1].axis('scaled')
            ax[1].axis('off')
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
            plt.tight_layout()
        else:
            ax.axis('scaled')
            ax.axis('off')
        fig.set_size_inches(imsize, imsize)
        plt.show()
