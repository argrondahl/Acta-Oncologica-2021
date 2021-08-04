import h5py
import numpy as np
import pandas as pd
import os
import cv2
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from deoxys.data import BasePreprocessor
from deoxys.customize import custom_preprocessor


@custom_preprocessor
class ElasticDeformPreprocesser(BasePreprocessor):
    """
    """
    SHOULD_AUGMENT = True

    def __init__(self, alpha=None, sigma=None, alpha_affine=None):
        """
        Parameters:
        ----------
        alpha : int
            scaling factor. Controls the intensity of the deformation.
            If alpha > threshold, the displacement become close to a affine transformation.
            If alpha is very large (>> threshold) the displacement become translations.

        sigma : float
            standard deviation for the filter, given in voxels. Elasticity coefficient.

        alpha_affine : float
            distorting the image grid. The intensity of affine transformation applied.

        """
        self.alpha = 6.20 if alpha is None else alpha
        self.sigma = 0.38 if sigma is None else sigma
        self.alpha_affine = 0.10 if alpha_affine is None else alpha_affine
        self.max_rotation = 50 # in degrees

    def affine(self, image, random_state):

        """Perform affine transformation (rotation and shift) on image


            rotation = random.uniform(-self.max_rotation, self.max_rotation)*(np.pi/180)
            transformation = random.uniform(-transform_std, transform_std)

            return [np.cos(rotation), np.sin(rotation), transformation
                    -np.sin(rotation), np.cos(rotation), transformation]
        """
        shape = image.shape
        shape_size = shape[:2]
        transform_std = self.alpha_affine*image.shape[1]

        center_square = np.float32(shape[:2]) // 2
        square_size = min(shape[:2]) // 3

        source = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        destination = source + random_state.uniform(-transform_std, transform_std, size=source.shape).astype(np.float32)
        M = cv2.getAffineTransform(source, destination)

        return cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)

    def stretch_indices(self, image, random_state):
        """Get stretching indices
        """

        shape = image.shape
        alpha = self.alpha*shape[1]
        stretching_std = self.sigma*shape[1]

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), stretching_std) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), stretching_std) * alpha

        x, y, channel = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

        return np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(channel, (-1, 1))

    def elastic_transform(self, image, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_ (with modifications).
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
         https://www.microsoft.com/en-us/research/wp-content/uploads/2003/08/icdar03.pdf

        Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        Borrowed from: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        image = self.affine(image, random_state)
        #from ipdb import set_trace; set_trace()
        indices = self.stretch_indices(image, random_state)

        return map_coordinates(image, indices, order=1, mode='reflect').reshape(image.shape)

    def _transform(self, images, targets):
        if not self.SHOULD_AUGMENT:
            return images, targets
        if random.randint(0,100) < 50:
            images = images.copy()
            images = images.astype(np.float32, copy=False)

            imgs_tar = cv2.merge((images, targets.astype(np.float32, copy=False)))

            transformed = self.elastic_transform(imgs_tar).astype(np.int32, copy=False)
            deformed_imgs = transformed[...,:-1]
            deformed_tars = transformed[...,-1]

            return deformed_imgs, deformed_tars.reshape(deformed_tars.shape[0],deformed_tars.shape[1],1)
        else:
            return images, targets

    def transform(self, images, targets):
        for i, (image, target) in enumerate(zip(images, targets)):
            images[i], targets[i] = self._transform(images[i], targets[i])

        return images, targets
