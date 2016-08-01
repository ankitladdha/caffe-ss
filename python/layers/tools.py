"""
Author: Ankit Laddha (aladdha@andrew.cmu.edu)

Functions to be used in all of the python layers
"""
__author__ = "Ankit Laddha <aladdha@andrew.cmu.edu>"


import numpy as np


class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.

    Code taken from the caffe examples
    """

    def __init__(self, mean=[128, 128, 128], scale=1.0):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = scale

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occuring in the vgg16 caffe
        prototxt
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)