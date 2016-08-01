"""
Data layer to load images and semantic labels from a list of images.

The prototxt could include the following parameters:
1. batch_size: Number of data points in each batch
2. im_size: Size of the image to be used.
             It will resize the image after loadingW
3. image_dir: Directory where images are present
4. gt_dir: Directory where the ground truth is present
5. file_list: The list of files to be used for training
6. image_ext: Extension of the images (e.g. jpg, png)
7. nclasses: Number of Classes
8. bgr_mean: Mean vector for preprocessing
9. mirror: Should is use mirroring for data augmentation?
10. random_seed: Seed for the random number generator
"""

from __future__ import division
from __future__ import print_function

__author__ = "Ankit Laddha <aladdha@andrew.cmu.edu>"

# imports
import caffe
import numpy as np
import yaml
import os
import random as rd
from tools import SimpleTransformer
import skimage.io
from multiprocessing import Process, Queue


class ImageSegData(caffe.Layer):

    def setup(self, bottom, top):
        """
        Setup the Data Layer. It will initialize the data blob
        """

        # Load the layer paramters
        layer_params = yaml.load(self.param_str)

        # Start a single worker CPU thread. This allows to load the data for next
        # batch while the current one is being processed in the GPU
        # - Multiple threads not supported
        self._q = Queue(1)
        self._worker = Process(target=worker, args=(self._q, layer_params))
        self._worker.start()

        # These paramters need to be there
        batch_size = layer_params['batch_size']
        im_size = layer_params['im_size']

        # Reshape the blobs
        assert (len(top) == 2), 'Top should have two blobs'
        top[0].reshape(batch_size, 3, im_size[0], im_size[1]) # Image
        top[1].reshape(batch_size, 1, im_size[0], im_size[1]) # GT

    def forward(self,  bottom, top):
        """
        Load the data into the CNN
        """

        # Wait for the worker to fill the queue
        while self._q.empty():
            pass

        # Get the data
        (ims, gts) = self._q.get()

        top[0].data[...] = ims
        top[1].data[...] = gts

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def worker(q, params):
    """
    Functionality of the worker thread

    :param q: Queue of length 1. Longer queues are not supported
    :param params: Layer paramters for the batchloader
    :return:
    """
    batch_loader = BatchLoader(params)
    while True:
        if q.empty():
            q.put(batch_loader.next_batch())


class BatchLoader():
    """
    Class definition for loading a batch of data. Use this class with a worker
    thread
    """
    def __init__(self, params):
        self._image_dir = params['image_dir']
        self._gt_dir = params['gt_dir']

        self._file_list = params['file_list']
        self._image_ext = params['image_ext']

        self._nclasses = params['nclasses']
        assert self._nclasses >= 1

        self._batch_size = params['batch_size']
        assert self._batch_size >= 1

        self._bgr_mean = np.array(params['bgr_mean'])

        # If _im_size is none then we assume that the batch size is 1
        if 'im_size' in params:
            self._im_size = np.array(params['im_size'])
        else:
            self._im_size = None
            self._batch_size = 1

        if 'mirror' in params:
            self._mirror = params['mirror'] == 1
        else:
            self._mirror = False

        if 'random_seed' in params:
            rd.seed(params['random_seed'])

        # Read the file list
        fid = open(self._file_list, 'r')
        self._list = [f.strip() for f in fid]
        rd.shuffle(self._list)

        self._transformer = SimpleTransformer(mean=self._bgr_mean)
        self._idx = 0

    def load_one_data_point(self, fname):
        """
        Load a single data point and preprocess it.

        Parameters
        ----------
        fname: the file for which we need to load the data

        Returns
        -------
        im: Processed Image
        gt: Processed Semantic Labeling
        """
        im_name = os.path.join(self._image_dir,
                               '{}.{}'.format(fname, self._image_ext))
        gt_name = os.path.join(self._gt_dir, '{}.txt'.format(fname))

        im = skimage.io.imread(im_name)
        gt = np.loadtxt(gt_name, delimiter=' ')

        [h, w, _] = im.shape

        assert h > self._im_size[0]
        assert w > self._im_size[1]

        h_start_idx = rd.randint(0, h - self._im_size[0])
        h_end_idx = h_start_idx + self._im_size[0]

        w_start_idx = rd.randint(0, w - self._im_size[1])
        w_end_idx = w_start_idx + self._im_size[1]

        final_im = im[h_start_idx:h_end_idx, w_start_idx:w_end_idx, :]

        final_gt = gt[h_start_idx:h_end_idx, w_start_idx:w_end_idx]

        if self._mirror and rd.randint(0, 1) == 1:
            final_im = final_im[:, ::-1, :]
            final_gt = final_gt[:, ::-1]

        final_im = self._transformer.preprocess(final_im)
        return final_im, final_gt

    def get_fname(self):
        """
        Randomly select next file to process.

        Get the next file in the list to load in the CNN. If it finishes
        the list then it randomly shuffles the list. This process ensures
        that we look at each file in the list in each epoch and we randomly
        select files rather than a fixed order.

        Returns
        -------
        fname: A randomly Selected filename
        """
        if self._idx >= len(self._list):
            rd.shuffle(self._list)
            self._idx = 0
        fname = self._list[self._idx]
        self._idx += 1
        return fname

    def next_batch(self):
        """
        Get the next batch to process

        Returns
        -------
        ims: numpy array of the images
        gts: numpy array of labels
        """
        if self._batch_size == 1:
            fname = self.get_fname()
            (ims, gts) = self.load_one_data_point(fname)
        else:
            ims = np.zeros((self._batch_size, 3, self._im_size[0],
                           self._im_size[1]))
            gts = np.zeros((self._batch_size, 1, self._im_size[0],
                               self._im_size[1]))
            for iter in range(self._batch_size):
                fname = self.get_fname()
                (im, gt) = self.load_one_data_point(fname)
                ims[iter, :] = im
                gts[iter, :] = gt

        return ims, gts
