from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np

from lib.config import config as cfg


class RoIDataLayer(object):
    def __init__(self, roidb, num_classes, random=False):
        self._roidb = roidb
        self._num_classes = num_classes
        self._random = random
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)

        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        # Restore the random state
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.FLAGS.ims_per_batch >= len(self._roidb):
            self._shuffle_roidb_inds()
        db_inds = self._perm[self._cur:self._cur + cfg.FLAGS.ims_per_batch]
        self._cur += cfg.FLAGS.ims_per_batch
        return db_inds

    def _get_next_minibatch(self):
        pass

    def forward(self):
        return self._get_next_minibatch()
