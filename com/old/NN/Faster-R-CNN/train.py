import time
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import lib.config.config as cfg
from lib.layer_utils.roi_data_layer import RoIDataLayer
from lib.nets.vgg16 import vgg16

from lib.utils.timer import Timer

try:
    import cPickle as pickle
except ImportError:
    import pickle
import os


def get_training_roidb(imdb):
    pass


def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        pass


class Train:
    def __init__(self):
        if cfg.FLAGS.network == 'vgg16':
            self.net = vgg16(batch_size=cfg.FLAGS.ims_per_batch)
        else:
            raise NotImplementedError

        self.imdb, self.roidb = combined_roidb("voc_2007_trainval")

        self.data_layer = RoIDataLayer(self.roidb, self.imdb.num_classes)
        self.ouput_dir = cfg.get_output_dir(self.imdb, 'default')

    def train(self):
        tf_config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=tf_config)
        with sess.graph.as_default():
            tf.set_random_seed(cfg.FLAGS.rng_seed)
            layers = self.net.create_architecture()
            loss = layers['total_loss']

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your chekpoint file has been compressed "
                      "with SNAPPY")

    def snapshot(self, sess, iter):
        if not os.path.exists(self.ouput_dir):
            os.makedirs(self.ouput_dir)

        # Store the model snapshot
        filename = 'vgg16_faster_rcnn_iter_{:d}'.format(iter) + ".ckpt"
        filename = os.path.join(self.ouput_dir, filename)
        self.saver.save(sess, filename)
        print("Wrote snapshot to {:s}".format(filename))

        nfilename = 'vgg16_faster_rcnn_iter_{:d'.format(iter) + ".pkl"
        nfilename = os.path.join(self.ouput_dir, nfilename)

        st0 = np.random.get_state()
        cur = self.data_layer._cur
        perm = self.data_layer._perm

        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename


if __name__ == '__main__':
    train = Train()
    train.train()
