import os
import os.path as osp
import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
FLAGS2 = {}

FLAGS2['pixel_means'] = np.array([[[102.9801, 115.9465, 122.7717]]])
tf.app.flags.DEFINE_integer('rng_seed', 3, "Tensorflow seed for reproducibility")

# network
tf.app.flags.DEFINE_string('network', 'vgg16', 'The network to be used as backbone')

# train
tf.app.flags.DEFINE_float('weight_decay', 0.0005, 'Weight decay for regularization')
tf.app.flags.DEFINE_integer('ims_per_batch', 1, 'Images to use per minibatch')


# rpn
tf.app.flags.DEFINE_integer('rpn_top_n', 300, 'Only useful when TEST.MODE is "top", specifies the number of top proposals to select')


# roi
tf.app.flags.DEFINE_integer('roi_pooling_size', 7, 'Size of the pooled region after roi pooling')


FLAGS2['root_dir'] = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
FLAGS2['root_dir'] = osp.abspath(osp.join(FLAGS2['root_dir'], 'data'))


def get_output_dir(imdb, weights_filename):
    outdir = osp.abspath(osp.join(FLAGS2['root_dir'], FLAGS2['root_dir'],  'default', imdb.name))
    if weights_filename is  None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir
