import tensorflow as tf
import tensorflow.contrib.slim as slim

import lib.config.config as cfg
from lib.nets.network import Network


class vgg16(Network):
    def __init__(self, batch_size=1):
        Network.__init__(self, batch_size=batch_size)

    def build_network(self, sess, is_training=True):
        with tf.variable_scope("vgg16", "vgg16"):
            pass

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

    def fix_variables(self, sess, pretrained_model):
        print("Fix VGG16 layers")
        with tf.variable_scope('Fix_VGG16'):
            with tf.device('/cpu:0'):
                pass

    def build_head(self, is_training):
        pass

    def build_rpn(self, net, is_training, initializer):
        pass

    def build_proposals(self, is_training, rpn_cls_prob, rpn_bbox_pred, rpn_cls_score):
        pass

    def build_predictions(self, net, rois, is_training, initializer, initializer_bbox):
        pass
