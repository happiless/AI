from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from layer_utils.proposal_top_layer import proposal_top_layer
from lib.config import config as cfg


class Network(object):

    def __init__(self, batch_size=1):
        self._feat_stride = [16, ]
        self._feat_compress = [1. / 16., ]
        self._batch_size = batch_size
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._act_summaries = {}
        self._score_summaries = {}
        self._train_summaries = {}
        self._event_summaries = {}
        self._variables_to_fix = {}

    # summaries
    def _add_image_summary(self, image, boxes):
        # add back mean
        image += cfg.FLAGS2['pixel_means']
        # bgr to rgb (opencv uses bgr)
        channels = tf.unstack(image, axis=1)
        image = tf.stack([channels[2], channels[1], channels[0]], axis=1)
        # dims for normalization
        width = tf.to_float(tf.shape(image[2]))
        height = tf.to_float(tf.shape(image[1]))
        # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1] / height,
                          cols[0] / width,
                          cols[3] / height,
                          cols[2] / width], axis=1)
        boxes = tf.expand_dims(boxes, dim=0)
        image = tf.image.draw_bounding_boxes(image, boxes)

        return tf.summary.image('ground_truth', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + 'zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name):
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[self._batch_size], [num_dim, -1], [input_shape[2]]]))
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name == 'rpn_cls_prob_reshape':
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name):
            rois, rpn_scores = tf.py_func(proposal_top_layer,
                                          rpn_cls_prob, rpn_bbox_pred,
                                          [self._im_info, self._mode, self._feat_stride,
                                           self._anchors, self._num_anchors],
                                          [tf.float32, tf.float32])
            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name):
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bboxes
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = cfg.FLAGS.roi_pooling_size * 2
            crops = tf.image.crop_and_resize(bottom, bboxes,
                                             tf.to_int32(batch_ids),
                                             [pre_pool_size, pre_pool_size],
                                             name='crops')
        return slim.max_pool2d(crops, [2, 2], padding='SAME')

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        with tf.variable_scope(name):
            pass

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name):
            pass

    def _anchor_component(self):
        with tf.variable_scope("ANCHOR_default"):
            pass

    def build_network(self, sess, is_training=True):
        raise NotImplementedError

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights,
                        bbox_outside_weights, similar=1.0, dim=[1]):
        pass

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope("loss_" + self.t_ag):
            pass

    def create_architecture(self, sess, mode, num_classes, tag=None,
                            anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[self._batch_size, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[self._batch_size, 3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.tag = tag

        self._num_classes = num_classes
        self._mode = mode
        self._anchor_scales = anchor_scales
        self._num_scales = len(anchor_scales)

        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)

        self.num_anchors = self._num_scales * self._num_ratios

        # 返回的training或者testing为True或者false
        training = mode == 'TRAIN'
        testing = mode == 'TEST'
        assert tag != None

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    def test_image(self, sess, image, im_info):
        pass

    def get_summary(self, sess, blobs):
        pass

    def train_step(self, sess, blobs, train_op):
        pass

    def train_step_with_summary(self, sess, blobs, train_op):
        pass

    def train_step_no_return(self, sess, blobs, train_op):
        pass

