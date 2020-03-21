# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from libs.box_utils.coordinate_convert import forward_convert


def ious_calu(boxes_1, boxes_2):
    '''

    :param boxes_1: [N, 4] [xmin, ymin, xmax, ymax]
    :param boxes_2: [M, 4] [xmin, ymin. xmax, ymax]
    :return:
    '''
    boxes_1 = tf.cast(boxes_1, tf.float32)
    boxes_2 = tf.cast(boxes_2, tf.float32)
    xmin_1, ymin_1, xmax_1, ymax_1 = tf.split(boxes_1, 4, axis=1)  # xmin_1 shape is [N, 1]..
    xmin_2, ymin_2, xmax_2, ymax_2 = tf.unstack(boxes_2, axis=1)  # xmin_2 shape is [M, ]..

    max_xmin = tf.maximum(xmin_1, xmin_2)
    min_xmax = tf.minimum(xmax_1, xmax_2)

    max_ymin = tf.maximum(ymin_1, ymin_2)
    min_ymax = tf.minimum(ymax_1, ymax_2)

    overlap_h = tf.maximum(0., min_ymax - max_ymin)  # avoid h < 0
    overlap_w = tf.maximum(0., min_xmax - max_xmin)

    overlaps = overlap_h * overlap_w

    area_1 = (xmax_1 - xmin_1) * (ymax_1 - ymin_1)  # [N, 1]
    area_2 = (xmax_2 - xmin_2) * (ymax_2 - ymin_2)  # [M, ]

    ious = overlaps / (area_1 + area_2 - overlaps)

    return ious


def clip_boxes_to_img_boundaries(decode_boxes, img_shape):
    '''

    :param decode_boxes:
    :return: decode boxes, and already clip to boundaries
    '''

    with tf.name_scope('clip_boxes_to_img_boundaries'):

        # xmin, ymin, xmax, ymax = tf.unstack(decode_boxes, axis=1)
        xmin = decode_boxes[:, 0]
        ymin = decode_boxes[:, 1]
        xmax = decode_boxes[:, 2]
        ymax = decode_boxes[:, 3]
        img_h, img_w = img_shape[1], img_shape[2]

        img_h, img_w = tf.cast(img_h, tf.float32), tf.cast(img_w, tf.float32)

        xmin = tf.maximum(tf.minimum(xmin, img_w-1.), 0.)
        ymin = tf.maximum(tf.minimum(ymin, img_h-1.), 0.)

        xmax = tf.maximum(tf.minimum(xmax, img_w-1.), 0.)
        ymax = tf.maximum(tf.minimum(ymax, img_h-1.), 0.)

        return tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))


def filter_outside_boxes(boxes, img_h, img_w):
    '''
    :param anchors:boxes with format [xmin, ymin, xmax, ymax]
    :param img_h: height of image
    :param img_w: width of image
    :return: indices of anchors that inside the image boundary
    '''

    with tf.name_scope('filter_outside_boxes'):
        xmin, ymin, xmax, ymax = tf.unstack(boxes, axis=1)

        xmin_index = tf.greater_equal(xmin, 0)
        ymin_index = tf.greater_equal(ymin, 0)
        xmax_index = tf.less_equal(xmax, tf.cast(img_w, tf.float32))
        ymax_index = tf.less_equal(ymax, tf.cast(img_h, tf.float32))

        indices = tf.transpose(tf.stack([xmin_index, ymin_index, xmax_index, ymax_index]))
        indices = tf.cast(indices, dtype=tf.int32)
        indices = tf.reduce_sum(indices, axis=1)
        indices = tf.where(tf.equal(indices, 4))
        # indices = tf.equal(indices, 4)
        return tf.reshape(indices, [-1])


def padd_boxes_with_zeros(boxes, scores, max_num_of_boxes):

    '''
    num of boxes less than max num of boxes, so it need to pad with zeros[0, 0, 0, 0]
    :param boxes:
    :param scores: [-1]
    :param max_num_of_boxes:
    :return:
    '''

    pad_num = tf.cast(max_num_of_boxes, tf.int32) - tf.shape(boxes)[0]

    zero_boxes = tf.zeros(shape=[pad_num, 4], dtype=boxes.dtype)
    zero_scores = tf.zeros(shape=[pad_num], dtype=scores.dtype)

    final_boxes = tf.concat([boxes, zero_boxes], axis=0)

    final_scores = tf.concat([scores, zero_scores], axis=0)

    return final_boxes, final_scores


def get_horizen_minAreaRectangle(boxs, img_shape, with_label=True):

    rpn_proposals_boxes_convert = tf.py_func(forward_convert,
                                             inp=[boxs, with_label],
                                             Tout=tf.float32)
    if with_label:
        rpn_proposals_boxes_convert = tf.reshape(rpn_proposals_boxes_convert, [-1, 9])

        boxes_shape = tf.shape(rpn_proposals_boxes_convert)
        x_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])
        y_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])

        label = tf.unstack(rpn_proposals_boxes_convert, axis=1)[-1]

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)

        ''' The following codes aims to avoid gtbox out_sde'''

        # img_h, img_w = img_shape[0], img_shape[1]
        # img_h = tf.cast(img_h, tf.float32)
        # img_w = tf.cast(img_w, tf.float32)
        # x_min = tf.maximum(x_min, 0)
        # y_min = tf.maximum(y_min, 0)
        # x_max = tf.minimum(x_max, img_w)
        # y_max = tf.minimum(y_max, img_h)
        return tf.transpose(tf.stack([x_min, y_min, x_max, y_max, label], axis=0))
    else:
        rpn_proposals_boxes_convert = tf.reshape(rpn_proposals_boxes_convert, [-1, 8])

        boxes_shape = tf.shape(rpn_proposals_boxes_convert)
        x_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])
        y_list = tf.strided_slice(rpn_proposals_boxes_convert, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)

        ''' The following codes aims to avoid gtbox out_sde'''

        # img_h, img_w = img_shape[0], img_shape[1]
        # img_h = tf.cast(img_h, tf.float32)
        # img_w = tf.cast(img_w, tf.float32)
        # x_min = tf.maximum(x_min, 0)
        # y_min = tf.maximum(y_min, 0)
        # x_max = tf.minimum(x_max, img_w)
        # y_max = tf.minimum(y_max, img_h)

    return tf.transpose(tf.stack([x_min, y_min, x_max, y_max], axis=0))