# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import tensorflow as tf


def forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)


def backward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


def get_horizen_minAreaRectangle(boxes, with_label=True):

    if with_label:
        boxes = tf.reshape(boxes, [-1, 9])

        boxes_shape = tf.shape(boxes)
        x_list = tf.strided_slice(boxes, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])
        y_list = tf.strided_slice(boxes, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1] - 1],
                                  strides=[1, 2])

        label = tf.unstack(boxes, axis=1)[-1]

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)
        return tf.transpose(tf.stack([x_min, y_min, x_max, y_max, label], axis=0))
    else:
        boxes = tf.reshape(boxes, [-1, 8])

        boxes_shape = tf.shape(boxes)
        x_list = tf.strided_slice(boxes, begin=[0, 0], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])
        y_list = tf.strided_slice(boxes, begin=[0, 1], end=[boxes_shape[0], boxes_shape[1]],
                                  strides=[1, 2])

        y_max = tf.reduce_max(y_list, axis=1)
        y_min = tf.reduce_min(y_list, axis=1)
        x_max = tf.reduce_max(x_list, axis=1)
        x_min = tf.reduce_min(x_list, axis=1)

    return tf.transpose(tf.stack([x_min, y_min, x_max, y_max], axis=0))


def sort_points(pts):

    x_argsort = np.argsort(pts[:, 0])

    point1 = pts[x_argsort[0]]  # point1 has the min X
    point1_index = x_argsort[0]

    tmp_point = pts[x_argsort[1]]  # maybe the point1
    if point1[0] == tmp_point[0]:
        # if has two same x, select the point which has smaller y as point1
        if tmp_point[1] < point1[1]:
            point1 = tmp_point
            point1_index = x_argsort[1]

    valid_index = np.array([True, True, True, True])
    valid_index[point1_index] = False
    # print (pts[valid_index], '\n***')
    point_a, point_b, point_c = pts[valid_index]

    vector_1a = point_a - point1
    vector_1b = point_b - point1
    vector_1c = point_c - point1

    # ensure point3
    if np.cross(vector_1a, vector_1b) * np.cross(vector_1a, vector_1c) < 0:
        point3 = point_a
        if np.cross(vector_1a, vector_1b) < 0:
            point2, point4 = point_b, point_c
        else:
            point2, point4 = point_c, point_b

    elif np.cross(vector_1b, vector_1a) * np.cross(vector_1b, vector_1c) < 0:
        point3 = point_b
        if np.cross(vector_1b, vector_1a) < 0:
            point2, point4 = point_a, point_c
        else:
            point2, point4 = point_c, point_a
    else:
        point3 = point_c
        if np.cross(vector_1c, vector_1a) < 0:
            point2, point4 = point_a, point_b
        else:
            point2, point4 = point_b, point_a

    return np.array([point1, point2, point3, point4], dtype=np.float32)


def sort_box_points(boxes, with_label=True):

    out_boxes = []
    boxes = np.float32(boxes)
    if with_label:
        circle_boxes = boxes[:, :-1]
    else:
        circle_boxes = boxes

    for points in circle_boxes:
        tmp_points = sort_points(points.reshape((-1, 2))).reshape((-1,))
        out_boxes.append(tmp_points)
    out_boxes = np.array(out_boxes)
    if with_label:
        tmp = np.concatenate([out_boxes, boxes[:, -1].reshape((-1, 1))], axis=1)
        return np.array(tmp, dtype=np.float32)
    else:
        return np.array(out_boxes, dtype=np.float32)


def coordinate_present_convert(coords, mode=1):
    """
    :param coords: shape [-1, 5]
    :param mode: -1 convert coords range to [-90, 90), 1 convert coords range to [-90, 0)
    :return: shape [-1, 5]
    """
    # angle range from [-90, 0) to [-180, 0)
    if mode == -1:
        w, h = coords[:, 2], coords[:, 3]

        remain_mask = np.greater(w, h)
        convert_mask = np.logical_not(remain_mask).astype(np.int32)
        remain_mask = remain_mask.astype(np.int32)

        remain_coords = coords * np.reshape(remain_mask, [-1, 1])

        coords[:, [2, 3]] = coords[:, [3, 2]]
        coords[:, 4] += 90

        convert_coords = coords * np.reshape(convert_mask, [-1, 1])

        coords_new = remain_coords + convert_coords

        coords_new[:, 4] -= 90

    # angle range from [-180, 0) to [-90, 0)
    elif mode == 1:
        coords[:, 4] += 90

        # theta = coords[:, 4]
        # remain_mask = np.logical_and(np.greater_equal(theta, -90), np.less(theta, 0))
        # convert_mask = np.logical_not(remain_mask)
        #
        # remain_coords = coords * np.reshape(remain_mask, [-1, 1])
        #
        # coords[:, [2, 3]] = coords[:, [3, 2]]
        # coords[:, 4] -= 90
        #
        # convert_coords = coords * np.reshape(convert_mask, [-1, 1])
        #
        # coords_new = remain_coords + convert_coords

        xlt, ylt = -1 * coords[:, 2] / 2.0, coords[:, 3] / 2.0
        xld, yld = -1 * coords[:, 2] / 2.0, -1 * coords[:, 3] / 2.0
        xrd, yrd = coords[:, 2] / 2.0, -1 * coords[:, 3] / 2.0
        xrt, yrt = coords[:, 2] / 2.0, coords[:, 3] / 2.0

        theta = -coords[:, -1] / 180 * np.pi

        xlt_ = np.cos(theta) * xlt + np.sin(theta) * ylt + coords[:, 0]
        ylt_ = -np.sin(theta) * xlt + np.cos(theta) * ylt + coords[:, 1]

        xrt_ = np.cos(theta) * xrt + np.sin(theta) * yrt + coords[:, 0]
        yrt_ = -np.sin(theta) * xrt + np.cos(theta) * yrt + coords[:, 1]

        xld_ = np.cos(theta) * xld + np.sin(theta) * yld + coords[:, 0]
        yld_ = -np.sin(theta) * xld + np.cos(theta) * yld + coords[:, 1]

        xrd_ = np.cos(theta) * xrd + np.sin(theta) * yrd + coords[:, 0]
        yrd_ = -np.sin(theta) * xrd + np.cos(theta) * yrd + coords[:, 1]

        convert_box = np.transpose(np.stack([xlt_, ylt_, xrt_, yrt_, xrd_, yrd_, xld_, yld_], axis=0))

        coords_new = backward_convert(convert_box, False)


    else:
        raise Exception('mode error!')

    return np.array(coords_new, dtype=np.float32)


if __name__ == '__main__':
    coord = np.array([[150, 150, 50, 100, -90, 1],
                      [150, 150, 100, 50, -90, 1],
                      [150, 150, 50, 100, -45, 1],
                      [150, 150, 100, 50, -45, 1]])

    coord1 = np.array([[150, 150, 100, 50, 0],
                      [150, 150, 100, 50, -90],
                      [150, 150, 100, 50, 45],
                      [150, 150, 100, 50, -45]])

    coord2 = forward_convert(coord)
    # coord3 = forward_convert(coord1, mode=-1)
    print(coord2)
    # print(coord3-coord2)
    # coord_label = np.array([[167., 203., 96., 132., 132., 96., 203., 167., 1.]])
    #
    # coord4 = back_forward_convert(coord_label, mode=1)
    # coord5 = back_forward_convert(coord_label)

    # print(coord4)
    # print(coord5)

    # coord3 = coordinate_present_convert(coord, -1)
    # print(coord3)
    # coord4 = coordinate_present_convert(coord3, mode=1)
    # print(coord4)