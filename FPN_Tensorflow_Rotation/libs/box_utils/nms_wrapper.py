# -*- coding: utf-8 -*-

from __future__ import absolute_import,print_function,division
from libs.box_utils.cython_utils.cython_nms import soft_nms
import numpy as np
import tensorflow as tf
import cv2

from libs.configs import cfgs
from libs.box_utils.rotate_utils.rotate_polygon_nms import rotate_gpu_nms

def soft_nms_cpu(boxes, sigma=0.5, Nt=0.3, threshold=0.001, max_keep=100):

    # boxes = np.asarray(boxes, dtype=np.float)
    # sigma = float(sigma)
    # Nt = float(Nt)
    # threshold = float(threshold)
    keep = soft_nms(boxes,
                    sigma=sigma,
                    Nt=Nt,
                    threshold=threshold,
                    method=2)  # default use Gaussian

    keep = np.array(keep[:max_keep])
    # print(type(keep), keep)
    return keep


def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size,
               use_gpu=True):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """
    # add this for speed up test.
    # valid_id = tf.reshape(tf.where(tf.greater(scores, 1e-4)), [-1])
    # decode_boxes = tf.gather(decode_boxes, valid_id)
    # scores = tf.gather(scores, valid_id)

    if use_gpu:
        det_tensor = tf.concat([decode_boxes, tf.expand_dims(scores, axis=1)], axis=1)
        keep = tf.py_func(nms_rotate_gpu,
                          inp=[det_tensor, iou_threshold, max_output_size, 0],  # int(cfgs.GPU_GROUP)
                          Tout=tf.int64)

    else:
        keep = tf.py_func(nms_rotate_cpu,
                          inp=[decode_boxes, scores, iou_threshold, max_output_size],
                          Tout=tf.int64)
    return keep


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):

    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]
    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            try:
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + cfgs.EPSILON)

            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def nms_rotate_gpu(dets, iou_threshold, max_keep, device_id=0):  # int(cfgs.GPU_GROUP)
    keep = rotate_gpu_nms(dets, iou_threshold, device_id)
    keep = np.array(keep[:max_keep])
    return keep
