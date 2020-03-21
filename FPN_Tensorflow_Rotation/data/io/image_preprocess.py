# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import cv2
from libs.label_name_dict.label_dict import NAME_LABEL_MAP


def max_length_limitation(length, length_limitation):
    return tf.cond(tf.less(length, length_limitation),
                   true_fn=lambda: length,
                   false_fn=lambda: length_limitation)


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len, max_len=1200):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 9]
    :param target_shortside_len:
    :return:
    '''

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (target_shortside_len, tf.minimum(target_shortside_len * w//h, max_len)),
                           false_fn=lambda: (tf.minimum(target_shortside_len * h//w, max_len),
                                             target_shortside_len))
    # new_h, new_w = 1200, 1200
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)

    x1, x2, x3, x4 = x1 * new_w//w, x2 * new_w//w, x3 * new_w//w, x4 * new_w//w
    y1, y2, y3, y4 = y1 * new_h//h, y2 * new_h//h, y3 * new_h//h, y4 * new_h//h

    img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
    return img_tensor, tf.transpose(tf.stack([x1, y1, x2, y2, x3, y3, x4, y4, label], axis=0))


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, length_limitation=1200, is_resize=True):
    if is_resize:
        img_h, img_w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

        new_h, new_w = tf.cond(tf.less(img_h, img_w),
                               true_fn=lambda: (target_shortside_len,
                                                max_length_limitation(target_shortside_len * img_w // img_h, length_limitation)),
                               false_fn=lambda: (max_length_limitation(target_shortside_len * img_h // img_w, length_limitation),
                                                 target_shortside_len))

        img_tensor = tf.expand_dims(img_tensor, axis=0)
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

        img_tensor = tf.squeeze(img_tensor, axis=0)  # ensure image tensor rank is 3
    return img_tensor


def flip_left_right(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.flip_left_right(img_tensor)

    x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)
    new_x1 = w - x1
    new_x2 = w - x2
    new_x3 = w - x3
    new_x4 = w - x4
    return img_tensor, tf.transpose(tf.stack([new_x1, y1, new_x2, y2, new_x3, y3, new_x4, y4, label], axis=0))


def random_flip_left_right(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_right(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label


def flip_up_down(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.flip_up_down(img_tensor)

    x1, y1, x2, y2, x3, y3, x4, y4, label = tf.unstack(gtboxes_and_label, axis=1)

    new_y1 = h - y1
    new_y2 = h - y2
    new_y3 = h - y3
    new_y4 = h - y4

    return img_tensor, tf.transpose(tf.stack([x1, new_y1, x2, new_y2, x3, new_y3, x4, new_y4, label], axis=0))


def random_flip_up_dowm(img_tensor, gtboxes_and_label):
    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_up_down(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor, gtboxes_and_label


def random_rgb2gray(img_tensor, gtboxes_and_label):
    '''
    :param img_tensor: tf.float32
    :return:
    '''
    def rgb2gray(img, gtboxes_and_label):

        label = gtboxes_and_label[:, -1]
        if NAME_LABEL_MAP['swimming-pool'] in label:
            # do not change color, because swimming-pool need color
            return img

        coin = np.random.rand()
        if coin < 0.3:
            img = np.asarray(img, dtype=np.float32)
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = r * 0.299 + g * 0.587 + b * 0.114
            img = np.stack([gray, gray, gray], axis=2)
            return img
        else:
            return img

    h, w, c = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1], tf.shape(img_tensor)[2]
    img_tensor = tf.py_func(rgb2gray,
                            inp=[img_tensor, gtboxes_and_label],
                            Tout=tf.float32)
    img_tensor = tf.reshape(img_tensor, shape=[h, w, c])

    return img_tensor


def rotate_img_np_OLD(img, gtboxes_and_label, r_theta):
    h, w, c = img.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, r_theta, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h))
    # print (M)
    new_points_list = []
    obj_num = len(gtboxes_and_label)
    for st in range(0, 7, 2):
        points = gtboxes_and_label[:, st:st+2]
        expand_points = np.concatenate((points, np.ones(shape=(obj_num, 1))), axis=1)
        new_points = np.dot(M, expand_points.T)
        new_points = new_points.T
        new_points_list.append(new_points)
    gtboxes = np.concatenate(new_points_list, axis=1)
    gtboxes_and_label = np.concatenate((gtboxes, gtboxes_and_label[:, -1].reshape(-1, 1)), axis=1)

    x1, y1, x2, y2, x3, y3, x4, y4 = np.split(gtboxes, 8, 1)

    xc = 0.25*(x1+x2+x3+x4)
    yc = 0.25*(y1+y2+y3+y4)

    valid = (xc>0) & (yc>0) & (xc<w) & (yc<h)
    valid = valid.reshape((-1, ))
    # print (valid)
    if np.sum(valid) != 0:
        gtboxes_and_label = gtboxes_and_label[valid]
    gtboxes_and_label = np.asarray(gtboxes_and_label, dtype=np.int32)

    return rotated_img, gtboxes_and_label

def rotate_img_np(img, gtboxes_and_label, r_theta):
    h, w, c = img.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, r_theta, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    nW, nH = int(h*sin + w*cos), int(h*cos + w*sin)  # new W and new H
    M[0, 2] += (nW/2) - center[0]
    M[1, 2] += (nH/2) - center[1]
    rotated_img = cv2.warpAffine(img, M, (nW, nH))
    # -------

    new_points_list = []
    obj_num = len(gtboxes_and_label)
    for st in range(0, 7, 2):
        points = gtboxes_and_label[:, st:st+2]
        expand_points = np.concatenate((points, np.ones(shape=(obj_num, 1))), axis=1)
        new_points = np.dot(M, expand_points.T)
        new_points = new_points.T
        new_points_list.append(new_points)
    gtboxes = np.concatenate(new_points_list, axis=1)
    gtboxes_and_label = np.concatenate((gtboxes, gtboxes_and_label[:, -1].reshape(-1, 1)), axis=1)
    gtboxes_and_label = np.asarray(gtboxes_and_label, dtype=np.int32)

    return rotated_img, gtboxes_and_label


def rotate_img(img_tensor, gtboxes_and_label):

    # thetas = tf.constant([-30, -60, -90, 30, 60, 90])
    thetas = tf.range(-90, 90+16, delta=15)
    # -90, -75, -60, -45, -30, -15,   0,  15,  30,  45,  60,  75,  90

    theta = tf.random_shuffle(thetas)[0]

    img_tensor, gtboxes_and_label = tf.py_func(rotate_img_np,
                                               inp=[img_tensor, gtboxes_and_label, theta],
                                               Tout=[tf.float32, tf.int32])

    h, w, c = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1], tf.shape(img_tensor)[2]
    img_tensor = tf.reshape(img_tensor, [h, w, c])
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])

    return img_tensor, gtboxes_and_label


def random_rotate_img(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.6),
                                            lambda: rotate_img(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor, gtboxes_and_label


def get_mask(img, boxes):
    h, w, _ = img.shape
    mask = np.zeros([h, w])
    for b in boxes:
        b = np.reshape(b[0:-1], [4, 2])
        rect = np.array(b, np.int32)
        cv2.fillConvexPoly(mask, rect, 1)
    # mask = cv2.resize(mask, dsize=(h // 16, w // 16))
    mask = np.expand_dims(mask, axis=-1)
    return np.array(mask, np.float32)




