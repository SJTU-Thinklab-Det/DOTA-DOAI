# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division
import numpy as np
import tfplot as tfp
import tensorflow as tf
from libs.box_utils.coordinate_convert import forward_convert, backward_convert
import cv2


def make_gt_mask(fet_h, fet_w, img_h, img_w, gtboxes):
    '''

    :param fet_h:
    :param fet_w:
    :param img_h:
    :param img_w:
    :param gtboxes: [xmin, ymin, xmax, ymax, label]. shape is (N, 5)
    :return:
    '''
    gtboxes = np.reshape(gtboxes, [-1, 5])
    # xmin, ymin, xmax, ymax, label = gtboxes[:, 0], gtboxes[:, 1], gtboxes[:, 2], gtboxes[:, 3], gtboxes[:, 4]

    areas = (gtboxes[:, 2]-gtboxes[:, 0])*(gtboxes[:, 3]-gtboxes[:, 1])
    arg_areas = np.argsort(-1*areas)  # sort from large to small
    gtboxes = gtboxes[arg_areas]

    fet_h, fet_w = int(fet_h), int(fet_w)
    mask = np.zeros(shape=[fet_h, fet_w], dtype=np.int32)
    for a_box in gtboxes:
        xmin, ymin, xmax, ymax, label = a_box[0], a_box[1], a_box[2], a_box[3], a_box[4]

        new_xmin, new_ymin, new_xmax, new_ymax = int(xmin*fet_w/float(img_w)), int(ymin*fet_h/float(img_h)),\
                                                 int(xmax*fet_w/float(img_w)), int(ymax*fet_h/float(img_h))

        new_xmin, new_ymin = max(0, new_xmin), max(0, new_ymin)
        new_xmax, new_ymax = min(fet_w, new_xmax), min(fet_h, new_ymax)

        mask[new_ymin:new_ymax, new_xmin:new_xmax] = np.int32(label)
    return mask


def make_r_gt_mask(fet_h, fet_w, img_h, img_w, gtboxes):
    gtboxes = np.reshape(gtboxes, [-1, 6])  # [x, y, w, h, theta, label]

    areas = gtboxes[:, 2] * gtboxes[:, 3]
    arg_areas = np.argsort(-1 * areas)  # sort from large to small
    gtboxes = gtboxes[arg_areas]

    fet_h, fet_w = int(fet_h), int(fet_w)
    mask = np.zeros(shape=[fet_h, fet_w], dtype=np.int32)
    for a_box in gtboxes:
        # print(a_box)
        box = cv2.boxPoints(((a_box[0], a_box[1]), (a_box[2], a_box[3]), a_box[4]))
        box = np.reshape(box, [-1, ])
        label = a_box[-1]
        new_box = []
        for i in range(8):
            if i % 2 == 0:
                x = box[i]
                new_x = int(x * fet_w / float(img_w))
                new_box.append(new_x)
            else:
                y = box[i]
                new_y = int(y*fet_h/float(img_h))
                new_box.append(new_y)

        new_box = np.int0(new_box).reshape([4, 2])
        color = int(label)
        # print(type(color), color)
        cv2.fillConvexPoly(mask, new_box, color=color)
    # print (mask.dtype)
    return mask


def vis_mask_tfsmry(mask, name):
    '''

    :param mask:[H, W]. It's a tensor, not array
    :return:
    '''

    def figure_attention(activation):
        fig, ax = tfp.subplots()
        im = ax.imshow(activation, cmap='jet')
        fig.colorbar(im)
        return fig

    heatmap = mask*10

    tfp.summary.plot(name, figure_attention, [heatmap])