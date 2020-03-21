# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import os, sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.tools import freeze_graph

sys.path.append('../../')
from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network_Concat

CKPT_PATH = '/home/omnisky/TF_Codes/horizen_code/output/trained_weights/FPN_Res101D_freezeC1C2_rmaskP2Concat_800_multiScale_WarmUpCosine/voc_399999model.ckpt'
OUT_DIR = '../../output/Pbs'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
PB_NAME = CKPT_PATH.split('/')[-2]+'.pb'


def build_detection_graph():
    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3],
                              name='input_img')  # is RGB. not GBR
    raw_shape = tf.shape(img_plac)
    raw_h, raw_w = tf.to_float(raw_shape[0]), tf.to_float(raw_shape[1])

    img_batch = tf.cast(img_plac, tf.float32)
    # do not resize
    # img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
    #                                                  target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
    #                                                  max_len=cfgs.IMG_MAX_LENGTH, is_resize=False)
    img_batch = img_batch / 255.0
    img_batch = (img_batch - tf.constant(cfgs.MXNET_MEAN)) / cfgs.MXNET_STD
    img_batch = tf.expand_dims(img_batch, axis=0)  # [1, None, None, 3]

    det_net = build_whole_network_Concat.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                          is_training=False)

    detected_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None,
        gtboxes_r_batch=None)

    xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                             detected_boxes[:, 2], detected_boxes[:, 3]

    resized_shape = tf.shape(img_batch)
    resized_h, resized_w = tf.to_float(resized_shape[1]), tf.to_float(resized_shape[2])

    xmin = xmin * raw_w / resized_w
    xmax = xmax * raw_w / resized_w

    ymin = ymin * raw_h / resized_h
    ymax = ymax * raw_h / resized_h

    boxes = tf.transpose(tf.stack([xmin, ymin, xmax, ymax]))
    dets = tf.concat([tf.reshape(detection_category, [-1, 1]),
                     tf.reshape(detection_scores, [-1, 1]),
                     boxes], axis=1, name='DetResults')

    return dets


def export_frozenPB():

    tf.reset_default_graph()

    dets = build_detection_graph()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("we have restred the weights from =====>>\n", CKPT_PATH)
        saver.restore(sess, CKPT_PATH)

        tf.train.write_graph(sess.graph_def, OUT_DIR, PB_NAME)
        freeze_graph.freeze_graph(input_graph=os.path.join(OUT_DIR, PB_NAME),
                                  input_saver='',
                                  input_binary=False,
                                  input_checkpoint=CKPT_PATH,
                                  output_node_names="DetResults",
                                  restore_op_name="save/restore_all",
                                  filename_tensor_name='save/Const:0',
                                  output_graph=os.path.join(OUT_DIR, PB_NAME.replace('.pb', '_Frozen.pb')),
                                  clear_devices=False,
                                  initializer_nodes='')

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    export_frozenPB()
