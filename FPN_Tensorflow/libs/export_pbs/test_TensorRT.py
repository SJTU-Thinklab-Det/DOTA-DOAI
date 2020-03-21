# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
import time
import cv2
sys.path.append('../../')

from libs.box_utils import draw_box_in_img


def get_graph(frozen_graph_file):
    with gfile.FastGFile(frozen_graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def test(frozen_graph_path, test_dir, mode=None):

    print("we are testing ====>>>>", frozen_graph_path)

    if mode == 'FP32':

        graph = trt.create_inference_graph(get_graph(frozen_graph_path), ["DetResults"],
                                           max_batch_size=1,
                                           max_workspace_size_bytes=1 << 10,
                                           precision_mode="FP32")  # Get optimized graph
        with gfile.FastGFile(frozen_graph_path.replace('.pb', '_tensorRT.pb'), 'wb') as f:
            f.write(graph.SerializeToString())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph,
                            input_map=None,
                            return_elements=None,
                            name="",
                            op_dict=None,
                            producer_op_list=None)

        img = graph.get_tensor_by_name("input_img:0")
        dets = graph.get_tensor_by_name("DetResults:0")

        with tf.Session(graph=graph) as sess:
            for img_path in os.listdir(test_dir)[:3]:
                nake_name = img_path.split('/')[-1]
                print(nake_name)
                a_img = cv2.imread(os.path.join(test_dir, img_path))[:, :, ::-1]
                st = time.time()
                dets_val = sess.run(dets, feed_dict={img: a_img})

                show_indices = dets_val[:, 1] >= 0.5
                dets_val = dets_val[show_indices]
                final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(a_img,
                                                                                    boxes=dets_val[:, 2:],
                                                                                    labels=dets_val[:, 0],
                                                                                    scores=dets_val[:, 1])
                cv2.imwrite(nake_name,
                            final_detections[:, :, ::-1])
                print("%s cost time: %f" % (img_path, time.time() - st))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    test('/home/omnisky/TF_Codes/horizen_code/output/Pbs/FPN_Res101D_freezeC1C2_rmaskP2Concat_800_multiScale_WarmUpCosine_Frozen.pb',
         '/home/omnisky/DataSets/Dota_clip/trainval800/images',
         'FP32')











