# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import tensorflow as tf
import cv2
import numpy as np
import math
from tqdm import tqdm
import argparse
from multiprocessing import Queue, Process
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.networks import build_whole_network
from help_utils import tools
from libs.label_name_dict.label_dict import *
from libs.box_utils import draw_box_in_img
from libs.box_utils.cython_utils.cython_nms import nms, soft_nms


def worker(gpu_id, images, det_net, args, result_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH,
                                                     is_resize=not args.multi_scale)
    if cfgs.NET_NAME in ['resnet152_v1d', 'resnet101_v1d', 'resnet50_v1d']:
        img_batch = (img_batch / 255 - tf.constant(cfgs.PIXEL_MEAN_)) / tf.constant(cfgs.PIXEL_STD)
    else:
        img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)

    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None,
        gtboxes_r_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model %d ...' % gpu_id)

        for img_path in images:

            # if '2043' not in img_path:
            #     continue

            img = cv2.imread(img_path)

            box_res = []
            label_res = []
            score_res = []

            imgH = img.shape[0]
            imgW = img.shape[1]

            img_short_side_len_list = cfgs.IMG_SHORT_SIDE_LEN if args.multi_scale else [cfgs.IMG_SHORT_SIDE_LEN]

            if imgH < args.h_len:
                temp = np.zeros([args.h_len, imgW, 3], np.float32)
                temp[0:imgH, :, :] = img
                img = temp
                imgH = args.h_len

            if imgW < args.w_len:
                temp = np.zeros([imgH, args.w_len, 3], np.float32)
                temp[:, 0:imgW, :] = img
                img = temp
                imgW = args.w_len

            for hh in range(0, imgH, args.h_len - args.h_overlap):
                if imgH - hh - 1 < args.h_len:
                    hh_ = imgH - args.h_len
                else:
                    hh_ = hh
                for ww in range(0, imgW, args.w_len - args.w_overlap):
                    if imgW - ww - 1 < args.w_len:
                        ww_ = imgW - args.w_len
                    else:
                        ww_ = ww
                    src_img = img[hh_:(hh_ + args.h_len), ww_:(ww_ + args.w_len), :]

                    for short_size in img_short_side_len_list:
                        max_len = cfgs.IMG_MAX_LENGTH
                        if args.h_len < args.w_len:
                            new_h, new_w = short_size, min(int(short_size * float(args.w_len) / args.h_len), max_len)
                        else:
                            new_h, new_w = min(int(short_size * float(args.h_len) / args.w_len), max_len), short_size
                        img_resize = cv2.resize(src_img, (new_w, new_h))

                        resized_img, det_boxes_h_, det_scores_h_, det_category_h_ = \
                            sess.run(
                                [img_batch, detection_boxes, detection_scores, detection_category],
                                feed_dict={img_plac: img_resize[:, :, ::-1]}
                            )

                        resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]
                        src_h, src_w = src_img.shape[0], src_img.shape[1]

                        if len(det_boxes_h_) > 0:
                            det_boxes_h_[:, 0::2] *= (src_w / resized_w)
                            det_boxes_h_[:, 1::2] *= (src_h / resized_h)

                            for ii in range(len(det_boxes_h_)):
                                box = det_boxes_h_[ii]
                                box[0] = box[0] + ww_
                                box[1] = box[1] + hh_
                                box[2] = box[2] + ww_
                                box[3] = box[3] + hh_
                                box_res.append(box)
                                label_res.append(det_category_h_[ii])
                                score_res.append(det_scores_h_[ii])

            box_res = np.array(box_res)
            label_res = np.array(label_res)
            score_res = np.array(score_res)

            filter_indices = score_res >= 0.05
            score_res = score_res[filter_indices]
            box_res = box_res[filter_indices]
            label_res = label_res[filter_indices]

            box_res_ = []
            label_res_ = []
            score_res_ = []

            threshold = {'roundabout': 0.35, 'tennis-court': 0.35, 'swimming-pool': 0.4, 'storage-tank': 0.3,
                         'soccer-ball-field': 0.3, 'small-vehicle': 0.4, 'ship': 0.35, 'plane': 0.35,
                         'large-vehicle': 0.4, 'helicopter': 0.4, 'harbor': 0.3, 'ground-track-field': 0.4,
                         'bridge': 0.3, 'basketball-court': 0.4, 'baseball-diamond': 0.3}

            for sub_class in range(1, cfgs.CLASS_NUM + 1):
                index = np.where(label_res == sub_class)[0]
                if len(index) == 0:
                    continue
                tmp_boxes_h = box_res[index]
                tmp_label_h = label_res[index]
                tmp_score_h = score_res[index]

                tmp_boxes_h = np.array(tmp_boxes_h)
                tmp = np.zeros([tmp_boxes_h.shape[0], tmp_boxes_h.shape[1] + 1])
                tmp[:, 0:-1] = tmp_boxes_h
                tmp[:, -1] = np.array(tmp_score_h)

                if cfgs.SOFT_NMS:
                    inx = soft_nms(np.array(tmp, np.float32), 0.5, Nt=threshold[LABEL_NAME_MAP[sub_class]],
                                   threshold=0.001, method=2)  # 2 means Gaussian
                else:
                    inx = nms(np.array(tmp, np.float32),
                              threshold[LABEL_NAME_MAP[sub_class]])

                box_res_.extend(np.array(tmp_boxes_h)[inx])
                score_res_.extend(np.array(tmp_score_h)[inx])
                label_res_.extend(np.array(tmp_label_h)[inx])

            result_dict = {'boxes': np.array(box_res_), 'scores': np.array(score_res_),
                           'labels': np.array(label_res_), 'image_id': img_path}
            result_queue.put_nowait(result_dict)


def test_dota(det_net, real_test_img_list, args, txt_name):

    save_path = os.path.join('./test_dota', cfgs.VERSION)

    nr_records = len(real_test_img_list)
    pbar = tqdm(total=nr_records)
    gpu_num = len(args.gpus.strip().split(','))

    nr_image = math.ceil(nr_records / gpu_num)
    result_queue = Queue(500)
    procs = []

    for i, gpu_id in enumerate(args.gpus.strip().split(',')):
        start = i * nr_image
        end = min(start + nr_image, nr_records)
        split_records = real_test_img_list[start:end]
        proc = Process(target=worker, args=(int(gpu_id), split_records, det_net, args, result_queue))
        print('process:%d, start:%d, end:%d' % (i, start, end))
        proc.start()
        procs.append(proc)

    for i in range(nr_records):
        res = result_queue.get()

        if args.show_box:

            nake_name = res['image_id'].split('/')[-1]
            tools.mkdir(os.path.join(save_path, 'dota_img_vis'))
            draw_path = os.path.join(save_path, 'dota_img_vis', nake_name)

            draw_img = np.array(cv2.imread(res['image_id']), np.float32)

            detected_indices = res['scores'] >= cfgs.SHOW_SCORE_THRSHOLD
            detected_scores = res['scores'][detected_indices]
            detected_boxes = res['boxes'][detected_indices]
            detected_categories = res['labels'][detected_indices]

            final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(draw_img,
                                                                                boxes=detected_boxes,
                                                                                labels=detected_categories,
                                                                                scores=detected_scores,
                                                                                method=0,
                                                                                in_graph=False)
            cv2.imwrite(draw_path, final_detections)

        else:
            CLASS_DOTA = NAME_LABEL_MAP.keys()
            write_handle = {}

            tools.mkdir(os.path.join(save_path, 'dota_res'))
            for sub_class in CLASS_DOTA:
                if sub_class == 'back_ground':
                    continue
                write_handle[sub_class] = open(os.path.join(save_path, 'dota_res', 'Task1_%s.txt' % sub_class), 'a+')

            hboxes = res['boxes']

            for i, hbox in enumerate(hboxes):
                command = '%s %.3f %.1f %.1f %.1f %.1f\n' % (res['image_id'].split('/')[-1].split('.')[0],
                                                             res['scores'][i],
                                                             hbox[0], hbox[1], hbox[2], hbox[3])

                write_handle[LABEL_NAME_MAP[res['labels'][i]]].write(command)

            for sub_class in CLASS_DOTA:
                if sub_class == 'back_ground':
                    continue
                write_handle[sub_class].close()

            fw = open(txt_name, 'a+')
            fw.write('{}\n'.format(res['image_id'].split('/')[-1]))
            fw.close()

        pbar.set_description("Test image %s" % res['image_id'].split('/')[-1])

        pbar.update(1)

    for p in procs:
        p.join()


def eval(num_imgs, args):

    txt_name = '{}.txt'.format(cfgs.VERSION)
    if not args.show_box:
        if not os.path.exists(txt_name):
            fw = open(txt_name, 'w')
            fw.close()

        fr = open(txt_name, 'r')
        img_filter = fr.readlines()
        print('****************************'*3)
        print('Already tested imgs:', img_filter)
        print('****************************'*3)
        fr.close()

        test_imgname_list = [os.path.join(args.test_dir, img_name) for img_name in os.listdir(args.test_dir)
                             if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')) and
                             (img_name + '\n' not in img_filter)]
    else:
        test_imgname_list = [os.path.join(args.test_dir, img_name) for img_name in os.listdir(args.test_dir)
                             if img_name.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))]

    assert len(test_imgname_list) != 0, 'test_dir has no imgs there.' \
                                        ' Note that, we only support img format of (.jpg, .png, and .tiff) '

    if num_imgs == np.inf:
        real_test_img_list = test_imgname_list
    else:
        real_test_img_list = test_imgname_list[: num_imgs]

    fpn = build_whole_network.DetectionNetwork(
        base_network_name=cfgs.NET_NAME,
        is_training=False)
    test_dota(det_net=fpn, real_test_img_list=real_test_img_list, args=args, txt_name=txt_name)

    if not args.show_box:
        os.remove(txt_name)


def parse_args():

    parser = argparse.ArgumentParser('evaluate the result.')

    parser.add_argument('--test_dir', dest='test_dir',
                        help='evaluate imgs dir ',
                        default='/data/dataset/DOTA/test/images/', type=str)
    parser.add_argument('--gpus', dest='gpus',
                        help='gpu id',
                        default='0,1,2,3,4,5,6,7', type=str)
    parser.add_argument('--eval_num', dest='eval_num',
                        help='the num of eval imgs',
                        default=np.inf, type=int)
    parser.add_argument('--show_box', '-s', default=False,
                        action='store_true')
    parser.add_argument('--multi_scale', '-ms', default=False,
                        action='store_true')
    parser.add_argument('--h_len', dest='h_len',
                        help='image height',
                        default=800, type=int)
    parser.add_argument('--w_len', dest='w_len',
                        help='image width',
                        default=800, type=int)
    parser.add_argument('--h_overlap', dest='h_overlap',
                        help='height overlap',
                        default=200, type=int)
    parser.add_argument('--w_overlap', dest='w_overlap',
                        help='width overlap',
                        default=200, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    print(20*"--")
    print(args)
    print(20*"--")
    eval(args.eval_num,
         args=args)


