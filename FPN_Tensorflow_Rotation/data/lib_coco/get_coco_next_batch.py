# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function, division

import sys, os
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, './PythonAPI/')
# sys.path.insert(0, os.path.abspath('data'))
for _ in sys.path:
    print (_)
from PythonAPI.pycocotools.coco import COCO
import cv2
import numpy as np
import os
from libs.label_name_dict import coco_dict


annotation_path = '/home/yjr/DataSet/COCO/2017/annotations/instances_train2017.json'
print ("load coco .... it will cost about 17s..")
coco = COCO(annotation_path)

imgId_list = coco.getImgIds()
imgId_list = np.array(imgId_list)

total_imgs = len(imgId_list)

# print (NAME_LABEL_DICT)


def next_img(step):

    if step % total_imgs == 0:
        np.random.shuffle(imgId_list)
    imgid = imgId_list[step % total_imgs]

    imgname = coco.loadImgs(ids=[imgid])[0]['file_name']
    # print (type(imgname), imgname)
    img = cv2.imread(os.path.join("/home/yjr/DataSet/COCO/2017/train2017", imgname))

    annotation = coco.imgToAnns[imgid]
    gtbox_and_label_list = []
    for ann in annotation:
        box = ann['bbox']

        box = [box[0], box[1], box[0]+box[2], box[1]+box[3]]  # [xmin, ymin, xmax, ymax]
        cat_id = ann['category_id']
        cat_name = coco_dict.originID_classes[cat_id] #ID_NAME_DICT[cat_id]
        label = coco_dict.NAME_LABEL_MAP[cat_name]
        gtbox_and_label_list.append(box + [label])
    gtbox_and_label_list = np.array(gtbox_and_label_list, dtype=np.int32)
    # print (img.shape, gtbox_and_label_list.shape)
    if gtbox_and_label_list.shape[0] == 0:
        return next_img(step+1)
    else:
        return imgid, img[:, :, ::-1], gtbox_and_label_list


if __name__ == '__main__':

    imgid, img,  gtbox = next_img(3234)

    print("::")
    from libs.box_utils.draw_box_in_img import draw_boxes_with_label_and_scores

    img = draw_boxes_with_label_and_scores(img_array=img, boxes=gtbox[:, :-1], labels=gtbox[:, -1],
                                           scores=np.ones(shape=(len(gtbox), )))
    print ("_----")


    cv2.imshow("test", img)
    cv2.waitKey(0)


