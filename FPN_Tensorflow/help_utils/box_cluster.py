# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET

from libs.label_name_dict.label_dict import *
import numpy as np
from libs.box_utils.cython_utils.cython_bbox import bbox_overlaps
import os
import cv2

def read_xml_gtbox_and_label(xml_path):
    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 9],
           and has [x1, y1, x2, y2, x3, y3, x4, y4, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(float(node.text))
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)

    return gtbox_label

def get_Data(data_id, gtbox_label_list):

    gtbox_label = np.concatenate(gtbox_label_list, axis=0)
    x_list = gtbox_label[:, 0:8:2]
    y_list = gtbox_label[:, 1:8:2]
    label = gtbox_label[:, -1]
    xmin = np.min(x_list, axis=1)
    ymin = np.min(y_list, axis=1)

    xmax = np.max(x_list, axis=1)
    ymax = np.max(y_list, axis=1)

    gtbox = np.transpose(np.stack([xmin, ymin, xmax, ymax], axis=0))
    if data_id == -1:
        gtbox = gtbox
    else:
        gtbox = gtbox[label==data_id]

    return gtbox

def wending(new_center, old_cetner, k):

    overlaps = bbox_overlaps(
        np.ascontiguousarray(new_center, dtype=np.float),
        np.ascontiguousarray(old_cetner, dtype=np.float))

    dis = []
    for i in range(k):
        dis.append(1- overlaps[i, i])

    if sum(dis) <=0.000001:
        return False
    else:
        return True


def cluster(boxes, k):

    center_id = np.random.choice(np.arange(len(boxes)), k, replace=False)
    new_center_boxes = [boxes[i] for i in center_id]
    old_center_boxes = [np.zeros_like(box) for box in new_center_boxes]

    i = 0
    while wending(new_center_boxes, old_center_boxes, k):
        overlaps = bbox_overlaps(
            np.ascontiguousarray(boxes, dtype=np.float),
            np.ascontiguousarray(new_center_boxes, dtype=np.float))
        argmax_id = np.argmax(overlaps, axis=1)
        for i in range(k):
            cluster_i_box = boxes[argmax_id==i]
            old_center_boxes[i] = new_center_boxes[i]
            new_center_boxes[i] = np.mean(cluster_i_box, axis=0)
        # if i % 1 == 0:
        #     print ("i", i)
        if i > 1000000:
            break
        i +=1

    return new_center_boxes

def decode_boxes(boxes):

    newboxes = []

    img = np.zeros(shape=(600, 600, 3), dtype=np.uint8)
    for box in boxes:
        xmin, ymin, xmax, ymax  = box[0], box[1], box[2], box[3]

        x = (xmin+xmax)/2.0
        y = (ymin+ymax)/2.0
        w = xmax - xmin
        h = ymax - ymin
        newboxes.append([x, y, w, h])

        cv2.rectangle(img, (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)),
                      color=(np.random.randint(255), np.random.randint(255),
                             np.random.randint(255)),
                      thickness=3)

    return np.array(newboxes), img


def cluster_for_all(k):

    xml_root = '/home/yjr/DataSet/Dota_clip/train/labeltxt'
    xml_files = [x for x in os.listdir(xml_root) if x.endswith('xml')][:20000]

    gt_list = []
    for i, f in enumerate(xml_files):
        tmp_gt = read_xml_gtbox_and_label(os.path.join(xml_root, f))
        gt_list.append(tmp_gt)
        if i %100 == 0:
            print(i)
    print("read_over")
    all_data = get_Data(-1, gt_list)

    center_boxes = cluster(all_data, k)

    center_boxes, vis_img = decode_boxes(center_boxes)
    cv2.imwrite("all.jpg", vis_img)

    for b in center_boxes:
        print ("box: ", b)
        print ("size: %f || ratio: %f " %(np.sqrt(b[2]*b[3]), b[2]/float(b[3])))
        print(20*"**")


    for i in range(1, 16):
        print (LABEl_NAME_MAP[i])
        all_data = get_Data(i, gt_list)

        center_boxes = cluster(all_data, k)

        center_boxes, vis_img = decode_boxes(center_boxes)
        cv2.imwrite("%s.jpg" %LABEl_NAME_MAP[i], vis_img)

        for b in center_boxes:
            print ("box: ", b)
            print ("size: %f || ratio: %f " % (np.sqrt(b[2] * b[3]), b[2] / float(b[3])))
            print(20 * "**")
        print(20 * "===")


if __name__ == "__main__":
    cluster_for_all(5)




