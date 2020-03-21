# -*- coding: utf-8 -*-

from __future__ import absolute_import,print_function,division
from libs.box_utils.cython_utils.cython_nms import soft_nms
import numpy as np

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