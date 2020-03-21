# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import math
import sys
import os


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)