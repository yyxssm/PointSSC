# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-12-10 10:38:01
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-26 14:21:36
# @Email:  cshzxie@gmail.com
#
# Note:
# - Replace float -> double, kFloat -> kDouble in chamfer.cu

import os
import sys
import torch
import unittest


from torch.autograd import gradcheck

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from extensions.chamfer_dist import ChamferFunction, ChamferDistanceL2, ChamferDistanceL1


class ChamferDistanceTestCase(unittest.TestCase):
    def test_chamfer_dist(self):
        x = torch.rand(4, 64, 3).double()
        y = torch.rand(4, 128, 3).double()
        x.requires_grad = True
        y.requires_grad = True
        print(gradcheck(ChamferFunction.apply, [x.cuda(), y.cuda()]))



if __name__ == '__main__':
    # unittest.main()
    # import pdb
    import time
    time_start = time.time()
    memory_start = torch.cuda.max_memory_allocated(0)
    x = torch.rand(32, 163840, 3)
    y = torch.rand(32, 163840, 3)
    l2_loss = ChamferDistanceL2(ignore_zeros=True)(x.cuda(), y.cuda())
    l1_loss = ChamferDistanceL1(ignore_zeros=True)(x.cuda(), y.cuda())
    print("L2 loss = {}".format(l2_loss))
    print("L1 loss = {}".format(l1_loss))
    time_end = time.time()
    memory_end = torch.cuda.max_memory_allocated(0)
    print("time cost: {:.4f}s".format(time_end - time_start))
    print("memory cost max: {:.2f} M".format((memory_end - memory_start) / 1024 / 1024))
    # pdb.set_trace()
