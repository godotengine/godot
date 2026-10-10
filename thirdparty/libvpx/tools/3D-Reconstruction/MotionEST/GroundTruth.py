##  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

#coding : utf - 8
import numpy as np
import numpy.linalg as LA
from MotionEST import MotionEST
"""Ground Truth:

  Load in ground truth motion field and mask
"""


class GroundTruth(MotionEST):
  """constructor:

    cur_f:current
    frame ref_f:reference
    frame blk_sz:block size
    gt_path:ground truth motion field file path
    """

  def __init__(self, cur_f, ref_f, blk_sz, gt_path, mf=None, mask=None):
    self.name = 'ground truth'
    super(GroundTruth, self).__init__(cur_f, ref_f, blk_sz)
    self.mask = np.zeros((self.num_row, self.num_col), dtype=bool)
    if gt_path:
      with open(gt_path) as gt_file:
        lines = gt_file.readlines()
        for i in xrange(len(lines)):
          info = lines[i].split(';')
          for j in xrange(len(info)):
            x, y = info[j].split(',')
            #-, - stands for nothing
            if x == '-' or y == '-':
              self.mask[i, -j - 1] = True
              continue
            #the order of original file is flipped on the x axis
            self.mf[i, -j - 1] = np.array([float(y), -float(x)], dtype=int)
    else:
      self.mf = mf
      self.mask = mask
