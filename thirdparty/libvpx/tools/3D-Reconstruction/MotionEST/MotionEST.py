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
import matplotlib.pyplot as plt
from Util import drawMF, MSE
"""The Base Class of Estimators"""


class MotionEST(object):
  """
    constructor:
        cur_f: current frame
        ref_f: reference frame
        blk_sz: block size
    """

  def __init__(self, cur_f, ref_f, blk_sz):
    self.cur_f = cur_f
    self.ref_f = ref_f
    self.blk_sz = blk_sz
    #convert RGB to YUV
    self.cur_yuv = np.array(self.cur_f.convert('YCbCr'), dtype=int)
    self.ref_yuv = np.array(self.ref_f.convert('YCbCr'), dtype=int)
    #frame size
    self.width = self.cur_f.size[0]
    self.height = self.cur_f.size[1]
    #motion field size
    self.num_row = self.height // self.blk_sz
    self.num_col = self.width // self.blk_sz
    #initialize motion field
    self.mf = np.zeros((self.num_row, self.num_col, 2))

  """estimation function Override by child classes"""

  def motion_field_estimation(self):
    pass

  """
    distortion of a block:
      cur_r: current row
      cur_c: current column
      mv: motion vector
      metric: distortion metric
  """

  def block_dist(self, cur_r, cur_c, mv, metric=MSE):
    cur_x = cur_c * self.blk_sz
    cur_y = cur_r * self.blk_sz
    h = min(self.blk_sz, self.height - cur_y)
    w = min(self.blk_sz, self.width - cur_x)
    cur_blk = self.cur_yuv[cur_y:cur_y + h, cur_x:cur_x + w, :]
    ref_x = int(cur_x + mv[1])
    ref_y = int(cur_y + mv[0])
    if 0 <= ref_x < self.width - w and 0 <= ref_y < self.height - h:
      ref_blk = self.ref_yuv[ref_y:ref_y + h, ref_x:ref_x + w, :]
    else:
      ref_blk = np.zeros((h, w, 3))
    return metric(cur_blk, ref_blk)

  """
    distortion of motion field
  """

  def distortion(self, mask=None, metric=MSE):
    loss = 0
    count = 0
    for i in xrange(self.num_row):
      for j in xrange(self.num_col):
        if mask is not None and mask[i, j]:
          continue
        loss += self.block_dist(i, j, self.mf[i, j], metric)
        count += 1
    return loss / count

  """evaluation compare the difference with ground truth"""

  def motion_field_evaluation(self, ground_truth):
    loss = 0
    count = 0
    gt = ground_truth.mf
    mask = ground_truth.mask
    for i in xrange(self.num_row):
      for j in xrange(self.num_col):
        if mask is not None and mask[i][j]:
          continue
        loss += LA.norm(gt[i, j] - self.mf[i, j])
        count += 1
    return loss / count

  """render the motion field"""

  def show(self, ground_truth=None, size=10):
    cur_mf = drawMF(self.cur_f, self.blk_sz, self.mf)
    if ground_truth is None:
      n_row = 1
    else:
      gt_mf = drawMF(self.cur_f, self.blk_sz, ground_truth)
      n_row = 2
    plt.figure(figsize=(n_row * size, size * self.height / self.width))
    plt.subplot(1, n_row, 1)
    plt.imshow(cur_mf)
    plt.title('Estimated Motion Field')
    if ground_truth is not None:
      plt.subplot(1, n_row, 2)
      plt.imshow(gt_mf)
      plt.title('Ground Truth')
    plt.tight_layout()
    plt.show()
