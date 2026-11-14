##  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

# coding: utf-8
import numpy as np
import numpy.linalg as LA
from Util import MSE
from MotionEST import MotionEST
"""Exhaust Search:"""


class Exhaust(MotionEST):
  """
    Constructor:
        cur_f: current frame
        ref_f: reference frame
        blk_sz: block size
        wnd_size: search window size
        metric: metric to compare the blocks distrotion
    """

  def __init__(self, cur_f, ref_f, blk_size, wnd_size, metric=MSE):
    self.name = 'exhaust'
    self.wnd_sz = wnd_size
    self.metric = metric
    super(Exhaust, self).__init__(cur_f, ref_f, blk_size)

  """
    search method:
        cur_r: start row
        cur_c: start column
    """

  def search(self, cur_r, cur_c):
    min_loss = self.block_dist(cur_r, cur_c, [0, 0], self.metric)
    cur_x = cur_c * self.blk_sz
    cur_y = cur_r * self.blk_sz
    ref_x = cur_x
    ref_y = cur_y
    #search all validate positions and select the one with minimum distortion
    for y in xrange(cur_y - self.wnd_sz, cur_y + self.wnd_sz):
      for x in xrange(cur_x - self.wnd_sz, cur_x + self.wnd_sz):
        if 0 <= x < self.width - self.blk_sz and 0 <= y < self.height - self.blk_sz:
          loss = self.block_dist(cur_r, cur_c, [y - cur_y, x - cur_x],
                                 self.metric)
          if loss < min_loss:
            min_loss = loss
            ref_x = x
            ref_y = y
    return ref_x, ref_y

  def motion_field_estimation(self):
    for i in xrange(self.num_row):
      for j in xrange(self.num_col):
        ref_x, ref_y = self.search(i, j)
        self.mf[i, j] = np.array(
            [ref_y - i * self.blk_sz, ref_x - j * self.blk_sz])


"""Exhaust with Neighbor Constraint"""


class ExhaustNeighbor(MotionEST):
  """
    Constructor:
        cur_f: current frame
        ref_f: reference frame
        blk_sz: block size
        wnd_size: search window size
        beta: neigbor loss weight
        metric: metric to compare the blocks distrotion
    """

  def __init__(self, cur_f, ref_f, blk_size, wnd_size, beta, metric=MSE):
    self.name = 'exhaust + neighbor'
    self.wnd_sz = wnd_size
    self.beta = beta
    self.metric = metric
    super(ExhaustNeighbor, self).__init__(cur_f, ref_f, blk_size)
    self.assign = np.zeros((self.num_row, self.num_col), dtype=bool)

  """
    estimate neighbor loss:
        cur_r: current row
        cur_c: current column
        mv: current motion vector
    """

  def neighborLoss(self, cur_r, cur_c, mv):
    loss = 0
    #accumulate difference between current block's motion vector with neighbors'
    for i, j in {(-1, 0), (1, 0), (0, 1), (0, -1)}:
      nb_r = cur_r + i
      nb_c = cur_c + j
      if 0 <= nb_r < self.num_row and 0 <= nb_c < self.num_col and self.assign[
          nb_r, nb_c]:
        loss += LA.norm(mv - self.mf[nb_r, nb_c])
    return loss

  """
    search method:
        cur_r: start row
        cur_c: start column
    """

  def search(self, cur_r, cur_c):
    dist_loss = self.block_dist(cur_r, cur_c, [0, 0], self.metric)
    nb_loss = self.neighborLoss(cur_r, cur_c, np.array([0, 0]))
    min_loss = dist_loss + self.beta * nb_loss
    cur_x = cur_c * self.blk_sz
    cur_y = cur_r * self.blk_sz
    ref_x = cur_x
    ref_y = cur_y
    #search all validate positions and select the one with minimum distortion
    # as well as weighted neighbor loss
    for y in xrange(cur_y - self.wnd_sz, cur_y + self.wnd_sz):
      for x in xrange(cur_x - self.wnd_sz, cur_x + self.wnd_sz):
        if 0 <= x < self.width - self.blk_sz and 0 <= y < self.height - self.blk_sz:
          dist_loss = self.block_dist(cur_r, cur_c, [y - cur_y, x - cur_x],
                                      self.metric)
          nb_loss = self.neighborLoss(cur_r, cur_c, [y - cur_y, x - cur_x])
          loss = dist_loss + self.beta * nb_loss
          if loss < min_loss:
            min_loss = loss
            ref_x = x
            ref_y = y
    return ref_x, ref_y

  def motion_field_estimation(self):
    for i in xrange(self.num_row):
      for j in xrange(self.num_col):
        ref_x, ref_y = self.search(i, j)
        self.mf[i, j] = np.array(
            [ref_y - i * self.blk_sz, ref_x - j * self.blk_sz])
        self.assign[i, j] = True


"""Exhaust with Neighbor Constraint and Feature Score"""


class ExhaustNeighborFeatureScore(MotionEST):
  """
    Constructor:
        cur_f: current frame
        ref_f: reference frame
        blk_sz: block size
        wnd_size: search window size
        beta: neigbor loss weight
        max_iter: maximum number of iterations
        metric: metric to compare the blocks distrotion
    """

  def __init__(self,
               cur_f,
               ref_f,
               blk_size,
               wnd_size,
               beta=1,
               max_iter=100,
               metric=MSE):
    self.name = 'exhaust + neighbor+feature score'
    self.wnd_sz = wnd_size
    self.beta = beta
    self.metric = metric
    self.max_iter = max_iter
    super(ExhaustNeighborFeatureScore, self).__init__(cur_f, ref_f, blk_size)
    self.fs = self.getFeatureScore()

  """
    get feature score of each block
    """

  def getFeatureScore(self):
    fs = np.zeros((self.num_row, self.num_col))
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        IxIx = 0
        IyIy = 0
        IxIy = 0
        #get ssd surface
        for x in xrange(self.blk_sz - 1):
          for y in xrange(self.blk_sz - 1):
            ox = c * self.blk_sz + x
            oy = r * self.blk_sz + y
            Ix = self.cur_yuv[oy, ox + 1, 0] - self.cur_yuv[oy, ox, 0]
            Iy = self.cur_yuv[oy + 1, ox, 0] - self.cur_yuv[oy, ox, 0]
            IxIx += Ix * Ix
            IyIy += Iy * Iy
            IxIy += Ix * Iy
        #get maximum and minimum eigenvalues
        lambda_max = 0.5 * ((IxIx + IyIy) + np.sqrt(4 * IxIy * IxIy +
                                                    (IxIx - IyIy)**2))
        lambda_min = 0.5 * ((IxIx + IyIy) - np.sqrt(4 * IxIy * IxIy +
                                                    (IxIx - IyIy)**2))
        fs[r, c] = lambda_max * lambda_min / (1e-6 + lambda_max + lambda_min)
        if fs[r, c] < 0:
          fs[r, c] = 0
    return fs

  """
    do exhaust search
    """

  def search(self, cur_r, cur_c):
    min_loss = self.block_dist(cur_r, cur_c, [0, 0], self.metric)
    cur_x = cur_c * self.blk_sz
    cur_y = cur_r * self.blk_sz
    ref_x = cur_x
    ref_y = cur_y
    #search all validate positions and select the one with minimum distortion
    for y in xrange(cur_y - self.wnd_sz, cur_y + self.wnd_sz):
      for x in xrange(cur_x - self.wnd_sz, cur_x + self.wnd_sz):
        if 0 <= x < self.width - self.blk_sz and 0 <= y < self.height - self.blk_sz:
          loss = self.block_dist(cur_r, cur_c, [y - cur_y, x - cur_x],
                                 self.metric)
          if loss < min_loss:
            min_loss = loss
            ref_x = x
            ref_y = y
    return ref_x, ref_y

  """
    add smooth constraint
    """

  def smooth(self, uvs, mvs):
    sm_uvs = np.zeros(uvs.shape)
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        avg_uv = np.array([0.0, 0.0])
        for i, j in {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}:
          if 0 <= i < self.num_row and 0 <= j < self.num_col:
            avg_uv += uvs[i, j] / 6.0
        for i, j in {(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1),
                     (r + 1, c + 1)}:
          if 0 <= i < self.num_row and 0 <= j < self.num_col:
            avg_uv += uvs[i, j] / 12.0
        sm_uvs[r, c] = (self.fs[r, c] * mvs[r, c] + self.beta * avg_uv) / (
            self.beta + self.fs[r, c])
    return sm_uvs

  def motion_field_estimation(self):
    #get matching results
    mvs = np.zeros(self.mf.shape)
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        ref_x, ref_y = self.search(r, c)
        mvs[r, c] = np.array([ref_y - r * self.blk_sz, ref_x - c * self.blk_sz])
    #add smoothness constraint
    uvs = np.zeros(self.mf.shape)
    for _ in xrange(self.max_iter):
      uvs = self.smooth(uvs, mvs)
    self.mf = uvs
