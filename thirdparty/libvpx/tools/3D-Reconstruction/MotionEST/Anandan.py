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
from scipy.ndimage.filters import gaussian_filter
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv
from MotionEST import MotionEST
"""Anandan Model"""


class Anandan(MotionEST):
  """
    constructor:
        cur_f: current frame
        ref_f: reference frame
        blk_sz: block size
        beta: smooth constrain weight
        k1,k2,k3: confidence coefficients
        max_iter: maximum number of iterations
    """

  def __init__(self, cur_f, ref_f, blk_sz, beta, k1, k2, k3, max_iter=100):
    super(Anandan, self).__init__(cur_f, ref_f, blk_sz)
    self.levels = int(np.log2(blk_sz))
    self.intensity_hierarchy()
    self.c_maxs = []
    self.c_mins = []
    self.e_maxs = []
    self.e_mins = []
    for l in xrange(self.levels + 1):
      c_max, c_min, e_max, e_min = self.get_curvature(self.cur_Is[l])
      self.c_maxs.append(c_max)
      self.c_mins.append(c_min)
      self.e_maxs.append(e_max)
      self.e_mins.append(e_min)
    self.beta = beta
    self.k1, self.k2, self.k3 = k1, k2, k3
    self.max_iter = max_iter

  """
    build intensity hierarchy
    """

  def intensity_hierarchy(self):
    level = 0
    self.cur_Is = []
    self.ref_Is = []
    #build each level itensity by using gaussian filters
    while level <= self.levels:
      cur_I = gaussian_filter(self.cur_yuv[:, :, 0], sigma=(2**level) * 0.56)
      ref_I = gaussian_filter(self.ref_yuv[:, :, 0], sigma=(2**level) * 0.56)
      self.ref_Is.append(ref_I)
      self.cur_Is.append(cur_I)
      level += 1

  """
    get curvature of each block
    """

  def get_curvature(self, I):
    c_max = np.zeros((self.num_row, self.num_col))
    c_min = np.zeros((self.num_row, self.num_col))
    e_max = np.zeros((self.num_row, self.num_col, 2))
    e_min = np.zeros((self.num_row, self.num_col, 2))
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        h11, h12, h21, h22 = 0, 0, 0, 0
        for i in xrange(r * self.blk_sz, r * self.blk_sz + self.blk_sz):
          for j in xrange(c * self.blk_sz, c * self.blk_sz + self.blk_sz):
            if 0 <= i < self.height - 1 and 0 <= j < self.width - 1:
              Ix = I[i][j + 1] - I[i][j]
              Iy = I[i + 1][j] - I[i][j]
              h11 += Iy * Iy
              h12 += Ix * Iy
              h21 += Ix * Iy
              h22 += Ix * Ix
        U, S, _ = LA.svd(np.array([[h11, h12], [h21, h22]]))
        c_max[r, c], c_min[r, c] = S[0], S[1]
        e_max[r, c] = U[:, 0]
        e_min[r, c] = U[:, 1]
    return c_max, c_min, e_max, e_min

  """
    get ssd of motion vector:
      cur_I: current intensity
      ref_I: reference intensity
      center: current position
      mv: motion vector
    """

  def get_ssd(self, cur_I, ref_I, center, mv):
    ssd = 0
    for r in xrange(int(center[0]), int(center[0]) + self.blk_sz):
      for c in xrange(int(center[1]), int(center[1]) + self.blk_sz):
        if 0 <= r < self.height and 0 <= c < self.width:
          tr, tc = r + int(mv[0]), c + int(mv[1])
          if 0 <= tr < self.height and 0 <= tc < self.width:
            ssd += (ref_I[tr, tc] - cur_I[r, c])**2
          else:
            ssd += cur_I[r, c]**2
    return ssd

  """
    get region match of level l
      l: current level
      last_mvs: matchine results of last level
      radius: movenment radius
    """

  def region_match(self, l, last_mvs, radius):
    mvs = np.zeros((self.num_row, self.num_col, 2))
    min_ssds = np.zeros((self.num_row, self.num_col))
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        center = np.array([r * self.blk_sz, c * self.blk_sz])
        #use overlap hierarchy policy
        init_mvs = []
        if last_mvs is None:
          init_mvs = [np.array([0, 0])]
        else:
          for i, j in {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}:
            if 0 <= i < last_mvs.shape[0] and 0 <= j < last_mvs.shape[1]:
              init_mvs.append(last_mvs[i, j])
        #use last matching results as the start position as current level
        min_ssd = None
        min_mv = None
        for init_mv in init_mvs:
          for i in xrange(-2, 3):
            for j in xrange(-2, 3):
              mv = init_mv + np.array([i, j]) * radius
              ssd = self.get_ssd(self.cur_Is[l], self.ref_Is[l], center, mv)
              if min_ssd is None or ssd < min_ssd:
                min_ssd = ssd
                min_mv = mv
        min_ssds[r, c] = min_ssd
        mvs[r, c] = min_mv
    return mvs, min_ssds

  """
    smooth motion field based on neighbor constraint
      uvs: current estimation
      mvs: matching results
      min_ssds: minimum ssd of matching results
      l: current level
    """

  def smooth(self, uvs, mvs, min_ssds, l):
    sm_uvs = np.zeros((self.num_row, self.num_col, 2))
    c_max = self.c_maxs[l]
    c_min = self.c_mins[l]
    e_max = self.e_maxs[l]
    e_min = self.e_mins[l]
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        w_max = c_max[r, c] / (
            self.k1 + self.k2 * min_ssds[r, c] + self.k3 * c_max[r, c])
        w_min = c_min[r, c] / (
            self.k1 + self.k2 * min_ssds[r, c] + self.k3 * c_min[r, c])
        w = w_max * w_min / (w_max + w_min + 1e-6)
        if w < 0:
          w = 0
        avg_uv = np.array([0.0, 0.0])
        for i, j in {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}:
          if 0 <= i < self.num_row and 0 <= j < self.num_col:
            avg_uv += 0.25 * uvs[i, j]
        sm_uvs[r, c] = (w * w * mvs[r, c] + self.beta * avg_uv) / (
            self.beta + w * w)
    return sm_uvs

  """
    motion field estimation
    """

  def motion_field_estimation(self):
    last_mvs = None
    for l in xrange(self.levels, -1, -1):
      mvs, min_ssds = self.region_match(l, last_mvs, 2**l)
      uvs = np.zeros(mvs.shape)
      for _ in xrange(self.max_iter):
        uvs = self.smooth(uvs, mvs, min_ssds, l)
      last_mvs = uvs
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        self.mf[r, c] = uvs[r, c]
