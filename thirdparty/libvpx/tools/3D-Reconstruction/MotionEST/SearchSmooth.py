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
"""Search & Smooth Model with Adapt Weights"""


class SearchSmoothAdapt(MotionEST):
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

  def __init__(self, cur_f, ref_f, blk_size, search, max_iter=100):
    self.search = search
    self.max_iter = max_iter
    super(SearchSmoothAdapt, self).__init__(cur_f, ref_f, blk_size)

  """
    get local diffiencial of refernce
    """

  def getRefLocalDiff(self, mvs):
    m, n = self.num_row, self.num_col
    localDiff = [[] for _ in xrange(m)]
    blk_sz = self.blk_sz
    for r in xrange(m):
      for c in xrange(n):
        I_row = 0
        I_col = 0
        #get ssd surface
        count = 0
        center = self.cur_yuv[r * blk_sz:(r + 1) * blk_sz,
                              c * blk_sz:(c + 1) * blk_sz, 0]
        ty = np.clip(r * blk_sz + int(mvs[r, c, 0]), 0, self.height - blk_sz)
        tx = np.clip(c * blk_sz + int(mvs[r, c, 1]), 0, self.width - blk_sz)
        target = self.ref_yuv[ty:ty + blk_sz, tx:tx + blk_sz, 0]
        for y, x in {(ty - blk_sz, tx), (ty + blk_sz, tx)}:
          if 0 <= y < self.height - blk_sz and 0 <= x < self.width - blk_sz:
            nb = self.ref_yuv[y:y + blk_sz, x:x + blk_sz, 0]
            I_row += np.sum(np.abs(nb - center)) - np.sum(
                np.abs(target - center))
            count += 1
        I_row //= (count * blk_sz * blk_sz)
        count = 0
        for y, x in {(ty, tx - blk_sz), (ty, tx + blk_sz)}:
          if 0 <= y < self.height - blk_sz and 0 <= x < self.width - blk_sz:
            nb = self.ref_yuv[y:y + blk_sz, x:x + blk_sz, 0]
            I_col += np.sum(np.abs(nb - center)) - np.sum(
                np.abs(target - center))
            count += 1
        I_col //= (count * blk_sz * blk_sz)
        localDiff[r].append(
            np.array([[I_row * I_row, I_row * I_col],
                      [I_col * I_row, I_col * I_col]]))
    return localDiff

  """
    add smooth constraint
    """

  def smooth(self, uvs, mvs):
    sm_uvs = np.zeros(uvs.shape)
    blk_sz = self.blk_sz
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        nb_uv = np.array([0.0, 0.0])
        for i, j in {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}:
          if 0 <= i < self.num_row and 0 <= j < self.num_col:
            nb_uv += uvs[i, j] / 6.0
          else:
            nb_uv += uvs[r, c] / 6.0
        for i, j in {(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1),
                     (r + 1, c + 1)}:
          if 0 <= i < self.num_row and 0 <= j < self.num_col:
            nb_uv += uvs[i, j] / 12.0
          else:
            nb_uv += uvs[r, c] / 12.0
        ssd_nb = self.block_dist(r, c, self.blk_sz * nb_uv)
        mv = mvs[r, c]
        ssd_mv = self.block_dist(r, c, mv)
        alpha = (ssd_nb - ssd_mv) / (ssd_mv + 1e-6)
        M = alpha * self.localDiff[r][c]
        P = M + np.identity(2)
        inv_P = LA.inv(P)
        sm_uvs[r, c] = np.dot(inv_P, nb_uv) + np.dot(
            np.matmul(inv_P, M), mv / blk_sz)
    return sm_uvs

  def block_matching(self):
    self.search.motion_field_estimation()

  def motion_field_estimation(self):
    self.localDiff = self.getRefLocalDiff(self.search.mf)
    #get matching results
    mvs = self.search.mf
    #add smoothness constraint
    uvs = mvs / self.blk_sz
    for _ in xrange(self.max_iter):
      uvs = self.smooth(uvs, mvs)
    self.mf = uvs * self.blk_sz


"""Search & Smooth Model with Fixed Weights"""


class SearchSmoothFix(MotionEST):
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

  def __init__(self, cur_f, ref_f, blk_size, search, beta, max_iter=100):
    self.search = search
    self.max_iter = max_iter
    self.beta = beta
    super(SearchSmoothFix, self).__init__(cur_f, ref_f, blk_size)

  """
    get local diffiencial of refernce
    """

  def getRefLocalDiff(self, mvs):
    m, n = self.num_row, self.num_col
    localDiff = [[] for _ in xrange(m)]
    blk_sz = self.blk_sz
    for r in xrange(m):
      for c in xrange(n):
        I_row = 0
        I_col = 0
        #get ssd surface
        count = 0
        center = self.cur_yuv[r * blk_sz:(r + 1) * blk_sz,
                              c * blk_sz:(c + 1) * blk_sz, 0]
        ty = np.clip(r * blk_sz + int(mvs[r, c, 0]), 0, self.height - blk_sz)
        tx = np.clip(c * blk_sz + int(mvs[r, c, 1]), 0, self.width - blk_sz)
        target = self.ref_yuv[ty:ty + blk_sz, tx:tx + blk_sz, 0]
        for y, x in {(ty - blk_sz, tx), (ty + blk_sz, tx)}:
          if 0 <= y < self.height - blk_sz and 0 <= x < self.width - blk_sz:
            nb = self.ref_yuv[y:y + blk_sz, x:x + blk_sz, 0]
            I_row += np.sum(np.abs(nb - center)) - np.sum(
                np.abs(target - center))
            count += 1
        I_row //= (count * blk_sz * blk_sz)
        count = 0
        for y, x in {(ty, tx - blk_sz), (ty, tx + blk_sz)}:
          if 0 <= y < self.height - blk_sz and 0 <= x < self.width - blk_sz:
            nb = self.ref_yuv[y:y + blk_sz, x:x + blk_sz, 0]
            I_col += np.sum(np.abs(nb - center)) - np.sum(
                np.abs(target - center))
            count += 1
        I_col //= (count * blk_sz * blk_sz)
        localDiff[r].append(
            np.array([[I_row * I_row, I_row * I_col],
                      [I_col * I_row, I_col * I_col]]))
    return localDiff

  """
    add smooth constraint
    """

  def smooth(self, uvs, mvs):
    sm_uvs = np.zeros(uvs.shape)
    blk_sz = self.blk_sz
    for r in xrange(self.num_row):
      for c in xrange(self.num_col):
        nb_uv = np.array([0.0, 0.0])
        for i, j in {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}:
          if 0 <= i < self.num_row and 0 <= j < self.num_col:
            nb_uv += uvs[i, j] / 6.0
          else:
            nb_uv += uvs[r, c] / 6.0
        for i, j in {(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1),
                     (r + 1, c + 1)}:
          if 0 <= i < self.num_row and 0 <= j < self.num_col:
            nb_uv += uvs[i, j] / 12.0
          else:
            nb_uv += uvs[r, c] / 12.0
        mv = mvs[r, c] / blk_sz
        M = self.localDiff[r][c]
        P = M + self.beta * np.identity(2)
        inv_P = LA.inv(P)
        sm_uvs[r, c] = np.dot(inv_P, self.beta * nb_uv) + np.dot(
            np.matmul(inv_P, M), mv)
    return sm_uvs

  def block_matching(self):
    self.search.motion_field_estimation()

  def motion_field_estimation(self):
    #get local structure
    self.localDiff = self.getRefLocalDiff(self.search.mf)
    #get matching results
    mvs = self.search.mf
    #add smoothness constraint
    uvs = mvs / self.blk_sz
    for _ in xrange(self.max_iter):
      uvs = self.smooth(uvs, mvs)
    self.mf = uvs * self.blk_sz
