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
"""Horn & Schunck Model"""


class HornSchunck(MotionEST):
  """
    constructor:
        cur_f: current frame
        ref_f: reference frame
        blk_sz: block size
        alpha: smooth constrain weight
        sigma: gaussian blur parameter
    """

  def __init__(self, cur_f, ref_f, blk_sz, alpha, sigma, max_iter=100):
    super(HornSchunck, self).__init__(cur_f, ref_f, blk_sz)
    self.cur_I, self.ref_I = self.getIntensity()
    #perform gaussian blur to smooth the intensity
    self.cur_I = gaussian_filter(self.cur_I, sigma=sigma)
    self.ref_I = gaussian_filter(self.ref_I, sigma=sigma)
    self.alpha = alpha
    self.max_iter = max_iter
    self.Ix, self.Iy, self.It = self.intensityDiff()

  """
    Build Frame Intensity
    """

  def getIntensity(self):
    cur_I = np.zeros((self.num_row, self.num_col))
    ref_I = np.zeros((self.num_row, self.num_col))
    #use average intensity as block's intensity
    for i in xrange(self.num_row):
      for j in xrange(self.num_col):
        r = i * self.blk_sz
        c = j * self.blk_sz
        cur_I[i, j] = np.mean(self.cur_yuv[r:r + self.blk_sz, c:c + self.blk_sz,
                                           0])
        ref_I[i, j] = np.mean(self.ref_yuv[r:r + self.blk_sz, c:c + self.blk_sz,
                                           0])
    return cur_I, ref_I

  """
    Get First Order Derivative
    """

  def intensityDiff(self):
    Ix = np.zeros((self.num_row, self.num_col))
    Iy = np.zeros((self.num_row, self.num_col))
    It = np.zeros((self.num_row, self.num_col))
    sz = self.blk_sz
    for i in xrange(self.num_row - 1):
      for j in xrange(self.num_col - 1):
        """
                Ix:
                (i  ,j) <--- (i  ,j+1)
                (i+1,j) <--- (i+1,j+1)
                """
        count = 0
        for r, c in {(i, j + 1), (i + 1, j + 1)}:
          if 0 <= r < self.num_row and 0 < c < self.num_col:
            Ix[i, j] += (
                self.cur_I[r, c] - self.cur_I[r, c - 1] + self.ref_I[r, c] -
                self.ref_I[r, c - 1])
            count += 2
        Ix[i, j] /= count
        """
                Iy:
                (i  ,j)      (i  ,j+1)
                   ^             ^
                   |             |
                (i+1,j)      (i+1,j+1)
                """
        count = 0
        for r, c in {(i + 1, j), (i + 1, j + 1)}:
          if 0 < r < self.num_row and 0 <= c < self.num_col:
            Iy[i, j] += (
                self.cur_I[r, c] - self.cur_I[r - 1, c] + self.ref_I[r, c] -
                self.ref_I[r - 1, c])
            count += 2
        Iy[i, j] /= count
        count = 0
        #It:
        for r in xrange(i, i + 2):
          for c in xrange(j, j + 2):
            if 0 <= r < self.num_row and 0 <= c < self.num_col:
              It[i, j] += (self.ref_I[r, c] - self.cur_I[r, c])
              count += 1
        It[i, j] /= count
    return Ix, Iy, It

  """
    Get weighted average of neighbor motion vectors
    for evaluation of laplacian
    """

  def averageMV(self):
    avg = np.zeros((self.num_row, self.num_col, 2))
    """
        1/12 ---  1/6 --- 1/12
         |         |       |
        1/6  --- -1/8 --- 1/6
         |         |       |
        1/12 ---  1/6 --- 1/12
        """
    for i in xrange(self.num_row):
      for j in xrange(self.num_col):
        for r, c in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
          if 0 <= i + r < self.num_row and 0 <= j + c < self.num_col:
            avg[i, j] += self.mf[i + r, j + c] / 6.0
        for r, c in {(-1, -1), (-1, 1), (1, -1), (1, 1)}:
          if 0 <= i + r < self.num_row and 0 <= j + c < self.num_col:
            avg[i, j] += self.mf[i + r, j + c] / 12.0
    return avg

  def motion_field_estimation(self):
    count = 0
    """
        u_{n+1} = ~u_n - Ix(Ix.~u_n+Iy.~v+It)/(IxIx+IyIy+alpha^2)
        v_{n+1} = ~v_n - Iy(Ix.~u_n+Iy.~v+It)/(IxIx+IyIy+alpha^2)
        """
    denom = self.alpha**2 + np.power(self.Ix, 2) + np.power(self.Iy, 2)
    while count < self.max_iter:
      avg = self.averageMV()
      self.mf[:, :, 1] = avg[:, :, 1] - self.Ix * (
          self.Ix * avg[:, :, 1] + self.Iy * avg[:, :, 0] + self.It) / denom
      self.mf[:, :, 0] = avg[:, :, 0] - self.Iy * (
          self.Ix * avg[:, :, 1] + self.Iy * avg[:, :, 0] + self.It) / denom
      count += 1
    self.mf *= self.blk_sz

  def motion_field_estimation_mat(self):
    row_idx = []
    col_idx = []
    data = []

    N = 2 * self.num_row * self.num_col
    b = np.zeros((N, 1))
    for i in xrange(self.num_row):
      for j in xrange(self.num_col):
        """(IxIx+alpha^2)u+IxIy.v-alpha^2~u IxIy.u+(IyIy+alpha^2)v-alpha^2~v"""
        u_idx = i * 2 * self.num_col + 2 * j
        v_idx = u_idx + 1
        b[u_idx, 0] = -self.Ix[i, j] * self.It[i, j]
        b[v_idx, 0] = -self.Iy[i, j] * self.It[i, j]
        #u: (IxIx+alpha^2)u
        row_idx.append(u_idx)
        col_idx.append(u_idx)
        data.append(self.Ix[i, j] * self.Ix[i, j] + self.alpha**2)
        #IxIy.v
        row_idx.append(u_idx)
        col_idx.append(v_idx)
        data.append(self.Ix[i, j] * self.Iy[i, j])

        #v: IxIy.u
        row_idx.append(v_idx)
        col_idx.append(u_idx)
        data.append(self.Ix[i, j] * self.Iy[i, j])
        #(IyIy+alpha^2)v
        row_idx.append(v_idx)
        col_idx.append(v_idx)
        data.append(self.Iy[i, j] * self.Iy[i, j] + self.alpha**2)

        #-alpha^2~u
        #-alpha^2~v
        for r, c in {(-1, 0), (1, 0), (0, -1), (0, 1)}:
          if 0 <= i + r < self.num_row and 0 <= j + c < self.num_col:
            u_nb = (i + r) * 2 * self.num_col + 2 * (j + c)
            v_nb = u_nb + 1

            row_idx.append(u_idx)
            col_idx.append(u_nb)
            data.append(-1 * self.alpha**2 / 6.0)

            row_idx.append(v_idx)
            col_idx.append(v_nb)
            data.append(-1 * self.alpha**2 / 6.0)
        for r, c in {(-1, -1), (-1, 1), (1, -1), (1, 1)}:
          if 0 <= i + r < self.num_row and 0 <= j + c < self.num_col:
            u_nb = (i + r) * 2 * self.num_col + 2 * (j + c)
            v_nb = u_nb + 1

            row_idx.append(u_idx)
            col_idx.append(u_nb)
            data.append(-1 * self.alpha**2 / 12.0)

            row_idx.append(v_idx)
            col_idx.append(v_nb)
            data.append(-1 * self.alpha**2 / 12.0)
    M = csc_matrix((data, (row_idx, col_idx)), shape=(N, N))
    M_inv = inv(M)
    uv = M_inv.dot(b)

    for i in xrange(self.num_row):
      for j in xrange(self.num_col):
        self.mf[i, j, 0] = uv[i * 2 * self.num_col + 2 * j + 1, 0] * self.blk_sz
        self.mf[i, j, 1] = uv[i * 2 * self.num_col + 2 * j, 0] * self.blk_sz
