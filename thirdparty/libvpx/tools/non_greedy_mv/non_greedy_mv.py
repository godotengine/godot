##  Copyright (c) 2020 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##

import sys
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
import numpy as np
import math


def draw_mv_ls(axis, mv_ls, mode=0):
  colors = np.array([(1., 0., 0., 1.)])
  segs = np.array([
      np.array([[ptr[0], ptr[1]], [ptr[0] + ptr[2], ptr[1] + ptr[3]]])
      for ptr in mv_ls
  ])
  line_segments = LineCollection(
      segs, linewidths=(1.,), colors=colors, linestyle='solid')
  axis.add_collection(line_segments)
  if mode == 0:
    axis.scatter(mv_ls[:, 0], mv_ls[:, 1], s=2, c='b')
  else:
    axis.scatter(
        mv_ls[:, 0] + mv_ls[:, 2], mv_ls[:, 1] + mv_ls[:, 3], s=2, c='b')


def draw_pred_block_ls(axis, mv_ls, bs, mode=0):
  colors = np.array([(0., 0., 0., 1.)])
  segs = []
  for ptr in mv_ls:
    if mode == 0:
      x = ptr[0]
      y = ptr[1]
    else:
      x = ptr[0] + ptr[2]
      y = ptr[1] + ptr[3]
    x_ls = [x, x + bs, x + bs, x, x]
    y_ls = [y, y, y + bs, y + bs, y]

    segs.append(np.column_stack([x_ls, y_ls]))
  line_segments = LineCollection(
      segs, linewidths=(.5,), colors=colors, linestyle='solid')
  axis.add_collection(line_segments)


def read_frame(fp, no_swap=0):
  plane = [None, None, None]
  for i in range(3):
    line = fp.readline()
    word_ls = line.split()
    word_ls = [int(item) for item in word_ls]
    rows = word_ls[0]
    cols = word_ls[1]

    line = fp.readline()
    word_ls = line.split()
    word_ls = [int(item) for item in word_ls]

    plane[i] = np.array(word_ls).reshape(rows, cols)
    if i > 0:
      plane[i] = plane[i].repeat(2, axis=0).repeat(2, axis=1)
  plane = np.array(plane)
  if no_swap == 0:
    plane = np.swapaxes(np.swapaxes(plane, 0, 1), 1, 2)
  return plane


def yuv_to_rgb(yuv):
  #mat = np.array([
  #    [1.164,   0   , 1.596  ],
  #    [1.164, -0.391, -0.813],
  #    [1.164, 2.018 , 0     ] ]
  #               )
  #c = np.array([[ -16 , -16 , -16  ],
  #              [ 0   , -128, -128 ],
  #              [ -128, -128,   0  ]])

  mat = np.array([[1, 0, 1.4075], [1, -0.3445, -0.7169], [1, 1.7790, 0]])
  c = np.array([[0, 0, 0], [0, -128, -128], [-128, -128, 0]])
  mat_c = np.dot(mat, c)
  v = np.array([mat_c[0, 0], mat_c[1, 1], mat_c[2, 2]])
  mat = mat.transpose()
  rgb = np.dot(yuv, mat) + v
  rgb = rgb.astype(int)
  rgb = rgb.clip(0, 255)
  return rgb / 255.


def read_feature_score(fp, mv_rows, mv_cols):
  line = fp.readline()
  word_ls = line.split()
  feature_score = np.array([math.log(float(v) + 1, 2) for v in word_ls])
  feature_score = feature_score.reshape(mv_rows, mv_cols)
  return feature_score

def read_mv_mode_arr(fp, mv_rows, mv_cols):
  line = fp.readline()
  word_ls = line.split()
  mv_mode_arr = np.array([int(v) for v in word_ls])
  mv_mode_arr = mv_mode_arr.reshape(mv_rows, mv_cols)
  return mv_mode_arr


def read_frame_dpl_stats(fp):
  line = fp.readline()
  word_ls = line.split()
  frame_idx = int(word_ls[1])
  mi_rows = int(word_ls[3])
  mi_cols = int(word_ls[5])
  bs = int(word_ls[7])
  ref_frame_idx = int(word_ls[9])
  rf_idx = int(word_ls[11])
  gf_frame_offset = int(word_ls[13])
  ref_gf_frame_offset = int(word_ls[15])
  mi_size = bs / 8
  mv_ls = []
  mv_rows = int((math.ceil(mi_rows * 1. / mi_size)))
  mv_cols = int((math.ceil(mi_cols * 1. / mi_size)))
  for i in range(mv_rows * mv_cols):
    line = fp.readline()
    word_ls = line.split()
    row = int(word_ls[0]) * 8.
    col = int(word_ls[1]) * 8.
    mv_row = int(word_ls[2]) / 8.
    mv_col = int(word_ls[3]) / 8.
    mv_ls.append([col, row, mv_col, mv_row])
  mv_ls = np.array(mv_ls)
  feature_score = read_feature_score(fp, mv_rows, mv_cols)
  mv_mode_arr = read_mv_mode_arr(fp, mv_rows, mv_cols)
  img = yuv_to_rgb(read_frame(fp))
  ref = yuv_to_rgb(read_frame(fp))
  return rf_idx, frame_idx, ref_frame_idx, gf_frame_offset, ref_gf_frame_offset, mv_ls, img, ref, bs, feature_score, mv_mode_arr


def read_dpl_stats_file(filename, frame_num=0):
  fp = open(filename)
  line = fp.readline()
  width = 0
  height = 0
  data_ls = []
  while (line):
    if line[0] == '=':
      data_ls.append(read_frame_dpl_stats(fp))
    line = fp.readline()
    if frame_num > 0 and len(data_ls) == frame_num:
      break
  return data_ls


if __name__ == '__main__':
  filename = sys.argv[1]
  data_ls = read_dpl_stats_file(filename, frame_num=5)
  for rf_idx, frame_idx, ref_frame_idx, gf_frame_offset, ref_gf_frame_offset, mv_ls, img, ref, bs, feature_score, mv_mode_arr in data_ls:
    fig, axes = plt.subplots(2, 2)

    axes[0][0].imshow(img)
    draw_mv_ls(axes[0][0], mv_ls)
    draw_pred_block_ls(axes[0][0], mv_ls, bs, mode=0)
    #axes[0].grid(color='k', linestyle='-')
    axes[0][0].set_ylim(img.shape[0], 0)
    axes[0][0].set_xlim(0, img.shape[1])

    if ref is not None:
      axes[0][1].imshow(ref)
      draw_mv_ls(axes[0][1], mv_ls, mode=1)
      draw_pred_block_ls(axes[0][1], mv_ls, bs, mode=1)
      #axes[1].grid(color='k', linestyle='-')
      axes[0][1].set_ylim(ref.shape[0], 0)
      axes[0][1].set_xlim(0, ref.shape[1])

    axes[1][0].imshow(feature_score)
    #feature_score_arr = feature_score.flatten()
    #feature_score_max = feature_score_arr.max()
    #feature_score_min = feature_score_arr.min()
    #step = (feature_score_max - feature_score_min) / 20.
    #feature_score_bins = np.arange(feature_score_min, feature_score_max, step)
    #axes[1][1].hist(feature_score_arr, bins=feature_score_bins)
    im = axes[1][1].imshow(mv_mode_arr)
    #axes[1][1].figure.colorbar(im, ax=axes[1][1])

    print rf_idx, frame_idx, ref_frame_idx, gf_frame_offset, ref_gf_frame_offset, len(mv_ls)

    flatten_mv_mode = mv_mode_arr.flatten()
    zero_mv_count = sum(flatten_mv_mode == 0);
    new_mv_count = sum(flatten_mv_mode == 1);
    ref_mv_count = sum(flatten_mv_mode == 2) + sum(flatten_mv_mode == 3);
    print zero_mv_count, new_mv_count, ref_mv_count
    plt.show()
