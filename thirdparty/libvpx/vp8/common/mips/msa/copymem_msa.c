/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp8_rtcd.h"
#include "vp8/common/mips/msa/vp8_macros_msa.h"

static void copy_8x4_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                         int32_t dst_stride) {
  uint64_t src0, src1, src2, src3;

  LD4(src, src_stride, src0, src1, src2, src3);
  SD4(src0, src1, src2, src3, dst, dst_stride);
}

static void copy_8x8_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                         int32_t dst_stride) {
  uint64_t src0, src1, src2, src3;

  LD4(src, src_stride, src0, src1, src2, src3);
  src += (4 * src_stride);
  SD4(src0, src1, src2, src3, dst, dst_stride);
  dst += (4 * dst_stride);

  LD4(src, src_stride, src0, src1, src2, src3);
  SD4(src0, src1, src2, src3, dst, dst_stride);
}

static void copy_16x16_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                           int32_t dst_stride) {
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 src8, src9, src10, src11, src12, src13, src14, src15;

  LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  src += (8 * src_stride);
  LD_UB8(src, src_stride, src8, src9, src10, src11, src12, src13, src14, src15);

  ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
  dst += (8 * dst_stride);
  ST_UB8(src8, src9, src10, src11, src12, src13, src14, src15, dst, dst_stride);
}

void vp8_copy_mem16x16_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                           int32_t dst_stride) {
  copy_16x16_msa(src, src_stride, dst, dst_stride);
}

void vp8_copy_mem8x8_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                         int32_t dst_stride) {
  copy_8x8_msa(src, src_stride, dst, dst_stride);
}

void vp8_copy_mem8x4_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                         int32_t dst_stride) {
  copy_8x4_msa(src, src_stride, dst, dst_stride);
}
