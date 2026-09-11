/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_UTIL_VPX_DEBUG_UTIL_H_
#define VPX_VPX_UTIL_VPX_DEBUG_UTIL_H_

#include "./vpx_config.h"

#include "vpx_dsp/prob.h"

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_BITSTREAM_DEBUG || CONFIG_MISMATCH_DEBUG
void bitstream_queue_set_frame_write(int frame_idx);
int bitstream_queue_get_frame_write(void);
void bitstream_queue_set_frame_read(int frame_idx);
int bitstream_queue_get_frame_read(void);
#endif

#if CONFIG_BITSTREAM_DEBUG
/* This is a debug tool used to detect bitstream error. On encoder side, it
 * pushes each bit and probability into a queue before the bit is written into
 * the Arithmetic coder. On decoder side, whenever a bit is read out from the
 * Arithmetic coder, it pops out the reference bit and probability from the
 * queue as well. If the two results do not match, this debug tool will report
 * an error.  This tool can be used to pin down the bitstream error precisely.
 * By combining gdb's backtrace method, we can detect which module causes the
 * bitstream error. */
int bitstream_queue_get_write(void);
int bitstream_queue_get_read(void);
void bitstream_queue_record_write(void);
void bitstream_queue_reset_write(void);
void bitstream_queue_pop(int *result, int *prob);
void bitstream_queue_push(int result, const int prob);
void bitstream_queue_set_skip_write(int skip);
void bitstream_queue_set_skip_read(int skip);
#endif  // CONFIG_BITSTREAM_DEBUG

#if CONFIG_MISMATCH_DEBUG
void mismatch_move_frame_idx_w(void);
void mismatch_move_frame_idx_r(void);
void mismatch_reset_frame(int num_planes);
void mismatch_record_block_pre(const uint8_t *src, int src_stride, int plane,
                               int pixel_c, int pixel_r, int blk_w, int blk_h,
                               int highbd);
void mismatch_record_block_tx(const uint8_t *src, int src_stride, int plane,
                              int pixel_c, int pixel_r, int blk_w, int blk_h,
                              int highbd);
void mismatch_check_block_pre(const uint8_t *src, int src_stride, int plane,
                              int pixel_c, int pixel_r, int blk_w, int blk_h,
                              int highbd);
void mismatch_check_block_tx(const uint8_t *src, int src_stride, int plane,
                             int pixel_c, int pixel_r, int blk_w, int blk_h,
                             int highbd);
#endif  // CONFIG_MISMATCH_DEBUG

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_UTIL_VPX_DEBUG_UTIL_H_
