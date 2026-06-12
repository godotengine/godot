/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPXSTATS_H_
#define VPX_VPXSTATS_H_

#include <stdio.h>

#include "vpx/vpx_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This structure is used to abstract the different ways of handling
 * first pass statistics
 */
typedef struct {
  vpx_fixed_buf_t buf;
  int pass;
  FILE *file;
  char *buf_ptr;
  size_t buf_alloc_sz;
} stats_io_t;

int stats_open_file(stats_io_t *stats, const char *fpf, int pass);
int stats_open_mem(stats_io_t *stats, int pass);
void stats_close(stats_io_t *stats, int last_pass);
void stats_write(stats_io_t *stats, const void *pkt, size_t len);
vpx_fixed_buf_t stats_get(stats_io_t *stats);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPXSTATS_H_
