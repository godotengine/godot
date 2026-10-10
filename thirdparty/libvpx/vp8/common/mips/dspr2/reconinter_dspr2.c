/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vpx/vpx_integer.h"

#if HAVE_DSPR2
inline void prefetch_load_int(unsigned char *src) {
  __asm__ __volatile__("pref   0,  0(%[src])   \n\t" : : [src] "r"(src));
}

__inline void vp8_copy_mem16x16_dspr2(unsigned char *RESTRICT src,
                                      int src_stride,
                                      unsigned char *RESTRICT dst,
                                      int dst_stride) {
  int r;
  unsigned int a0, a1, a2, a3;

  for (r = 16; r--;) {
    /* load src data in cache memory */
    prefetch_load_int(src + src_stride);

    /* use unaligned memory load and store */
    __asm__ __volatile__(
        "ulw    %[a0], 0(%[src])            \n\t"
        "ulw    %[a1], 4(%[src])            \n\t"
        "ulw    %[a2], 8(%[src])            \n\t"
        "ulw    %[a3], 12(%[src])           \n\t"
        "sw     %[a0], 0(%[dst])            \n\t"
        "sw     %[a1], 4(%[dst])            \n\t"
        "sw     %[a2], 8(%[dst])            \n\t"
        "sw     %[a3], 12(%[dst])           \n\t"
        : [a0] "=&r"(a0), [a1] "=&r"(a1), [a2] "=&r"(a2), [a3] "=&r"(a3)
        : [src] "r"(src), [dst] "r"(dst));

    src += src_stride;
    dst += dst_stride;
  }
}

__inline void vp8_copy_mem8x8_dspr2(unsigned char *RESTRICT src, int src_stride,
                                    unsigned char *RESTRICT dst,
                                    int dst_stride) {
  int r;
  unsigned int a0, a1;

  /* load src data in cache memory */
  prefetch_load_int(src + src_stride);

  for (r = 8; r--;) {
    /* use unaligned memory load and store */
    __asm__ __volatile__(
        "ulw    %[a0], 0(%[src])            \n\t"
        "ulw    %[a1], 4(%[src])            \n\t"
        "sw     %[a0], 0(%[dst])            \n\t"
        "sw     %[a1], 4(%[dst])            \n\t"
        : [a0] "=&r"(a0), [a1] "=&r"(a1)
        : [src] "r"(src), [dst] "r"(dst));

    src += src_stride;
    dst += dst_stride;
  }
}

__inline void vp8_copy_mem8x4_dspr2(unsigned char *RESTRICT src, int src_stride,
                                    unsigned char *RESTRICT dst,
                                    int dst_stride) {
  int r;
  unsigned int a0, a1;

  /* load src data in cache memory */
  prefetch_load_int(src + src_stride);

  for (r = 4; r--;) {
    /* use unaligned memory load and store */
    __asm__ __volatile__(
        "ulw    %[a0], 0(%[src])            \n\t"
        "ulw    %[a1], 4(%[src])            \n\t"
        "sw     %[a0], 0(%[dst])            \n\t"
        "sw     %[a1], 4(%[dst])            \n\t"
        : [a0] "=&r"(a0), [a1] "=&r"(a1)
        : [src] "r"(src), [dst] "r"(dst));

    src += src_stride;
    dst += dst_stride;
  }
}

#endif
