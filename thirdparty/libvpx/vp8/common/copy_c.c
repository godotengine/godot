/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include <string.h>

#include "./vp8_rtcd.h"
#include "vpx/vpx_integer.h"

/* Copy 2 macroblocks to a buffer */
void vp8_copy32xn_c(const unsigned char *src_ptr, int src_stride,
                    unsigned char *dst_ptr, int dst_stride,
                    int height)
{
    int r;

    for (r = 0; r < height; r++)
    {
        memcpy(dst_ptr, src_ptr, 32);

        src_ptr += src_stride;
        dst_ptr += dst_stride;

    }
}
