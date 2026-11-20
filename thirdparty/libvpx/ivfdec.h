/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#ifndef VPX_IVFDEC_H_
#define VPX_IVFDEC_H_

#include "./tools_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int file_is_ivf(struct VpxInputContext *input);

int ivf_read_frame(FILE *infile, uint8_t **buffer, size_t *bytes_read,
                   size_t *buffer_size);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif  // VPX_IVFDEC_H_
