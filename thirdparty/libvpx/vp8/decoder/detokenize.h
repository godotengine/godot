/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP8_DECODER_DETOKENIZE_H_
#define VP8_DECODER_DETOKENIZE_H_

#include "onyxd_int.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp8_reset_mb_tokens_context(MACROBLOCKD *x);
int vp8_decode_mb_tokens(VP8D_COMP *, MACROBLOCKD *);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP8_DECODER_DETOKENIZE_H_
