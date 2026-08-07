/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_TOKENIZE_H_
#define VPX_VP8_ENCODER_TOKENIZE_H_

#include "vp8/common/entropy.h"
#include "block.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  short Token;
  short Extra;
} TOKENVALUE;

typedef struct {
  const vp8_prob *context_tree;
  short Extra;
  unsigned char Token;
  unsigned char skip_eob_node;
} TOKENEXTRA;

int rd_cost_mby(MACROBLOCKD *);

extern const short *const vp8_dct_value_cost_ptr;
/* TODO: The Token field should be broken out into a separate char array to
 *  improve cache locality, since it's needed for costing when the rest of the
 *  fields are not.
 */
extern const TOKENVALUE *const vp8_dct_value_tokens_ptr;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_TOKENIZE_H_
