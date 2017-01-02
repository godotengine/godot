/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#ifndef VP8_DECODER_TREEREADER_H_
#define VP8_DECODER_TREEREADER_H_

#include "./vpx_config.h"
#include "vp8/common/treecoder.h"
#include "dboolhuff.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef BOOL_DECODER vp8_reader;

#define vp8_read vp8dx_decode_bool
#define vp8_read_literal vp8_decode_value
#define vp8_read_bit(R) vp8_read(R, vp8_prob_half)


/* Intent of tree data structure is to make decoding trivial. */

static INLINE int vp8_treed_read(
    vp8_reader *const r,        /* !!! must return a 0 or 1 !!! */
    vp8_tree t,
    const vp8_prob *const p
)
{
    register vp8_tree_index i = 0;

    while ((i = t[ i + vp8_read(r, p[i>>1])]) > 0) ;

    return -i;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP8_DECODER_TREEREADER_H_
