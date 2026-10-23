/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_COMMON_ENTROPY_H_
#define VPX_VP8_COMMON_ENTROPY_H_

#include "treecoder.h"
#include "blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Coefficient token alphabet */

#define ZERO_TOKEN 0         /* 0         Extra Bits 0+0 */
#define ONE_TOKEN 1          /* 1         Extra Bits 0+1 */
#define TWO_TOKEN 2          /* 2         Extra Bits 0+1 */
#define THREE_TOKEN 3        /* 3         Extra Bits 0+1 */
#define FOUR_TOKEN 4         /* 4         Extra Bits 0+1 */
#define DCT_VAL_CATEGORY1 5  /* 5-6       Extra Bits 1+1 */
#define DCT_VAL_CATEGORY2 6  /* 7-10      Extra Bits 2+1 */
#define DCT_VAL_CATEGORY3 7  /* 11-18     Extra Bits 3+1 */
#define DCT_VAL_CATEGORY4 8  /* 19-34     Extra Bits 4+1 */
#define DCT_VAL_CATEGORY5 9  /* 35-66     Extra Bits 5+1 */
#define DCT_VAL_CATEGORY6 10 /* 67+       Extra Bits 11+1 */
#define DCT_EOB_TOKEN 11     /* EOB       Extra Bits 0+0 */

#define MAX_ENTROPY_TOKENS 12
#define ENTROPY_NODES 11

extern const vp8_tree_index vp8_coef_tree[];

extern const struct vp8_token_struct vp8_coef_encodings[MAX_ENTROPY_TOKENS];

typedef struct {
  vp8_tree_p tree;
  const vp8_prob *prob;
  int Len;
  int base_val;
} vp8_extra_bit_struct;

extern const vp8_extra_bit_struct
    vp8_extra_bits[12]; /* indexed by token value */

#define PROB_UPDATE_BASELINE_COST 7

#define MAX_PROB 255
#define DCT_MAX_VALUE 2048

/* Coefficients are predicted via a 3-dimensional probability table. */

/* Outside dimension.  0 = Y no DC, 1 = Y2, 2 = UV, 3 = Y with DC */

#define BLOCK_TYPES 4

/* Middle dimension is a coarsening of the coefficient's
   position within the 4x4 DCT. */

#define COEF_BANDS 8
extern DECLARE_ALIGNED(16, const unsigned char, vp8_coef_bands[16]);

/* Inside dimension is 3-valued measure of nearby complexity, that is,
   the extent to which nearby coefficients are nonzero.  For the first
   coefficient (DC, unless block type is 0), we look at the (already encoded)
   blocks above and to the left of the current block.  The context index is
   then the number (0,1,or 2) of these blocks having nonzero coefficients.
   After decoding a coefficient, the measure is roughly the size of the
   most recently decoded coefficient (0 for 0, 1 for 1, 2 for >1).
   Note that the intuitive meaning of this measure changes as coefficients
   are decoded, e.g., prior to the first token, a zero means that my neighbors
   are empty while, after the first token, because of the use of end-of-block,
   a zero means we just decoded a zero and hence guarantees that a non-zero
   coefficient will appear later in this block.  However, this shift
   in meaning is perfectly OK because our context depends also on the
   coefficient band (and since zigzag positions 0, 1, and 2 are in
   distinct bands). */

/*# define DC_TOKEN_CONTEXTS        3*/ /* 00, 0!0, !0!0 */
#define PREV_COEF_CONTEXTS 3

extern DECLARE_ALIGNED(16, const unsigned char,
                       vp8_prev_token_class[MAX_ENTROPY_TOKENS]);

extern const vp8_prob vp8_coef_update_probs[BLOCK_TYPES][COEF_BANDS]
                                           [PREV_COEF_CONTEXTS][ENTROPY_NODES];

struct VP8Common;
void vp8_default_coef_probs(struct VP8Common *);

extern DECLARE_ALIGNED(16, const int, vp8_default_zig_zag1d[16]);
extern DECLARE_ALIGNED(16, const short, vp8_default_inv_zig_zag[16]);
extern DECLARE_ALIGNED(16, const short, vp8_default_zig_zag_mask[16]);
extern const int vp8_mb_feature_data_bits[MB_LVL_MAX];

void vp8_coef_tree_initialize(void);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_ENTROPY_H_
