/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp8/common/blockd.h"
#include "onyxd_int.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_ports/compiler_attributes.h"
#include "vpx_ports/mem.h"
#include "detokenize.h"

void vp8_reset_mb_tokens_context(MACROBLOCKD *x) {
  ENTROPY_CONTEXT *a_ctx = ((ENTROPY_CONTEXT *)x->above_context);
  ENTROPY_CONTEXT *l_ctx = ((ENTROPY_CONTEXT *)x->left_context);

  memset(a_ctx, 0, sizeof(ENTROPY_CONTEXT_PLANES) - 1);
  memset(l_ctx, 0, sizeof(ENTROPY_CONTEXT_PLANES) - 1);

  /* Clear entropy contexts for Y2 blocks */
  if (!x->mode_info_context->mbmi.is_4x4) {
    a_ctx[8] = l_ctx[8] = 0;
  }
}

/*
    ------------------------------------------------------------------------------
    Residual decoding (Paragraph 13.2 / 13.3)
*/
static const uint8_t kBands[16 + 1] = {
  0, 1, 2, 3, 6, 4, 5, 6, 6,
  6, 6, 6, 6, 6, 6, 7, 0 /* extra entry as sentinel */
};

static const uint8_t kCat3[] = { 173, 148, 140, 0 };
static const uint8_t kCat4[] = { 176, 155, 140, 135, 0 };
static const uint8_t kCat5[] = { 180, 157, 141, 134, 130, 0 };
static const uint8_t kCat6[] = { 254, 254, 243, 230, 196, 177,
                                 153, 140, 133, 130, 129, 0 };
static const uint8_t *const kCat3456[] = { kCat3, kCat4, kCat5, kCat6 };
static const uint8_t kZigzag[16] = { 0, 1,  4,  8,  5, 2,  3,  6,
                                     9, 12, 13, 10, 7, 11, 14, 15 };

#define VP8GetBit vp8dx_decode_bool
#define NUM_PROBAS 11
#define NUM_CTX 3

/* for const-casting */
typedef const uint8_t (*ProbaArray)[NUM_CTX][NUM_PROBAS];

// With corrupt / fuzzed streams the calculation of br->value may overflow. See
// b/148271109.
static VPX_NO_UNSIGNED_OVERFLOW_CHECK int GetSigned(BOOL_DECODER *br,
                                                    int value_to_sign) {
  int split = (br->range + 1) >> 1;
  VP8_BD_VALUE bigsplit = (VP8_BD_VALUE)split << (VP8_BD_VALUE_SIZE - 8);
  int v;

  if (br->count < 0) vp8dx_bool_decoder_fill(br);

  if (br->value < bigsplit) {
    br->range = split;
    v = value_to_sign;
  } else {
    br->range = br->range - split;
    br->value = br->value - bigsplit;
    v = -value_to_sign;
  }
  br->range += br->range;
  br->value += br->value;
  br->count--;

  return v;
}
/*
   Returns the position of the last non-zero coeff plus one
   (and 0 if there's no coeff at all)
*/
static int GetCoeffs(BOOL_DECODER *br, ProbaArray prob, int ctx, int n,
                     int16_t *out) {
  const uint8_t *p = prob[n][ctx];
  if (!VP8GetBit(br, p[0])) { /* first EOB is more a 'CBP' bit. */
    return 0;
  }
  while (1) {
    ++n;
    if (!VP8GetBit(br, p[1])) {
      p = prob[kBands[n]][0];
    } else { /* non zero coeff */
      int v, j;
      if (!VP8GetBit(br, p[2])) {
        p = prob[kBands[n]][1];
        v = 1;
      } else {
        if (!VP8GetBit(br, p[3])) {
          if (!VP8GetBit(br, p[4])) {
            v = 2;
          } else {
            v = 3 + VP8GetBit(br, p[5]);
          }
        } else {
          if (!VP8GetBit(br, p[6])) {
            if (!VP8GetBit(br, p[7])) {
              v = 5 + VP8GetBit(br, 159);
            } else {
              v = 7 + 2 * VP8GetBit(br, 165);
              v += VP8GetBit(br, 145);
            }
          } else {
            const uint8_t *tab;
            const int bit1 = VP8GetBit(br, p[8]);
            const int bit0 = VP8GetBit(br, p[9 + bit1]);
            const int cat = 2 * bit1 + bit0;
            v = 0;
            for (tab = kCat3456[cat]; *tab; ++tab) {
              v += v + VP8GetBit(br, *tab);
            }
            v += 3 + (8 << cat);
          }
        }
        p = prob[kBands[n]][2];
      }
      j = kZigzag[n - 1];

      out[j] = GetSigned(br, v);

      if (n == 16 || !VP8GetBit(br, p[0])) { /* EOB */
        return n;
      }
    }
    if (n == 16) {
      return 16;
    }
  }
}

int vp8_decode_mb_tokens(VP8D_COMP *dx, MACROBLOCKD *x) {
  BOOL_DECODER *bc = x->current_bc;
  const FRAME_CONTEXT *const fc = &dx->common.fc;
  char *eobs = x->eobs;

  int i;
  int nonzeros;
  int eobtotal = 0;

  short *qcoeff_ptr;
  ProbaArray coef_probs;
  ENTROPY_CONTEXT *a_ctx = ((ENTROPY_CONTEXT *)x->above_context);
  ENTROPY_CONTEXT *l_ctx = ((ENTROPY_CONTEXT *)x->left_context);
  ENTROPY_CONTEXT *a;
  ENTROPY_CONTEXT *l;
  int skip_dc = 0;

  qcoeff_ptr = &x->qcoeff[0];

  if (!x->mode_info_context->mbmi.is_4x4) {
    a = a_ctx + 8;
    l = l_ctx + 8;

    coef_probs = fc->coef_probs[1];

    nonzeros = GetCoeffs(bc, coef_probs, (*a + *l), 0, qcoeff_ptr + 24 * 16);
    *a = *l = (nonzeros > 0);

    eobs[24] = nonzeros;
    eobtotal += nonzeros - 16;

    coef_probs = fc->coef_probs[0];
    skip_dc = 1;
  } else {
    coef_probs = fc->coef_probs[3];
    skip_dc = 0;
  }

  for (i = 0; i < 16; ++i) {
    a = a_ctx + (i & 3);
    l = l_ctx + ((i & 0xc) >> 2);

    nonzeros = GetCoeffs(bc, coef_probs, (*a + *l), skip_dc, qcoeff_ptr);
    *a = *l = (nonzeros > 0);

    nonzeros += skip_dc;
    eobs[i] = nonzeros;
    eobtotal += nonzeros;
    qcoeff_ptr += 16;
  }

  coef_probs = fc->coef_probs[2];

  a_ctx += 4;
  l_ctx += 4;
  for (i = 16; i < 24; ++i) {
    a = a_ctx + ((i > 19) << 1) + (i & 1);
    l = l_ctx + ((i > 19) << 1) + ((i & 3) > 1);

    nonzeros = GetCoeffs(bc, coef_probs, (*a + *l), 0, qcoeff_ptr);
    *a = *l = (nonzeros > 0);

    eobs[i] = nonzeros;
    eobtotal += nonzeros;
    qcoeff_ptr += 16;
  }

  return eobtotal;
}
