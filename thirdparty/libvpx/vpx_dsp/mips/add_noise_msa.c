/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <stdlib.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/macros_msa.h"

void vpx_plane_add_noise_msa(uint8_t *start_ptr, const int8_t *noise,
                             int blackclamp, int whiteclamp, int width,
                             int height, int32_t pitch) {
  int i, j;
  v16u8 pos0, pos1, ref0, ref1;
  v16i8 black_clamp, white_clamp, both_clamp;

  black_clamp = __msa_fill_b(blackclamp);
  white_clamp = __msa_fill_b(whiteclamp);
  both_clamp = black_clamp + white_clamp;
  both_clamp = -both_clamp;

  for (i = 0; i < height / 2; ++i) {
    uint8_t *pos0_ptr = start_ptr + (2 * i) * pitch;
    const int8_t *ref0_ptr = noise + (rand() & 0xff);
    uint8_t *pos1_ptr = start_ptr + (2 * i + 1) * pitch;
    const int8_t *ref1_ptr = noise + (rand() & 0xff);
    for (j = width / 16; j--;) {
      pos0 = LD_UB(pos0_ptr);
      ref0 = LD_UB(ref0_ptr);
      pos1 = LD_UB(pos1_ptr);
      ref1 = LD_UB(ref1_ptr);
      pos0 = __msa_subsus_u_b(pos0, black_clamp);
      pos1 = __msa_subsus_u_b(pos1, black_clamp);
      pos0 = __msa_subsus_u_b(pos0, both_clamp);
      pos1 = __msa_subsus_u_b(pos1, both_clamp);
      pos0 = __msa_subsus_u_b(pos0, white_clamp);
      pos1 = __msa_subsus_u_b(pos1, white_clamp);
      pos0 += ref0;
      ST_UB(pos0, pos0_ptr);
      pos1 += ref1;
      ST_UB(pos1, pos1_ptr);
      pos0_ptr += 16;
      pos1_ptr += 16;
      ref0_ptr += 16;
      ref1_ptr += 16;
    }
  }
}
