// Copyright 2010 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Quantizer initialization
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./vp8i.h"

static WEBP_INLINE int clip(int v, int M) {
  return v < 0 ? 0 : v > M ? M : v;
}

// Paragraph 14.1
static const uint8_t kDcTable[128] = {
  4,     5,   6,   7,   8,   9,  10,  10,
  11,   12,  13,  14,  15,  16,  17,  17,
  18,   19,  20,  20,  21,  21,  22,  22,
  23,   23,  24,  25,  25,  26,  27,  28,
  29,   30,  31,  32,  33,  34,  35,  36,
  37,   37,  38,  39,  40,  41,  42,  43,
  44,   45,  46,  46,  47,  48,  49,  50,
  51,   52,  53,  54,  55,  56,  57,  58,
  59,   60,  61,  62,  63,  64,  65,  66,
  67,   68,  69,  70,  71,  72,  73,  74,
  75,   76,  76,  77,  78,  79,  80,  81,
  82,   83,  84,  85,  86,  87,  88,  89,
  91,   93,  95,  96,  98, 100, 101, 102,
  104, 106, 108, 110, 112, 114, 116, 118,
  122, 124, 126, 128, 130, 132, 134, 136,
  138, 140, 143, 145, 148, 151, 154, 157
};

static const uint16_t kAcTable[128] = {
  4,     5,   6,   7,   8,   9,  10,  11,
  12,   13,  14,  15,  16,  17,  18,  19,
  20,   21,  22,  23,  24,  25,  26,  27,
  28,   29,  30,  31,  32,  33,  34,  35,
  36,   37,  38,  39,  40,  41,  42,  43,
  44,   45,  46,  47,  48,  49,  50,  51,
  52,   53,  54,  55,  56,  57,  58,  60,
  62,   64,  66,  68,  70,  72,  74,  76,
  78,   80,  82,  84,  86,  88,  90,  92,
  94,   96,  98, 100, 102, 104, 106, 108,
  110, 112, 114, 116, 119, 122, 125, 128,
  131, 134, 137, 140, 143, 146, 149, 152,
  155, 158, 161, 164, 167, 170, 173, 177,
  181, 185, 189, 193, 197, 201, 205, 209,
  213, 217, 221, 225, 229, 234, 239, 245,
  249, 254, 259, 264, 269, 274, 279, 284
};

//------------------------------------------------------------------------------
// Paragraph 9.6

void VP8ParseQuant(VP8Decoder* const dec) {
  VP8BitReader* const br = &dec->br_;
  const int base_q0 = VP8GetValue(br, 7);
  const int dqy1_dc = VP8Get(br) ? VP8GetSignedValue(br, 4) : 0;
  const int dqy2_dc = VP8Get(br) ? VP8GetSignedValue(br, 4) : 0;
  const int dqy2_ac = VP8Get(br) ? VP8GetSignedValue(br, 4) : 0;
  const int dquv_dc = VP8Get(br) ? VP8GetSignedValue(br, 4) : 0;
  const int dquv_ac = VP8Get(br) ? VP8GetSignedValue(br, 4) : 0;

  const VP8SegmentHeader* const hdr = &dec->segment_hdr_;
  int i;

  for (i = 0; i < NUM_MB_SEGMENTS; ++i) {
    int q;
    if (hdr->use_segment_) {
      q = hdr->quantizer_[i];
      if (!hdr->absolute_delta_) {
        q += base_q0;
      }
    } else {
      if (i > 0) {
        dec->dqm_[i] = dec->dqm_[0];
        continue;
      } else {
        q = base_q0;
      }
    }
    {
      VP8QuantMatrix* const m = &dec->dqm_[i];
      m->y1_mat_[0] = kDcTable[clip(q + dqy1_dc, 127)];
      m->y1_mat_[1] = kAcTable[clip(q + 0,       127)];

      m->y2_mat_[0] = kDcTable[clip(q + dqy2_dc, 127)] * 2;
      // For all x in [0..284], x*155/100 is bitwise equal to (x*101581) >> 16.
      // The smallest precision for that is '(x*6349) >> 12' but 16 is a good
      // word size.
      m->y2_mat_[1] = (kAcTable[clip(q + dqy2_ac, 127)] * 101581) >> 16;
      if (m->y2_mat_[1] < 8) m->y2_mat_[1] = 8;

      m->uv_mat_[0] = kDcTable[clip(q + dquv_dc, 117)];
      m->uv_mat_[1] = kAcTable[clip(q + dquv_ac, 127)];

      m->uv_quant_ = q + dquv_ac;   // for dithering strength evaluation
    }
  }
}

//------------------------------------------------------------------------------

