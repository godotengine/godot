/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_mem/vpx_mem.h"

#include "vp9/common/vp9_onyxc_int.h"
#include "vp9/common/vp9_seg_common.h"

const vpx_prob vp9_kf_y_mode_prob[INTRA_MODES][INTRA_MODES][INTRA_MODES - 1] = {
  {  // above = dc
    { 137,  30,  42, 148, 151, 207,  70,  52,  91 },  // left = dc
    {  92,  45, 102, 136, 116, 180,  74,  90, 100 },  // left = v
    {  73,  32,  19, 187, 222, 215,  46,  34, 100 },  // left = h
    {  91,  30,  32, 116, 121, 186,  93,  86,  94 },  // left = d45
    {  72,  35,  36, 149,  68, 206,  68,  63, 105 },  // left = d135
    {  73,  31,  28, 138,  57, 124,  55, 122, 151 },  // left = d117
    {  67,  23,  21, 140, 126, 197,  40,  37, 171 },  // left = d153
    {  86,  27,  28, 128, 154, 212,  45,  43,  53 },  // left = d207
    {  74,  32,  27, 107,  86, 160,  63, 134, 102 },  // left = d63
    {  59,  67,  44, 140, 161, 202,  78,  67, 119 }   // left = tm
  }, {  // above = v
    {  63,  36, 126, 146, 123, 158,  60,  90,  96 },  // left = dc
    {  43,  46, 168, 134, 107, 128,  69, 142,  92 },  // left = v
    {  44,  29,  68, 159, 201, 177,  50,  57,  77 },  // left = h
    {  58,  38,  76, 114,  97, 172,  78, 133,  92 },  // left = d45
    {  46,  41,  76, 140,  63, 184,  69, 112,  57 },  // left = d135
    {  38,  32,  85, 140,  46, 112,  54, 151, 133 },  // left = d117
    {  39,  27,  61, 131, 110, 175,  44,  75, 136 },  // left = d153
    {  52,  30,  74, 113, 130, 175,  51,  64,  58 },  // left = d207
    {  47,  35,  80, 100,  74, 143,  64, 163,  74 },  // left = d63
    {  36,  61, 116, 114, 128, 162,  80, 125,  82 }   // left = tm
  }, {  // above = h
    {  82,  26,  26, 171, 208, 204,  44,  32, 105 },  // left = dc
    {  55,  44,  68, 166, 179, 192,  57,  57, 108 },  // left = v
    {  42,  26,  11, 199, 241, 228,  23,  15,  85 },  // left = h
    {  68,  42,  19, 131, 160, 199,  55,  52,  83 },  // left = d45
    {  58,  50,  25, 139, 115, 232,  39,  52, 118 },  // left = d135
    {  50,  35,  33, 153, 104, 162,  64,  59, 131 },  // left = d117
    {  44,  24,  16, 150, 177, 202,  33,  19, 156 },  // left = d153
    {  55,  27,  12, 153, 203, 218,  26,  27,  49 },  // left = d207
    {  53,  49,  21, 110, 116, 168,  59,  80,  76 },  // left = d63
    {  38,  72,  19, 168, 203, 212,  50,  50, 107 }   // left = tm
  }, {  // above = d45
    { 103,  26,  36, 129, 132, 201,  83,  80,  93 },  // left = dc
    {  59,  38,  83, 112, 103, 162,  98, 136,  90 },  // left = v
    {  62,  30,  23, 158, 200, 207,  59,  57,  50 },  // left = h
    {  67,  30,  29,  84,  86, 191, 102,  91,  59 },  // left = d45
    {  60,  32,  33, 112,  71, 220,  64,  89, 104 },  // left = d135
    {  53,  26,  34, 130,  56, 149,  84, 120, 103 },  // left = d117
    {  53,  21,  23, 133, 109, 210,  56,  77, 172 },  // left = d153
    {  77,  19,  29, 112, 142, 228,  55,  66,  36 },  // left = d207
    {  61,  29,  29,  93,  97, 165,  83, 175, 162 },  // left = d63
    {  47,  47,  43, 114, 137, 181, 100,  99,  95 }   // left = tm
  }, {  // above = d135
    {  69,  23,  29, 128,  83, 199,  46,  44, 101 },  // left = dc
    {  53,  40,  55, 139,  69, 183,  61,  80, 110 },  // left = v
    {  40,  29,  19, 161, 180, 207,  43,  24,  91 },  // left = h
    {  60,  34,  19, 105,  61, 198,  53,  64,  89 },  // left = d45
    {  52,  31,  22, 158,  40, 209,  58,  62,  89 },  // left = d135
    {  44,  31,  29, 147,  46, 158,  56, 102, 198 },  // left = d117
    {  35,  19,  12, 135,  87, 209,  41,  45, 167 },  // left = d153
    {  55,  25,  21, 118,  95, 215,  38,  39,  66 },  // left = d207
    {  51,  38,  25, 113,  58, 164,  70,  93,  97 },  // left = d63
    {  47,  54,  34, 146, 108, 203,  72, 103, 151 }   // left = tm
  }, {  // above = d117
    {  64,  19,  37, 156,  66, 138,  49,  95, 133 },  // left = dc
    {  46,  27,  80, 150,  55, 124,  55, 121, 135 },  // left = v
    {  36,  23,  27, 165, 149, 166,  54,  64, 118 },  // left = h
    {  53,  21,  36, 131,  63, 163,  60, 109,  81 },  // left = d45
    {  40,  26,  35, 154,  40, 185,  51,  97, 123 },  // left = d135
    {  35,  19,  34, 179,  19,  97,  48, 129, 124 },  // left = d117
    {  36,  20,  26, 136,  62, 164,  33,  77, 154 },  // left = d153
    {  45,  18,  32, 130,  90, 157,  40,  79,  91 },  // left = d207
    {  45,  26,  28, 129,  45, 129,  49, 147, 123 },  // left = d63
    {  38,  44,  51, 136,  74, 162,  57,  97, 121 }   // left = tm
  }, {  // above = d153
    {  75,  17,  22, 136, 138, 185,  32,  34, 166 },  // left = dc
    {  56,  39,  58, 133, 117, 173,  48,  53, 187 },  // left = v
    {  35,  21,  12, 161, 212, 207,  20,  23, 145 },  // left = h
    {  56,  29,  19, 117, 109, 181,  55,  68, 112 },  // left = d45
    {  47,  29,  17, 153,  64, 220,  59,  51, 114 },  // left = d135
    {  46,  16,  24, 136,  76, 147,  41,  64, 172 },  // left = d117
    {  34,  17,  11, 108, 152, 187,  13,  15, 209 },  // left = d153
    {  51,  24,  14, 115, 133, 209,  32,  26, 104 },  // left = d207
    {  55,  30,  18, 122,  79, 179,  44,  88, 116 },  // left = d63
    {  37,  49,  25, 129, 168, 164,  41,  54, 148 }   // left = tm
  }, {  // above = d207
    {  82,  22,  32, 127, 143, 213,  39,  41,  70 },  // left = dc
    {  62,  44,  61, 123, 105, 189,  48,  57,  64 },  // left = v
    {  47,  25,  17, 175, 222, 220,  24,  30,  86 },  // left = h
    {  68,  36,  17, 106, 102, 206,  59,  74,  74 },  // left = d45
    {  57,  39,  23, 151,  68, 216,  55,  63,  58 },  // left = d135
    {  49,  30,  35, 141,  70, 168,  82,  40, 115 },  // left = d117
    {  51,  25,  15, 136, 129, 202,  38,  35, 139 },  // left = d153
    {  68,  26,  16, 111, 141, 215,  29,  28,  28 },  // left = d207
    {  59,  39,  19, 114,  75, 180,  77, 104,  42 },  // left = d63
    {  40,  61,  26, 126, 152, 206,  61,  59,  93 }   // left = tm
  }, {  // above = d63
    {  78,  23,  39, 111, 117, 170,  74, 124,  94 },  // left = dc
    {  48,  34,  86, 101,  92, 146,  78, 179, 134 },  // left = v
    {  47,  22,  24, 138, 187, 178,  68,  69,  59 },  // left = h
    {  56,  25,  33, 105, 112, 187,  95, 177, 129 },  // left = d45
    {  48,  31,  27, 114,  63, 183,  82, 116,  56 },  // left = d135
    {  43,  28,  37, 121,  63, 123,  61, 192, 169 },  // left = d117
    {  42,  17,  24, 109,  97, 177,  56,  76, 122 },  // left = d153
    {  58,  18,  28, 105, 139, 182,  70,  92,  63 },  // left = d207
    {  46,  23,  32,  74,  86, 150,  67, 183,  88 },  // left = d63
    {  36,  38,  48,  92, 122, 165,  88, 137,  91 }   // left = tm
  }, {  // above = tm
    {  65,  70,  60, 155, 159, 199,  61,  60,  81 },  // left = dc
    {  44,  78, 115, 132, 119, 173,  71, 112,  93 },  // left = v
    {  39,  38,  21, 184, 227, 206,  42,  32,  64 },  // left = h
    {  58,  47,  36, 124, 137, 193,  80,  82,  78 },  // left = d45
    {  49,  50,  35, 144,  95, 205,  63,  78,  59 },  // left = d135
    {  41,  53,  52, 148,  71, 142,  65, 128,  51 },  // left = d117
    {  40,  36,  28, 143, 143, 202,  40,  55, 137 },  // left = d153
    {  52,  34,  29, 129, 183, 227,  42,  35,  43 },  // left = d207
    {  42,  44,  44, 104, 105, 164,  64, 130,  80 },  // left = d63
    {  43,  81,  53, 140, 169, 204,  68,  84,  72 }   // left = tm
  }
};

const vpx_prob vp9_kf_uv_mode_prob[INTRA_MODES][INTRA_MODES - 1] = {
  { 144,  11,  54, 157, 195, 130,  46,  58, 108 },  // y = dc
  { 118,  15, 123, 148, 131, 101,  44,  93, 131 },  // y = v
  { 113,  12,  23, 188, 226, 142,  26,  32, 125 },  // y = h
  { 120,  11,  50, 123, 163, 135,  64,  77, 103 },  // y = d45
  { 113,   9,  36, 155, 111, 157,  32,  44, 161 },  // y = d135
  { 116,   9,  55, 176,  76,  96,  37,  61, 149 },  // y = d117
  { 115,   9,  28, 141, 161, 167,  21,  25, 193 },  // y = d153
  { 120,  12,  32, 145, 195, 142,  32,  38,  86 },  // y = d207
  { 116,  12,  64, 120, 140, 125,  49, 115, 121 },  // y = d63
  { 102,  19,  66, 162, 182, 122,  35,  59, 128 }   // y = tm
};

static const vpx_prob default_if_y_probs[BLOCK_SIZE_GROUPS][INTRA_MODES - 1] = {
  {  65,  32,  18, 144, 162, 194,  41,  51,  98 },  // block_size < 8x8
  { 132,  68,  18, 165, 217, 196,  45,  40,  78 },  // block_size < 16x16
  { 173,  80,  19, 176, 240, 193,  64,  35,  46 },  // block_size < 32x32
  { 221, 135,  38, 194, 248, 121,  96,  85,  29 }   // block_size >= 32x32
};

static const vpx_prob default_if_uv_probs[INTRA_MODES][INTRA_MODES - 1] = {
  { 120,   7,  76, 176, 208, 126,  28,  54, 103 },  // y = dc
  {  48,  12, 154, 155, 139,  90,  34, 117, 119 },  // y = v
  {  67,   6,  25, 204, 243, 158,  13,  21,  96 },  // y = h
  {  97,   5,  44, 131, 176, 139,  48,  68,  97 },  // y = d45
  {  83,   5,  42, 156, 111, 152,  26,  49, 152 },  // y = d135
  {  80,   5,  58, 178,  74,  83,  33,  62, 145 },  // y = d117
  {  86,   5,  32, 154, 192, 168,  14,  22, 163 },  // y = d153
  {  85,   5,  32, 156, 216, 148,  19,  29,  73 },  // y = d207
  {  77,   7,  64, 116, 132, 122,  37, 126, 120 },  // y = d63
  { 101,  21, 107, 181, 192, 103,  19,  67, 125 }   // y = tm
};

const vpx_prob vp9_kf_partition_probs[PARTITION_CONTEXTS]
                                     [PARTITION_TYPES - 1] = {
  // 8x8 -> 4x4
  { 158,  97,  94 },  // a/l both not split
  {  93,  24,  99 },  // a split, l not split
  {  85, 119,  44 },  // l split, a not split
  {  62,  59,  67 },  // a/l both split
  // 16x16 -> 8x8
  { 149,  53,  53 },  // a/l both not split
  {  94,  20,  48 },  // a split, l not split
  {  83,  53,  24 },  // l split, a not split
  {  52,  18,  18 },  // a/l both split
  // 32x32 -> 16x16
  { 150,  40,  39 },  // a/l both not split
  {  78,  12,  26 },  // a split, l not split
  {  67,  33,  11 },  // l split, a not split
  {  24,   7,   5 },  // a/l both split
  // 64x64 -> 32x32
  { 174,  35,  49 },  // a/l both not split
  {  68,  11,  27 },  // a split, l not split
  {  57,  15,   9 },  // l split, a not split
  {  12,   3,   3 },  // a/l both split
};

static const vpx_prob default_partition_probs[PARTITION_CONTEXTS]
                                             [PARTITION_TYPES - 1] = {
  // 8x8 -> 4x4
  { 199, 122, 141 },  // a/l both not split
  { 147,  63, 159 },  // a split, l not split
  { 148, 133, 118 },  // l split, a not split
  { 121, 104, 114 },  // a/l both split
  // 16x16 -> 8x8
  { 174,  73,  87 },  // a/l both not split
  {  92,  41,  83 },  // a split, l not split
  {  82,  99,  50 },  // l split, a not split
  {  53,  39,  39 },  // a/l both split
  // 32x32 -> 16x16
  { 177,  58,  59 },  // a/l both not split
  {  68,  26,  63 },  // a split, l not split
  {  52,  79,  25 },  // l split, a not split
  {  17,  14,  12 },  // a/l both split
  // 64x64 -> 32x32
  { 222,  34,  30 },  // a/l both not split
  {  72,  16,  44 },  // a split, l not split
  {  58,  32,  12 },  // l split, a not split
  {  10,   7,   6 },  // a/l both split
};

static const vpx_prob default_inter_mode_probs[INTER_MODE_CONTEXTS]
                                              [INTER_MODES - 1] = {
  {2,       173,   34},  // 0 = both zero mv
  {7,       145,   85},  // 1 = one zero mv + one a predicted mv
  {7,       166,   63},  // 2 = two predicted mvs
  {7,       94,    66},  // 3 = one predicted/zero and one new mv
  {8,       64,    46},  // 4 = two new mvs
  {17,      81,    31},  // 5 = one intra neighbour + x
  {25,      29,    30},  // 6 = two intra neighbours
};

/* Array indices are identical to previously-existing INTRAMODECONTEXTNODES. */
const vpx_tree_index vp9_intra_mode_tree[TREE_SIZE(INTRA_MODES)] = {
  -DC_PRED, 2,                      /* 0 = DC_NODE */
  -TM_PRED, 4,                      /* 1 = TM_NODE */
  -V_PRED, 6,                       /* 2 = V_NODE */
  8, 12,                            /* 3 = COM_NODE */
  -H_PRED, 10,                      /* 4 = H_NODE */
  -D135_PRED, -D117_PRED,           /* 5 = D135_NODE */
  -D45_PRED, 14,                    /* 6 = D45_NODE */
  -D63_PRED, 16,                    /* 7 = D63_NODE */
  -D153_PRED, -D207_PRED             /* 8 = D153_NODE */
};

const vpx_tree_index vp9_inter_mode_tree[TREE_SIZE(INTER_MODES)] = {
  -INTER_OFFSET(ZEROMV), 2,
  -INTER_OFFSET(NEARESTMV), 4,
  -INTER_OFFSET(NEARMV), -INTER_OFFSET(NEWMV)
};

const vpx_tree_index vp9_partition_tree[TREE_SIZE(PARTITION_TYPES)] = {
  -PARTITION_NONE, 2,
  -PARTITION_HORZ, 4,
  -PARTITION_VERT, -PARTITION_SPLIT
};

static const vpx_prob default_intra_inter_p[INTRA_INTER_CONTEXTS] = {
  9, 102, 187, 225
};

static const vpx_prob default_comp_inter_p[COMP_INTER_CONTEXTS] = {
  239, 183, 119,  96,  41
};

static const vpx_prob default_comp_ref_p[REF_CONTEXTS] = {
  50, 126, 123, 221, 226
};

static const vpx_prob default_single_ref_p[REF_CONTEXTS][2] = {
  {  33,  16 },
  {  77,  74 },
  { 142, 142 },
  { 172, 170 },
  { 238, 247 }
};

static const struct tx_probs default_tx_probs = {
  { { 3, 136, 37 },
    { 5, 52,  13 } },

  { { 20, 152 },
    { 15, 101 } },

  { { 100 },
    { 66  } }
};

void tx_counts_to_branch_counts_32x32(const unsigned int *tx_count_32x32p,
                                      unsigned int (*ct_32x32p)[2]) {
  ct_32x32p[0][0] = tx_count_32x32p[TX_4X4];
  ct_32x32p[0][1] = tx_count_32x32p[TX_8X8] +
                    tx_count_32x32p[TX_16X16] +
                    tx_count_32x32p[TX_32X32];
  ct_32x32p[1][0] = tx_count_32x32p[TX_8X8];
  ct_32x32p[1][1] = tx_count_32x32p[TX_16X16] +
                    tx_count_32x32p[TX_32X32];
  ct_32x32p[2][0] = tx_count_32x32p[TX_16X16];
  ct_32x32p[2][1] = tx_count_32x32p[TX_32X32];
}

void tx_counts_to_branch_counts_16x16(const unsigned int *tx_count_16x16p,
                                      unsigned int (*ct_16x16p)[2]) {
  ct_16x16p[0][0] = tx_count_16x16p[TX_4X4];
  ct_16x16p[0][1] = tx_count_16x16p[TX_8X8] + tx_count_16x16p[TX_16X16];
  ct_16x16p[1][0] = tx_count_16x16p[TX_8X8];
  ct_16x16p[1][1] = tx_count_16x16p[TX_16X16];
}

void tx_counts_to_branch_counts_8x8(const unsigned int *tx_count_8x8p,
                                    unsigned int (*ct_8x8p)[2]) {
  ct_8x8p[0][0] = tx_count_8x8p[TX_4X4];
  ct_8x8p[0][1] = tx_count_8x8p[TX_8X8];
}

static const vpx_prob default_skip_probs[SKIP_CONTEXTS] = {
  192, 128, 64
};

static const vpx_prob default_switchable_interp_prob[SWITCHABLE_FILTER_CONTEXTS]
                                                    [SWITCHABLE_FILTERS - 1] = {
  { 235, 162, },
  { 36, 255, },
  { 34, 3, },
  { 149, 144, },
};

static void init_mode_probs(FRAME_CONTEXT *fc) {
  vp9_copy(fc->uv_mode_prob, default_if_uv_probs);
  vp9_copy(fc->y_mode_prob, default_if_y_probs);
  vp9_copy(fc->switchable_interp_prob, default_switchable_interp_prob);
  vp9_copy(fc->partition_prob, default_partition_probs);
  vp9_copy(fc->intra_inter_prob, default_intra_inter_p);
  vp9_copy(fc->comp_inter_prob, default_comp_inter_p);
  vp9_copy(fc->comp_ref_prob, default_comp_ref_p);
  vp9_copy(fc->single_ref_prob, default_single_ref_p);
  fc->tx_probs = default_tx_probs;
  vp9_copy(fc->skip_probs, default_skip_probs);
  vp9_copy(fc->inter_mode_probs, default_inter_mode_probs);
}

const vpx_tree_index vp9_switchable_interp_tree
                         [TREE_SIZE(SWITCHABLE_FILTERS)] = {
  -EIGHTTAP, 2,
  -EIGHTTAP_SMOOTH, -EIGHTTAP_SHARP
};

void vp9_adapt_mode_probs(VP9_COMMON *cm) {
  int i, j;
  FRAME_CONTEXT *fc = cm->fc;
  const FRAME_CONTEXT *pre_fc = &cm->frame_contexts[cm->frame_context_idx];
  const FRAME_COUNTS *counts = &cm->counts;

  for (i = 0; i < INTRA_INTER_CONTEXTS; i++)
    fc->intra_inter_prob[i] = mode_mv_merge_probs(pre_fc->intra_inter_prob[i],
                                                  counts->intra_inter[i]);
  for (i = 0; i < COMP_INTER_CONTEXTS; i++)
    fc->comp_inter_prob[i] = mode_mv_merge_probs(pre_fc->comp_inter_prob[i],
                                                 counts->comp_inter[i]);
  for (i = 0; i < REF_CONTEXTS; i++)
    fc->comp_ref_prob[i] = mode_mv_merge_probs(pre_fc->comp_ref_prob[i],
                                               counts->comp_ref[i]);
  for (i = 0; i < REF_CONTEXTS; i++)
    for (j = 0; j < 2; j++)
      fc->single_ref_prob[i][j] = mode_mv_merge_probs(
          pre_fc->single_ref_prob[i][j], counts->single_ref[i][j]);

  for (i = 0; i < INTER_MODE_CONTEXTS; i++)
    vpx_tree_merge_probs(vp9_inter_mode_tree, pre_fc->inter_mode_probs[i],
                counts->inter_mode[i], fc->inter_mode_probs[i]);

  for (i = 0; i < BLOCK_SIZE_GROUPS; i++)
    vpx_tree_merge_probs(vp9_intra_mode_tree, pre_fc->y_mode_prob[i],
                counts->y_mode[i], fc->y_mode_prob[i]);

  for (i = 0; i < INTRA_MODES; ++i)
    vpx_tree_merge_probs(vp9_intra_mode_tree, pre_fc->uv_mode_prob[i],
                         counts->uv_mode[i], fc->uv_mode_prob[i]);

  for (i = 0; i < PARTITION_CONTEXTS; i++)
    vpx_tree_merge_probs(vp9_partition_tree, pre_fc->partition_prob[i],
                         counts->partition[i], fc->partition_prob[i]);

  if (cm->interp_filter == SWITCHABLE) {
    for (i = 0; i < SWITCHABLE_FILTER_CONTEXTS; i++)
      vpx_tree_merge_probs(vp9_switchable_interp_tree,
                           pre_fc->switchable_interp_prob[i],
                           counts->switchable_interp[i],
                           fc->switchable_interp_prob[i]);
  }

  if (cm->tx_mode == TX_MODE_SELECT) {
    int j;
    unsigned int branch_ct_8x8p[TX_SIZES - 3][2];
    unsigned int branch_ct_16x16p[TX_SIZES - 2][2];
    unsigned int branch_ct_32x32p[TX_SIZES - 1][2];

    for (i = 0; i < TX_SIZE_CONTEXTS; ++i) {
      tx_counts_to_branch_counts_8x8(counts->tx.p8x8[i], branch_ct_8x8p);
      for (j = 0; j < TX_SIZES - 3; ++j)
        fc->tx_probs.p8x8[i][j] = mode_mv_merge_probs(
            pre_fc->tx_probs.p8x8[i][j], branch_ct_8x8p[j]);

      tx_counts_to_branch_counts_16x16(counts->tx.p16x16[i], branch_ct_16x16p);
      for (j = 0; j < TX_SIZES - 2; ++j)
        fc->tx_probs.p16x16[i][j] = mode_mv_merge_probs(
            pre_fc->tx_probs.p16x16[i][j], branch_ct_16x16p[j]);

      tx_counts_to_branch_counts_32x32(counts->tx.p32x32[i], branch_ct_32x32p);
      for (j = 0; j < TX_SIZES - 1; ++j)
        fc->tx_probs.p32x32[i][j] = mode_mv_merge_probs(
            pre_fc->tx_probs.p32x32[i][j], branch_ct_32x32p[j]);
    }
  }

  for (i = 0; i < SKIP_CONTEXTS; ++i)
    fc->skip_probs[i] = mode_mv_merge_probs(
        pre_fc->skip_probs[i], counts->skip[i]);
}

static void set_default_lf_deltas(struct loopfilter *lf) {
  lf->mode_ref_delta_enabled = 1;
  lf->mode_ref_delta_update = 1;

  lf->ref_deltas[INTRA_FRAME] = 1;
  lf->ref_deltas[LAST_FRAME] = 0;
  lf->ref_deltas[GOLDEN_FRAME] = -1;
  lf->ref_deltas[ALTREF_FRAME] = -1;

  lf->mode_deltas[0] = 0;
  lf->mode_deltas[1] = 0;
}

void vp9_setup_past_independence(VP9_COMMON *cm) {
  // Reset the segment feature data to the default stats:
  // Features disabled, 0, with delta coding (Default state).
  struct loopfilter *const lf = &cm->lf;

  int i;
  vp9_clearall_segfeatures(&cm->seg);
  cm->seg.abs_delta = SEGMENT_DELTADATA;

  if (cm->last_frame_seg_map && !cm->frame_parallel_decode)
    memset(cm->last_frame_seg_map, 0, (cm->mi_rows * cm->mi_cols));

  if (cm->current_frame_seg_map)
    memset(cm->current_frame_seg_map, 0, (cm->mi_rows * cm->mi_cols));

  // Reset the mode ref deltas for loop filter
  vp9_zero(lf->last_ref_deltas);
  vp9_zero(lf->last_mode_deltas);
  set_default_lf_deltas(lf);

  // To force update of the sharpness
  lf->last_sharpness_level = -1;

  vp9_default_coef_probs(cm);
  init_mode_probs(cm->fc);
  vp9_init_mv_probs(cm);
  cm->fc->initialized = 1;

  if (cm->frame_type == KEY_FRAME ||
      cm->error_resilient_mode || cm->reset_frame_context == 3) {
    // Reset all frame contexts.
    for (i = 0; i < FRAME_CONTEXTS; ++i)
      cm->frame_contexts[i] = *cm->fc;
  } else if (cm->reset_frame_context == 2) {
    // Reset only the frame context specified in the frame header.
    cm->frame_contexts[cm->frame_context_idx] = *cm->fc;
  }

  // prev_mip will only be allocated in encoder.
  if (frame_is_intra_only(cm) && cm->prev_mip && !cm->frame_parallel_decode)
    memset(cm->prev_mip, 0,
           cm->mi_stride * (cm->mi_rows + 1) * sizeof(*cm->prev_mip));

  vp9_zero(cm->ref_frame_sign_bias);

  cm->frame_context_idx = 0;
}
