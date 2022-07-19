/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "findnearmv.h"

const unsigned char vp8_mbsplit_offset[4][16] = {
    { 0,  8,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0},
    { 0,  2,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0},
    { 0,  2,  8, 10,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0},
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15}
};

/* Predict motion vectors using those from already-decoded nearby blocks.
   Note that we only consider one 4x4 subblock from each candidate 16x16
   macroblock.   */
void vp8_find_near_mvs
(
    MACROBLOCKD *xd,
    const MODE_INFO *here,
    int_mv *nearest,
    int_mv *nearby,
    int_mv *best_mv,
    int cnt[4],
    int refframe,
    int *ref_frame_sign_bias
)
{
    const MODE_INFO *above = here - xd->mode_info_stride;
    const MODE_INFO *left = here - 1;
    const MODE_INFO *aboveleft = above - 1;
    int_mv            near_mvs[4];
    int_mv           *mv = near_mvs;
    int             *cntx = cnt;
    enum {CNT_INTRA, CNT_NEAREST, CNT_NEAR, CNT_SPLITMV};

    /* Zero accumulators */
    mv[0].as_int = mv[1].as_int = mv[2].as_int = 0;
    cnt[0] = cnt[1] = cnt[2] = cnt[3] = 0;

    /* Process above */
    if (above->mbmi.ref_frame != INTRA_FRAME)
    {
        if (above->mbmi.mv.as_int)
        {
            (++mv)->as_int = above->mbmi.mv.as_int;
            mv_bias(ref_frame_sign_bias[above->mbmi.ref_frame], refframe, mv, ref_frame_sign_bias);
            ++cntx;
        }

        *cntx += 2;
    }

    /* Process left */
    if (left->mbmi.ref_frame != INTRA_FRAME)
    {
        if (left->mbmi.mv.as_int)
        {
            int_mv this_mv;

            this_mv.as_int = left->mbmi.mv.as_int;
            mv_bias(ref_frame_sign_bias[left->mbmi.ref_frame], refframe, &this_mv, ref_frame_sign_bias);

            if (this_mv.as_int != mv->as_int)
            {
                (++mv)->as_int = this_mv.as_int;
                ++cntx;
            }

            *cntx += 2;
        }
        else
            cnt[CNT_INTRA] += 2;
    }

    /* Process above left */
    if (aboveleft->mbmi.ref_frame != INTRA_FRAME)
    {
        if (aboveleft->mbmi.mv.as_int)
        {
            int_mv this_mv;

            this_mv.as_int = aboveleft->mbmi.mv.as_int;
            mv_bias(ref_frame_sign_bias[aboveleft->mbmi.ref_frame], refframe, &this_mv, ref_frame_sign_bias);

            if (this_mv.as_int != mv->as_int)
            {
                (++mv)->as_int = this_mv.as_int;
                ++cntx;
            }

            *cntx += 1;
        }
        else
            cnt[CNT_INTRA] += 1;
    }

    /* If we have three distinct MV's ... */
    if (cnt[CNT_SPLITMV])
    {
        /* See if above-left MV can be merged with NEAREST */
        if (mv->as_int == near_mvs[CNT_NEAREST].as_int)
            cnt[CNT_NEAREST] += 1;
    }

    cnt[CNT_SPLITMV] = ((above->mbmi.mode == SPLITMV)
                        + (left->mbmi.mode == SPLITMV)) * 2
                       + (aboveleft->mbmi.mode == SPLITMV);

    /* Swap near and nearest if necessary */
    if (cnt[CNT_NEAR] > cnt[CNT_NEAREST])
    {
        int tmp;
        tmp = cnt[CNT_NEAREST];
        cnt[CNT_NEAREST] = cnt[CNT_NEAR];
        cnt[CNT_NEAR] = tmp;
        tmp = near_mvs[CNT_NEAREST].as_int;
        near_mvs[CNT_NEAREST].as_int = near_mvs[CNT_NEAR].as_int;
        near_mvs[CNT_NEAR].as_int = tmp;
    }

    /* Use near_mvs[0] to store the "best" MV */
    if (cnt[CNT_NEAREST] >= cnt[CNT_INTRA])
        near_mvs[CNT_INTRA] = near_mvs[CNT_NEAREST];

    /* Set up return values */
    best_mv->as_int = near_mvs[0].as_int;
    nearest->as_int = near_mvs[CNT_NEAREST].as_int;
    nearby->as_int = near_mvs[CNT_NEAR].as_int;
}


static void invert_and_clamp_mvs(int_mv *inv, int_mv *src, MACROBLOCKD *xd)
{
    inv->as_mv.row = src->as_mv.row * -1;
    inv->as_mv.col = src->as_mv.col * -1;
    vp8_clamp_mv2(inv, xd);
    vp8_clamp_mv2(src, xd);
}


int vp8_find_near_mvs_bias
(
    MACROBLOCKD *xd,
    const MODE_INFO *here,
    int_mv mode_mv_sb[2][MB_MODE_COUNT],
    int_mv best_mv_sb[2],
    int cnt[4],
    int refframe,
    int *ref_frame_sign_bias
)
{
    int sign_bias = ref_frame_sign_bias[refframe];

    vp8_find_near_mvs(xd,
                      here,
                      &mode_mv_sb[sign_bias][NEARESTMV],
                      &mode_mv_sb[sign_bias][NEARMV],
                      &best_mv_sb[sign_bias],
                      cnt,
                      refframe,
                      ref_frame_sign_bias);

    invert_and_clamp_mvs(&mode_mv_sb[!sign_bias][NEARESTMV],
                         &mode_mv_sb[sign_bias][NEARESTMV], xd);
    invert_and_clamp_mvs(&mode_mv_sb[!sign_bias][NEARMV],
                         &mode_mv_sb[sign_bias][NEARMV], xd);
    invert_and_clamp_mvs(&best_mv_sb[!sign_bias],
                         &best_mv_sb[sign_bias], xd);

    return sign_bias;
}


vp8_prob *vp8_mv_ref_probs(
    vp8_prob p[VP8_MVREFS-1], const int near_mv_ref_ct[4]
)
{
    p[0] = vp8_mode_contexts [near_mv_ref_ct[0]] [0];
    p[1] = vp8_mode_contexts [near_mv_ref_ct[1]] [1];
    p[2] = vp8_mode_contexts [near_mv_ref_ct[2]] [2];
    p[3] = vp8_mode_contexts [near_mv_ref_ct[3]] [3];
    /*p[3] = vp8_mode_contexts [near_mv_ref_ct[1] + near_mv_ref_ct[2] + near_mv_ref_ct[3]] [3];*/
    return p;
}

