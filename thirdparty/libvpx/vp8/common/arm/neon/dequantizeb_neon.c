/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>

#include "vp8/common/blockd.h"

void vp8_dequantize_b_neon(BLOCKD *d, short *DQC) {
    int16x8x2_t qQ, qDQC, qDQ;

    qQ   = vld2q_s16(d->qcoeff);
    qDQC = vld2q_s16(DQC);

    qDQ.val[0] = vmulq_s16(qQ.val[0], qDQC.val[0]);
    qDQ.val[1] = vmulq_s16(qQ.val[1], qDQC.val[1]);

    vst2q_s16(d->dqcoeff, qDQ);
}
