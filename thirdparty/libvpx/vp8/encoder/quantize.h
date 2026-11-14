/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_ENCODER_QUANTIZE_H_
#define VPX_VP8_ENCODER_QUANTIZE_H_

#ifdef __cplusplus
extern "C" {
#endif

struct VP8_COMP;
struct macroblock;
extern void vp8_quantize_mb(struct macroblock *x);
extern void vp8_quantize_mby(struct macroblock *x);
extern void vp8_quantize_mbuv(struct macroblock *x);
extern void vp8_set_quantizer(struct VP8_COMP *cpi, int Q);
extern void vp8cx_frame_init_quantizer(struct VP8_COMP *cpi);
extern void vp8_update_zbin_extra(struct VP8_COMP *cpi, struct macroblock *x);
extern void vp8cx_mb_init_quantizer(struct VP8_COMP *cpi, struct macroblock *x,
                                    int ok_to_skip);
extern void vp8cx_init_quantizer(struct VP8_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_QUANTIZE_H_
