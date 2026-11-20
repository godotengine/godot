/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_DECODER_DECODERTHREADING_H_
#define VPX_VP8_DECODER_DECODERTHREADING_H_

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_MULTITHREAD
int vp8mt_decode_mb_rows(VP8D_COMP *pbi, MACROBLOCKD *xd);
void vp8_decoder_remove_threads(VP8D_COMP *pbi);
void vp8_decoder_create_threads(VP8D_COMP *pbi);
void vp8mt_alloc_temp_buffers(VP8D_COMP *pbi, int width, int prev_mb_rows);
void vp8mt_de_alloc_temp_buffers(VP8D_COMP *pbi, int mb_rows);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_DECODER_DECODERTHREADING_H_
