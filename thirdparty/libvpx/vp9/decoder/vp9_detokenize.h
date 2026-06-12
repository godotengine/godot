/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_DECODER_VP9_DETOKENIZE_H_
#define VPX_VP9_DECODER_VP9_DETOKENIZE_H_

#include "vpx_dsp/bitreader.h"
#include "vp9/decoder/vp9_decoder.h"
#include "vp9/common/vp9_scan.h"

#ifdef __cplusplus
extern "C" {
#endif

int vp9_decode_block_tokens(TileWorkerData *twd, int plane, const ScanOrder *sc,
                            int x, int y, TX_SIZE tx_size, int seg_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_DECODER_VP9_DETOKENIZE_H_
