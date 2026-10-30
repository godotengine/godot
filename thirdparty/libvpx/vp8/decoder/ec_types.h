/*
 *  Copyright (c) 2011 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP8_DECODER_EC_TYPES_H_
#define VPX_VP8_DECODER_EC_TYPES_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_OVERLAPS 16

/* The area (pixel area in Q6) the block pointed to by bmi overlaps
 * another block with.
 */
typedef struct {
  int overlap;
  union b_mode_info *bmi;
} OVERLAP_NODE;

/* Structure to keep track of overlapping blocks on a block level. */
typedef struct {
  /* TODO(holmer): This array should be exchanged for a linked list */
  OVERLAP_NODE overlaps[MAX_OVERLAPS];
} B_OVERLAP;

/* Structure used to hold all the overlaps of a macroblock. The overlaps of a
 * macroblock is further divided into block overlaps.
 */
typedef struct {
  B_OVERLAP overlaps[16];
} MB_OVERLAP;

/* Structure for keeping track of motion vectors and which reference frame they
 * refer to. Used for motion vector interpolation.
 */
typedef struct {
  MV mv;
  MV_REFERENCE_FRAME ref_frame;
} EC_BLOCK;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_DECODER_EC_TYPES_H_
