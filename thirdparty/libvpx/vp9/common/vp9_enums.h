/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VP9_COMMON_VP9_ENUMS_H_
#define VPX_VP9_COMMON_VP9_ENUMS_H_

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MI_SIZE_LOG2 3
#define MI_BLOCK_SIZE_LOG2 (6 - MI_SIZE_LOG2)  // 64 = 2^6

#define MI_SIZE (1 << MI_SIZE_LOG2)              // pixels per mi-unit
#define MI_BLOCK_SIZE (1 << MI_BLOCK_SIZE_LOG2)  // mi-units per max block

#define MI_MASK (MI_BLOCK_SIZE - 1)

// Bitstream profiles indicated by 2-3 bits in the uncompressed header.
// 00: Profile 0.  8-bit 4:2:0 only.
// 10: Profile 1.  8-bit 4:4:4, 4:2:2, and 4:4:0.
// 01: Profile 2.  10-bit and 12-bit color only, with 4:2:0 sampling.
// 110: Profile 3. 10-bit and 12-bit color only, with 4:2:2/4:4:4/4:4:0
//                 sampling.
// 111: Undefined profile.
typedef enum BITSTREAM_PROFILE {
  PROFILE_0,
  PROFILE_1,
  PROFILE_2,
  PROFILE_3,
  MAX_PROFILES
} BITSTREAM_PROFILE;

typedef enum PARSE_RECON_FLAG { PARSE = 1, RECON = 2 } PARSE_RECON_FLAG;

#define BLOCK_4X4 0
#define BLOCK_4X8 1
#define BLOCK_8X4 2
#define BLOCK_8X8 3
#define BLOCK_8X16 4
#define BLOCK_16X8 5
#define BLOCK_16X16 6
#define BLOCK_16X32 7
#define BLOCK_32X16 8
#define BLOCK_32X32 9
#define BLOCK_32X64 10
#define BLOCK_64X32 11
#define BLOCK_64X64 12
#define BLOCK_SIZES 13
#define BLOCK_INVALID BLOCK_SIZES
typedef uint8_t BLOCK_SIZE;

typedef enum PARTITION_TYPE {
  PARTITION_NONE,
  PARTITION_HORZ,
  PARTITION_VERT,
  PARTITION_SPLIT,
  PARTITION_TYPES,
  PARTITION_INVALID = PARTITION_TYPES
} PARTITION_TYPE;

typedef char PARTITION_CONTEXT;
#define PARTITION_PLOFFSET 4  // number of probability models per block size
#define PARTITION_CONTEXTS (4 * PARTITION_PLOFFSET)

// block transform size
typedef uint8_t TX_SIZE;
#define TX_4X4 ((TX_SIZE)0)    // 4x4 transform
#define TX_8X8 ((TX_SIZE)1)    // 8x8 transform
#define TX_16X16 ((TX_SIZE)2)  // 16x16 transform
#define TX_32X32 ((TX_SIZE)3)  // 32x32 transform
#define TX_SIZES ((TX_SIZE)4)

// frame transform mode
typedef enum {
  ONLY_4X4 = 0,        // only 4x4 transform used
  ALLOW_8X8 = 1,       // allow block transform size up to 8x8
  ALLOW_16X16 = 2,     // allow block transform size up to 16x16
  ALLOW_32X32 = 3,     // allow block transform size up to 32x32
  TX_MODE_SELECT = 4,  // transform specified for each block
  TX_MODES = 5,
} TX_MODE;

typedef enum {
  DCT_DCT = 0,    // DCT  in both horizontal and vertical
  ADST_DCT = 1,   // ADST in vertical, DCT in horizontal
  DCT_ADST = 2,   // DCT  in vertical, ADST in horizontal
  ADST_ADST = 3,  // ADST in both directions
  TX_TYPES = 4
} TX_TYPE;

typedef enum {
  VP9_LAST_FLAG = 1 << 0,
  VP9_GOLD_FLAG = 1 << 1,
  VP9_ALT_FLAG = 1 << 2,
} VP9_REFFRAME;

typedef enum { PLANE_TYPE_Y = 0, PLANE_TYPE_UV = 1, PLANE_TYPES } PLANE_TYPE;

#define DC_PRED 0    // Average of above and left pixels
#define V_PRED 1     // Vertical
#define H_PRED 2     // Horizontal
#define D45_PRED 3   // Directional 45  deg = round(arctan(1/1) * 180/pi)
#define D135_PRED 4  // Directional 135 deg = 180 - 45
#define D117_PRED 5  // Directional 117 deg = 180 - 63
#define D153_PRED 6  // Directional 153 deg = 180 - 27
#define D207_PRED 7  // Directional 207 deg = 180 + 27
#define D63_PRED 8   // Directional 63  deg = round(arctan(2/1) * 180/pi)
#define TM_PRED 9    // True-motion
#define NEARESTMV 10
#define NEARMV 11
#define ZEROMV 12
#define NEWMV 13
#define MB_MODE_COUNT 14
typedef uint8_t PREDICTION_MODE;

#define INTRA_MODES (TM_PRED + 1)

#define INTER_MODES (1 + NEWMV - NEARESTMV)

#define SKIP_CONTEXTS 3
#define INTER_MODE_CONTEXTS 7

/* Segment Feature Masks */
#define MAX_MV_REF_CANDIDATES 2

#define INTRA_INTER_CONTEXTS 4
#define COMP_INTER_CONTEXTS 5
#define REF_CONTEXTS 5

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_ENUMS_H_
