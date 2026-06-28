/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab textwidth=70: */

/*
 * Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file basisu_sgd.h
 * @~English
 *
 * @brief Declare global data for Basis LZ supercompression with ETC1S.
 *
 * These functions are private and should not be used outside the library.
 */

#ifndef _BASIS_SGD_H_
#define _BASIS_SGD_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// This must be the same value as cSliceDescFlagsFrameIsIFrame so we can just
// invert the bit when passing back & forth. As FrameIsIFrame is within
// a C namespace it can't easily be accessed from a c header.
enum bu_image_flags__bits_e { ETC1S_P_FRAME = 0x02 };

typedef uint32_t buFlags;

typedef struct ktxBasisLzGlobalHeader {
    uint16_t endpointCount;
    uint16_t selectorCount;
    uint32_t endpointsByteLength;
    uint32_t selectorsByteLength;
    uint32_t tablesByteLength;
    uint32_t extendedByteLength;
} ktxBasisLzGlobalHeader;

// This header is followed by imageCount "slice" descriptions.

// 1, or 2 slices per image (i.e. layer, face & slice).
// These offsets are relative to start of a mip level as given by the
// main levelIndex.
typedef struct ktxBasisLzEtc1sImageDesc {
    buFlags imageFlags;
    uint32_t rgbSliceByteOffset;
    uint32_t rgbSliceByteLength;
    uint32_t alphaSliceByteOffset;
    uint32_t alphaSliceByteLength;
} ktxBasisLzEtc1sImageDesc;

#define BGD_ETC1S_IMAGE_DESCS(bgd) \
        reinterpret_cast<ktxBasisLzEtc1sImageDesc*>(bgd + sizeof(ktxBasisLzGlobalHeader))

// The are followed in the global data by these ...
//    uint8_t[endpointsByteLength] endpointsData;
//    uint8_t[selectorsByteLength] selectorsData;
//    uint8_t[tablesByteLength] tablesData;

#define BGD_ENDPOINTS_ADDR(bgd, imageCount) \
    (bgd + sizeof(ktxBasisLzGlobalHeader) + sizeof(ktxBasisLzEtc1sImageDesc) * imageCount)

#define BGD_SELECTORS_ADDR(bgd, bgdh, imageCount) (BGD_ENDPOINTS_ADDR(bgd, imageCount) + bgdh.endpointsByteLength)

#define BGD_TABLES_ADDR(bgd, bgdh, imageCount) (BGD_SELECTORS_ADDR(bgd, bgdh, imageCount) + bgdh.selectorsByteLength)

#define BGD_EXTENDED_ADDR(bgd, bgdh, imageCount) (BGD_TABLES_ADDR(bgd, bgdh, imageCount) + bgdh.tablesByteLength)

// Just because this is a convenient place to put it for basis_{en,trans}code.
enum alpha_content_e {
    eNone,
    eAlpha,
    eGreen
};

#ifdef __cplusplus
}
#endif

#endif /* _BASIS_SGD_H_ */
