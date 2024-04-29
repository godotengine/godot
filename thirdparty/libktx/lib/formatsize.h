/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file
 * @~English
 *
 * @brief Struct for returning size information about an image format.
 *
 * @author Mark Callow, www.edgewise-consulting.com
 */

#ifndef _FORMATSIZE_H_
#define _FORMATSIZE_H_

#include "ktx.h"

typedef enum ktxFormatSizeFlagBits {
    KTX_FORMAT_SIZE_PACKED_BIT                = 0x00000001,
    KTX_FORMAT_SIZE_COMPRESSED_BIT            = 0x00000002,
    KTX_FORMAT_SIZE_PALETTIZED_BIT            = 0x00000004,
    KTX_FORMAT_SIZE_DEPTH_BIT                 = 0x00000008,
    KTX_FORMAT_SIZE_STENCIL_BIT               = 0x00000010,
} ktxFormatSizeFlagBits;

typedef ktx_uint32_t ktxFormatSizeFlags;

/**
 * @brief Structure for holding size information for a texture format.
 */
typedef struct ktxFormatSize {
    ktxFormatSizeFlags  flags;
    unsigned int        paletteSizeInBits;  // For KTX1.
    unsigned int        blockSizeInBits;
    unsigned int        blockWidth;         // in texels
    unsigned int        blockHeight;        // in texels
    unsigned int        blockDepth;         // in texels
    unsigned int        minBlocksX;         // Minimum required number of blocks
    unsigned int        minBlocksY;
} ktxFormatSize;

#ifdef __cplusplus
extern "C" {
#endif

bool ktxFormatSize_initFromDfd(ktxFormatSize* This, ktx_uint32_t* pDfd);

#ifdef __cplusplus
} // extern "C"
#endif

#endif /* _FORMATSIZE_H_ */
