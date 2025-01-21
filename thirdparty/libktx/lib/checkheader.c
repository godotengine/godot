/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* $Id$ */

/*
 * Copyright 2010-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file checkheader.c
 * @~English
 *
 * @brief Function to verify a KTX file header
 *
 * @author Mark Callow, HI Corporation
 */

/*
 * Author: Georg Kolling, Imagination Technology with modifications
 * by Mark Callow, HI Corporation.
 */
#include <assert.h>
#include <string.h>

#include "ktx.h"
#include "ktxint.h"
#include "vkformat_enum.h"

bool isProhibitedFormat(VkFormat format);
bool isValidFormat(VkFormat format);

/**
 * @internal
 * @~English
 * @brief Check a KTX file header.
 *
 * As well as checking that the header identifies a KTX file, the function
 * sanity checks the values and returns information about the texture in a
 * struct KTX_supplementary_info.
 *
 * @param pHeader   pointer to the KTX header to check
 * @param pSuppInfo pointer to a KTX_supplementary_info structure in which to
 *                  return information about the texture.
 *
 * @author Georg Kolling, Imagination Technology
 * @author Mark Callow, HI Corporation
 */

KTX_error_code  ktxCheckHeader1_(KTX_header* pHeader,
                                 KTX_supplemental_info* pSuppInfo)
{
    ktx_uint8_t identifier_reference[12] = KTX_IDENTIFIER_REF;
    ktx_uint32_t max_dim;

    assert(pHeader != NULL && pSuppInfo != NULL);

    /* Compare identifier, is this a KTX file? */
    if (memcmp(pHeader->identifier, identifier_reference, 12) != 0)
    {
        return KTX_UNKNOWN_FILE_FORMAT;
    }

    if (pHeader->endianness == KTX_ENDIAN_REF_REV)
    {
        /* Convert endianness of pHeader fields. */
        _ktxSwapEndian32(&pHeader->glType, 12);

        if (pHeader->glTypeSize != 1 &&
            pHeader->glTypeSize != 2 &&
            pHeader->glTypeSize != 4)
        {
            /* Only 8-, 16-, and 32-bit types supported so far. */
            return KTX_FILE_DATA_ERROR;
        }
    }
    else if (pHeader->endianness != KTX_ENDIAN_REF)
    {
        return KTX_FILE_DATA_ERROR;
    }

    /* Check glType and glFormat */
    pSuppInfo->compressed = 0;
    if (pHeader->glType == 0 || pHeader->glFormat == 0)
    {
        if (pHeader->glType + pHeader->glFormat != 0)
        {
            /* either both or none of glType, glFormat must be zero */
            return KTX_FILE_DATA_ERROR;
        }
        pSuppInfo->compressed = 1;
    }

    if (pHeader->glFormat == pHeader->glInternalformat) {
        // glInternalFormat is either unsized (which is no longer and should
        // never have been supported by libktx) or glFormat is sized.
        return KTX_FILE_DATA_ERROR;
    }

    /* Check texture dimensions. KTX files can store 8 types of textures:
       1D, 2D, 3D, cube, and array variants of these. There is currently
       no GL extension for 3D array textures. */
    if ((pHeader->pixelWidth == 0) ||
        (pHeader->pixelDepth > 0 && pHeader->pixelHeight == 0))
    {
        /* texture must have width */
        /* texture must have height if it has depth */
        return KTX_FILE_DATA_ERROR;
    }


    if (pHeader->pixelDepth > 0)
    {
        if (pHeader->numberOfArrayElements > 0)
        {
            /* No 3D array textures yet. */
            return KTX_UNSUPPORTED_FEATURE;
        }
        pSuppInfo->textureDimension = 3;
    }
    else if (pHeader->pixelHeight > 0)
    {
        pSuppInfo->textureDimension = 2;
    }
    else
    {
        pSuppInfo->textureDimension = 1;
    }

    if (pHeader->numberOfFaces == 6)
    {
        if (pSuppInfo->textureDimension != 2)
        {
            /* cube map needs 2D faces */
            return KTX_FILE_DATA_ERROR;
        }
    }
    else if (pHeader->numberOfFaces != 1)
    {
        /* numberOfFaces must be either 1 or 6 */
        return KTX_FILE_DATA_ERROR;
    }

    /* Check number of mipmap levels */
    if (pHeader->numberOfMipLevels == 0)
    {
        pSuppInfo->generateMipmaps = 1;
        pHeader->numberOfMipLevels = 1;
    }
    else
    {
        pSuppInfo->generateMipmaps = 0;
    }

    /* This test works for arrays too because height or depth will be 0. */
    max_dim = MAX(MAX(pHeader->pixelWidth, pHeader->pixelHeight), pHeader->pixelDepth);
    if (max_dim < ((ktx_uint32_t)1 << (pHeader->numberOfMipLevels - 1)))
    {
        /* Can't have more mip levels than 1 + log2(max(width, height, depth)) */
        return KTX_FILE_DATA_ERROR;
    }

    return KTX_SUCCESS;
}

/**
 * @internal
 * @~English
 * @brief Check a KTX2 file header.
 *
 * As well as checking that the header identifies a KTX 2 file, the function
 * sanity checks the values and returns information about the texture in a
 * struct KTX_supplementary_info.
 *
 * @param pHeader   pointer to the KTX header to check
 * @param pSuppInfo pointer to a KTX_supplementary_info structure in which to
 *                  return information about the texture.
 *
 * @author Mark Callow, HI Corporation
 */
KTX_error_code ktxCheckHeader2_(KTX_header2* pHeader,
                                KTX_supplemental_info* pSuppInfo)
{
// supp info is compressed, generateMipmaps and num dimensions. Don't need
// compressed as formatSize gives us that. I think the other 2 aren't needed.
    ktx_uint8_t identifier_reference[12] = KTX2_IDENTIFIER_REF;

    assert(pHeader != NULL && pSuppInfo != NULL);
    ktx_uint32_t max_dim;

    /* Compare identifier, is this a KTX file? */
    if (memcmp(pHeader->identifier, identifier_reference, 12) != 0)
    {
        return KTX_UNKNOWN_FILE_FORMAT;
    }

    /* Check format */
    if (isProhibitedFormat(pHeader->vkFormat))
    {
        return KTX_FILE_DATA_ERROR;
    }
    if (!isValidFormat(pHeader->vkFormat))
    {
        return KTX_UNSUPPORTED_FEATURE;
    }
    if (pHeader->supercompressionScheme == KTX_SS_BASIS_LZ && pHeader->vkFormat != VK_FORMAT_UNDEFINED)
    {
        return KTX_FILE_DATA_ERROR;
    }

    /* Check texture dimensions. KTX files can store 8 types of textures:
       1D, 2D, 3D, cube, and array variants of these. There is currently
       no extension for 3D array textures in any 3D API. */
    if ((pHeader->pixelWidth == 0) ||
        (pHeader->pixelDepth > 0 && pHeader->pixelHeight == 0))
    {
        /* texture must have width */
        /* texture must have height if it has depth */
        return KTX_FILE_DATA_ERROR;
    }

    if (pHeader->pixelDepth > 0)
    {
        if (pHeader->layerCount > 0)
        {
            /* No 3D array textures yet. */
            return KTX_UNSUPPORTED_FEATURE;
        }
        pSuppInfo->textureDimension = 3;
    }
    else if (pHeader->pixelHeight > 0)
    {
        pSuppInfo->textureDimension = 2;
    }
    else
    {
        pSuppInfo->textureDimension = 1;
    }

    if (pHeader->faceCount == 6)
    {
        if (pSuppInfo->textureDimension != 2)
        {
            /* cube map needs 2D faces */
            return KTX_FILE_DATA_ERROR;
        }
        if (pHeader->pixelDepth != 0)
        {
            /* cube map cannot have depth */
            return KTX_FILE_DATA_ERROR;
        }
        if (pHeader->pixelWidth != pHeader->pixelHeight)
        {
            /* cube map needs square faces */
            return KTX_FILE_DATA_ERROR;
        }
    }
    else if (pHeader->faceCount != 1)
    {
        /* numberOfFaces must be either 1 or 6 */
        return KTX_FILE_DATA_ERROR;
    }

    // Check number of mipmap levels
    if (pHeader->levelCount == 0)
    {
        pSuppInfo->generateMipmaps = 1;
        pHeader->levelCount = 1;
    }
    else
    {
        pSuppInfo->generateMipmaps = 0;
    }

    // Check supercompression
    switch (pHeader->supercompressionScheme) {
      case KTX_SS_NONE:
      case KTX_SS_BASIS_LZ:
      case KTX_SS_ZSTD:
      case KTX_SS_ZLIB:
        break;
      default:
        // Unsupported supercompression
        return KTX_UNSUPPORTED_FEATURE;
    }

    // This test works for arrays too because height or depth will be 0.
    max_dim = MAX(MAX(pHeader->pixelWidth, pHeader->pixelHeight), pHeader->pixelDepth);
    if (max_dim < ((ktx_uint32_t)1 << (pHeader->levelCount - 1)))
    {
        // Can't have more mip levels than 1 + log2(max(width, height, depth))
        return KTX_FILE_DATA_ERROR;
    }

    return KTX_SUCCESS;

}
