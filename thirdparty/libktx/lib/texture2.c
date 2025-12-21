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
 * @brief ktxTexture2 implementation. Support for KTX2 format.
 *
 * @author Mark Callow, github.com/MarkCallow
 */

#if defined(_WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <zstd.h>
#include <zstd_errors.h>
#include <KHR/khr_df.h>

#include "dfdutils/dfd.h"
#include "ktx.h"
#include "ktxint.h"
#include "filestream.h"
#include "memstream.h"
#include "texture2.h"
#include "unused.h"

// FIXME: Test this #define and put it in a header somewhere.
//#define IS_BIG_ENDIAN (1 == *(unsigned char *)&(const int){0x01000000ul})
#define IS_BIG_ENDIAN 0

extern uint32_t vkFormatTypeSize(VkFormat format);
extern bool isProhibitedFormat(VkFormat format);

struct ktxTexture_vtbl ktxTexture2_vtbl;
struct ktxTexture_vtblInt ktxTexture2_vtblInt;

#if !defined(BITFIELD_ORDER_FROM_MSB)
// Most compilers, including all those tested so far, including clang, gcc
// and msvc, order bitfields from the lsb so these struct declarations work.
// Could this be because I've only tested on little-endian machines?
// These are preferred as they are much easier to manually initialize
// and verify.
struct sampleType {
    uint32_t bitOffset: 16;
    uint32_t bitLength: 8;
    // MSVC 14.44 introduced a warning when mixing enums of different types.
    // To avoid doing that make separate channelId and qualifier fields.
    uint32_t channelId : 4;
    uint32_t datatypeQualifiers : 4;
    uint32_t samplePosition0 : 8;
    uint32_t samplePosition1: 8;
    uint32_t samplePosition2: 8;
    uint32_t samplePosition3: 8;
    uint32_t lower;
    uint32_t upper;
};

struct BDFD {
    uint32_t vendorId: 17;
    uint32_t descriptorType: 15;
    uint32_t versionNumber: 16;
    uint32_t descriptorBlockSize: 16;
    uint32_t model: 8;
    uint32_t primaries: 8;
    uint32_t transfer: 8;
    uint32_t flags: 8;
    uint32_t texelBlockDimension0: 8;
    uint32_t texelBlockDimension1: 8;
    uint32_t texelBlockDimension2: 8;
    uint32_t texelBlockDimension3: 8;
    uint32_t bytesPlane0: 8;
    uint32_t bytesPlane1: 8;
    uint32_t bytesPlane2: 8;
    uint32_t bytesPlane3: 8;
    uint32_t bytesPlane4: 8;
    uint32_t bytesPlane5: 8;
    uint32_t bytesPlane6: 8;
    uint32_t bytesPlane7: 8;
    struct sampleType samples[6];
};

struct BDFD e5b9g9r9_ufloat_comparator = {
    .vendorId = 0,
    .descriptorType = 0,
    .versionNumber = 2,
    .descriptorBlockSize = sizeof(struct BDFD),
    .model = KHR_DF_MODEL_RGBSDA,
    .primaries = KHR_DF_PRIMARIES_BT709,
    .transfer = KHR_DF_TRANSFER_LINEAR,
    .flags = KHR_DF_FLAG_ALPHA_STRAIGHT,
    .texelBlockDimension0 = 0,
    .texelBlockDimension1 = 0,
    .texelBlockDimension2 = 0,
    .texelBlockDimension3 = 0,
    .bytesPlane0 = 4,
    .bytesPlane1 = 0,
    .bytesPlane2 = 0,
    .bytesPlane3 = 0,
    .bytesPlane4 = 0,
    .bytesPlane5 = 0,
    .bytesPlane6 = 0,
    .bytesPlane7 = 0,
    // gcc likes this way. It does not like, e.g.,
    // .samples[0].bitOffset = 0, etc. which is accepted by both clang & msvc.
    // I find the standards docs impenetrable so I don't know which is correct.
    .samples[0] = {
        .bitOffset = 0,
        .bitLength = 8,
        .channelId = KHR_DF_CHANNEL_RGBSDA_RED,
        .datatypeQualifiers = 0,
        .samplePosition0 = 0,
        .samplePosition1 = 0,
        .samplePosition2 = 0,
        .samplePosition3 = 0,
        .lower = 0,
        .upper = 8448,
    },
    .samples[1] = {
        .bitOffset = 27,
        .bitLength = 4,
        .channelId = KHR_DF_CHANNEL_RGBSDA_RED,
        // The constant is defined to be ORed with a channelId into
        // an 8-bit value. Shift to make it suitable for the 4-bit field.
        .datatypeQualifiers = (KHR_DF_SAMPLE_DATATYPE_EXPONENT >> 4U),
        .samplePosition0 = 0,
        .samplePosition1 = 0,
        .samplePosition2 = 0,
        .samplePosition3 = 0,
        .lower = 15,
        .upper = 31,
    },
    .samples[2] = {
        .bitOffset = 9,
        .bitLength = 8,
        .channelId = KHR_DF_CHANNEL_RGBSDA_GREEN,
        .datatypeQualifiers = 0,
        .samplePosition0 = 0,
        .samplePosition1 = 0,
        .samplePosition2 = 0,
        .samplePosition3 = 0,
        .lower = 0,
        .upper = 8448,
    },
    .samples[3] = {
        .bitOffset = 27,
        .bitLength = 4,
        .channelId = KHR_DF_CHANNEL_RGBSDA_GREEN,
        // Ditto comment in samples[1].
        .datatypeQualifiers = (KHR_DF_SAMPLE_DATATYPE_EXPONENT >> 4U),
        .samplePosition0 = 0,
        .samplePosition1 = 0,
        .samplePosition2 = 0,
        .samplePosition3 = 0,
        .lower = 15,
        .upper = 31,
    },
    .samples[4] = {
        .bitOffset = 18,
        .bitLength = 8,
        .channelId = KHR_DF_CHANNEL_RGBSDA_BLUE,
        .datatypeQualifiers = 0,
        .samplePosition0 = 0,
        .samplePosition1 = 0,
        .samplePosition2 = 0,
        .samplePosition3 = 0,
        .lower = 0,
        .upper = 8448,
    },
    .samples[5] = {
        .bitOffset = 27,
        .bitLength = 4,
        .channelId = KHR_DF_CHANNEL_RGBSDA_BLUE,
        // Ditto comment in samples[1].
        .datatypeQualifiers = (KHR_DF_SAMPLE_DATATYPE_EXPONENT >> 4U),
        .samplePosition0 = 0,
        .samplePosition1 = 0,
        .samplePosition2 = 0,
        .samplePosition3 = 0,
        .lower = 15,
        .upper = 31,
    }
};
#else
// For compilers which order bitfields from the msb rather than lsb.
#define shift(x,val) ((val) << KHR_DF_SHIFT_ ## x)
#define sampleshift(x,val) ((val) << KHR_DF_SAMPLESHIFT_ ## x)
#define e5b9g9r9_bdbwordcount KHR_DFDSIZEWORDS(6)
ktx_uint32_t e5b9g9r9_ufloat_comparator[e5b9g9r9_bdbwordcount] = {
    0,    // descriptorType & vendorId
    shift(DESCRIPTORBLOCKSIZE, e5b9g9r9_bdbwordcount * sizeof(ktx_uint32_t)) | shift(VERSIONNUMBER, 2),
    // N.B. Allow various values of primaries, transfer & flags
    shift(FLAGS, KHR_DF_FLAG_ALPHA_STRAIGHT) | shift(TRANSFER, KHR_DF_TRANSFER_LINEAR) | shift(PRIMARIES, KHR_DF_PRIMARIES_BT709) | shift(MODEL, KHR_DF_MODEL_RGBSDA),
    0,    // texelBlockDimension3~0
    shift(BYTESPLANE0, 4),  // All other bytesPlane fields are 0.
    0,    // bytesPlane7~4
    sampleshift(CHANNELID, KHR_DF_CHANNEL_RGBSDA_RED) | sampleshift(BITLENGTH, 8) | sampleshift(BITOFFSET, 0),
    0,    // samplePosition3~0
    0,    // sampleLower
    8448, // sampleUpper
    sampleshift(CHANNELID, KHR_DF_CHANNEL_RGBSDA_RED | KHR_DF_SAMPLE_DATATYPE_EXPONENT) | sampleshift(BITLENGTH, 4) | sampleshift(BITOFFSET, 27),
    0,    // samplePosition3~0
    15,   // sampleLower
    31,   // sampleUpper
    sampleshift(CHANNELID, KHR_DF_CHANNEL_RGBSDA_GREEN) | sampleshift(BITLENGTH, 8) | sampleshift(BITOFFSET, 9),
    0,    // samplePosition3~0
    0,    // sampleLower
    8448, // sampleUpper
    sampleshift(CHANNELID, KHR_DF_CHANNEL_RGBSDA_GREEN | KHR_DF_SAMPLE_DATATYPE_EXPONENT) | sampleshift(BITLENGTH, 4) | sampleshift(BITOFFSET, 27),
    0,    // samplePosition3~0
    15,   // sampleLower
    31,   // sampleUpper
    sampleshift(CHANNELID, KHR_DF_CHANNEL_RGBSDA_BLUE) | sampleshift(BITLENGTH, 8) | sampleshift(BITOFFSET, 18),
    0,    // samplePosition3~0
    0,    // sampleLower
    8448, // sampleUpper
    sampleshift(CHANNELID, KHR_DF_CHANNEL_RGBSDA_BLUE | KHR_DF_SAMPLE_DATATYPE_EXPONENT) | sampleshift(BITLENGTH, 4) | sampleshift(BITOFFSET, 27),
    0,    // samplePosition3~0
    15,   // sampleLower
    31,   // sampleUpper
};
#endif

/* Helper constant:
   Minimal size of basic descriptor block to safely read its size */
#define KHR_DFD_SIZEFOR_DESCRIPTORBLOCKSIZE \
    ((KHR_DF_WORD_DESCRIPTORBLOCKSIZE + 1) * sizeof(uint32_t))

/**
 * @private
 * @~English
 * @brief Initialize a ktxFormatSize object from the info in a DFD.
 *
 * This is used instead of referring to the DFD directly so code dealing
 * with format info can be common to KTX 1 & 2.
 *
 * @param[in] This   pointer the ktxFormatSize to initialize.
 * @param[in] pDFD   pointer to the DFD whose data to use.
 *
 * @return    KTX_TRUE on success, otherwise KTX_FALSE.
 */
bool
ktxFormatSize_initFromDfd(ktxFormatSize* This, ktx_uint32_t* pDfd)
{
    uint32_t* pBdb = pDfd + 1;
    // pDfd[0] contains totalSize in bytes, check if it has at least
    // KHR_DFD_SIZEFOR_DESCRIPTORBLOCKSIZE bytes
    if (pDfd[0] < KHR_DFD_SIZEFOR_DESCRIPTORBLOCKSIZE || *pBdb != 0) {
        // Either decriptorType or vendorId is not 0
        return false;
    }
    // Iterate through all block descriptors and check if sum of their sizes
    // is equal to the totalSize in pDfd[0]
    uint32_t descriptorSize = pDfd[0] - sizeof(uint32_t);
    while(descriptorSize > KHR_DFD_SIZEFOR_DESCRIPTORBLOCKSIZE) {
        uint32_t descriptorBlockSize = KHR_DFDVAL(pBdb, DESCRIPTORBLOCKSIZE);
        if (descriptorBlockSize <= descriptorSize) {
            descriptorSize -= descriptorBlockSize;
            pBdb += descriptorBlockSize / sizeof(uint32_t);
        } else {
            break;
        }
    }
    if (descriptorSize != 0) {
        return false;
    }

    // reset pBdb pointer to the first block descriptor
    pBdb = pDfd + 1;

    // Check the DFD is of the expected version.
    if (KHR_DFDVAL(pBdb, VERSIONNUMBER) != KHR_DF_VERSIONNUMBER_1_3) {
        return false;
    }

    // DFD has supported type and version. Process it.
    This->blockWidth = KHR_DFDVAL(pBdb, TEXELBLOCKDIMENSION0) + 1;
    This->blockHeight = KHR_DFDVAL(pBdb, TEXELBLOCKDIMENSION1) + 1;
    This->blockDepth = KHR_DFDVAL(pBdb, TEXELBLOCKDIMENSION2) + 1;
    if (KHR_DFDVAL(pBdb, BYTESPLANE0) == 0) {
        // The DFD uses the deprecated way of indicating a supercompressed
        // texture. Reconstruct the original values.
        reconstructDFDBytesPlanesFromSamples(pDfd);
    }
    This->blockSizeInBits = KHR_DFDVAL(pBdb, BYTESPLANE0) * 8;
    // Account for ETC1S with possible second slice.
    This->blockSizeInBits += KHR_DFDVAL(pBdb, BYTESPLANE1) * 8;
    This->paletteSizeInBits = 0; // No paletted formats in ktx v2.
    This->flags = 0;
    This->minBlocksX = This->minBlocksY = 1;
    if (KHR_DFDVAL(pBdb, MODEL) >= KHR_DF_MODEL_DXT1A) {
        // A block compressed format. Entire block is a single sample.
        This->flags |= KTX_FORMAT_SIZE_COMPRESSED_BIT;
        if (KHR_DFDVAL(pBdb, MODEL) == KHR_DF_MODEL_ETC1S) {
            // Special case the only multi-plane format we handle.
            This->blockSizeInBits += KHR_DFDVAL(pBdb, BYTESPLANE1) * 8;
        }
        if (KHR_DFDVAL(pBdb, MODEL) == KHR_DF_MODEL_PVRTC) {
            This->minBlocksX = This->minBlocksY = 2;
        }
    } else {
        // An uncompressed format.

        // Special case depth & depth stencil formats
        if (KHR_DFDSVAL(pBdb, 0, CHANNELID) == KHR_DF_CHANNEL_RGBSDA_DEPTH) {
            if (KHR_DFDSAMPLECOUNT(pBdb) == 1) {
                This->flags |= KTX_FORMAT_SIZE_DEPTH_BIT;
            } else if (KHR_DFDSAMPLECOUNT(pBdb) == 2) {
                This->flags |= KTX_FORMAT_SIZE_STENCIL_BIT;
                This->flags |= KTX_FORMAT_SIZE_DEPTH_BIT;
                This->flags |= KTX_FORMAT_SIZE_PACKED_BIT;
            } else {
                return false;
            }
        } else if (KHR_DFDSVAL(pBdb, 0, CHANNELID) == KHR_DF_CHANNEL_RGBSDA_STENCIL) {
            This->flags |= KTX_FORMAT_SIZE_STENCIL_BIT;
        } else if (KHR_DFDSAMPLECOUNT(pBdb) == 6
#if !defined(BITFIELD_ORDER_FROM_MSB)
                   && !memcmp(((uint32_t*)&e5b9g9r9_ufloat_comparator) + KHR_DF_WORD_TEXELBLOCKDIMENSION0, &pBdb[KHR_DF_WORD_TEXELBLOCKDIMENSION0], sizeof(e5b9g9r9_ufloat_comparator)-(KHR_DF_WORD_TEXELBLOCKDIMENSION0)*sizeof(uint32_t))) {
#else
                   && !memcmp(&e5b9g9r9_ufloat_comparator[KHR_DF_WORD_TEXELBLOCKDIMENSION0], &pBdb[KHR_DF_WORD_TEXELBLOCKDIMENSION0], sizeof(e5b9g9r9_ufloat_comparator)-(KHR_DF_WORD_TEXELBLOCKDIMENSION0)*sizeof(uint32_t))) {
#endif
            // Special case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 as  interpretDFD
            // only handles "simple formats", i.e. where channels are described
            // in contiguous bits.
            This->flags |= KTX_FORMAT_SIZE_PACKED_BIT;
        } else {
            InterpretedDFDChannel rgba[4];
            uint32_t wordBytes;
            enum InterpretDFDResult result;

            result = interpretDFD(pDfd, &rgba[0], &rgba[1], &rgba[2], &rgba[3],
                                  &wordBytes);
            if (result >= i_UNSUPPORTED_ERROR_BIT)
                return false;
            if (result & i_PACKED_FORMAT_BIT)
                This->flags |= KTX_FORMAT_SIZE_PACKED_BIT;
            if (result & i_COMPRESSED_FORMAT_BIT)
                This->flags |= KTX_FORMAT_SIZE_COMPRESSED_BIT;
            if (result & i_YUVSDA_FORMAT_BIT)
                This->flags |= KTX_FORMAT_SIZE_YUVSDA_BIT;
        }
    }
    return true;
}

/**
 * @private
 * @~English
 * @brief Create a DFD for a VkFormat.
 *
 * This KTX-specific function adds support for combined depth stencil formats
 * which are not supported by @e dfdutils' @c vk2dfd function because they
 * are not seen outside a Vulkan device. KTX has its own definitions for
 * these that enable uploading, with some effort.
 *
 * @param[in] vkFormat   the format for which to create a DFD.
 */
static uint32_t*
ktxVk2dfd(ktx_uint32_t vkFormat)
{
    return vk2dfd(vkFormat);
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Do the part of ktxTexture2 construction that is common to
 *        new textures and those constructed from a stream.
 *
 * @param[in] This      pointer to a ktxTexture2-sized block of memory to
 *                      initialize.
 * @param[in] numLevels the number of levels the texture must have.
 *
 * @return    KTX_SUCCESS on success, other KTX_* enum values on error.
 * @exception KTX_OUT_OF_MEMORY Not enough memory for the texture data.
 */
static KTX_error_code
ktxTexture2_constructCommon(ktxTexture2* This, ktx_uint32_t numLevels)
{
    assert(This != NULL);
    ktx_size_t privateSize;

    This->classId = ktxTexture2_c;
    This->vtbl = &ktxTexture2_vtbl;
    This->_protected->_vtbl = ktxTexture2_vtblInt;
    privateSize = sizeof(ktxTexture2_private)
                + sizeof(ktxLevelIndexEntry) * (numLevels - 1);
    This->_private = (ktxTexture2_private*)malloc(privateSize);
    if (This->_private == NULL) {
        return KTX_OUT_OF_MEMORY;
    }
    memset(This->_private, 0, privateSize);
    return KTX_SUCCESS;
}

/*
 * In hindsight this function should have been `#if KTX_FEATURE_WRITE`.
 * In the interest of not breaking an app that may be using this via
 * `ktxTexture2_Create` in `libktx_read` we won't change it.
 */
/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Construct a new, empty, ktxTexture2.
 *
 * @param[in] This       pointer to a ktxTexture2-sized block of memory to
 *                       initialize.
 * @param[in] createInfo pointer to a ktxTextureCreateInfo struct with
 *                       information describing the texture.
 * @param[in] storageAllocation
 *                       enum indicating whether or not to allocate storage
 *                       for the texture images.
 * @return    KTX_SUCCESS on success, other KTX_* enum values on error.
 * @exception KTX_OUT_OF_MEMORY Not enough memory for the texture or image data.
 * @exception KTX_UNSUPPORTED_TEXTURE_TYPE
 *                              The request VkFormat is one of the
 *                              prohibited formats or is otherwise
 *                              unsupported.
 */
static KTX_error_code
ktxTexture2_construct(ktxTexture2* This,
                      const ktxTextureCreateInfo* const createInfo,
                      ktxTextureCreateStorageEnum storageAllocation)
{
    ktxFormatSize formatSize;
    KTX_error_code result;

    memset(This, 0, sizeof(*This));

    if (createInfo->vkFormat != VK_FORMAT_UNDEFINED) {
        if (isProhibitedFormat(createInfo->vkFormat))
            return KTX_UNSUPPORTED_TEXTURE_TYPE;
        This->pDfd = ktxVk2dfd(createInfo->vkFormat);
        if (!This->pDfd)
            return KTX_INVALID_VALUE;  // Format is unknown or unsupported.

#ifdef _DEBUG
        // If this fires, an unsupported format or incorrect DFD
        // has crept into vk2dfd.
        assert(ktxFormatSize_initFromDfd(&formatSize, This->pDfd));
#else
        (void)ktxFormatSize_initFromDfd(&formatSize, This->pDfd);
#endif

    } else {
        // TODO: Validate createInfo->pDfd.
        This->pDfd = (ktx_uint32_t*)malloc(*createInfo->pDfd);
        if (!This->pDfd)
            return KTX_OUT_OF_MEMORY;
        memcpy(This->pDfd, createInfo->pDfd, *createInfo->pDfd);
        if (!ktxFormatSize_initFromDfd(&formatSize, This->pDfd)) {
            result = KTX_UNSUPPORTED_TEXTURE_TYPE;
            goto cleanup;
        }
    }

    result =  ktxTexture_construct(ktxTexture(This), createInfo, &formatSize);

    if (result != KTX_SUCCESS)
        return result;
    result = ktxTexture2_constructCommon(This, createInfo->numLevels);
    if (result != KTX_SUCCESS)
        goto cleanup;;

    This->vkFormat = createInfo->vkFormat;

    // The typeSize cannot be reconstructed just from the DFD as the BDFD
    // does not capture the packing expressed by the [m]PACK[n] layout
    // information in the VkFormat, so we calculate the typeSize directly
    // from the vkFormat
    This->_protected->_typeSize = vkFormatTypeSize(createInfo->vkFormat);

    This->supercompressionScheme = KTX_SS_NONE;

    This->_private->_requiredLevelAlignment
                        = ktxTexture2_calcRequiredLevelAlignment(This);

    // Create levelIndex. Offsets are from start of the KTX2 stream.
    ktxLevelIndexEntry* levelIndex = This->_private->_levelIndex;

    This->_private->_firstLevelFileOffset = 0;

    for (ktx_uint32_t level = 0; level < This->numLevels; level++) {
        levelIndex[level].uncompressedByteLength =
            ktxTexture_calcLevelSize(ktxTexture(This), level,
                                     KTX_FORMAT_VERSION_TWO);
        levelIndex[level].byteLength =
            levelIndex[level].uncompressedByteLength;
        levelIndex[level].byteOffset =
            ktxTexture_calcLevelOffset(ktxTexture(This), level);
    }

    // Allocate storage, if requested.
    if (storageAllocation == KTX_TEXTURE_CREATE_ALLOC_STORAGE) {
        This->dataSize
                = ktxTexture_calcDataSizeTexture(ktxTexture(This));
        This->pData = malloc(This->dataSize);
        if (This->pData == NULL) {
            result = KTX_OUT_OF_MEMORY;
            goto cleanup;
        }
    }
    return result;

cleanup:
    ktxTexture2_destruct(This);
    return result;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Construct a ktxTexture by copying a source ktxTexture.
 *
 * @param[in] This pointer to a ktxTexture2-sized block of memory to
 *                 initialize.
 * @param[in] orig pointer to the source texture to copy.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_OUT_OF_MEMORY Not enough memory for the texture data.
 */
KTX_error_code
ktxTexture2_constructCopy(ktxTexture2* This, ktxTexture2* orig)
{
    KTX_error_code result;

    memcpy(This, orig, sizeof(ktxTexture2));
    // Zero all the pointers to make error handling easier
    This->_protected = NULL;
    This->_private = NULL;
    This->pDfd = NULL;
    This->kvData = NULL;
    This->kvDataHead = NULL;
    This->pData = NULL;

    This->_protected =
                    (ktxTexture_protected*)malloc(sizeof(ktxTexture_protected));
    if (!This->_protected)
        return KTX_OUT_OF_MEMORY;
    // Must come before memcpy of _protected so as to close an active stream.
    if (!orig->pData && ktxTexture_isActiveStream((ktxTexture*)orig))
        ktxTexture2_LoadImageData(orig, NULL, 0);
    memcpy(This->_protected, orig->_protected, sizeof(ktxTexture_protected));

    ktx_size_t privateSize = sizeof(ktxTexture2_private)
                           + sizeof(ktxLevelIndexEntry) * (orig->numLevels - 1);
    This->_private = (ktxTexture2_private*)malloc(privateSize);
    if (This->_private == NULL) {
        result = KTX_OUT_OF_MEMORY;
        goto cleanup;
    }
    memcpy(This->_private, orig->_private, privateSize);
    if (orig->_private->_sgdByteLength > 0) {
        This->_private->_supercompressionGlobalData
                        = (ktx_uint8_t*)malloc(orig->_private->_sgdByteLength);
        if (!This->_private->_supercompressionGlobalData) {
            result = KTX_OUT_OF_MEMORY;
            goto cleanup;
        }
        memcpy(This->_private->_supercompressionGlobalData,
               orig->_private->_supercompressionGlobalData,
               orig->_private->_sgdByteLength);
    }

    This->pDfd = (ktx_uint32_t*)malloc(*orig->pDfd);
    if (!This->pDfd) {
        result = KTX_OUT_OF_MEMORY;
        goto cleanup;
    }
    memcpy(This->pDfd, orig->pDfd, *orig->pDfd);

    if (orig->kvDataHead) {
        ktxHashList_ConstructCopy(&This->kvDataHead, orig->kvDataHead);
    } else if (orig->kvData) {
        This->kvData = (ktx_uint8_t*)malloc(orig->kvDataLen);
        if (!This->kvData) {
            result = KTX_OUT_OF_MEMORY;
            goto cleanup;
        }
        memcpy(This->kvData, orig->kvData, orig->kvDataLen);
    }

    // Can't share the image data as the data pointer is exposed in the
    // ktxTexture2 structure. Changing it to a ref-counted pointer would
    // break code. Maybe that's okay as we're still pre-release. But,
    // since this constructor will be mostly be used when transcoding
    // supercompressed images, it is probably not too big a deal to make
    // a copy of the data.
    This->pData = (ktx_uint8_t*)malloc(This->dataSize);
    if (This->pData == NULL) {
        result = KTX_OUT_OF_MEMORY;
        goto cleanup;
    }
    memcpy(This->pData, orig->pData, orig->dataSize);
    return KTX_SUCCESS;

cleanup:
    if (This->_protected) free(This->_protected);
    if (This->_private) {
        if (This->_private->_supercompressionGlobalData)
            free(This->_private->_supercompressionGlobalData);
        free(This->_private);
    }
    if (This->pDfd) free (This->pDfd);
    if (This->kvDataHead) ktxHashList_Destruct(&This->kvDataHead);

    return result;
}

bool isSrgbFormat(VkFormat format);
bool isNotSrgbFormatButHasSrgbVariant(VkFormat format);

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Construct a ktxTexture from a ktxStream reading from a KTX source.
 *
 * The KTX header, which must have been read prior to calling this, is passed
 * to the function.
 *
 * The stream object is copied into the constructed ktxTexture2.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture.
 *
 * If either KTX_TEXTURE_CREATE_SKIP_KVDATA_BIT or
 * KTX_TEXTURE_CREATE_RAW_KVDATA_BIT is set then the ktxTexture's orientation
 * fields will be set to defaults even if the KTX source contains
 * KTXorientation metadata.
 *
 * @param[in] This pointer to a ktxTexture2-sized block of memory to
 *                 initialize.
 * @param[in] pStream pointer to the stream to read.
 * @param[in] pHeader pointer to a KTX header that has already been read from
 *            the stream.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_DATA_ERROR
 *                              Source data is inconsistent with the KTX
 *                              specification.
 * @exception KTX_FILE_READ_ERROR
 *                              An error occurred while reading the source.
 * @exception KTX_FILE_UNEXPECTED_EOF
 *                              Not enough data in the source.
 * @exception KTX_OUT_OF_MEMORY Not enough memory to load either the images or
 *                              the key-value data.
 * @exception KTX_UNKNOWN_FILE_FORMAT
 *                              The source is not in KTX format.
 * @exception KTX_UNSUPPORTED_TEXTURE_TYPE
 *                              The source describes a texture type not
 *                              supported by OpenGL or Vulkan, e.g, a 3D array.
 */
KTX_error_code
ktxTexture2_constructFromStreamAndHeader(ktxTexture2* This, ktxStream* pStream,
                                        KTX_header2* pHeader,
                                        ktxTextureCreateFlags createFlags)
{
    ktxTexture2_private* private;
    KTX_error_code result;
    KTX_supplemental_info suppInfo;
    ktxStream* stream;
    struct BDFD* pBDFD;
    ktx_size_t levelIndexSize;

    assert(pHeader != NULL && pStream != NULL);

    memset(This, 0, sizeof(*This));
    result = ktxTexture_constructFromStream(ktxTexture(This), pStream,
                                            createFlags);
    if (result != KTX_SUCCESS)
        return result;

    result = ktxCheckHeader2_(pHeader, &suppInfo);
    if (result != KTX_SUCCESS)
        goto cleanup;
    // ktxCheckHeader2_ has done the max(1, levelCount) on pHeader->levelCount.
    result = ktxTexture2_constructCommon(This, pHeader->levelCount);
    if (result != KTX_SUCCESS)
        goto cleanup;
    private = This->_private;

    stream = ktxTexture2_getStream(This);

    /*
     * Initialize from pHeader->info.
     */
    This->vkFormat = pHeader->vkFormat;
    This->supercompressionScheme = pHeader->supercompressionScheme;

    This->_protected->_typeSize = pHeader->typeSize;
    // Can these be done by a ktxTexture_constructFromStream?
    This->numDimensions = suppInfo.textureDimension;
    This->baseWidth = pHeader->pixelWidth;
    assert(suppInfo.textureDimension > 0 && suppInfo.textureDimension < 4);
    switch (suppInfo.textureDimension) {
      case 1:
        This->baseHeight = This->baseDepth = 1;
        break;
      case 2:
        This->baseHeight = pHeader->pixelHeight;
        This->baseDepth = 1;
        break;
      case 3:
        This->baseHeight = pHeader->pixelHeight;
        This->baseDepth = pHeader->pixelDepth;
        break;
    }
    if (pHeader->layerCount > 0) {
        This->numLayers = pHeader->layerCount;
        This->isArray = KTX_TRUE;
    } else {
        This->numLayers = 1;
        This->isArray = KTX_FALSE;
    }
    This->numFaces = pHeader->faceCount;
    if (pHeader->faceCount == 6)
        This->isCubemap = KTX_TRUE;
    else
        This->isCubemap = KTX_FALSE;
    // ktxCheckHeader2_ does the max(1, levelCount) and sets
    // suppInfo.generateMipmaps when it was originally 0.
    This->numLevels = pHeader->levelCount;
    This->generateMipmaps = suppInfo.generateMipmaps;

    // Read level index
    levelIndexSize = sizeof(ktxLevelIndexEntry) * This->numLevels;
    result = stream->read(stream, &private->_levelIndex, levelIndexSize);
    if (result != KTX_SUCCESS)
        goto cleanup;
    // Rebase index to start of data and save file offset.
    private->_firstLevelFileOffset
                    = private->_levelIndex[This->numLevels-1].byteOffset;
    for (ktx_uint32_t level = 0; level < This->numLevels; level++) {
        private->_levelIndex[level].byteOffset
                                        -= private->_firstLevelFileOffset;
        if (This->supercompressionScheme == KTX_SS_NONE &&
            private->_levelIndex[level].byteLength != private->_levelIndex[level].uncompressedByteLength) {
            // For non-supercompressed files the levels must have matching byte lengths
            result = KTX_FILE_DATA_ERROR;
        }
    }
    if (result != KTX_SUCCESS)
        goto cleanup;

    // Read DFD
    if (pHeader->dataFormatDescriptor.byteOffset == 0 || pHeader->dataFormatDescriptor.byteLength < 16) {
        // Missing or too small DFD
        result = KTX_FILE_DATA_ERROR;
        goto cleanup;
    }
    This->pDfd =
            (ktx_uint32_t*)malloc(pHeader->dataFormatDescriptor.byteLength);
    if (!This->pDfd) {
        result = KTX_OUT_OF_MEMORY;
        goto cleanup;
    }
    result = stream->read(stream, This->pDfd,
                          pHeader->dataFormatDescriptor.byteLength);
    if (result != KTX_SUCCESS)
        goto cleanup;

    if (pHeader->dataFormatDescriptor.byteLength != This->pDfd[0]) {
        // DFD byteLength does not match dfdTotalSize
        result = KTX_FILE_DATA_ERROR;
        goto cleanup;
    }
    pBDFD = (struct BDFD*)(This->pDfd + 1);
    if (pBDFD->descriptorBlockSize < 24 || (pBDFD->descriptorBlockSize - 24) % 16 != 0) {
        // BDFD has invalid size
        result = KTX_FILE_DATA_ERROR;
        goto cleanup;
    }
    if (pBDFD->transfer > KHR_DF_TRANSFER_HLG_UNNORMALIZED_OETF) {
          // Invalid transfer function
          result = KTX_FILE_DATA_ERROR;
          goto cleanup;
    }
    // No test for VK_FORMAT_UNDEFINED is needed here because:
    // - any transfer function is allowed when vkFormat is UNDEFINED as with,
    //   e.g., some Basis Universal formats;
    // - the following tests return false for VK_FORMAT_UNDEFINED.
    if (isSrgbFormat(This->vkFormat) && pBDFD->transfer != KHR_DF_TRANSFER_SRGB) {
          // Invalid transfer function
          result = KTX_FILE_DATA_ERROR;
          goto cleanup;
    }
    if (isNotSrgbFormatButHasSrgbVariant(This->vkFormat)
        && pBDFD->transfer == KHR_DF_TRANSFER_SRGB) {
          // Invalid transfer function
          result = KTX_FILE_DATA_ERROR;
          goto cleanup;
    }

    if (!ktxFormatSize_initFromDfd(&This->_protected->_formatSize, This->pDfd)) {
        result = KTX_UNSUPPORTED_TEXTURE_TYPE;
        goto cleanup;
    }
    This->isCompressed = (This->_protected->_formatSize.flags & KTX_FORMAT_SIZE_COMPRESSED_BIT);

    if (This->supercompressionScheme == KTX_SS_BASIS_LZ && pBDFD->model != KHR_DF_MODEL_ETC1S) {
        result = KTX_FILE_DATA_ERROR;
        goto cleanup;
    }

    // Check compatibility with the KHR_texture_basisu glTF extension, if needed.
    if (createFlags & KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT) {
        uint32_t max_dim = MAX(MAX(pHeader->pixelWidth, pHeader->pixelHeight), pHeader->pixelDepth);
        uint32_t full_mip_pyramid_level_count = 1 + (uint32_t)log2(max_dim);
        if (pHeader->levelCount != 1 && pHeader->levelCount != full_mip_pyramid_level_count) {
            // KHR_texture_basisu requires full mip pyramid or single mip level
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }
        if (This->numDimensions != 2 || This->isArray || This->isCubemap) {
            // KHR_texture_basisu requires 2D textures.
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }
        if ((This->baseWidth % 4) != 0 || (This->baseHeight % 4) != 0) {
            // KHR_texture_basisu requires width and height to be a multiple of 4.
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }
        if (pBDFD->model != KHR_DF_MODEL_ETC1S && pBDFD->model != KHR_DF_MODEL_UASTC) {
            // KHR_texture_basisu requires BasisLZ or UASTC
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }
        if (pBDFD->model == KHR_DF_MODEL_UASTC &&
            This->supercompressionScheme != KTX_SS_NONE &&
            This->supercompressionScheme != KTX_SS_ZSTD) {
            // KHR_texture_basisu only allows NONE and ZSTD supercompression for UASTC
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }
    }

    uint32_t sampleCount = KHR_DFDSAMPLECOUNT(This->pDfd + 1);
    if (sampleCount == 0) {
        // Invalid sample count
        result = KTX_FILE_DATA_ERROR;
        goto cleanup;
    }
    if (pBDFD->model == KHR_DF_MODEL_ETC1S || pBDFD->model == KHR_DF_MODEL_UASTC) {
        if (sampleCount < 1 || sampleCount > 2 || (sampleCount == 2 && pBDFD->model == KHR_DF_MODEL_UASTC)) {
            // Invalid sample count
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }
        if (pBDFD->texelBlockDimension0 != 3 || pBDFD->texelBlockDimension1 != 3 ||
            pBDFD->texelBlockDimension2 != 0 || pBDFD->texelBlockDimension3 != 0) {
            // Texel block dimension must be 4x4x1x1 (offset by one)
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }
    }

    This->_private->_requiredLevelAlignment
                          = ktxTexture2_calcRequiredLevelAlignment(This);

    // Make an empty hash list.
    ktxHashList_Construct(&This->kvDataHead);
    // Load KVData.
    if (pHeader->keyValueData.byteLength > 0) {
        uint32_t expectedOffset = pHeader->dataFormatDescriptor.byteOffset + pHeader->dataFormatDescriptor.byteLength;
        expectedOffset = (expectedOffset + 3) & ~0x3; // 4 byte aligned
        if (pHeader->keyValueData.byteOffset != expectedOffset) {
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }
        if (!(createFlags & KTX_TEXTURE_CREATE_SKIP_KVDATA_BIT)) {
            ktx_uint32_t kvdLen = pHeader->keyValueData.byteLength;
            ktx_uint8_t* pKvd;

            pKvd = malloc(kvdLen);
            if (pKvd == NULL) {
                result = KTX_OUT_OF_MEMORY;
                goto cleanup;
            }

            result = stream->read(stream, pKvd, kvdLen);
            if (result != KTX_SUCCESS) {
                free(pKvd);
                goto cleanup;
            }

            if (IS_BIG_ENDIAN) {
                /* Swap the counts inside the key & value data. */
                ktx_uint8_t* src = pKvd;
                ktx_uint8_t* end = pKvd + kvdLen;
                while (src < end) {
                    ktx_uint32_t keyAndValueByteSize = *((ktx_uint32_t*)src);
                    _ktxSwapEndian32(&keyAndValueByteSize, 1);
                    src += _KTX_PAD4(keyAndValueByteSize);
                }
            }

            if (!(createFlags & KTX_TEXTURE_CREATE_RAW_KVDATA_BIT)) {
                char* orientationStr;
                ktx_uint32_t orientationLen;
                ktx_uint32_t animData[3];
                ktx_uint32_t animDataLen;

                result = ktxHashList_Deserialize(&This->kvDataHead,
                                                 kvdLen, pKvd);
                free(pKvd);
                if (result != KTX_SUCCESS) {
                    goto cleanup;
                }

                result = ktxHashList_FindValue(&This->kvDataHead,
                                               KTX_ORIENTATION_KEY,
                                               &orientationLen,
                                               (void**)&orientationStr);
                assert(result != KTX_INVALID_VALUE);
                if (result == KTX_SUCCESS) {
                    // Length includes the terminating NUL.
                    if (orientationLen != This->numDimensions + 1) {
                        // There needs to be an entry for each dimension of
                        // the texture.
                        result = KTX_FILE_DATA_ERROR;
                        goto cleanup;
                    } else {
                        switch (This->numDimensions) {
                          case 3:
                            This->orientation.z = orientationStr[2];
                            FALLTHROUGH;
                          case 2:
                            This->orientation.y = orientationStr[1];
                            FALLTHROUGH;
                          case 1:
                            This->orientation.x = orientationStr[0];
                        }
                    }
                } else {
                    result = KTX_SUCCESS; // Not finding orientation is okay.
                }
                result = ktxHashList_FindValue(&This->kvDataHead,
                                               KTX_ANIMDATA_KEY,
                                               &animDataLen,
                                               (void**)animData);
                assert(result != KTX_INVALID_VALUE);
                if (result == KTX_SUCCESS) {
                    if (animDataLen != sizeof(animData)) {
                        result = KTX_FILE_DATA_ERROR;
                        goto cleanup;
                    }
                    if (This->isArray) {
                        This->isVideo = KTX_TRUE;
                        This->duration = animData[0];
                        This->timescale = animData[1];
                        This->loopcount = animData[2];
                    } else {
                        // animData is only valid for array textures.
                        result = KTX_FILE_DATA_ERROR;
                        goto cleanup;
                    }
                } else {
                    result = KTX_SUCCESS; // Not finding video is okay.
                }
            } else {
                This->kvDataLen = kvdLen;
                This->kvData = pKvd;
            }
        } else {
            stream->skip(stream, pHeader->keyValueData.byteLength);
        }
    } else if (pHeader->keyValueData.byteOffset != 0) {
        // Non-zero KVD byteOffset with zero byteLength
        result = KTX_FILE_DATA_ERROR;
        goto cleanup;
    }

    if (pHeader->supercompressionGlobalData.byteLength > 0) {
        switch (This->supercompressionScheme) {
          case KTX_SS_BASIS_LZ:
            break;
          case KTX_SS_NONE:
          case KTX_SS_ZSTD:
          case KTX_SS_ZLIB:
            // In these cases SGD is not allowed
            result = KTX_FILE_DATA_ERROR;
            break;
          default:
            // We don't support other supercompression schemes
            result = KTX_UNSUPPORTED_FEATURE;
            break;
        }
        if (result != KTX_SUCCESS)
            goto cleanup;

        // There could be padding here so seek to the next item.
        result = stream->setpos(stream,
                             pHeader->supercompressionGlobalData.byteOffset);
        if (result != KTX_SUCCESS)
            goto cleanup;

        // Read supercompressionGlobalData
        private->_supercompressionGlobalData =
          (ktx_uint8_t*)malloc(pHeader->supercompressionGlobalData.byteLength);
        if (!private->_supercompressionGlobalData) {
            result = KTX_OUT_OF_MEMORY;
            goto cleanup;
        }
        private->_sgdByteLength
                            = pHeader->supercompressionGlobalData.byteLength;
        result = stream->read(stream, private->_supercompressionGlobalData,
                              private->_sgdByteLength);

        if (result != KTX_SUCCESS)
            goto cleanup;
    } else if (pHeader->supercompressionGlobalData.byteOffset != 0) {
        // Non-zero SGD byteOffset with zero byteLength
        result = KTX_FILE_DATA_ERROR;
        goto cleanup;
    } else if (This->supercompressionScheme == KTX_SS_BASIS_LZ) {
        // SGD is required for BasisLZ
        result = KTX_FILE_DATA_ERROR;
        goto cleanup;
    }

    // Calculate size of the image data. Level 0 is the last level in the data.
    This->dataSize = private->_levelIndex[0].byteOffset
                     + private->_levelIndex[0].byteLength;

    /*
     * Load the images, if requested.
     */
    if (createFlags & KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT) {
        result = ktxTexture2_LoadImageData(This, NULL, 0);
    }
    if (result != KTX_SUCCESS)
        goto cleanup;

    return result;

cleanup:
    ktxTexture2_destruct(This);
    return result;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Construct a ktxTexture from a ktxStream reading from a KTX source.
 *
 * The stream object is copied into the constructed ktxTexture2.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture.
 *
 * @param[in] This pointer to a ktxTexture2-sized block of memory to
 *            initialize.
 * @param[in] pStream pointer to the stream to read.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return    KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_READ_ERROR
 *                              An error occurred while reading the source.
 *
 * For other exceptions see ktxTexture2_constructFromStreamAndHeader().
 */
static KTX_error_code
ktxTexture2_constructFromStream(ktxTexture2* This, ktxStream* pStream,
                                ktxTextureCreateFlags createFlags)
{
    KTX_header2 header;
    KTX_error_code result;

    // Read header.
    result = pStream->read(pStream, &header, KTX2_HEADER_SIZE);
    if (result != KTX_SUCCESS)
        return result;

#if IS_BIG_ENDIAN
    // byte swap the header
#endif
    return ktxTexture2_constructFromStreamAndHeader(This, pStream,
                                                    &header, createFlags);
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Construct a ktxTexture from a stdio stream reading from a KTX source.
 *
 * See ktxTextureInt_constructFromStream for details.
 *
 * @note Do not close the stdio stream until you are finished with the texture
 *       object.
 *
 * @param[in] This pointer to a ktxTextureInt-sized block of memory to
 *                 initialize.
 * @param[in] stdioStream a stdio FILE pointer opened on the source.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE Either @p stdiostream or @p This is null.
 *
 * For other exceptions, see ktxTexture_constructFromStream().
 */
static KTX_error_code
ktxTexture2_constructFromStdioStream(ktxTexture2* This, FILE* stdioStream,
                                     ktxTextureCreateFlags createFlags)
{
    KTX_error_code result;
    ktxStream stream;

    if (stdioStream == NULL || This == NULL)
        return KTX_INVALID_VALUE;

    result = ktxFileStream_construct(&stream, stdioStream, KTX_FALSE);
    if (result == KTX_SUCCESS)
        result = ktxTexture2_constructFromStream(This, &stream, createFlags);
    return result;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Construct a ktxTexture from a named KTX file.
 *
 * The file name must be encoded in utf-8. On Windows convert unicode names
 * to utf-8 with @c WideCharToMultiByte(CP_UTF8, ...) before calling.
 *
 * See ktxTextureInt_constructFromStream for details.
 *
 * @param[in] This pointer to a ktxTextureInt-sized block of memory to
 *                 initialize.
 * @param[in] filename    pointer to a char array containing the file name.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_OPEN_FAILED The file could not be opened.
 * @exception KTX_INVALID_VALUE @p filename is @c NULL.
 *
 * For other exceptions, see ktxTexture_constructFromStream().
 */
static KTX_error_code
ktxTexture2_constructFromNamedFile(ktxTexture2* This,
                                   const char* const filename,
                                   ktxTextureCreateFlags createFlags)
{
    KTX_error_code result;
    ktxStream stream;
    FILE* file;

    if (This == NULL || filename == NULL)
        return KTX_INVALID_VALUE;

    file = ktxFOpenUTF8(filename, "rb");
    if (!file)
       return KTX_FILE_OPEN_FAILED;

    result = ktxFileStream_construct(&stream, file, KTX_TRUE);
    if (result == KTX_SUCCESS)
        result = ktxTexture2_constructFromStream(This, &stream, createFlags);

    return result;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Construct a ktxTexture from KTX-formatted data in memory.
 *
 * See ktxTextureInt_constructFromStream for details.
 *
 * @param[in] This  pointer to a ktxTextureInt-sized block of memory to
 *                  initialize.
 * @param[in] bytes pointer to the memory containing the serialized KTX data.
 * @param[in] size  length of the KTX data in bytes.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE Either @p bytes is NULL or @p size is 0.
 *
 * For other exceptions, see ktxTexture_constructFromStream().
 */
static KTX_error_code
ktxTexture2_constructFromMemory(ktxTexture2* This,
                                  const ktx_uint8_t* bytes, ktx_size_t size,
                                  ktxTextureCreateFlags createFlags)
{
    KTX_error_code result;
    ktxStream stream;

    if (bytes == NULL || size == 0)
        return KTX_INVALID_VALUE;

    result = ktxMemStream_construct_ro(&stream, bytes, size);
    if (result == KTX_SUCCESS)
        result = ktxTexture2_constructFromStream(This, &stream, createFlags);

    return result;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Destruct a ktxTexture2, freeing and internal memory.
 *
 * @param[in] This pointer to a ktxTexture2-sized block of memory to
 *                 initialize.
 */
void
ktxTexture2_destruct(ktxTexture2* This)
{
    if (This->pDfd) free(This->pDfd);
    if (This->_private) {
      ktx_uint8_t* sgd = This->_private->_supercompressionGlobalData;
      if (sgd) free(sgd);
      free(This->_private);
    }
    ktxTexture_destruct(ktxTexture(This));
}

/*
 * In hindsight this function should have been `#if KTX_FEATURE_WRITE`.
 * In the interest of not breaking an app that may be using this in
 * `libktx_read` we won't change it.
 */
/**
 * @memberof ktxTexture2
 * @ingroup writer
 * @~English
 * @brief Create a new empty ktxTexture2.
 *
 * The address of the newly created ktxTexture2 is written to the location
 * pointed at by @p newTex.
 *
 * @param[in] createInfo pointer to a ktxTextureCreateInfo struct with
 *                       information describing the texture.
 * @param[in] storageAllocation
 *                       enum indicating whether or not to allocate storage
 *                       for the texture images.
 * @param[in,out] newTex pointer to a location in which store the address of
 *                       the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @c glInternalFormat in @p createInfo is not a
 *                              valid OpenGL internal format value.
 * @exception KTX_INVALID_VALUE @c numDimensions in @p createInfo is not 1, 2
 *                              or 3.
 * @exception KTX_INVALID_VALUE One of <tt>base{Width,Height,Depth}</tt> in
 *                              @p createInfo is 0.
 * @exception KTX_INVALID_VALUE @c numFaces in @p createInfo is not 1 or 6.
 * @exception KTX_INVALID_VALUE @c numLevels in @p createInfo is 0.
 * @exception KTX_INVALID_OPERATION
 *                              The <tt>base{Width,Height,Depth}</tt> specified
 *                              in @p createInfo are inconsistent with
 *                              @c numDimensions.
 * @exception KTX_INVALID_OPERATION
 *                              @p createInfo is requesting a 3D array or
 *                              3D cubemap texture.
 * @exception KTX_INVALID_OPERATION
 *                              @p createInfo is requesting a cubemap with
 *                              non-square or non-2D images.
 * @exception KTX_INVALID_OPERATION
 *                              @p createInfo is requesting more mip levels
 *                              than needed for the specified
 *                              <tt>base{Width,Height,Depth}</tt>.
 * @exception KTX_OUT_OF_MEMORY Not enough memory for the texture's images.
 */
KTX_error_code
ktxTexture2_Create(const ktxTextureCreateInfo* const createInfo,
                  ktxTextureCreateStorageEnum storageAllocation,
                  ktxTexture2** newTex)
{
    KTX_error_code result;

    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture2* tex = (ktxTexture2*)malloc(sizeof(ktxTexture2));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture2_construct(tex, createInfo, storageAllocation);
    if (result != KTX_SUCCESS) {
        free(tex);
    } else {
        *newTex = tex;
    }
    return result;
}

/**
 * @memberof ktxTexture2
 * @ingroup writer
 * @~English
 * @brief Create a ktxTexture2 by making a copy of a ktxTexture2.
 *
 * The address of the newly created ktxTexture2 is written to the location
 * pointed at by @p newTex.
 *
 * @param[in]     orig   pointer to the texture to copy.
 * @param[in,out] newTex pointer to a location in which store the address of
 *                       the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_OUT_OF_MEMORY Not enough memory for the texture data.
 */
 KTX_error_code
 ktxTexture2_CreateCopy(ktxTexture2* orig, ktxTexture2** newTex)
 {
    KTX_error_code result;

    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture2* tex = (ktxTexture2*)malloc(sizeof(ktxTexture2));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture2_constructCopy(tex, orig);
    if (result != KTX_SUCCESS) {
        free(tex);
    } else {
        *newTex = tex;
    }
    return result;

 }

/**
 * @defgroup reader Reader
 * @brief Read KTX-formatted data.
 * @{
 */

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Create a ktxTexture2 from a stdio stream reading from a KTX source.
 *
 * The address of a newly created ktxTexture2 reflecting the contents of the
 * stdio stream is written to the location pointed at by @p newTex.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture.
 *
 * @param[in] stdioStream stdio FILE pointer created from the desired file.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 * @param[in,out] newTex  pointer to a location in which store the address of
 *                        the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p newTex is @c NULL.
 * @exception KTX_FILE_DATA_ERROR
 *                              Source data is inconsistent with the KTX
 *                              specification.
 * @exception KTX_FILE_READ_ERROR
 *                              An error occurred while reading the source.
 * @exception KTX_FILE_UNEXPECTED_EOF
 *                              Not enough data in the source.
 * @exception KTX_OUT_OF_MEMORY Not enough memory to create the texture object,
 *                              load the images or load the key-value data.
 * @exception KTX_UNKNOWN_FILE_FORMAT
 *                              The source is not in KTX format.
 * @exception KTX_UNSUPPORTED_TEXTURE_TYPE
 *                              The source describes a texture type not
 *                              supported by OpenGL or Vulkan, e.g, a 3D array.
 */
KTX_error_code
ktxTexture2_CreateFromStdioStream(FILE* stdioStream,
                                  ktxTextureCreateFlags createFlags,
                                  ktxTexture2** newTex)
{
    KTX_error_code result;
    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture2* tex = (ktxTexture2*)malloc(sizeof(ktxTexture2));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture2_constructFromStdioStream(tex, stdioStream,
                                                  createFlags);
    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture2*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Create a ktxTexture2 from a named KTX file.
 *
 * The address of a newly created ktxTexture2 reflecting the contents of the
 * file is written to the location pointed at by @p newTex.
 *
 * The file name must be encoded in utf-8. On Windows convert unicode names
 * to utf-8 with @c WideCharToMultiByte(CP_UTF8, ...) before calling.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture.
 *
 * @param[in] filename    pointer to a char array containing the file name.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 * @param[in,out] newTex  pointer to a location in which store the address of
 *                        the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.

 * @exception KTX_FILE_OPEN_FAILED The file could not be opened.
 * @exception KTX_INVALID_VALUE @p filename is @c NULL.
 *
 * For other exceptions, see ktxTexture2_CreateFromStdioStream().
 */
KTX_error_code
ktxTexture2_CreateFromNamedFile(const char* const filename,
                                ktxTextureCreateFlags createFlags,
                                ktxTexture2** newTex)
{
    KTX_error_code result;

    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture2* tex = (ktxTexture2*)malloc(sizeof(ktxTexture2));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture2_constructFromNamedFile(tex, filename, createFlags);
    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture2*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Create a ktxTexture2 from KTX-formatted data in memory.
 *
 * The address of a newly created ktxTexture2 reflecting the contents of the
 * serialized KTX data is written to the location pointed at by @p newTex.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture.
 *
 * @param[in] bytes pointer to the memory containing the serialized KTX data.
 * @param[in] size  length of the KTX data in bytes.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 * @param[in,out] newTex  pointer to a location in which store the address of
 *                        the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE Either @p bytes is NULL or @p size is 0.
 *
 * For other exceptions, see ktxTexture2_CreateFromStdioStream().
 */
KTX_error_code
ktxTexture2_CreateFromMemory(const ktx_uint8_t* bytes, ktx_size_t size,
                             ktxTextureCreateFlags createFlags,
                             ktxTexture2** newTex)
{
    KTX_error_code result;
    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture2* tex = (ktxTexture2*)malloc(sizeof(ktxTexture2));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture2_constructFromMemory(tex, bytes, size,
                                             createFlags);
    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture2*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Create a ktxTexture2 from KTX-formatted data from a stream.
 *
 * The address of a newly created ktxTexture2 reflecting the contents of the
 * serialized KTX data is written to the location pointed at by @p newTex.
 *
 * The create flag KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT should not be set,
 * if the ktxTexture is ultimately to be uploaded to OpenGL or Vulkan. This
 * will minimize memory usage by allowing, for example, loading the images
 * directly from the source into a Vulkan staging buffer.
 *
 * The create flag KTX_TEXTURE_CREATE_RAW_KVDATA_BIT should not be used. It is
 * provided solely to enable implementation of the @e libktx v1 API on top of
 * ktxTexture.
 *
 * @param[in] stream pointer to the stream to read KTX data from.
 * @param[in] createFlags bitmask requesting specific actions during creation.
 * @param[in,out] newTex  pointer to a location in which store the address of
 *                        the newly created texture.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE Either @p bytes is NULL or @p size is 0.
 *
 * For other exceptions, see ktxTexture2_CreateFromStdioStream().
 */
KTX_error_code
ktxTexture2_CreateFromStream(ktxStream* stream,
                             ktxTextureCreateFlags createFlags,
                             ktxTexture2** newTex)
{
    KTX_error_code result;
    if (newTex == NULL)
        return KTX_INVALID_VALUE;

    ktxTexture2* tex = (ktxTexture2*)malloc(sizeof(ktxTexture2));
    if (tex == NULL)
        return KTX_OUT_OF_MEMORY;

    result = ktxTexture2_constructFromStream(tex, stream, createFlags);
    if (result == KTX_SUCCESS)
        *newTex = (ktxTexture2*)tex;
    else {
        free(tex);
        *newTex = NULL;
    }
    return result;
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Destroy a ktxTexture2 object.
 *
 * This frees the memory associated with the texture contents and the memory
 * of the ktxTexture2 object. This does @e not delete any OpenGL or Vulkan
 * texture objects created by ktxTexture2_GLUpload or ktxTexture2_VkUpload.
 *
 * @param[in] This pointer to the ktxTexture2 object to destroy
 */
void
ktxTexture2_Destroy(ktxTexture2* This)
{
    ktxTexture2_destruct(This);
    free(This);
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Calculate the size of the image data for the specified number
 *        of levels.
 *
 * The data size is the sum of the sizes of each level up to the number
 * specified and includes any @c mipPadding between levels. It does
 * not include initial @c mipPadding required in the file.
 *
 * @param[in] This     pointer to the ktxTexture object of interest.
 * @param[in] levels   number of levels whose data size to return.
 *
 * @return the data size in bytes.
 */
ktx_size_t
ktxTexture2_calcDataSizeLevels(ktxTexture2* This, ktx_uint32_t levels)
{
    ktx_size_t dataSize = 0;

    assert(This != NULL);
    assert(This->supercompressionScheme == KTX_SS_NONE);
    assert(levels <= This->numLevels);
    for (ktx_uint32_t i = levels - 1; i > 0; i--) {
        ktx_size_t levelSize = ktxTexture_calcLevelSize(ktxTexture(This), i,
                                                        KTX_FORMAT_VERSION_TWO);
        dataSize += _KTX_PADN(This->_private->_requiredLevelAlignment,
                              levelSize);
    }
    dataSize += ktxTexture_calcLevelSize(ktxTexture(This), 0,
                                         KTX_FORMAT_VERSION_TWO);
    return dataSize;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 *
 * @copydoc ktxTexture::ktxTexture_doCalcFaceLodSize
 */
ktx_size_t
ktxTexture2_calcFaceLodSize(ktxTexture2* This, ktx_uint32_t level)
{
    assert(This != NULL);
    assert(This->supercompressionScheme == KTX_SS_NONE);
    /*
     * For non-array cubemaps this is the size of a face. For everything
     * else it is the size of the level.
     */
    if (This->isCubemap && !This->isArray)
        return ktxTexture_calcImageSize(ktxTexture(This), level,
                                        KTX_FORMAT_VERSION_TWO);
    else
        return This->_private->_levelIndex[level].uncompressedByteLength;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Return the offset of a level in bytes from the start of the image
 *        data in a ktxTexture.
 *
 * Since the offset is from the start of the image data, it does not include the initial
 * @c mipPadding required in the file.
 *
 * @param[in]     This  pointer to the ktxTexture object of interest.
 * @param[in]     level level whose offset to return.
 *
 * @return the data size in bytes.
 */
ktx_size_t
ktxTexture2_calcLevelOffset(ktxTexture2* This, ktx_uint32_t level)
{
  assert (This != NULL);
  assert(This->supercompressionScheme == KTX_SS_NONE);
  assert (level < This->numLevels);
  ktx_size_t levelOffset = 0;
  for (ktx_uint32_t i = This->numLevels - 1; i > level; i--) {
      ktx_size_t levelSize;
      levelSize = ktxTexture_calcLevelSize(ktxTexture(This), i,
                                           KTX_FORMAT_VERSION_TWO);
      levelOffset += _KTX_PADN(This->_private->_requiredLevelAlignment,
                               levelSize);
  }
  return levelOffset;
}


/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Retrieve the offset of a level's first image within the KTX2 file.
 *
 * @param[in] This pointer to the ktxTexture object of interest.
 */
ktx_uint64_t ktxTexture2_levelFileOffset(ktxTexture2* This, ktx_uint32_t level)
{
    assert(This->_private->_firstLevelFileOffset != 0);
    return This->_private->_levelIndex[level].byteOffset
           + This->_private->_firstLevelFileOffset;
}

// Recursive function to return the greatest common divisor of a and b.
static uint32_t
gcd(uint32_t a, uint32_t b) {
    if (a == 0)
        return b;
    return gcd(b % a, a);
}

// Function to return the least common multiple of a & 4.
uint32_t
lcm4(uint32_t a)
{
    if (!(a & 0x03))
        return a;  // a is a multiple of 4.
    return (a*4) / gcd(a, 4);
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Return the required alignment for levels of this texture.
 *
 * @param[in] This       pointer to the ktxTexture2 object of interest.
 *
 * @return    The required alignment for levels.
 */
 ktx_uint32_t
 ktxTexture2_calcRequiredLevelAlignment(ktxTexture2* This)
 {
    ktx_uint32_t alignment;
    if (This->supercompressionScheme != KTX_SS_NONE)
        alignment = 1;
    else
        alignment = lcm4(This->_protected->_formatSize.blockSizeInBits / 8);
    return alignment;
 }

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Return what the required alignment for levels of this texture will be after inflation.
 *
 * @param[in] This       pointer to the ktxTexture2 object of interest.
 *
 * @return    The required alignment for levels.
 */
ktx_uint32_t
ktxTexture2_calcPostInflationLevelAlignment(ktxTexture2* This)
{
    ktx_uint32_t alignment;

    // Should actually work for none supercompressed but don't want to
    // encourage use of it.
    assert(This->supercompressionScheme != KTX_SS_NONE && This->supercompressionScheme != KTX_SS_BASIS_LZ);

    if (This->vkFormat != VK_FORMAT_UNDEFINED)
        alignment = lcm4(This->_protected->_formatSize.blockSizeInBits / 8);
    else
        alignment = 16;

    return alignment;
}


/**
 * @memberof ktxTexture2
 * @~English
 * @brief Return information about the components of an image in a texture.
 *
 * @param[in]     This           pointer to the ktxTexture object of interest.
 * @param[in,out] pNumComponents pointer to location in which to write the
 *                               number of components in the textures images.
 * @param[in,out] pComponentByteLength
 *                               pointer to the location in which to write
 *                               byte length of a component.
 */
void
ktxTexture2_GetComponentInfo(ktxTexture2* This, uint32_t* pNumComponents,
                             uint32_t* pComponentByteLength)
{
    // FIXME Need to handle packed case.
    getDFDComponentInfoUnpacked(This->pDfd, pNumComponents,
                                pComponentByteLength);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Return the number of components in an image of the texture.
 *
 * Returns the number of components indicated by the DFD's sample information
 * in accordance with the color model. For uncompressed formats it will be the actual
 * number of components in the image. For block-compressed formats, it will be 1 or 2
 * according to the format's DFD color model. For Basis compressed textures, the
 * function examines the ids of the channels indicated by the DFD and uses that
 * information to determine and return the number of components in the image
 * @e before encoding and deflation so it can be used to help choose a suitable
 * transcode target format.
 *
 * @param[in]     This           pointer to the ktxTexture object of interest.
 *
 * @return the number of components.
 */
ktx_uint32_t
ktxTexture2_GetNumComponents(ktxTexture2* This)
{
    uint32_t* pBdb = This->pDfd + 1;
    uint32_t dfdNumComponents = getDFDNumComponents(This->pDfd);
    uint32_t colorModel = KHR_DFDVAL(pBdb, MODEL);
    if (colorModel < KHR_DF_MODEL_DXT1A) {
        return dfdNumComponents;
    } else {
        switch (colorModel) {
          case KHR_DF_MODEL_ETC1S:
          {
            uint32_t channel0Id = KHR_DFDSVAL(pBdb, 0, CHANNELID);
            if (dfdNumComponents == 1) {
                if (channel0Id == KHR_DF_CHANNEL_ETC1S_RGB)
                    return 3;
                else
                    return 1;
            } else {
                uint32_t channel1Id = KHR_DFDSVAL(pBdb, 1, CHANNELID);
                if (channel0Id == KHR_DF_CHANNEL_ETC1S_RGB
                    && channel1Id == KHR_DF_CHANNEL_ETC1S_AAA)
                    return 4;
                else {
                    // An invalid combination of channel Ids should never
                    // have been set during creation or should have been
                    // caught when the file was loaded.
                    assert(channel0Id == KHR_DF_CHANNEL_ETC1S_RRR
                           && channel1Id == KHR_DF_CHANNEL_ETC1S_GGG);
                    return 2;
                }
            }
            break;
          }
          case KHR_DF_MODEL_UASTC:
            switch (KHR_DFDSVAL(pBdb, 0, CHANNELID)) {
              case KHR_DF_CHANNEL_UASTC_RRR:
                return 1;
              case KHR_DF_CHANNEL_UASTC_RRRG:
                return 2;
              case KHR_DF_CHANNEL_UASTC_RGB:
                return 3;
              case KHR_DF_CHANNEL_UASTC_RGBA:
                return 4;
              default:
                // Same comment as for the assert in the ETC1 case.
                assert(false);
                return 1;
            }
            break;
          default:
            return dfdNumComponents;
        }
    }
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Find the offset of an image within a ktxTexture's image data.
 *
 * As there is no such thing as a 3D cubemap we make the 3rd location parameter
 * do double duty. Only works for non-supercompressed textures as
 * there is no way to tell where an image is for a supercompressed one.
 *
 * @param[in]     This      pointer to the ktxTexture object of interest.
 * @param[in]     level     mip level of the image.
 * @param[in]     layer     array layer of the image.
 * @param[in]     faceSlice cube map face or depth slice of the image.
 * @param[in,out] pOffset   pointer to location to store the offset.
 *
 * @return  KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_OPERATION
 *                         @p level, @p layer or @p faceSlice exceed the
 *                         dimensions of the texture.
 * @exception KTX_INVALID_OPERATION Texture is supercompressed.
 * @exception KTX_INVALID_VALID @p This is NULL.
 */
KTX_error_code
ktxTexture2_GetImageOffset(ktxTexture2* This, ktx_uint32_t level,
                          ktx_uint32_t layer, ktx_uint32_t faceSlice,
                          ktx_size_t* pOffset)
{
    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (level >= This->numLevels || layer >= This->numLayers)
        return KTX_INVALID_OPERATION;

    if (This->supercompressionScheme != KTX_SS_NONE)
        return KTX_INVALID_OPERATION;

    if (This->isCubemap) {
        if (faceSlice >= This->numFaces)
            return KTX_INVALID_OPERATION;
    } else {
        ktx_uint32_t maxSlice = MAX(1, This->baseDepth >> level);
        if (faceSlice >= maxSlice)
            return KTX_INVALID_OPERATION;
    }

    // Get the offset of the start of the level.
    *pOffset = ktxTexture2_levelDataOffset(This, level);

    // All layers, faces & slices within a level are the same size.
    if (layer != 0) {
        ktx_size_t layerSize;
        layerSize = ktxTexture_layerSize(ktxTexture(This), level,
                                         KTX_FORMAT_VERSION_TWO);
        *pOffset += layer * layerSize;
    }
    if (faceSlice != 0) {
        ktx_size_t imageSize;
        imageSize = ktxTexture2_GetImageSize(This, level);
        *pOffset += faceSlice * imageSize;
    }
    return KTX_SUCCESS;
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Retrieve the transfer function of the images.
 *
 * @param[in]     This      pointer to the ktxTexture2 object of interest.
 *
 * @return A @c khr_df_transfer enum value specifying the transfer function.
 */
khr_df_transfer_e
ktxTexture2_GetTransferFunction_e(ktxTexture2* This)
{
    return KHR_DFDVAL(This->pDfd+1, TRANSFER);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Retrieve the transfer function of the images.
 * @deprecated Use ktxTexture2\_GetTransferFunction\_e. Now that the KTX
 * specification allows setting of non-linear transfer functions other than
 * sRGB, it is possible for the transfer function to be an EOTF so this
 * name is no longer appropriate.
 *
 * @param[in]     This      pointer to the ktxTexture2 object of interest.
 *
 * @return A @c khr_df_transfer enum value specifying the transfer function.
 */
khr_df_transfer_e
ktxTexture2_GetOETF_e(ktxTexture2* This)
{
    return KHR_DFDVAL(This->pDfd+1, TRANSFER);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Retrieve the transfer function of the images.
 * @deprecated Use ktxTexture2\_GetTransferFunction\_e.
 *
 * @param[in]     This      pointer to the ktxTexture2 object of interest.
 *
 * @return A @c khr_df_transfer enum value specifying the transfer function,
 *         returned as @c ktx_uint32_t.
 */
ktx_uint32_t
ktxTexture2_GetOETF(ktxTexture2* This)
{
    return KHR_DFDVAL(This->pDfd+1, TRANSFER);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Retrieve the DFD color model of the images.
 *
 * @param[in]     This      pointer to the ktxTexture2 object of interest.
 *
 * @return A @c khr_df_transfer enum value specifying the color model.
 */
khr_df_model_e
ktxTexture2_GetColorModel_e(ktxTexture2* This)
{
    return KHR_DFDVAL(This->pDfd+1, MODEL);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Retrieve whether the RGB components have been premultiplied by the alpha component.
 *
 * @param[in]     This      pointer to the ktxTexture2 object of interest.
 *
 * @return KTX\_TRUE if the components are premultiplied, KTX_FALSE otherwise.
 */
ktx_bool_t
ktxTexture2_GetPremultipliedAlpha(ktxTexture2* This)
{
    return KHR_DFDVAL(This->pDfd+1, FLAGS) & KHR_DF_FLAG_ALPHA_PREMULTIPLIED;
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Retrieve the color primaries of the images.
 *
 * @param[in]     This      pointer to the ktxTexture2 object of interest.
 *
 * @return A @c khr_df_primaries enum value specifying the primaries.
 */
khr_df_primaries_e
ktxTexture2_GetPrimaries_e(ktxTexture2* This)
{
    return KHR_DFDVAL(This->pDfd+1, PRIMARIES);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Query if the images are in a transcodable format.
 *
 * @param[in]     This     pointer to the ktxTexture2 object of interest.
 */
ktx_bool_t
ktxTexture2_NeedsTranscoding(ktxTexture2* This)
{
    if (KHR_DFDVAL(This->pDfd + 1, MODEL) == KHR_DF_MODEL_ETC1S)
        return true;
    else if (KHR_DFDVAL(This->pDfd + 1, MODEL) == KHR_DF_MODEL_UASTC)
        return true;
    else
        return false;
}

#if KTX_FEATURE_WRITE
/*
 * @memberof ktxTexture2
 * @ingroup writer
 * @~English
 * @brief Set the transfer function  for the images in a texture.
 *
 * @param[in]     This     pointer to the ktxTexture2
 * @param[in]     tf       enumerator of the transfer function to set
 *
 * @return  KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_OPERATION The transfer function is not valid for the
 *                                  vkFormat of the texture.
 * @exception KTX_INVALID_VALUE The transfer function is not allowed by the
 *                              KTX spec.
 */
ktx_error_code_e
ktxTexture2_SetTransferFunction(ktxTexture2* This, khr_df_transfer_e tf)
{
    if (isSrgbFormat(This->vkFormat) && tf != KHR_DF_TRANSFER_SRGB)
        return KTX_INVALID_OPERATION;

    if (isNotSrgbFormatButHasSrgbVariant(This->vkFormat) && tf == KHR_DF_TRANSFER_SRGB)
        return KTX_INVALID_OPERATION;

    KHR_DFDSETVAL(This->pDfd + 1, TRANSFER, tf);
    return KTX_SUCCESS;
}

/**
 * @memberof ktxTexture2
 * @ingroup writer
 * @~English
 * @brief Set the transfer function for the images in a texture.
 * @deprecated Use ktxTexture2\_SetTransferFunction.
 *
 * @param[in]     This     pointer to the ktxTexture2
 * @param[in]     tf       enumerator of the transfer function to set
 */
ktx_error_code_e
ktxTexture2_SetOETF(ktxTexture2* This, khr_df_transfer_e tf)
{
    return ktxTexture2_SetTransferFunction(This, tf);
}

/**
 * @memberof ktxTexture2
 * @ingroup writer
 * @~English
 * @brief Set the primaries  for the images in a texture.
 *
 * @param[in]     This           pointer to the ktxTexture2
 * @param[in]     primaries      enumerator of the primaries to set
 */
ktx_error_code_e
ktxTexture2_SetPrimaries(ktxTexture2* This, khr_df_primaries_e primaries)
{
    KHR_DFDSETVAL(This->pDfd + 1, PRIMARIES, primaries);
    return KTX_SUCCESS;
}
#endif

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Return the total size in bytes of the uncompressed data of a
 *        ktxTexture2.
 *
 * If supercompressionScheme == @c KTX_SS_NONE or
 * @c KTX_SS_BASIS_LZ, returns the value of @c This->dataSize
 * else if supercompressionScheme == @c KTX_SS_ZSTD or @c KTX_SS_ZLIB, it
 * returns the sum of the uncompressed sizes of each mip level plus space for
 * the level padding. With no supercompression the data size and uncompressed
 * data size are the same. For Basis supercompression the uncompressed size
 * cannot be known until the data is transcoded so the compressed size is
 * returned.
 *
 * @param[in]     This     pointer to the ktxTexture1 object of interest.
 */
ktx_size_t
ktxTexture2_GetDataSizeUncompressed(ktxTexture2* This)
{
    switch (This->supercompressionScheme) {
      case KTX_SS_BASIS_LZ:
      case KTX_SS_NONE:
        return This->dataSize;
      case KTX_SS_ZSTD:
      case KTX_SS_ZLIB:
      {
            ktx_size_t uncompressedSize = 0;
            ktx_uint32_t uncompressedLevelAlignment;
            ktxLevelIndexEntry* levelIndex = This->_private->_levelIndex;

            uncompressedLevelAlignment =
                ktxTexture2_calcPostInflationLevelAlignment(This);

            for (ktx_int32_t level = This->numLevels - 1; level >= 1; level--) {
                ktx_size_t uncompressedLevelSize;
                uncompressedLevelSize = levelIndex[level].uncompressedByteLength;
                uncompressedLevelSize = _KTX_PADN(uncompressedLevelAlignment,
                                                  uncompressedLevelSize);
                uncompressedSize += uncompressedLevelSize;
            }
            uncompressedSize += levelIndex[0].uncompressedByteLength;
            return uncompressedSize;
      }
      case KTX_SS_BEGIN_VENDOR_RANGE:
      case KTX_SS_END_VENDOR_RANGE:
      case KTX_SS_BEGIN_RESERVED:
      default:
        return 0;
    }
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Calculate & return the size in bytes of an image at the specified
 *        mip level.
 *
 * For arrays, this is the size of a layer, for cubemaps, the size of a face
 * and for 3D textures, the size of a depth slice.
 *
 * The size reflects the padding of each row to KTX_GL_UNPACK_ALIGNMENT.
 *
 * @param[in]     This     pointer to the ktxTexture2 object of interest.
 * @param[in]     level    level of interest. *
 */
ktx_size_t
ktxTexture2_GetImageSize(ktxTexture2* This, ktx_uint32_t level)
{
    return ktxTexture_calcImageSize(ktxTexture(This), level,
                                    KTX_FORMAT_VERSION_TWO);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Calculate & return the size in bytes of all the  images in the specified
 *        mip level.
 *
 * For arrays, this is the size of all layers in the level, for cubemaps, the size of all
 * faces in the level and for 3D textures, the size of all depth slices in the level.
 *
 * @param[in]     This     pointer to the ktxTexture2 object of interest.
 * @param[in]     level    level of interest. *
 */
ktx_size_t
ktxTexture2_GetLevelSize(ktxTexture2* This, ktx_uint32_t level)
{
    return ktxTexture_calcLevelSize(ktxTexture(This), level,
                                    KTX_FORMAT_VERSION_TWO);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Iterate over the mip levels in a ktxTexture2 object.
 *
 * This is almost identical to ktxTexture_IterateLevelFaces(). The difference is
 * that the blocks of image data for non-array cube maps include all faces of
 * a mip level.
 *
 * This function works even if @p This->pData == 0 so it can be used to
 * obtain offsets and sizes for each level by callers who have loaded the data
 * externally.
 *
 * Intended for use only when supercompressionScheme == SUPERCOMPRESSION_NONE.
 *
 * @param[in]     This     handle of the ktxTexture opened on the data.
 * @param[in,out] iterCb   the address of a callback function which is called
 *                         with the data for each image block.
 * @param[in,out] userdata the address of application-specific data which is
 *                         passed to the callback along with the image data.
 *
 * @return  KTX_SUCCESS on success, other KTX_* enum values on error. The
 *          following are returned directly by this function. @p iterCb may
 *          return these for other causes or may return additional errors.
 *
 * @exception KTX_FILE_DATA_ERROR   Mip level sizes are increasing not
 *                                  decreasing
 * @exception KTX_INVALID_OPERATION supercompressionScheme != SUPERCOMPRESSION_NONE.
 * @exception KTX_INVALID_VALUE     @p This is @c NULL or @p iterCb is @c NULL.
 *
 */
KTX_error_code
ktxTexture2_IterateLevels(ktxTexture2* This, PFNKTXITERCB iterCb, void* userdata)
{
    KTX_error_code  result = KTX_SUCCESS;
    //ZSTD_DCtx* dctx;
    //ktx_uint8_t* decompBuf;
    ktxLevelIndexEntry* levelIndex = This->_private->_levelIndex;

    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (iterCb == NULL)
        return KTX_INVALID_VALUE;

    if (This->supercompressionScheme != KTX_SS_NONE)
        return KTX_INVALID_OPERATION;

    for (ktx_int32_t level = This->numLevels - 1; level >= 0; level--)
    {
        ktx_uint32_t width, height, depth;
        ktx_uint64_t levelSize;
        ktx_uint64_t offset;

        /* Array textures have the same number of layers at each mip level. */
        width = MAX(1, This->baseWidth  >> level);
        height = MAX(1, This->baseHeight >> level);
        depth = MAX(1, This->baseDepth  >> level);

        levelSize = levelIndex[level].uncompressedByteLength;
        offset = ktxTexture2_levelDataOffset(This, level);

        /* All array layers are passed in a group because that is how
         * GL & Vulkan need them. Hence no
         *    for (layer = 0; layer < This->numLayers)
         */
        result = iterCb(level, 0, width, height, depth,
                        levelSize, This->pData + offset, userdata);
        if (result != KTX_SUCCESS)
            break;
    }

    return result;
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Iterate over the images in a ktxTexture2 object while loading the
 *        image data.
 *
 * This operates similarly to ktxTexture_IterateLevelFaces() except that it
 * loads the images from the ktxTexture2's source to a temporary buffer
 * while iterating. If supercompressionScheme == KTX_SS_ZSTD or KTX_SS_ZLIB,
 * it will inflate the data before passing it to the callback. The callback function
 * must copy the image data if it wishes to preserve it as the temporary buffer
 * is reused for each level and is freed when this function exits.
 *
 * This function is helpful for reducing memory usage when uploading the data
 * to a graphics API.
 *
 * Intended for use only when supercompressionScheme == KTX_SS_NONE,
 * KTX_SS_ZSTD or KTX_SS_ZLIB. As there is no access to the ktxTexture's data on
 * conclusion of this function, destroying the texture on completion is recommended.
 *
 * @param[in]     This     pointer to the ktxTexture2 object of interest.
 * @param[in,out] iterCb   the address of a callback function which is called
 *                         with the data for each image.
 * @param[in,out] userdata the address of application-specific data which is
 *                         passed to the callback along with the image data.
 *
 * @return  KTX_SUCCESS on success, other KTX_* enum values on error. The
 *          following are returned directly by this function. @p iterCb may
 *          return these for other causes or may return additional errors.
 *
 * @exception KTX_FILE_DATA_ERROR   mip level sizes are increasing not
 *                                  decreasing
 * @exception KTX_INVALID_OPERATION the ktxTexture2 was not created from a
 *                                  stream, i.e there is no data to load, or
 *                                  this ktxTexture2's images have already
 *                                  been loaded.
 * @exception KTX_INVALID_OPERATION
 *                          supercompressionScheme != KTX_SS_NONE,
 *                          supercompressionScheme != KTX_SS_ZSTD, and
 *                          supercompressionScheme != KTX_SS_ZLIB.
 * @exception KTX_INVALID_VALUE     @p This is @c NULL or @p iterCb is @c NULL.
 * @exception KTX_OUT_OF_MEMORY     not enough memory to allocate a block to
 *                                  hold the base level image.
 */
KTX_error_code
ktxTexture2_IterateLoadLevelFaces(ktxTexture2* This, PFNKTXITERCB iterCb,
                                  void* userdata)
{
    DECLARE_PROTECTED(ktxTexture);
    ktxStream* stream = (ktxStream *)&prtctd->_stream;
    ktxLevelIndexEntry* levelIndex;
    ktx_size_t      dataSize = 0, uncompressedDataSize = 0;
    KTX_error_code  result = KTX_SUCCESS;
    ktx_uint8_t*    dataBuf = NULL;
    ktx_uint8_t*    uncompressedDataBuf = NULL;
    ktx_uint8_t*    pData;
    ZSTD_DCtx*      dctx = NULL;

    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (This->classId != ktxTexture2_c)
        return KTX_INVALID_OPERATION;

    if (This->supercompressionScheme != KTX_SS_NONE &&
        This->supercompressionScheme != KTX_SS_ZSTD &&
        This->supercompressionScheme != KTX_SS_ZLIB)
        return KTX_INVALID_OPERATION;

    if (iterCb == NULL)
        return KTX_INVALID_VALUE;

    if (prtctd->_stream.data.file == NULL)
        // This Texture not created from a stream or images are already loaded.
        return KTX_INVALID_OPERATION;

    levelIndex = This->_private->_levelIndex;

    // Allocate memory sufficient for the base level
    dataSize = levelIndex[0].byteLength;
    dataBuf = malloc(dataSize);
    if (!dataBuf)
        return KTX_OUT_OF_MEMORY;
    if (This->supercompressionScheme == KTX_SS_ZSTD || This->supercompressionScheme == KTX_SS_ZLIB) {
        uncompressedDataSize = levelIndex[0].uncompressedByteLength;
        uncompressedDataBuf = malloc(uncompressedDataSize);
        if (!uncompressedDataBuf) {
            result = KTX_OUT_OF_MEMORY;
            goto cleanup;
        }
        if (This->supercompressionScheme == KTX_SS_ZSTD) {
            dctx = ZSTD_createDCtx();
        }
        pData = uncompressedDataBuf;
    } else {
        pData = dataBuf;
    }

    for (ktx_int32_t level = This->numLevels - 1; level >= 0; --level)
    {
        ktx_size_t   levelSize;
        GLsizei      width, height, depth;

        // Array textures have the same number of layers at each mip level.
        width = MAX(1, This->baseWidth  >> level);
        height = MAX(1, This->baseHeight >> level);
        depth = MAX(1, This->baseDepth  >> level);

        levelSize = levelIndex[level].byteLength;
        if (dataSize < levelSize) {
            // Levels cannot be larger than the base level
            result = KTX_FILE_DATA_ERROR;
            goto cleanup;
        }

        // Use setpos so we skip any padding.
        result = stream->setpos(stream,
                                ktxTexture2_levelFileOffset(This, level));
        if (result != KTX_SUCCESS)
            goto cleanup;

        result = stream->read(stream, dataBuf, levelSize);
        if (result != KTX_SUCCESS)
            goto cleanup;

        if (This->supercompressionScheme == KTX_SS_ZSTD) {
            levelSize =
                ZSTD_decompressDCtx(dctx, uncompressedDataBuf,
                                  uncompressedDataSize,
                                  dataBuf,
                                  levelSize);
            if (ZSTD_isError(levelSize)) {
                ZSTD_ErrorCode error = ZSTD_getErrorCode(levelSize);
                switch(error) {
                  case ZSTD_error_dstSize_tooSmall:
                    result = KTX_DECOMPRESS_LENGTH_ERROR; // inflatedDataCapacity too small.
                    goto cleanup;
                  case ZSTD_error_checksum_wrong:
                    result =  KTX_DECOMPRESS_CHECKSUM_ERROR;
                    goto cleanup;
                  case ZSTD_error_memory_allocation:
                    result = KTX_OUT_OF_MEMORY;
                    goto cleanup;
                  default:
                    result = KTX_FILE_DATA_ERROR;
                    goto cleanup;
                }
            }

            // We don't fix up the texture's dataSize, levelIndex or
            // _requiredAlignment because after this function completes there
            // is no way to get at the texture's data.
            //nindex[level].byteOffset = levelOffset;
            //nindex[level].uncompressedByteLength = nindex[level].byteLength =
                                                                //levelByteLength;
        } else if (This->supercompressionScheme == KTX_SS_ZLIB) {
            result = ktxUncompressZLIBInt(uncompressedDataBuf,
                                            &uncompressedDataSize,
                                            dataBuf,
                                            levelSize);
            if (result != KTX_SUCCESS)
                return result;
        }

        if (levelIndex[level].uncompressedByteLength != levelSize) {
            result = KTX_DECOMPRESS_LENGTH_ERROR;
            goto cleanup;
        }


#if IS_BIG_ENDIAN
        switch (prtctd->_typeSize) {
          case 2:
            _ktxSwapEndian16((ktx_uint16_t*)pData, levelSize / 2);
            break;
          case 4:
            _ktxSwapEndian32((ktx_uint32_t*)pDest, levelSize / 4);
            break;
          case 8:
            _ktxSwapEndian64((ktx_uint64_t*)pDest, levelSize / 8);
            break;
        }
#endif

        // With the exception of non-array cubemaps the entire level
        // is passed at once because that is how OpenGL and Vulkan need them.
        // Vulkan could take all the faces at once too but we iterate
        // them separately for OpenGL.
        if (This->isCubemap && !This->isArray) {
            ktx_uint8_t* pFace = pData;
            struct blockCount {
                ktx_uint32_t x, y;
            } blockCount;
            ktx_size_t faceSize;

            blockCount.x
              = (uint32_t)ceilf((float)width / prtctd->_formatSize.blockWidth);
            blockCount.y
              = (uint32_t)ceilf((float)height / prtctd->_formatSize.blockHeight);
            blockCount.x = MAX(prtctd->_formatSize.minBlocksX, blockCount.x);
            blockCount.y = MAX(prtctd->_formatSize.minBlocksX, blockCount.y);
            faceSize = blockCount.x * blockCount.y
                       * prtctd->_formatSize.blockSizeInBits / 8;

            for (ktx_uint32_t face = 0; face < This->numFaces; ++face) {
                result = iterCb(level, face,
                                width, height, depth,
                                (ktx_uint32_t)faceSize, pFace, userdata);
                pFace += faceSize;
                if (result != KTX_SUCCESS)
                    goto cleanup;
            }
        } else {
            result = iterCb(level, 0,
                             width, height, depth,
                             (ktx_uint32_t)levelSize, pData, userdata);
            if (result != KTX_SUCCESS)
                goto cleanup;
       }
    }

    // No further need for this.
    stream->destruct(stream);
    This->_private->_firstLevelFileOffset = 0;
cleanup:
    free(dataBuf);
    if (uncompressedDataBuf) free(uncompressedDataBuf);
    if (dctx) ZSTD_freeDCtx(dctx);

    return result;
}

KTX_error_code
ktxTexture2_inflateZstdInt(ktxTexture2* This, ktx_uint8_t* pDeflatedData,
                           ktx_uint8_t* pInflatedData,
                           ktx_size_t inflatedDataCapacity);

KTX_error_code
ktxTexture2_inflateZLIBInt(ktxTexture2* This, ktx_uint8_t* pDeflatedData,
                           ktx_uint8_t* pInflatedData,
                           ktx_size_t inflatedDataCapacity);

typedef enum {
    LOADDATA_DONT_INFLATE_ON_LOAD,
    LOADDATA_INFLATE_ON_LOAD
} ktxTexture2InflateFlagEnum;

/**
 * @memberof ktxTexture2
 * @internal
 * @~English
 * @brief Load all the image data from the ktxTexture2's source.
 *
 * The data will be inflated if requested and supercompressionScheme == @c KTX_SS_ZSTD
 * or @c KTX_SS_ZLIB.
 * The data is loaded into the provided buffer or to an internally allocated
 * buffer, if @p pBuffer is @c NULL. Callers providing their own buffer must
 * ensure the buffer large enough to hold the inflated data for files deflated
 * with Zstd or ZLIB. See ktxTexture2\_GetDataSizeUncompressed().
 *
 * The texture's levelIndex, dataSize, DFD  and supercompressionScheme will
 * all be updated after successful inflation to reflect the inflated data.
 *
 * @param[in] This pointer to the ktxTexture object of interest.
 * @param[in] pBuffer pointer to the buffer in which to load the image data.
 * @param[in] bufSize size of the buffer pointed at by @p pBuffer.
 * @param[in] inflateHandling enum indicating whether or not to inflate
 *                            supercompressed data.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p This is NULL.
 * @exception KTX_INVALID_VALUE @p bufSize is less than the the image data size.
 * @exception KTX_INVALID_OPERATION
 *                              The data has already been loaded or the
 *                              ktxTexture was not created from a KTX source.
 * @exception KTX_OUT_OF_MEMORY Insufficient memory for the image data.
 */
ktx_error_code_e
ktxTexture2_loadImageDataInt(ktxTexture2* This,
                             ktx_uint8_t* pBuffer, ktx_size_t bufSize,
                             ktxTexture2InflateFlagEnum inflateHandling)
{
    DECLARE_PROTECTED(ktxTexture);
    DECLARE_PRIVATE(ktxTexture2);
    ktx_uint8_t*    pDest;
    ktx_uint8_t*    pDeflatedData = NULL;
    ktx_uint8_t*    pReadBuf;
    KTX_error_code  result = KTX_SUCCESS;
    ktx_size_t outputDataCapacity;
    ktx_bool_t doInflate = false;

    if (This == NULL)
        return KTX_INVALID_VALUE;

    if (This->pData != NULL)
        return KTX_INVALID_OPERATION; // Data already loaded.

    if (prtctd->_stream.data.file == NULL)
        // This Texture not created from a stream or images already loaded;
        return KTX_INVALID_OPERATION;

    if (inflateHandling == LOADDATA_INFLATE_ON_LOAD) {
        outputDataCapacity = ktxTexture2_GetDataSizeUncompressed(This);
        if (This->supercompressionScheme == KTX_SS_ZSTD || This->supercompressionScheme == KTX_SS_ZLIB)
            doInflate = true;
    } else {
        outputDataCapacity = This->dataSize;
    }

    if (pBuffer == NULL) {
        This->pData = malloc(outputDataCapacity);
        if (This->pData == NULL)
            return KTX_OUT_OF_MEMORY;
        pDest = This->pData;
    } else if (bufSize < outputDataCapacity) {
        return KTX_INVALID_VALUE;
    } else {
        pDest = pBuffer;
    }

    if (doInflate) {
        // Create buffer to hold deflated data.
        pDeflatedData = malloc(This->dataSize);
        if (pDeflatedData == NULL)
            return KTX_OUT_OF_MEMORY;
        pReadBuf = pDeflatedData;
    } else {
        pReadBuf = pDest;
    }

    // Seek to data for first level as there may be padding between the
    // metadata/sgd and the image data.

    result = prtctd->_stream.setpos(&prtctd->_stream,
                                    private->_firstLevelFileOffset);
    if (result != KTX_SUCCESS)
        goto cleanup;

    result = prtctd->_stream.read(&prtctd->_stream, pReadBuf,
                                  This->dataSize);
    if (result != KTX_SUCCESS)
        goto cleanup;

    if (doInflate) {
        assert(pDeflatedData != NULL);
        if (This->supercompressionScheme == KTX_SS_ZSTD) {
            result = ktxTexture2_inflateZstdInt(This, pDeflatedData, pDest,
                                                outputDataCapacity);
        } else if (This->supercompressionScheme == KTX_SS_ZLIB) {
            result = ktxTexture2_inflateZLIBInt(This, pDeflatedData, pDest,
                                                outputDataCapacity);
        }
        if (result != KTX_SUCCESS) {
            if (pBuffer == NULL) {
                free(This->pData);
                This->pData = 0;
            }
            goto cleanup;
        }
    }

    if (IS_BIG_ENDIAN) {
        // Perform endianness conversion on texture data.
        // To avoid mip padding, need to convert each level individually.
        for (ktx_uint32_t level = 0; level < This->numLevels; ++level)
        {
            ktx_size_t levelOffset;
            ktx_size_t levelByteLength;

            levelByteLength = private->_levelIndex[level].byteLength;
            levelOffset = ktxTexture2_levelDataOffset(This, level);
            pDest = This->pData + levelOffset;
            switch (prtctd->_typeSize) {
              case 2:
                _ktxSwapEndian16((ktx_uint16_t*)pDest, levelByteLength / 2);
                break;
              case 4:
                _ktxSwapEndian32((ktx_uint32_t*)pDest, levelByteLength / 4);
                break;
              case 8:
                _ktxSwapEndian64((ktx_uint64_t*)pDest, levelByteLength / 8);
                break;
            }
        }
    }

    // No further need for stream or file offset.
    prtctd->_stream.destruct(&prtctd->_stream);
    private->_firstLevelFileOffset = 0;

cleanup:
    free(pDeflatedData);

    return result;
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Load all the image data from the ktxTexture2's source.
 *
 * The data will be inflated if supercompressionScheme == @c KTX_SS_ZSTD or
 * @c KTX_SS_ZLIB.
 * The data is loaded into the provided buffer or to an internally allocated
 * buffer, if @p pBuffer is @c NULL. Callers providing their own buffer must
 * ensure the buffer large enough to hold the inflated data for files deflated
 * with Zstd or ZLIB. See ktxTexture2\_GetDataSizeUncompressed().
 *
 * The texture's levelIndex, dataSize, DFD  and supercompressionScheme will
 * all be updated after successful inflation to reflect the inflated data.
 *
 * @param[in] This pointer to the ktxTexture object of interest.
 * @param[in] pBuffer pointer to the buffer in which to load the image data.
 * @param[in] bufSize size of the buffer pointed at by @p pBuffer.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p This is NULL.
 * @exception KTX_INVALID_VALUE @p bufSize is less than the the image data size.
 * @exception KTX_INVALID_OPERATION
 *                              The data has already been loaded or the
 *                              ktxTexture was not created from a KTX source.
 * @exception KTX_OUT_OF_MEMORY Insufficient memory for the image data.
 */
ktx_error_code_e
ktxTexture2_LoadImageData(ktxTexture2* This,
                          ktx_uint8_t* pBuffer, ktx_size_t bufSize)
{
    return ktxTexture2_loadImageDataInt(This, pBuffer, bufSize, LOADDATA_INFLATE_ON_LOAD);
}

/**
 * @memberof ktxTexture2
 * @~English
 * @brief Load all the image data from the ktxTexture2's source without inflatiion..
 *
 * The data will be not be inflated if supercompressionScheme == @c KTX_SS_ZSTD or
 * @c KTX_SS_ZLIB. This function is provided to support some rare testing scenarios.
 * Generally use of ktxTexture2\_LoadImageData is highly recommended. For supercompressionScheme
 * values other than those mentioned, the result of this function is the same as
 * ktxTexture2\_LoadImageData.
 *
 * The data is loaded into the provided buffer or to an internally allocated
 * buffer, if @p pBuffer is @c NULL.
 *
 * @param[in] This pointer to the ktxTexture object of interest.
 * @param[in] pBuffer pointer to the buffer in which to load the image data.
 * @param[in] bufSize size of the buffer pointed at by @p pBuffer.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_INVALID_VALUE @p This is NULL.
 * @exception KTX_INVALID_VALUE @p bufSize is less than the the image data size.
 * @exception KTX_INVALID_OPERATION
 *                              The data has already been loaded or the
 *                              ktxTexture was not created from a KTX source.
 * @exception KTX_OUT_OF_MEMORY Insufficient memory for the image data.
 */
ktx_error_code_e
ktxTexture2_LoadDeflatedImageData(ktxTexture2* This,
                                  ktx_uint8_t* pBuffer, ktx_size_t bufSize)
{
    return ktxTexture2_loadImageDataInt(This, pBuffer, bufSize, LOADDATA_DONT_INFLATE_ON_LOAD);
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Retrieve the offset of a level's first image within the ktxTexture2's
 *        image data.
 *
 * @param[in] This pointer to the ktxTexture2 object of interest.
 */
ktx_uint64_t ktxTexture2_levelDataOffset(ktxTexture2* This, ktx_uint32_t level)
{
    return This->_private->_levelIndex[level].byteOffset;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Inflate the data in a ktxTexture2 object using Zstandard.
 *
 * The texture's levelIndex, dataSize, DFD, data pointer, and supercompressionScheme will
 * all be updated after successful inflation to reflect the inflated data.
 *
 * @param[in] This                    pointer to the ktxTexture2 object of interest.
 * @param[in] pDeflatedData pointer to a buffer containing the deflated data
 *                         of the entire texture.
 * @param[in,out] pInflatedData pointer to a buffer in which to write the inflated
 *                             data.
 * @param[in] inflatedDataCapacity capacity of the buffer pointed at by
 *                                @p pInflatedData.
 */
KTX_error_code
ktxTexture2_inflateZstdInt(ktxTexture2* This, ktx_uint8_t* pDeflatedData,
                           ktx_uint8_t* pInflatedData,
                           ktx_size_t inflatedDataCapacity)
{
    ktx_uint32_t levelIndexByteLength =
                            This->numLevels * sizeof(ktxLevelIndexEntry);
    uint64_t levelOffset = 0;
    ktxLevelIndexEntry* cindex = This->_private->_levelIndex;
    ktxLevelIndexEntry* nindex = NULL;
    ktx_uint32_t uncompressedLevelAlignment;
    ktx_error_code_e result = KTX_SUCCESS;

    ZSTD_DCtx* dctx = NULL;

    if (pDeflatedData == NULL)
        return KTX_INVALID_VALUE;

    if (pInflatedData == NULL)
        return KTX_INVALID_VALUE;

    if (This->supercompressionScheme != KTX_SS_ZSTD)
        return KTX_INVALID_OPERATION;

    nindex = malloc(levelIndexByteLength);
    if (nindex == NULL)
        return KTX_OUT_OF_MEMORY;

    uncompressedLevelAlignment =
        ktxTexture2_calcPostInflationLevelAlignment(This);

    ktx_size_t inflatedByteLength = 0;
    dctx = ZSTD_createDCtx();
    if (dctx == NULL) {
        result = KTX_OUT_OF_MEMORY;
        goto cleanup;
    }
    for (int32_t level = This->numLevels - 1; level >= 0; level--) {
        size_t levelByteLength =
            ZSTD_decompressDCtx(dctx, pInflatedData + levelOffset,
                              inflatedDataCapacity,
                              &pDeflatedData[cindex[level].byteOffset],
                              cindex[level].byteLength);
        if (ZSTD_isError(levelByteLength)) {
            ZSTD_ErrorCode error = ZSTD_getErrorCode(levelByteLength);
            switch(error) {
              case ZSTD_error_dstSize_tooSmall:
                result = KTX_DECOMPRESS_LENGTH_ERROR; // inflatedDataCapacity too small.
                goto cleanup;
              case ZSTD_error_checksum_wrong:
                result = KTX_DECOMPRESS_CHECKSUM_ERROR;
                goto cleanup;
              case ZSTD_error_memory_allocation:
                result =  KTX_OUT_OF_MEMORY;
                goto cleanup;
             default:
                result = KTX_FILE_DATA_ERROR;
                goto cleanup;
            }
        }

        if (This->_private->_levelIndex[level].uncompressedByteLength != levelByteLength) {
            result = KTX_DECOMPRESS_LENGTH_ERROR;
            goto cleanup;
        }

        nindex[level].byteOffset = levelOffset;
        nindex[level].uncompressedByteLength = nindex[level].byteLength =
                                                            levelByteLength;
        ktx_size_t paddedLevelByteLength
              = _KTX_PADN(uncompressedLevelAlignment, levelByteLength);
        inflatedByteLength += paddedLevelByteLength;
        levelOffset += paddedLevelByteLength;
        inflatedDataCapacity -= paddedLevelByteLength;
    }

    // Now modify the texture.

    This->dataSize = inflatedByteLength;
    This->supercompressionScheme = KTX_SS_NONE;
    memcpy(cindex, nindex, levelIndexByteLength); // Update level index
    This->_private->_requiredLevelAlignment = uncompressedLevelAlignment;

cleanup:
    ZSTD_freeDCtx(dctx);
    free(nindex);
    return result;
}

/**
 * @memberof ktxTexture2 @private
 * @~English
 * @brief Inflate the data in a ktxTexture2 object using miniz (ZLIB).
 *
 * The texture's levelIndex, dataSize, DFD, data pointer, and supercompressionScheme will
 * all be updated after successful inflation to reflect the inflated data.
 *
 * @param[in] This              pointer to the ktxTexture2 object of interest.
 * @param[in] pDeflatedData     pointer to a buffer containing the deflated
 *                              data of the entire texture.
 * @param[in,out] pInflatedData pointer to a buffer in which to write the
 *                              inflated data.
 * @param[in] inflatedDataCapacity capacity of the buffer pointed at by
 *                                @p pInflatedData.
 */
KTX_error_code
ktxTexture2_inflateZLIBInt(ktxTexture2* This, ktx_uint8_t* pDeflatedData,
                           ktx_uint8_t* pInflatedData,
                           ktx_size_t inflatedDataCapacity)
{
    ktx_uint32_t levelIndexByteLength =
                            This->numLevels * sizeof(ktxLevelIndexEntry);
    uint64_t levelOffset = 0;
    ktxLevelIndexEntry* cindex = This->_private->_levelIndex;
    ktxLevelIndexEntry* nindex;
    ktx_uint32_t uncompressedLevelAlignment;

    if (pDeflatedData == NULL)
        return KTX_INVALID_VALUE;

    if (pInflatedData == NULL)
        return KTX_INVALID_VALUE;

    if (This->supercompressionScheme != KTX_SS_ZLIB)
        return KTX_INVALID_OPERATION;

    nindex = malloc(levelIndexByteLength);
    if (nindex == NULL)
        return KTX_OUT_OF_MEMORY;

    uncompressedLevelAlignment =
        ktxTexture2_calcPostInflationLevelAlignment(This);

    ktx_size_t inflatedByteLength = 0;
    for (int32_t level = This->numLevels - 1; level >= 0; level--) {
        size_t levelByteLength = inflatedDataCapacity;
        KTX_error_code result = ktxUncompressZLIBInt(pInflatedData + levelOffset,
                                                    &levelByteLength,
                                                    &pDeflatedData[cindex[level].byteOffset],
                                                    cindex[level].byteLength);
        if (result != KTX_SUCCESS) {
            free(nindex);
            return result;
        }

        if (This->_private->_levelIndex[level].uncompressedByteLength != levelByteLength) {
            free(nindex);
            return KTX_DECOMPRESS_LENGTH_ERROR;
        }

        nindex[level].byteOffset = levelOffset;
        nindex[level].uncompressedByteLength = nindex[level].byteLength =
                                                            levelByteLength;
        ktx_size_t paddedLevelByteLength
              = _KTX_PADN(uncompressedLevelAlignment, levelByteLength);
        inflatedByteLength += paddedLevelByteLength;
        levelOffset += paddedLevelByteLength;
        inflatedDataCapacity -= paddedLevelByteLength;
    }

    // Now modify the texture.

    This->dataSize = inflatedByteLength;
    This->supercompressionScheme = KTX_SS_NONE;
    memcpy(cindex, nindex, levelIndexByteLength); // Update level index
    free(nindex);
    This->_private->_requiredLevelAlignment = uncompressedLevelAlignment;

    return KTX_SUCCESS;
}

#if !KTX_FEATURE_WRITE

/*
 * Stubs for writer functions that return a proper error code
 */

KTX_error_code
ktxTexture2_SetImageFromMemory(ktxTexture2* This, ktx_uint32_t level,
                               ktx_uint32_t layer, ktx_uint32_t faceSlice,
                               const ktx_uint8_t* src, ktx_size_t srcSize)
{
    UNUSED(This);
    UNUSED(level);
    UNUSED(layer);
    UNUSED(faceSlice);
    UNUSED(src);
    UNUSED(srcSize);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture2_SetImageFromStdioStream(ktxTexture2* This, ktx_uint32_t level,
                                    ktx_uint32_t layer, ktx_uint32_t faceSlice,
                                    FILE* src, ktx_size_t srcSize)
{
    UNUSED(This);
    UNUSED(level);
    UNUSED(layer);
    UNUSED(faceSlice);
    UNUSED(src);
    UNUSED(srcSize);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture2_WriteToStdioStream(ktxTexture2* This, FILE* dstsstr)
{
    UNUSED(This);
    UNUSED(dstsstr);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture2_WriteToNamedFile(ktxTexture2* This, const char* const dstname)
{
    UNUSED(This);
    UNUSED(dstname);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture2_WriteToMemory(ktxTexture2* This,
                          ktx_uint8_t** ppDstBytes, ktx_size_t* pSize)
{
    UNUSED(This);
    UNUSED(ppDstBytes);
    UNUSED(pSize);
    return KTX_INVALID_OPERATION;
}

KTX_error_code
ktxTexture2_WriteToStream(ktxTexture2* This,
                          ktxStream* dststr)
{
    UNUSED(This);
    UNUSED(dststr);
    return KTX_INVALID_OPERATION;
}

#endif

/*
 * Initialized here at the end to avoid the need for multiple declarations of
 * the virtual functions.
 */

struct ktxTexture_vtblInt ktxTexture2_vtblInt = {
    (PFNCALCDATASIZELEVELS)ktxTexture2_calcDataSizeLevels,
    (PFNCALCFACELODSIZE)ktxTexture2_calcFaceLodSize,
    (PFNCALCLEVELOFFSET)ktxTexture2_calcLevelOffset
};

struct ktxTexture_vtbl ktxTexture2_vtbl = {
    (PFNKTEXDESTROY)ktxTexture2_Destroy,
    (PFNKTEXGETIMAGEOFFSET)ktxTexture2_GetImageOffset,
    (PFNKTEXGETDATASIZEUNCOMPRESSED)ktxTexture2_GetDataSizeUncompressed,
    (PFNKTEXGETIMAGESIZE)ktxTexture2_GetImageSize,
    (PFNKTEXGETLEVELSIZE)ktxTexture2_GetLevelSize,
    (PFNKTEXITERATELEVELS)ktxTexture2_IterateLevels,
    (PFNKTEXITERATELOADLEVELFACES)ktxTexture2_IterateLoadLevelFaces,
    (PFNKTEXNEEDSTRANSCODING)ktxTexture2_NeedsTranscoding,
    (PFNKTEXLOADIMAGEDATA)ktxTexture2_LoadImageData,
    (PFNKTEXSETIMAGEFROMMEMORY)ktxTexture2_SetImageFromMemory,
    (PFNKTEXSETIMAGEFROMSTDIOSTREAM)ktxTexture2_SetImageFromStdioStream,
    (PFNKTEXWRITETOSTDIOSTREAM)ktxTexture2_WriteToStdioStream,
    (PFNKTEXWRITETONAMEDFILE)ktxTexture2_WriteToNamedFile,
    (PFNKTEXWRITETOMEMORY)ktxTexture2_WriteToMemory,
    (PFNKTEXWRITETOSTREAM)ktxTexture2_WriteToStream,
};

/** @} */

