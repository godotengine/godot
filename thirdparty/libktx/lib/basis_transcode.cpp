/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file basis_transcode.cpp
 * @~English
 *
 * @brief Functions for transcoding Basis Universal BasisLZ/ETC1S and UASTC textures.
 *
 * Two worlds collide here too. More uglyness!
 *
 * @author Mark Callow, www.edgewise-consulting.com
 */

#include <inttypes.h>
#include <stdio.h>
#include <KHR/khr_df.h>

#include "dfdutils/dfd.h"
#include "ktx.h"
#include "ktxint.h"
#include "texture2.h"
#include "vkformat_enum.h"
#include "vk_format.h"
#include "basis_sgd.h"
#include "transcoder/basisu_file_headers.h"
#include "transcoder/basisu_transcoder.h"
#include "transcoder/basisu_transcoder_internal.h"

#undef DECLARE_PRIVATE
#undef DECLARE_PROTECTED
#define DECLARE_PRIVATE(n,t2) ktxTexture2_private& n = *(t2->_private)
#define DECLARE_PROTECTED(n,t2) ktxTexture_protected& n = *(t2->_protected)

using namespace basisu;
using namespace basist;

inline bool isPow2(uint32_t x) { return x && ((x & (x - 1U)) == 0U); }

inline bool isPow2(uint64_t x) { return x && ((x & (x - 1U)) == 0U); }

KTX_error_code
ktxTexture2_transcodeLzEtc1s(ktxTexture2* This,
                           alpha_content_e alphaContent,
                           ktxTexture2* prototype,
                           ktx_transcode_fmt_e outputFormat,
                           ktx_transcode_flags transcodeFlags);
KTX_error_code
ktxTexture2_transcodeUastc(ktxTexture2* This,
                           alpha_content_e alphaContent,
                           ktxTexture2* prototype,
                           ktx_transcode_fmt_e outputFormat,
                           ktx_transcode_flags transcodeFlags);

/**
 * @memberof ktxTexture2
 * @ingroup reader
 * @~English
 * @brief Transcode a KTX2 texture with BasisLZ/ETC1S or UASTC images.
 *
 * If the texture contains BasisLZ supercompressed images, Inflates them from
 * back to ETC1S then transcodes them to the specified block-compressed
 * format. If the texture contains UASTC images, inflates them, if they have been
 * supercompressed with zstd, then transcodes then to the specified format, The
 * transcoded images replace the original images and the texture's fields including
 * the DFD are modified to reflect the new format.
 *
 * These types of textures must be transcoded to a desired target
 * block-compressed format before they can be uploaded to a GPU via a
 * graphics API.
 *
 * The following block compressed transcode targets are available: @c KTX_TTF_ETC1_RGB,
 * @c KTX_TTF_ETC2_RGBA, @c KTX_TTF_BC1_RGB, @c KTX_TTF_BC3_RGBA,
 * @c KTX_TTF_BC4_R, @c KTX_TTF_BC5_RG, @c KTX_TTF_BC7_RGBA,
 * @c @c KTX_TTF_PVRTC1_4_RGB, @c KTX_TTF_PVRTC1_4_RGBA,
 * @c KTX_TTF_PVRTC2_4_RGB, @c KTX_TTF_PVRTC2_4_RGBA, @c KTX_TTF_ASTC_4x4_RGBA,
 * @c KTX_TTF_ETC2_EAC_R11, @c KTX_TTF_ETC2_EAC_RG11, @c KTX_TTF_ETC and
 * @c KTX_TTF_BC1_OR_3.
 *
 * @c KTX_TTF_ETC automatically selects between @c KTX_TTF_ETC1_RGB and
 * @c KTX_TTF_ETC2_RGBA according to whether an alpha channel is available. @c KTX_TTF_BC1_OR_3
 * does likewise between @c KTX_TTF_BC1_RGB and @c KTX_TTF_BC3_RGBA. Note that if
 * @c KTX_TTF_PVRTC1_4_RGBA or @c KTX_TTF_PVRTC2_4_RGBA is specified and there is no alpha
 * channel @c KTX_TTF_PVRTC1_4_RGB or @c KTX_TTF_PVRTC2_4_RGB respectively will be selected.
 *
 * Transcoding to ATC & FXT1 formats is not supported by libktx as there
 * are no equivalent Vulkan formats.
 *
 * The following uncompressed transcode targets are also available: @c KTX_TTF_RGBA32,
 * @c KTX_TTF_RGB565, KTX_TTF_BGR565 and KTX_TTF_RGBA4444.
 *
 * The following @p transcodeFlags are available.
 *
 * @sa ktxtexture2_CompressBasis().
 *
 * @param[in]   This         pointer to the ktxTexture2 object of interest.
 * @param[in]   outputFormat a value from the ktx_texture_transcode_fmt_e enum
 *                                             specifying the target format.
 * @param[in]   transcodeFlags  bitfield of flags modifying the transcode
 *                                                operation. @sa ktx_texture_decode_flags_e.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_DATA_ERROR
 *                              Supercompression global data is corrupted.
 * @exception KTX_INVALID_OPERATION
 *                              The texture's format is not transcodable (not
 *                              ETC1S/BasisLZ or UASTC).
 * @exception KTX_INVALID_OPERATION
 *                              Supercompression global data is missing, i.e.,
 *                              the texture object is invalid.
 * @exception KTX_INVALID_OPERATION
 *                              Image data is missing, i.e., the texture object
 *                              is invalid.
 * @exception KTX_INVALID_OPERATION
 *                              @p outputFormat is PVRTC1 but the texture does
 *                              does not have power-of-two dimensions.
 * @exception KTX_INVALID_VALUE @p outputFormat is invalid.
 * @exception KTX_TRANSCODE_FAILED
 *                              Something went wrong during transcoding.
 * @exception KTX_UNSUPPORTED_FEATURE
 *                              KTX_TF_PVRTC_DECODE_TO_NEXT_POW2 was requested
 *                              or the specified transcode target has not been
 *                              included in the library being used.
 * @exception KTX_OUT_OF_MEMORY Not enough memory to carry out transcoding.
 */
 KTX_error_code
 ktxTexture2_TranscodeBasis(ktxTexture2* This,
                            ktx_transcode_fmt_e outputFormat,
                            ktx_transcode_flags transcodeFlags)
{
    uint32_t* BDB = This->pDfd + 1;
    khr_df_model_e colorModel = (khr_df_model_e)KHR_DFDVAL(BDB, MODEL);
    if (colorModel != KHR_DF_MODEL_UASTC
        // Constructor has checked color model matches BASIS_LZ.
        && This->supercompressionScheme != KTX_SS_BASIS_LZ)
    {
        return KTX_INVALID_OPERATION; // Not in a transcodable format.
    }

    DECLARE_PRIVATE(priv, This);
    if (This->supercompressionScheme == KTX_SS_BASIS_LZ) {
        if (!priv._supercompressionGlobalData || priv._sgdByteLength == 0)
            return KTX_INVALID_OPERATION;
    }

    if (transcodeFlags & KTX_TF_PVRTC_DECODE_TO_NEXT_POW2) {
         debug_printf("ktxTexture_TranscodeBasis: KTX_TF_PVRTC_DECODE_TO_NEXT_POW2 currently unsupported\n");
         return KTX_UNSUPPORTED_FEATURE;
    }

    if (outputFormat == KTX_TTF_PVRTC1_4_RGB
        || outputFormat == KTX_TTF_PVRTC1_4_RGBA) {
         if ((!isPow2(This->baseWidth)) || (!isPow2(This->baseHeight))) {
             debug_printf("ktxTexture_TranscodeBasis: PVRTC1 only supports power of 2 dimensions\n");
             return KTX_INVALID_OPERATION;
        }
    }

    const bool srgb = (KHR_DFDVAL(BDB, TRANSFER) == KHR_DF_TRANSFER_SRGB);
    alpha_content_e alphaContent = eNone;
    if (colorModel == KHR_DF_MODEL_ETC1S) {
        if (KHR_DFDSAMPLECOUNT(BDB) == 2) {
            uint32_t channelId = KHR_DFDSVAL(BDB, 1, CHANNELID);
            if (channelId == KHR_DF_CHANNEL_ETC1S_AAA) {
                alphaContent = eAlpha;
            } else if (channelId == KHR_DF_CHANNEL_ETC1S_GGG){
                alphaContent = eGreen;
            } else {
                return KTX_FILE_DATA_ERROR;
            }
        }
    } else {
        uint32_t channelId = KHR_DFDSVAL(BDB, 0, CHANNELID);
        if (channelId == KHR_DF_CHANNEL_UASTC_RGBA)
            alphaContent = eAlpha;
        else if (channelId == KHR_DF_CHANNEL_UASTC_RRRG)
            alphaContent = eGreen;
    }

    VkFormat vkFormat;

    // Do some format mapping.
    switch (outputFormat) {
      case KTX_TTF_BC1_OR_3:
        outputFormat = alphaContent != eNone ? KTX_TTF_BC3_RGBA
                                             : KTX_TTF_BC1_RGB;
        break;
      case KTX_TTF_ETC:
        outputFormat = alphaContent != eNone ? KTX_TTF_ETC2_RGBA
                                             : KTX_TTF_ETC1_RGB;
        break;
      case KTX_TTF_PVRTC1_4_RGBA:
        // This transcoder does not write opaque alpha blocks.
        outputFormat = alphaContent != eNone  ? KTX_TTF_PVRTC1_4_RGBA
                                              : KTX_TTF_PVRTC1_4_RGB;
        break;
      case KTX_TTF_PVRTC2_4_RGBA:
        // This transcoder does not write opaque alpha blocks.
        outputFormat = alphaContent != eNone ? KTX_TTF_PVRTC2_4_RGBA
                                              : KTX_TTF_PVRTC2_4_RGB;
        break;
      default:
        /*NOP*/;
    }

    switch (outputFormat) {
      case KTX_TTF_ETC1_RGB:
        vkFormat = srgb ? VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK
                        : VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
        break;
      case KTX_TTF_ETC2_RGBA:
        vkFormat = srgb ? VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK
                        : VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
        break;
      case KTX_TTF_ETC2_EAC_R11:
        vkFormat = VK_FORMAT_EAC_R11_UNORM_BLOCK;
        break;
      case KTX_TTF_ETC2_EAC_RG11:
        vkFormat = VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
        break;
      case KTX_TTF_BC1_RGB:
        // Transcoding doesn't support BC1 alpha.
        vkFormat = srgb ? VK_FORMAT_BC1_RGB_SRGB_BLOCK
                        : VK_FORMAT_BC1_RGB_UNORM_BLOCK;
        break;
      case KTX_TTF_BC3_RGBA:
        vkFormat = srgb ? VK_FORMAT_BC3_SRGB_BLOCK
                        : VK_FORMAT_BC3_UNORM_BLOCK;
        break;
      case KTX_TTF_BC4_R:
        vkFormat = VK_FORMAT_BC4_UNORM_BLOCK;
        break;
      case KTX_TTF_BC5_RG:
        vkFormat = VK_FORMAT_BC5_UNORM_BLOCK;
        break;
      case KTX_TTF_PVRTC1_4_RGB:
      case KTX_TTF_PVRTC1_4_RGBA:
        vkFormat = srgb ? VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG
                        : VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG;
        break;
      case KTX_TTF_PVRTC2_4_RGB:
      case KTX_TTF_PVRTC2_4_RGBA:
        vkFormat = srgb ? VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG
                        : VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG;
        break;
      case KTX_TTF_BC7_RGBA:
        vkFormat = srgb ? VK_FORMAT_BC7_SRGB_BLOCK
                        : VK_FORMAT_BC7_UNORM_BLOCK;
        break;
      case KTX_TTF_ASTC_4x4_RGBA:
        vkFormat = srgb ? VK_FORMAT_ASTC_4x4_SRGB_BLOCK
                        : VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        break;
      case KTX_TTF_RGB565:
        vkFormat = VK_FORMAT_R5G6B5_UNORM_PACK16;
        break;
      case KTX_TTF_BGR565:
        vkFormat = VK_FORMAT_B5G6R5_UNORM_PACK16;
        break;
      case KTX_TTF_RGBA4444:
        vkFormat = VK_FORMAT_R4G4B4A4_UNORM_PACK16;
        break;
      case KTX_TTF_RGBA32:
        vkFormat = srgb ? VK_FORMAT_R8G8B8A8_SRGB
                        : VK_FORMAT_R8G8B8A8_UNORM;
        break;
      default:
        return KTX_INVALID_VALUE;
    }

    basis_tex_format textureFormat;
    if (colorModel == KHR_DF_MODEL_UASTC)
        textureFormat = basis_tex_format::cUASTC4x4;
    else
        textureFormat = basis_tex_format::cETC1S;

    if (!basis_is_format_supported((transcoder_texture_format)outputFormat,
                                    textureFormat)) {
        return KTX_UNSUPPORTED_FEATURE;
    }


    // Create a prototype texture to use for calculating sizes in the target
    // format and, as useful side effects, provide us with a properly sized
    // data allocation and the DFD for the target format.
    ktxTextureCreateInfo createInfo;
    createInfo.glInternalformat = 0;
    createInfo.vkFormat = vkFormat;
    createInfo.baseWidth = This->baseWidth;
    createInfo.baseHeight = This->baseHeight;
    createInfo.baseDepth = This->baseDepth;
    createInfo.generateMipmaps = This->generateMipmaps;
    createInfo.isArray = This->isArray;
    createInfo.numDimensions = This->numDimensions;
    createInfo.numFaces = This->numFaces;
    createInfo.numLayers = This->numLayers;
    createInfo.numLevels = This->numLevels;
    createInfo.pDfd = nullptr;

    KTX_error_code result;
    ktxTexture2* prototype;
    result = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE,
                                &prototype);

    if (result != KTX_SUCCESS) {
        assert(result == KTX_OUT_OF_MEMORY); // The only run time error
        return result;
    }

    if (!This->pData) {
        if (ktxTexture_isActiveStream((ktxTexture*)This)) {
             // Load pending. Complete it.
            result = ktxTexture2_LoadImageData(This, NULL, 0);
            if (result != KTX_SUCCESS)
            {
                ktxTexture2_Destroy(prototype);
                return result;
            }
        } else {
            // No data to transcode.
            ktxTexture2_Destroy(prototype);
            return KTX_INVALID_OPERATION;
        }
    }

    // Transcoder global initialization. Requires ~9 milliseconds when compiled
    // and executed natively on a Core i7 2.2 GHz. If this is too slow, the
    // tables it computes can easily be moved to be compiled in.
    static bool transcoderInitialized;
    if (!transcoderInitialized) {
        basisu_transcoder_init();
        transcoderInitialized = true;
    }

    if (textureFormat == basis_tex_format::cETC1S) {
        result = ktxTexture2_transcodeLzEtc1s(This, alphaContent,
                                            prototype, outputFormat,
                                            transcodeFlags);
    } else {
        result = ktxTexture2_transcodeUastc(This, alphaContent,
                                            prototype, outputFormat,
                                            transcodeFlags);
    }

    if (result == KTX_SUCCESS) {
        // Fix up the current texture
        DECLARE_PROTECTED(thisPrtctd, This);
        DECLARE_PRIVATE(protoPriv, prototype);
        DECLARE_PROTECTED(protoPrtctd, prototype);
        memcpy(&thisPrtctd._formatSize, &protoPrtctd._formatSize,
               sizeof(ktxFormatSize));
        This->vkFormat = vkFormat;
        This->isCompressed = prototype->isCompressed;
        This->supercompressionScheme = KTX_SS_NONE;
        priv._requiredLevelAlignment = protoPriv._requiredLevelAlignment;
        // Copy the levelIndex from the prototype to This.
        memcpy(priv._levelIndex, protoPriv._levelIndex,
               This->numLevels * sizeof(ktxLevelIndexEntry));
        // Move the DFD and data from the prototype to This.
        free(This->pDfd);
        This->pDfd = prototype->pDfd;
        prototype->pDfd = 0;
        free(This->pData);
        This->pData = prototype->pData;
        This->dataSize = prototype->dataSize;
        prototype->pData = 0;
        prototype->dataSize = 0;
        // Free SGD data
        This->_private->_sgdByteLength = 0;
        if (This->_private->_supercompressionGlobalData) {
            free(This->_private->_supercompressionGlobalData);
            This->_private->_supercompressionGlobalData = NULL;
        }
    }
    ktxTexture2_Destroy(prototype);
    return result;
 }

/**
 * @memberof ktxTexture2 @private
 * @ingroup reader
 * @~English
 * @brief Transcode a KTX2 texture with BasisLZ supercompressed ETC1S images.
 *
 * Inflates the images from BasisLZ supercompression back to ETC1S
 * then transcodes them to the specified block-compressed format. The
 * transcoded images replace the original images and the texture's fields
 * including the DFD are modified to reflect the new format.
 *
 * BasisLZ supercompressed textures must be transcoded to a desired target
 * block-compressed format before they can be uploaded to a GPU via a graphics
 * API.
 *
 * The following block compressed transcode targets are available: @c KTX_TTF_ETC1_RGB,
 * @c KTX_TTF_ETC2_RGBA, @c KTX_TTF_BC1_RGB, @c KTX_TTF_BC3_RGBA,
 * @c KTX_TTF_BC4_R, @c KTX_TTF_BC5_RG, @c KTX_TTF_BC7_RGBA,
 * @c @c KTX_TTF_PVRTC1_4_RGB, @c KTX_TTF_PVRTC1_4_RGBA,
 * @c KTX_TTF_PVRTC2_4_RGB, @c KTX_TTF_PVRTC2_4_RGBA, @c KTX_TTF_ASTC_4x4_RGBA,
 * @c KTX_TTF_ETC2_EAC_R11, @c KTX_TTF_ETC2_EAC_RG11, @c KTX_TTF_ETC and
 * @c KTX_TTF_BC1_OR_3.
 *
 * @c KTX_TTF_ETC automatically selects between @c KTX_TTF_ETC1_RGB and
 * @c KTX_TTF_ETC2_RGBA according to whether an alpha channel is available. @c KTX_TTF_BC1_OR_3
 * does likewise between @c KTX_TTF_BC1_RGB and @c KTX_TTF_BC3_RGBA. Note that if
 * @c KTX_TTF_PVRTC1_4_RGBA or @c KTX_TTF_PVRTC2_4_RGBA is specified and there is no alpha
 * channel @c KTX_TTF_PVRTC1_4_RGB or @c KTX_TTF_PVRTC2_4_RGB respectively will be selected.
 *
 * ATC & FXT1 formats are not supported by KTX2 & libktx as there are no equivalent Vulkan formats.
 *
 * The following uncompressed transcode targets are also available: @c KTX_TTF_RGBA32,
 * @c KTX_TTF_RGB565, KTX_TTF_BGR565 and KTX_TTF_RGBA4444.
 *
 * The following @p transcodeFlags are available.
 *
 * @sa ktxtexture2_CompressBasis().
 *
 * @param[in]   This         pointer to the ktxTexture2 object of interest.
 * @param[in]   outputFormat a value from the ktx_texture_transcode_fmt_e enum
 *                           specifying the target format.
 * @param[in]   transcodeFlags  bitfield of flags modifying the transcode
 *                           operation. @sa ktx_texture_decode_flags_e.
 *
 * @return      KTX_SUCCESS on success, other KTX_* enum values on error.
 *
 * @exception KTX_FILE_DATA_ERROR
 *                              Supercompression global data is corrupted.
 * @exception KTX_INVALID_OPERATION
 *                              The texture's format is not transcodable (not
 *                              ETC1S/BasisLZ or UASTC).
 * @exception KTX_INVALID_OPERATION
 *                              Supercompression global data is missing, i.e.,
 *                              the texture object is invalid.
 * @exception KTX_INVALID_OPERATION
 *                              Image data is missing, i.e., the texture object
 *                              is invalid.
 * @exception KTX_INVALID_OPERATION
 *                              @p outputFormat is PVRTC1 but the texture does
 *                              does not have power-of-two dimensions.
 * @exception KTX_INVALID_VALUE @p outputFormat is invalid.
 * @exception KTX_TRANSCODE_FAILED
 *                              Something went wrong during transcoding. The
 *                              texture object will be corrupted.
 * @exception KTX_UNSUPPORTED_FEATURE
 *                              KTX_TF_PVRTC_DECODE_TO_NEXT_POW2 was requested
 *                              or the specified transcode target has not been
 *                              included in the library being used.
 * @exception KTX_OUT_OF_MEMORY Not enough memory to carry out transcoding.
 */
KTX_error_code
ktxTexture2_transcodeLzEtc1s(ktxTexture2* This,
                             alpha_content_e alphaContent,
                             ktxTexture2* prototype,
                             ktx_transcode_fmt_e outputFormat,
                             ktx_transcode_flags transcodeFlags)
{
    DECLARE_PRIVATE(priv, This);
    DECLARE_PRIVATE(protoPriv, prototype);
    KTX_error_code result = KTX_SUCCESS;

    assert(This->supercompressionScheme == KTX_SS_BASIS_LZ);

    uint8_t* bgd = priv._supercompressionGlobalData;
    ktxBasisLzGlobalHeader& bgdh = *reinterpret_cast<ktxBasisLzGlobalHeader*>(bgd);
    if (!(bgdh.endpointsByteLength && bgdh.selectorsByteLength && bgdh.tablesByteLength)) {
        debug_printf("ktxTexture_TranscodeBasis: missing endpoints, selectors or tables");
        return KTX_FILE_DATA_ERROR;
    }

    // Compute some helpful numbers.
    //
    // firstImages contains the indices of the first images for each level to
    // ease finding the correct slice description when iterating from smallest
    // level to largest or when randomly accessing them (t.b.c). The last array
    // entry contains the total number of images, for calculating the offsets
    // of the endpoints, etc.
    uint32_t* firstImages = new uint32_t[This->numLevels+1];

    // Temporary invariant value
    uint32_t layersFaces = This->numLayers * This->numFaces;
    firstImages[0] = 0;
    for (uint32_t level = 1; level <= This->numLevels; level++) {
        // NOTA BENE: numFaces * depth is only reasonable because they can't
        // both be > 1. I.e there are no 3d cubemaps.
        firstImages[level] = firstImages[level - 1]
                           + layersFaces * MAX(This->baseDepth >> (level - 1), 1);
    }
    uint32_t& imageCount = firstImages[This->numLevels];

    if (BGD_TABLES_ADDR(0, bgdh, imageCount) + bgdh.tablesByteLength > priv._sgdByteLength) {
        return KTX_FILE_DATA_ERROR;
    }
    // FIXME: Do more validation.

    // Prepare low-level transcoder for transcoding slices.
    basist::basisu_lowlevel_etc1s_transcoder bit;

    // basisu_transcoder_state is used to find the previous frame when
    // decoding a video P-Frame. It tracks the previous frame for each mip
    // level. For cube map array textures we need to find the previous frame
    // for each face so we a state per face. Although providing this is only
    // needed for video, it is easier to always pass our own.
    std::vector<basisu_transcoder_state> xcoderStates;
    xcoderStates.resize(This->isVideo ? This->numFaces : 1);

    bit.decode_palettes(bgdh.endpointCount, BGD_ENDPOINTS_ADDR(bgd, imageCount),
                        bgdh.endpointsByteLength,
                        bgdh.selectorCount, BGD_SELECTORS_ADDR(bgd, bgdh, imageCount),
                        bgdh.selectorsByteLength);

    bit.decode_tables(BGD_TABLES_ADDR(bgd, bgdh, imageCount),
                      bgdh.tablesByteLength);

    // Find matching VkFormat and calculate output sizes.

    const bool isVideo = This->isVideo;

    ktx_uint8_t* pXcodedData = prototype->pData;
    // Inconveniently, the output buffer size parameter of transcode_image
    // has to be in pixels for uncompressed output and in blocks for
    // compressed output. The only reason for humouring the API is so
    // its buffer size tests provide a real check. An alternative is to
    // always provide the size in bytes which will always pass.
    ktx_uint32_t outputBlockByteLength
                      = prototype->_protected->_formatSize.blockSizeInBits / 8;
    ktx_size_t xcodedDataLength
                      = prototype->dataSize / outputBlockByteLength;
    ktxLevelIndexEntry* protoLevelIndex;
    uint64_t levelOffsetWrite;
    const ktxBasisLzEtc1sImageDesc* imageDescs = BGD_ETC1S_IMAGE_DESCS(bgd);

    // Finally we're ready to transcode the slices.

    // FIXME: Iframe flag needs to be queryable by the application. In Basis
    // the app can query file_info and image_info from the transcoder which
    // returns a structure with lots of info about the image.

    protoLevelIndex = protoPriv._levelIndex;
    levelOffsetWrite = 0;
    for (int32_t level = This->numLevels - 1; level >= 0; level--) {
        uint64_t levelOffset = ktxTexture2_levelDataOffset(This, level);
        uint64_t writeOffset = levelOffsetWrite;
        uint64_t writeOffsetBlocks = levelOffsetWrite / outputBlockByteLength;
        uint32_t levelWidth = MAX(1, This->baseWidth >> level);
        uint32_t levelHeight = MAX(1, This->baseHeight >> level);
        // ETC1S texel block dimensions
        const uint32_t bw = 4, bh = 4;
        uint32_t levelBlocksX = (levelWidth + (bw - 1)) / bw;
        uint32_t levelBlocksY = (levelHeight + (bh - 1)) / bh;
        uint32_t depth = MAX(1, This->baseDepth >> level);
        //uint32_t faceSlices = This->numFaces == 1 ? depth : This->numFaces;
        uint32_t faceSlices = This->numFaces * depth;
        uint32_t numImages = This->numLayers * faceSlices;
        uint32_t image = firstImages[level];
        uint32_t endImage = image + numImages;
        ktx_size_t levelImageSizeOut, levelSizeOut;
        uint32_t stateIndex = 0;

        levelSizeOut = 0;
        // FIXME: Figure out a way to get the size out of the transcoder.
        levelImageSizeOut = ktxTexture2_GetImageSize(prototype, level);
        for (; image < endImage; image++) {
            const ktxBasisLzEtc1sImageDesc& imageDesc = imageDescs[image];

            basisu_transcoder_state& xcoderState = xcoderStates[stateIndex];
            // We have face0 [face1 ...] within each layer. Use `stateIndex`
            // rather than a double loop of layers and faceSlices as this
            // works for 3d texture and non-array cube maps as well as
            // cube map arrays without special casing.
            if (++stateIndex == xcoderStates.size())
                stateIndex = 0;

            if (alphaContent != eNone)
            {
                // The slice descriptions should have alpha information.
                if (imageDesc.alphaSliceByteOffset == 0
                    || imageDesc.alphaSliceByteLength == 0)
                    return KTX_FILE_DATA_ERROR;
            }

            bool status;
            status = bit.transcode_image(
                      (transcoder_texture_format)outputFormat,
                      pXcodedData + writeOffset,
                      (uint32_t)(xcodedDataLength - writeOffsetBlocks),
                      This->pData,
                      (uint32_t)This->dataSize,
                      levelBlocksX,
                      levelBlocksY,
                      levelWidth,
                      levelHeight,
                      level,
                      (uint32_t)(levelOffset + imageDesc.rgbSliceByteOffset),
                      imageDesc.rgbSliceByteLength,
                      (uint32_t)(levelOffset + imageDesc.alphaSliceByteOffset),
                      imageDesc.alphaSliceByteLength,
                      transcodeFlags,
                      alphaContent != eNone,
                      isVideo,
                      // Our P-Frame flag is in the same bit as
                      // cSliceDescFlagsFrameIsIFrame. We have to
                      // invert it to make it an I-Frame flag.
                      //
                      // API currently doesn't have any way to pass
                      // the I-Frame flag.
                      //imageDesc.imageFlags ^ cSliceDescFlagsFrameIsIFrame,
                      0, // output_row_pitch_in_blocks_or_pixels
                      &xcoderState,
                      0  // output_rows_in_pixels
                      );
            if (!status) {
                result = KTX_TRANSCODE_FAILED;
                goto cleanup;
            }

            writeOffset += levelImageSizeOut;
            levelSizeOut += levelImageSizeOut;
        } // end images loop
        protoLevelIndex[level].byteOffset = levelOffsetWrite;
        protoLevelIndex[level].byteLength = levelSizeOut;
        protoLevelIndex[level].uncompressedByteLength = levelSizeOut;
        levelOffsetWrite += levelSizeOut;
        assert(levelOffsetWrite == writeOffset);
        // In case of transcoding to uncompressed.
        levelOffsetWrite = _KTX_PADN(protoPriv._requiredLevelAlignment,
                                     levelOffsetWrite);
    } // level loop

    result = KTX_SUCCESS;

cleanup:
    delete[] firstImages;
    return result;
}


KTX_error_code
ktxTexture2_transcodeUastc(ktxTexture2* This,
                           alpha_content_e alphaContent,
                           ktxTexture2* prototype,
                           ktx_transcode_fmt_e outputFormat,
                           ktx_transcode_flags transcodeFlags)
{
    assert(This->supercompressionScheme != KTX_SS_BASIS_LZ);

    ktx_uint8_t* pXcodedData = prototype->pData;
    ktx_uint32_t outputBlockByteLength
                      = prototype->_protected->_formatSize.blockSizeInBits / 8;
    ktx_size_t xcodedDataLength
                      = prototype->dataSize / outputBlockByteLength;
    DECLARE_PRIVATE(protoPriv, prototype);
    ktxLevelIndexEntry* protoLevelIndex = protoPriv._levelIndex;
    ktx_size_t levelOffsetWrite = 0;

    basisu_lowlevel_uastc_transcoder uit;
    // See comment on same declaration in transcodeEtc1s.
    std::vector<basisu_transcoder_state> xcoderStates;
    xcoderStates.resize(This->isVideo ? This->numFaces : 1);

    for (ktx_int32_t level = This->numLevels - 1; level >= 0; level--)
    {
        ktx_uint32_t depth;
        uint64_t writeOffset = levelOffsetWrite;
        uint64_t writeOffsetBlocks = levelOffsetWrite / outputBlockByteLength;
        ktx_size_t levelImageSizeIn, levelImageOffsetIn;
        ktx_size_t levelImageSizeOut, levelSizeOut;
        ktx_uint32_t levelImageCount;
        uint32_t levelWidth = MAX(1, This->baseWidth >> level);
        uint32_t levelHeight = MAX(1, This->baseHeight >> level);
        // UASTC texel block dimensions
        const uint32_t bw = 4, bh = 4;
        uint32_t levelBlocksX = (levelWidth + (bw - 1)) / bw;
        uint32_t levelBlocksY = (levelHeight + (bh - 1)) / bh;
        uint32_t stateIndex = 0;

        depth = MAX(1, This->baseDepth  >> level);

        levelImageCount = This->numLayers * This->numFaces * depth;
        levelImageSizeIn = ktxTexture_calcImageSize(ktxTexture(This), level,
                                                    KTX_FORMAT_VERSION_TWO);
        levelImageSizeOut = ktxTexture_calcImageSize(ktxTexture(prototype),
                                                     level,
                                                     KTX_FORMAT_VERSION_TWO);

        levelImageOffsetIn = ktxTexture2_levelDataOffset(This, level);
        levelSizeOut = 0;
        bool status;
        for (uint32_t image = 0; image < levelImageCount; image++) {
            basisu_transcoder_state& xcoderState = xcoderStates[stateIndex];
            // See comment before same lines in transcodeEtc1s.
            if (++stateIndex == xcoderStates.size())
                stateIndex = 0;

            status = uit.transcode_image(
                          (transcoder_texture_format)outputFormat,
                          pXcodedData + writeOffset,
                          (uint32_t)(xcodedDataLength - writeOffsetBlocks),
                          This->pData,
                          (uint32_t)This->dataSize,
                          levelBlocksX,
                          levelBlocksY,
                          levelWidth,
                          levelHeight,
                          level,
                          (uint32_t)levelImageOffsetIn,
                          (uint32_t)levelImageSizeIn,
                          transcodeFlags,
                          alphaContent != eNone,
                          This->isVideo, // is_video
                          //imageDesc.imageFlags ^ cSliceDescFlagsFrameIsIFrame,
                          0, // output_row_pitch_in_blocks_or_pixels
                          &xcoderState, // pState
                          0, // output_rows_in_pixels,
                          -1, // channel0
                          -1  // channel1
                          );
            if (!status)
                return KTX_TRANSCODE_FAILED;
            writeOffset += levelImageSizeOut;
            levelSizeOut += levelImageSizeOut;
            levelImageOffsetIn += levelImageSizeIn;
        }
        protoLevelIndex[level].byteOffset = levelOffsetWrite;
        // writeOffset will be equal to total size of the images in the level.
        protoLevelIndex[level].byteLength = levelSizeOut;
        protoLevelIndex[level].uncompressedByteLength = levelSizeOut;
        levelOffsetWrite += levelSizeOut;
    }
    // In case of transcoding to uncompressed.
    levelOffsetWrite = _KTX_PADN(protoPriv._requiredLevelAlignment,
                                 levelOffsetWrite);
    return KTX_SUCCESS;
}
