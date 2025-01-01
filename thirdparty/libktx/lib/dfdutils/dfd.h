/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @~English
 * @brief Header file defining the data format descriptor utilities API.
 */

/*
 * Author: Andrew Garrard
 */

#ifndef _DFD_H_
#define _DFD_H_

#include <KHR/khr_df.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Qualifier suffix to the format, in Vulkan terms. */
enum VkSuffix {
    s_UNORM,   /*!< Unsigned normalized format. */
    s_SNORM,   /*!< Signed normalized format. */
    s_USCALED, /*!< Unsigned scaled format. */
    s_SSCALED, /*!< Signed scaled format. */
    s_UINT,    /*!< Unsigned integer format. */
    s_SINT,    /*!< Signed integer format. */
    s_SFLOAT,  /*!< Signed float format. */
    s_UFLOAT,  /*!< Unsigned float format. */
    s_SRGB,    /*!< sRGB normalized format. */
    s_S10_5    /*!< 2's complement fixed-point; 5 fractional bits. */
};

/** Compression scheme, in Vulkan terms. */
enum VkCompScheme {
    c_BC1_RGB,       /*!< BC1, aka DXT1, no alpha. */
    c_BC1_RGBA,      /*!< BC1, aka DXT1, punch-through alpha. */
    c_BC2,           /*!< BC2, aka DXT2 and DXT3. */
    c_BC3,           /*!< BC3, aka DXT4 and DXT5. */
    c_BC4,           /*!< BC4. */
    c_BC5,           /*!< BC5. */
    c_BC6H,          /*!< BC6h HDR format. */
    c_BC7,           /*!< BC7. */
    c_ETC2_R8G8B8,   /*!< ETC2 no alpha. */
    c_ETC2_R8G8B8A1, /*!< ETC2 punch-through alpha. */
    c_ETC2_R8G8B8A8, /*!< ETC2 independent alpha. */
    c_EAC_R11,       /*!< R11 ETC2 single-channel. */
    c_EAC_R11G11,    /*!< R11G11 ETC2 dual-channel. */
    c_ASTC,          /*!< ASTC. */
    c_ETC1S,         /*!< ETC1S. */
    c_PVRTC,         /*!< PVRTC(1). */
    c_PVRTC2         /*!< PVRTC2. */
};

typedef unsigned int uint32_t;

#if !defined(LIBKTX)
#include <vulkan/vulkan_core.h>
#else
#include "../vkformat_enum.h"
#endif

uint32_t* vk2dfd(enum VkFormat format);
enum VkFormat dfd2vk(uint32_t* dfd);

/* Create a Data Format Descriptor for an unpacked format. */
uint32_t *createDFDUnpacked(int bigEndian, int numChannels, int bytes,
                            int redBlueSwap, enum VkSuffix suffix);

/* Create a Data Format Descriptor for a packed padded format. */
uint32_t *createDFDPackedShifted(int bigEndian, int numChannels,
                                 int bits[], int shiftBits[],
                                 int channels[], enum VkSuffix suffix);

/* Create a Data Format Descriptor for a packed format. */
uint32_t *createDFDPacked(int bigEndian, int numChannels,
                          int bits[], int channels[],
                          enum VkSuffix suffix);

/* Create a Data Format Descriptor for a 4:2:2 format. */
uint32_t *createDFD422(int bigEndian, int numChannels,
                       int bits[], int shiftBits[], int channels[],
                       int position_xs[], int position_ys[],
                       enum VkSuffix suffix);

/* Create a Data Format Descriptor for a compressed format. */
uint32_t *createDFDCompressed(enum VkCompScheme compScheme,
                              int bwidth, int bheight, int bdepth,
                              enum VkSuffix suffix);

/* Create a Data Format Descriptor for a depth/stencil format. */
uint32_t *createDFDDepthStencil(int depthBits,
                                int stencilBits,
                                int sizeBytes);

/* Create a Data Format Descriptor for an alpha-only format */
uint32_t *createDFDAlpha(int bigEndian, int bytes,
                         enum VkSuffix suffix);

/** @brief Result of interpreting the data format descriptor. */
enum InterpretDFDResult {
    i_LITTLE_ENDIAN_FORMAT_BIT = 0, /*!< Confirmed little-endian (default for 8bpc). */
    i_BIG_ENDIAN_FORMAT_BIT    = 1u << 0u, /*!< Confirmed big-endian. */
    i_PACKED_FORMAT_BIT        = 1u << 1u, /*!< Packed format. */
    i_SRGB_FORMAT_BIT          = 1u << 2u, /*!< sRGB transfer function. */
    i_NORMALIZED_FORMAT_BIT    = 1u << 3u, /*!< Normalized (UNORM or SNORM). */
    i_SIGNED_FORMAT_BIT        = 1u << 4u, /*!< Format is signed. */
    i_FIXED_FORMAT_BIT         = 1u << 5u, /*!< Format is a fixed-point representation. */
    i_FLOAT_FORMAT_BIT         = 1u << 6u, /*!< Format is floating point. */
    i_COMPRESSED_FORMAT_BIT    = 1u << 7u, /*!< Format is block compressed (422). */
    i_YUVSDA_FORMAT_BIT        = 1u << 8u, /*!< Color model is YUVSDA. */
    i_UNSUPPORTED_ERROR_BIT    = 1u << 9u, /*!< Format not successfully interpreted. */
    /** "NONTRIVIAL_ENDIANNESS" means not big-endian, not little-endian
     * (a channel has bits that are not consecutive in either order). **/
    i_UNSUPPORTED_NONTRIVIAL_ENDIANNESS     = i_UNSUPPORTED_ERROR_BIT,
    /** "MULTIPLE_SAMPLE_LOCATIONS" is an error because only single-sample
     * texel blocks (with coordinates 0,0,0,0 for all samples) are supported. **/
    i_UNSUPPORTED_MULTIPLE_SAMPLE_LOCATIONS = i_UNSUPPORTED_ERROR_BIT + 1,
    /** "MULTIPLE_PLANES" is an error because only contiguous data is supported. */
    i_UNSUPPORTED_MULTIPLE_PLANES           = i_UNSUPPORTED_ERROR_BIT + 2,
    /** Only channels R, G, B and A are supported. */
    i_UNSUPPORTED_CHANNEL_TYPES             = i_UNSUPPORTED_ERROR_BIT + 3,
    /** Only channels with the same flags are supported
     * (e.g. we don't support float red with integer green). */
    i_UNSUPPORTED_MIXED_CHANNELS            = i_UNSUPPORTED_ERROR_BIT + 4,
    /** Only 2x1 block is supported for YUVSDA model. */
    i_UNSUPPORTED_BLOCK_DIMENSIONS          = i_UNSUPPORTED_ERROR_BIT + 5,
};

/** @brief Interpretation of a channel from the data format descriptor. */
typedef struct _InterpretedDFDChannel {
    uint32_t offset; /*!< Offset in bits for packed, bytes for unpacked. */
    uint32_t size;   /*!< Size in bits for packed, bytes for unpacked. */
} InterpretedDFDChannel;

/* Interpret a Data Format Descriptor. */
enum InterpretDFDResult interpretDFD(const uint32_t *DFD,
                                     InterpretedDFDChannel *R,
                                     InterpretedDFDChannel *G,
                                     InterpretedDFDChannel *B,
                                     InterpretedDFDChannel *A,
                                     uint32_t *wordBytes);

/* Returns the string representation.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringVendorID(khr_df_vendorid_e value);

/* Returns the string representation.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringDescriptorType(khr_df_khr_descriptortype_e value);

/* Returns the string representation.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringVersionNumber(khr_df_versionnumber_e value);

/* Returns the string representation of a bit in a khr_df_flags_e.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringFlagsBit(uint32_t bit_index, bool bit_value);

/* Returns the string representation.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringTransferFunction(khr_df_transfer_e value);

/* Returns the string representation.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringColorPrimaries(khr_df_primaries_e value);

/* Returns the string representation.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringColorModel(khr_df_model_e value);

/* Returns the string representation of a bit in a khr_df_sample_datatype_qualifiers_e.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringSampleDatatypeQualifiersBit(uint32_t bit_index, bool bit_value);

/* Returns the string representation.
 * If there is no direct match or the value is invalid returns NULL */
const char* dfdToStringChannelId(khr_df_model_e model, khr_df_model_channels_e value);

/* Print a human-readable interpretation of a data format descriptor. */
void printDFD(uint32_t *DFD, uint32_t dataSize);

/* Print a JSON interpretation of a data format descriptor. */
void printDFDJSON(uint32_t *DFD, uint32_t dataSize, uint32_t base_indent, uint32_t indent_width, bool minified);

/* Get the number of components & component size from a DFD for an
 * unpacked format.
 */
void
getDFDComponentInfoUnpacked(const uint32_t* DFD, uint32_t* numComponents,
                            uint32_t* componentByteLength);

/* Return the number of components described by a DFD. */
uint32_t getDFDNumComponents(const uint32_t* DFD);

/* Reconstruct and return the value of bytesPlane0 as it should be for the data
 * post-inflation from variable-rate compression.
 */
uint32_t
reconstructDFDBytesPlane0FromSamples(const uint32_t* DFD);
/* Deprecated. For backward compatibility. */
void
recreateBytesPlane0FromSampleInfo(const uint32_t* DFD, uint32_t* bytesPlane0);

/** @brief Colourspace primaries information.
 *
 * Structure to store the 1931 CIE x,y chromaticities of the red, green, and blue
 * display primaries and the reference white point of a colourspace.
 */
typedef struct _Primaries {
    float Rx; /*!< Red x. */
    float Ry; /*!< Red y. */
    float Gx; /*!< Green x. */
    float Gy; /*!< Green y. */
    float Bx; /*!< Blue x. */
    float By; /*!< Blue y. */
    float Wx; /*!< White x. */
    float Wy; /*!< White y. */
} Primaries;

khr_df_primaries_e findMapping(const Primaries *p, float latitude);
bool getPrimaries(khr_df_primaries_e primaries, Primaries *p);

#ifdef __cplusplus
}
#endif

#endif /* _DFD_H_ */
