/* The Khronos Data Format Specification (version 1.4.0) */
/*
** Copyright 2015-2025 The Khronos Group Inc.
** SPDX-License-Identifier: Apache-2.0
*/

/* This header defines a structure that can describe the layout of image
   formats in memory. This means that the data format is transparent to
   the application, and the expectation is that this should be used when
   the layout is defined external to the API. Many Khronos APIs deliberately
   keep the internal layout of images opaque, to allow proprietary layouts
   and optimisations. This structure is not appropriate for describing
   opaque layouts. */

/* We stick to standard C89 constructs for simplicity and portability. */

#ifndef _KHR_DATA_FORMAT_H_
#define _KHR_DATA_FORMAT_H_

/* Accessors */
typedef enum _khr_word_e {
    KHR_DF_WORD_VENDORID = 0U,
    KHR_DF_WORD_DESCRIPTORTYPE = 0U,
    KHR_DF_WORD_VERSIONNUMBER = 1U,
    KHR_DF_WORD_DESCRIPTORBLOCKSIZE = 1U,
    KHR_DF_WORD_MODEL = 2U,
    KHR_DF_WORD_PRIMARIES = 2U,
    KHR_DF_WORD_TRANSFER = 2U,
    KHR_DF_WORD_FLAGS = 2U,
    KHR_DF_WORD_TEXELBLOCKDIMENSION0 = 3U,
    KHR_DF_WORD_TEXELBLOCKDIMENSION1 = 3U,
    KHR_DF_WORD_TEXELBLOCKDIMENSION2 = 3U,
    KHR_DF_WORD_TEXELBLOCKDIMENSION3 = 3U,
    KHR_DF_WORD_BYTESPLANE0 = 4U,
    KHR_DF_WORD_BYTESPLANE1 = 4U,
    KHR_DF_WORD_BYTESPLANE2 = 4U,
    KHR_DF_WORD_BYTESPLANE3 = 4U,
    KHR_DF_WORD_BYTESPLANE4 = 5U,
    KHR_DF_WORD_BYTESPLANE5 = 5U,
    KHR_DF_WORD_BYTESPLANE6 = 5U,
    KHR_DF_WORD_BYTESPLANE7 = 5U,
    KHR_DF_WORD_SAMPLESTART = 6U,
    KHR_DF_WORD_SAMPLEWORDS = 4U
} khr_df_word_e;

typedef enum _khr_df_shift_e {
    KHR_DF_SHIFT_VENDORID = 0U,
    KHR_DF_SHIFT_DESCRIPTORTYPE = 17U,
    KHR_DF_SHIFT_VERSIONNUMBER = 0U,
    KHR_DF_SHIFT_DESCRIPTORBLOCKSIZE = 16U,
    KHR_DF_SHIFT_MODEL = 0U,
    KHR_DF_SHIFT_PRIMARIES = 8U,
    KHR_DF_SHIFT_TRANSFER = 16U,
    KHR_DF_SHIFT_FLAGS = 24U,
    KHR_DF_SHIFT_TEXELBLOCKDIMENSION0 = 0U,
    KHR_DF_SHIFT_TEXELBLOCKDIMENSION1 = 8U,
    KHR_DF_SHIFT_TEXELBLOCKDIMENSION2 = 16U,
    KHR_DF_SHIFT_TEXELBLOCKDIMENSION3 = 24U,
    KHR_DF_SHIFT_BYTESPLANE0 = 0U,
    KHR_DF_SHIFT_BYTESPLANE1 = 8U,
    KHR_DF_SHIFT_BYTESPLANE2 = 16U,
    KHR_DF_SHIFT_BYTESPLANE3 = 24U,
    KHR_DF_SHIFT_BYTESPLANE4 = 0U,
    KHR_DF_SHIFT_BYTESPLANE5 = 8U,
    KHR_DF_SHIFT_BYTESPLANE6 = 16U,
    KHR_DF_SHIFT_BYTESPLANE7 = 24U
} khr_df_shift_e;

typedef enum _khr_df_mask_e {
    KHR_DF_MASK_VENDORID = 0x1FFFFU,
    KHR_DF_MASK_DESCRIPTORTYPE = 0x7FFFU,
    KHR_DF_MASK_VERSIONNUMBER = 0xFFFFU,
    KHR_DF_MASK_DESCRIPTORBLOCKSIZE = 0xFFFFU,
    KHR_DF_MASK_MODEL = 0xFFU,
    KHR_DF_MASK_PRIMARIES = 0xFFU,
    KHR_DF_MASK_TRANSFER = 0xFFU,
    KHR_DF_MASK_FLAGS = 0xFFU,
    KHR_DF_MASK_TEXELBLOCKDIMENSION0 = 0xFFU,
    KHR_DF_MASK_TEXELBLOCKDIMENSION1 = 0xFFU,
    KHR_DF_MASK_TEXELBLOCKDIMENSION2 = 0xFFU,
    KHR_DF_MASK_TEXELBLOCKDIMENSION3 = 0xFFU,
    KHR_DF_MASK_BYTESPLANE0 = 0xFFU,
    KHR_DF_MASK_BYTESPLANE1 = 0xFFU,
    KHR_DF_MASK_BYTESPLANE2 = 0xFFU,
    KHR_DF_MASK_BYTESPLANE3 = 0xFFU,
    KHR_DF_MASK_BYTESPLANE4 = 0xFFU,
    KHR_DF_MASK_BYTESPLANE5 = 0xFFU,
    KHR_DF_MASK_BYTESPLANE6 = 0xFFU,
    KHR_DF_MASK_BYTESPLANE7 = 0xFFU
} khr_df_mask_e;

/* Helper macro:
   Extract field X from basic descriptor block BDB */
#define KHR_DFDVAL(BDB, X) \
    (((BDB)[KHR_DF_WORD_ ## X] >> (KHR_DF_SHIFT_ ## X)) \
     & (KHR_DF_MASK_ ## X))

/* Helper macro:
   Set field X of basic descriptor block BDB */
#define KHR_DFDSETVAL(BDB, X, val) \
    ((BDB)[KHR_DF_WORD_ ## X] = \
     ((BDB)[KHR_DF_WORD_ ## X] & \
      ~((KHR_DF_MASK_ ## X) << (KHR_DF_SHIFT_ ## X))) | \
     (((uint32_t)(val) & (KHR_DF_MASK_ ## X)) << (KHR_DF_SHIFT_ ## X)))

/* Offsets relative to the start of a sample */
typedef enum _khr_df_sampleword_e {
    KHR_DF_SAMPLEWORD_BITOFFSET = 0U,
    KHR_DF_SAMPLEWORD_BITLENGTH = 0U,
    KHR_DF_SAMPLEWORD_CHANNELID = 0U,
    KHR_DF_SAMPLEWORD_QUALIFIERS = 0U,
    KHR_DF_SAMPLEWORD_SAMPLEPOSITION0 = 1U,
    KHR_DF_SAMPLEWORD_SAMPLEPOSITION1 = 1U,
    KHR_DF_SAMPLEWORD_SAMPLEPOSITION2 = 1U,
    KHR_DF_SAMPLEWORD_SAMPLEPOSITION3 = 1U,
    KHR_DF_SAMPLEWORD_SAMPLEPOSITION_ALL = 1U,
    KHR_DF_SAMPLEWORD_SAMPLELOWER = 2U,
    KHR_DF_SAMPLEWORD_SAMPLEUPPER = 3U
} khr_df_sampleword_e;

typedef enum _khr_df_sampleshift_e {
    KHR_DF_SAMPLESHIFT_BITOFFSET = 0U,
    KHR_DF_SAMPLESHIFT_BITLENGTH = 16U,
    KHR_DF_SAMPLESHIFT_CHANNELID = 24U,
    /* N.B. Qualifiers are defined as an offset into a byte */
    KHR_DF_SAMPLESHIFT_QUALIFIERS = 24U,
    KHR_DF_SAMPLESHIFT_SAMPLEPOSITION0 = 0U,
    KHR_DF_SAMPLESHIFT_SAMPLEPOSITION1 = 8U,
    KHR_DF_SAMPLESHIFT_SAMPLEPOSITION2 = 16U,
    KHR_DF_SAMPLESHIFT_SAMPLEPOSITION3 = 24U,
    KHR_DF_SAMPLESHIFT_SAMPLEPOSITION_ALL = 0U,
    KHR_DF_SAMPLESHIFT_SAMPLELOWER = 0U,
    KHR_DF_SAMPLESHIFT_SAMPLEUPPER = 0U
} khr_df_sampleshift_e;

typedef enum _khr_df_samplemask_e {
    KHR_DF_SAMPLEMASK_BITOFFSET = 0xFFFFU,
    KHR_DF_SAMPLEMASK_BITLENGTH = 0xFFU,
    KHR_DF_SAMPLEMASK_CHANNELID = 0xFU,
    /* N.B. Qualifiers are defined as an offset into a byte */
    KHR_DF_SAMPLEMASK_QUALIFIERS = 0xF0U,
    KHR_DF_SAMPLEMASK_SAMPLEPOSITION0 = 0xFFU,
    KHR_DF_SAMPLEMASK_SAMPLEPOSITION1 = 0xFFU,
    KHR_DF_SAMPLEMASK_SAMPLEPOSITION2 = 0xFFU,
    KHR_DF_SAMPLEMASK_SAMPLEPOSITION3 = 0xFFU,
    /* ISO C restricts enum values to range of int hence the
       cast. We do it verbosely instead of using -1 to ensure
       it is a 32-bit value even if int is 64 bits. */
    KHR_DF_SAMPLEMASK_SAMPLEPOSITION_ALL = (int) 0xFFFFFFFFU,
    KHR_DF_SAMPLEMASK_SAMPLELOWER = (int) 0xFFFFFFFFU,
    KHR_DF_SAMPLEMASK_SAMPLEUPPER = (int) 0xFFFFFFFFU
} khr_df_samplemask_e;

/* Helper macro:
   Extract field X of sample S from basic descriptor block BDB */
#define KHR_DFDSVAL(BDB, S, X) \
    (((BDB)[KHR_DF_WORD_SAMPLESTART + \
            ((S) * KHR_DF_WORD_SAMPLEWORDS) + \
            KHR_DF_SAMPLEWORD_ ## X] >> (KHR_DF_SAMPLESHIFT_ ## X)) \
     & (KHR_DF_SAMPLEMASK_ ## X))

/* Helper macro:
   Set field X of sample S of basic descriptor block BDB */
#define KHR_DFDSETSVAL(BDB, S, X, val) \
    ((BDB)[KHR_DF_WORD_SAMPLESTART + \
           ((S) * KHR_DF_WORD_SAMPLEWORDS) + \
           KHR_DF_SAMPLEWORD_ ## X] = \
     ((BDB)[KHR_DF_WORD_SAMPLESTART + \
            ((S) * KHR_DF_WORD_SAMPLEWORDS) + \
            KHR_DF_SAMPLEWORD_ ## X] & \
      ~((uint32_t)(KHR_DF_SAMPLEMASK_ ## X) << (KHR_DF_SAMPLESHIFT_ ## X))) | \
     (((uint32_t)(val) & (uint32_t)(KHR_DF_SAMPLEMASK_ ## X)) << (KHR_DF_SAMPLESHIFT_ ## X)))

/* Helper macro:
   Number of samples in basic descriptor block BDB */
#define KHR_DFDSAMPLECOUNT(BDB) \
    (((KHR_DFDVAL(BDB, DESCRIPTORBLOCKSIZE) >> 2) - \
      KHR_DF_WORD_SAMPLESTART) \
     / KHR_DF_WORD_SAMPLEWORDS)

/* Helper macro:
   Size in words of basic descriptor block for S samples */
#define KHR_DFDSIZEWORDS(S) \
    (KHR_DF_WORD_SAMPLESTART + \
     (S) * KHR_DF_WORD_SAMPLEWORDS)

/* Vendor ids */
typedef enum _khr_df_vendorid_e {
    /* Standard Khronos descriptor */
    KHR_DF_VENDORID_KHRONOS = 0U,
    KHR_DF_VENDORID_MAX     = 0x1FFFFU
} khr_df_vendorid_e;

/* Descriptor types */
typedef enum _khr_df_khr_descriptortype_e {
    /* Default Khronos basic descriptor block */
    KHR_DF_KHR_DESCRIPTORTYPE_BASICFORMAT = 0U,
    /* Extension descriptor block for additional planes */
    KHR_DF_KHR_DESCRIPTORTYPE_ADDITIONAL_PLANES = 0x6001U,
    /* Extension descriptor block for additional dimensions */
    KHR_DF_KHR_DESCRIPTORTYPE_ADDITIONAL_DIMENSIONS = 0x6002U,
    /* Bit indicates modifying requires understanding this extension */
    KHR_DF_KHR_DESCRIPTORTYPE_NEEDED_FOR_WRITE_BIT = 0x2000U,
    /* Bit indicates processing requires understanding this extension */
    KHR_DF_KHR_DESCRIPTORTYPE_NEEDED_FOR_DECODE_BIT = 0x4000U,
    KHR_DF_KHR_DESCRIPTORTYPE_MAX         = 0x7FFFU
} khr_df_khr_descriptortype_e;

/* Descriptor block version */
typedef enum _khr_df_versionnumber_e {
    /* Standard Khronos descriptor */
    KHR_DF_VERSIONNUMBER_1_0 = 0U, /* Version 1.0 of the specification */
    KHR_DF_VERSIONNUMBER_1_1 = 0U, /* Version 1.1 did not bump the version number */
    KHR_DF_VERSIONNUMBER_1_2 = 1U, /* Version 1.2 increased the version number */
    KHR_DF_VERSIONNUMBER_1_3 = 2U, /* Version 1.3 increased the version number */
    KHR_DF_VERSIONNUMBER_1_4 = 2U, /* Version 1.4.0 did not bump the block version number */
    KHR_DF_VERSIONNUMBER_LATEST = KHR_DF_VERSIONNUMBER_1_4,
    KHR_DF_VERSIONNUMBER_MAX = 0xFFFFU
} khr_df_versionnumber_e;

/* Model in which the color coordinate space is defined.
   There is no requirement that a color format use all the
   channel types that are defined in the color model. */
typedef enum _khr_df_model_e {
    /* No interpretation of color channels defined */
    KHR_DF_MODEL_UNSPECIFIED  = 0U,
    /* Color primaries (red, green, blue) + alpha, depth and stencil */
    KHR_DF_MODEL_RGBSDA       = 1U,
    /* Color differences (Y', Cb, Cr) + alpha, depth and stencil */
    KHR_DF_MODEL_YUVSDA       = 2U,
    /* Color differences (Y', I, Q) + alpha, depth and stencil */
    KHR_DF_MODEL_YIQSDA       = 3U,
    /* Perceptual color (CIE L*a*b*) + alpha, depth and stencil */
    KHR_DF_MODEL_LABSDA       = 4U,
    /* Subtractive colors (cyan, magenta, yellow, black) + alpha */
    KHR_DF_MODEL_CMYKA        = 5U,
    /* Non-color coordinate data (X, Y, Z, W) */
    KHR_DF_MODEL_XYZW         = 6U,
    /* Hue, saturation, value, hue angle on color circle, plus alpha */
    KHR_DF_MODEL_HSVA_ANG     = 7U,
    /* Hue, saturation, lightness, hue angle on color circle, plus alpha */
    KHR_DF_MODEL_HSLA_ANG     = 8U,
    /* Hue, saturation, value, hue on color hexagon, plus alpha */
    KHR_DF_MODEL_HSVA_HEX     = 9U,
    /* Hue, saturation, lightness, hue on color hexagon, plus alpha */
    KHR_DF_MODEL_HSLA_HEX     = 10U,
    /* Lightweight approximate color difference (luma, orange, green) */
    KHR_DF_MODEL_YCGCOA       = 11U,
    /* ITU BT.2020 constant luminance YcCbcCrc */
    KHR_DF_MODEL_YCCBCCRC     = 12U,
    /* ITU BT.2100 constant intensity ICtCp */
    KHR_DF_MODEL_ICTCP        = 13U,
    /* CIE 1931 XYZ color coordinates (X, Y, Z) */
    KHR_DF_MODEL_CIEXYZ       = 14U,
    /* CIE 1931 xyY color coordinates (X, Y, Y) */
    KHR_DF_MODEL_CIEXYY       = 15U,

    /* Compressed formats start at 128. */
    /* These compressed formats should generally have a single sample,
       sited at the 0,0 position of the texel block. Where multiple
       channels are used to distinguish formats, these should be cosited. */
    /* Direct3D (and S3) compressed formats */
    /* Note that premultiplied status is recorded separately */
    /* DXT1 "channels" are RGB (0), Alpha (1) */
    /* DXT1/BC1 with one channel is opaque */
    /* DXT1/BC1 with a cosited alpha sample is transparent */
    KHR_DF_MODEL_DXT1A         = 128U,
    KHR_DF_MODEL_BC1A          = 128U,
    /* DXT2/DXT3/BC2, with explicit 4-bit alpha */
    KHR_DF_MODEL_DXT2          = 129U,
    KHR_DF_MODEL_DXT3          = 129U,
    KHR_DF_MODEL_BC2           = 129U,
    /* DXT4/DXT5/BC3, with interpolated alpha */
    KHR_DF_MODEL_DXT4          = 130U,
    KHR_DF_MODEL_DXT5          = 130U,
    KHR_DF_MODEL_BC3           = 130U,
    /* ATI1n/DXT5A/BC4 - single channel interpolated 8-bit data */
    /* (The UNORM/SNORM variation is recorded in the channel data) */
    KHR_DF_MODEL_ATI1N         = 131U,
    KHR_DF_MODEL_DXT5A         = 131U,
    KHR_DF_MODEL_BC4           = 131U,
    /* ATI2n_XY/DXN/BC5 - two channel interpolated 8-bit data */
    /* (The UNORM/SNORM variation is recorded in the channel data) */
    KHR_DF_MODEL_ATI2N_XY      = 132U,
    KHR_DF_MODEL_DXN           = 132U,
    KHR_DF_MODEL_BC5           = 132U,
    /* BC6H - DX11 format for 16-bit float channels */
    KHR_DF_MODEL_BC6H          = 133U,
    /* BC7 - DX11 format */
    KHR_DF_MODEL_BC7           = 134U,
    /* Gap left for future desktop expansion */

    /* Mobile compressed formats follow */
    /* A format of ETC1 indicates that the format shall be decodable
       by an ETC1-compliant decoder and not rely on ETC2 features */
    KHR_DF_MODEL_ETC1          = 160U,
    /* A format of ETC2 is permitted to use ETC2 encodings on top of
       the baseline ETC1 specification */
    /* The ETC2 format has channels "red", "green", "RGB" and "alpha",
       which should be cosited samples */
    /* Punch-through alpha can be distinguished from full alpha by
       the plane size in bytes required for the texel block */
    KHR_DF_MODEL_ETC2          = 161U,
    /* Adaptive Scalable Texture Compression */
    /* ASTC HDR vs LDR is determined by the float flag in the channel */
    /* ASTC block size can be distinguished by texel block size */
    KHR_DF_MODEL_ASTC          = 162U,
    /* ETC1S is a simplified subset of ETC1 */
    KHR_DF_MODEL_ETC1S         = 163U,
    /* PowerVR Texture Compression */
    KHR_DF_MODEL_PVRTC         = 164U,
    KHR_DF_MODEL_PVRTC2        = 165U,
    KHR_DF_MODEL_UASTC         = 166U,
    /* Proprietary formats (ATITC, etc.) should follow */
    KHR_DF_MODEL_MAX = 0xFFU
} khr_df_model_e;

/* Definition of channel names for each color model */
typedef enum _khr_df_model_channels_e {
    /* Unspecified format with nominal channel numbering */
    KHR_DF_CHANNEL_UNSPECIFIED_0  = 0U,
    KHR_DF_CHANNEL_UNSPECIFIED_1  = 1U,
    KHR_DF_CHANNEL_UNSPECIFIED_2  = 2U,
    KHR_DF_CHANNEL_UNSPECIFIED_3  = 3U,
    KHR_DF_CHANNEL_UNSPECIFIED_4  = 4U,
    KHR_DF_CHANNEL_UNSPECIFIED_5  = 5U,
    KHR_DF_CHANNEL_UNSPECIFIED_6  = 6U,
    KHR_DF_CHANNEL_UNSPECIFIED_7  = 7U,
    KHR_DF_CHANNEL_UNSPECIFIED_8  = 8U,
    KHR_DF_CHANNEL_UNSPECIFIED_9  = 9U,
    KHR_DF_CHANNEL_UNSPECIFIED_10 = 10U,
    KHR_DF_CHANNEL_UNSPECIFIED_11 = 11U,
    KHR_DF_CHANNEL_UNSPECIFIED_12 = 12U,
    KHR_DF_CHANNEL_UNSPECIFIED_13 = 13U,
    KHR_DF_CHANNEL_UNSPECIFIED_14 = 14U,
    KHR_DF_CHANNEL_UNSPECIFIED_15 = 15U,
    /* MODEL_RGBSDA - red, green, blue, stencil, depth, alpha */
    KHR_DF_CHANNEL_RGBSDA_RED     =  0U,
    KHR_DF_CHANNEL_RGBSDA_R       =  0U,
    KHR_DF_CHANNEL_RGBSDA_GREEN   =  1U,
    KHR_DF_CHANNEL_RGBSDA_G       =  1U,
    KHR_DF_CHANNEL_RGBSDA_BLUE    =  2U,
    KHR_DF_CHANNEL_RGBSDA_B       =  2U,
    KHR_DF_CHANNEL_RGBSDA_STENCIL = 13U,
    KHR_DF_CHANNEL_RGBSDA_S       = 13U,
    KHR_DF_CHANNEL_RGBSDA_DEPTH   = 14U,
    KHR_DF_CHANNEL_RGBSDA_D       = 14U,
    KHR_DF_CHANNEL_RGBSDA_ALPHA   = 15U,
    KHR_DF_CHANNEL_RGBSDA_A       = 15U,
    /* MODEL_YUVSDA - luma, Cb, Cr, stencil, depth, alpha */
    KHR_DF_CHANNEL_YUVSDA_Y       =  0U,
    KHR_DF_CHANNEL_YUVSDA_CB      =  1U,
    KHR_DF_CHANNEL_YUVSDA_U       =  1U,
    KHR_DF_CHANNEL_YUVSDA_CR      =  2U,
    KHR_DF_CHANNEL_YUVSDA_V       =  2U,
    KHR_DF_CHANNEL_YUVSDA_STENCIL = 13U,
    KHR_DF_CHANNEL_YUVSDA_S       = 13U,
    KHR_DF_CHANNEL_YUVSDA_DEPTH   = 14U,
    KHR_DF_CHANNEL_YUVSDA_D       = 14U,
    KHR_DF_CHANNEL_YUVSDA_ALPHA   = 15U,
    KHR_DF_CHANNEL_YUVSDA_A       = 15U,
    /* MODEL_YIQSDA - luma, in-phase, quadrature, stencil, depth, alpha */
    KHR_DF_CHANNEL_YIQSDA_Y       =  0U,
    KHR_DF_CHANNEL_YIQSDA_I       =  1U,
    KHR_DF_CHANNEL_YIQSDA_Q       =  2U,
    KHR_DF_CHANNEL_YIQSDA_STENCIL = 13U,
    KHR_DF_CHANNEL_YIQSDA_S       = 13U,
    KHR_DF_CHANNEL_YIQSDA_DEPTH   = 14U,
    KHR_DF_CHANNEL_YIQSDA_D       = 14U,
    KHR_DF_CHANNEL_YIQSDA_ALPHA   = 15U,
    KHR_DF_CHANNEL_YIQSDA_A       = 15U,
    /* MODEL_LABSDA - CIELAB/L*a*b* luma, red-green, blue-yellow, stencil, depth, alpha */
    KHR_DF_CHANNEL_LABSDA_L       =  0U,
    KHR_DF_CHANNEL_LABSDA_A       =  1U,
    KHR_DF_CHANNEL_LABSDA_B       =  2U,
    KHR_DF_CHANNEL_LABSDA_STENCIL = 13U,
    KHR_DF_CHANNEL_LABSDA_S       = 13U,
    KHR_DF_CHANNEL_LABSDA_DEPTH   = 14U,
    KHR_DF_CHANNEL_LABSDA_D       = 14U,
    KHR_DF_CHANNEL_LABSDA_ALPHA   = 15U,
    /* NOTE: KHR_DF_CHANNEL_LABSDA_A is not a synonym for alpha! */
    /* MODEL_CMYKA - cyan, magenta, yellow, key/blacK, alpha */
    KHR_DF_CHANNEL_CMYKSDA_CYAN    =  0U,
    KHR_DF_CHANNEL_CMYKSDA_C       =  0U,
    KHR_DF_CHANNEL_CMYKSDA_MAGENTA =  1U,
    KHR_DF_CHANNEL_CMYKSDA_M       =  1U,
    KHR_DF_CHANNEL_CMYKSDA_YELLOW  =  2U,
    KHR_DF_CHANNEL_CMYKSDA_Y       =  2U,
    KHR_DF_CHANNEL_CMYKSDA_KEY     =  3U,
    KHR_DF_CHANNEL_CMYKSDA_BLACK   =  3U,
    KHR_DF_CHANNEL_CMYKSDA_K       =  3U,
    KHR_DF_CHANNEL_CMYKSDA_ALPHA   = 15U,
    KHR_DF_CHANNEL_CMYKSDA_A       = 15U,
    /* MODEL_XYZW - coordinates x, y, z, w */
    KHR_DF_CHANNEL_XYZW_X = 0U,
    KHR_DF_CHANNEL_XYZW_Y = 1U,
    KHR_DF_CHANNEL_XYZW_Z = 2U,
    KHR_DF_CHANNEL_XYZW_W = 3U,
    /* MODEL_HSVA_ANG - value (luma), saturation, hue, alpha, angular projection, conical space */
    KHR_DF_CHANNEL_HSVA_ANG_VALUE      = 0U,
    KHR_DF_CHANNEL_HSVA_ANG_V          = 0U,
    KHR_DF_CHANNEL_HSVA_ANG_SATURATION = 1U,
    KHR_DF_CHANNEL_HSVA_ANG_S          = 1U,
    KHR_DF_CHANNEL_HSVA_ANG_HUE        = 2U,
    KHR_DF_CHANNEL_HSVA_ANG_H          = 2U,
    KHR_DF_CHANNEL_HSVA_ANG_ALPHA      = 15U,
    KHR_DF_CHANNEL_HSVA_ANG_A          = 15U,
    /* MODEL_HSLA_ANG - lightness (luma), saturation, hue, alpha, angular projection, double conical space */
    KHR_DF_CHANNEL_HSLA_ANG_LIGHTNESS  = 0U,
    KHR_DF_CHANNEL_HSLA_ANG_L          = 0U,
    KHR_DF_CHANNEL_HSLA_ANG_SATURATION = 1U,
    KHR_DF_CHANNEL_HSLA_ANG_S          = 1U,
    KHR_DF_CHANNEL_HSLA_ANG_HUE        = 2U,
    KHR_DF_CHANNEL_HSLA_ANG_H          = 2U,
    KHR_DF_CHANNEL_HSLA_ANG_ALPHA      = 15U,
    KHR_DF_CHANNEL_HSLA_ANG_A          = 15U,
    /* MODEL_HSVA_HEX - value (luma), saturation, hue, alpha, hexagonal projection, conical space */
    KHR_DF_CHANNEL_HSVA_HEX_VALUE      = 0U,
    KHR_DF_CHANNEL_HSVA_HEX_V          = 0U,
    KHR_DF_CHANNEL_HSVA_HEX_SATURATION = 1U,
    KHR_DF_CHANNEL_HSVA_HEX_S          = 1U,
    KHR_DF_CHANNEL_HSVA_HEX_HUE        = 2U,
    KHR_DF_CHANNEL_HSVA_HEX_H          = 2U,
    KHR_DF_CHANNEL_HSVA_HEX_ALPHA      = 15U,
    KHR_DF_CHANNEL_HSVA_HEX_A          = 15U,
    /* MODEL_HSLA_HEX - lightness (luma), saturation, hue, alpha, hexagonal projection, double conical space */
    KHR_DF_CHANNEL_HSLA_HEX_LIGHTNESS  = 0U,
    KHR_DF_CHANNEL_HSLA_HEX_L          = 0U,
    KHR_DF_CHANNEL_HSLA_HEX_SATURATION = 1U,
    KHR_DF_CHANNEL_HSLA_HEX_S          = 1U,
    KHR_DF_CHANNEL_HSLA_HEX_HUE        = 2U,
    KHR_DF_CHANNEL_HSLA_HEX_H          = 2U,
    KHR_DF_CHANNEL_HSLA_HEX_ALPHA      = 15U,
    KHR_DF_CHANNEL_HSLA_HEX_A          = 15U,
    /* MODEL_YCGCOA - luma, green delta, orange delta, alpha */
    KHR_DF_CHANNEL_YCGCOA_Y       =  0U,
    KHR_DF_CHANNEL_YCGCOA_CG      =  1U,
    KHR_DF_CHANNEL_YCGCOA_CO      =  2U,
    KHR_DF_CHANNEL_YCGCOA_ALPHA   = 15U,
    KHR_DF_CHANNEL_YCGCOA_A       = 15U,
    /* MODEL_CIEXYZ - CIE 1931 X, Y, Z */
    KHR_DF_CHANNEL_CIEXYZ_X = 0U,
    KHR_DF_CHANNEL_CIEXYZ_Y = 1U,
    KHR_DF_CHANNEL_CIEXYZ_Z = 2U,
    /* MODEL_CIEXYY - CIE 1931 x, y, Y */
    KHR_DF_CHANNEL_CIEXYY_X        = 0U,
    KHR_DF_CHANNEL_CIEXYY_YCHROMA  = 1U,
    KHR_DF_CHANNEL_CIEXYY_YLUMA    = 2U,

    /* Compressed formats */
    /* MODEL_DXT1A/MODEL_BC1A */
    KHR_DF_CHANNEL_DXT1A_COLOR = 0U,
    KHR_DF_CHANNEL_BC1A_COLOR  = 0U,
    KHR_DF_CHANNEL_DXT1A_ALPHAPRESENT = 1U,
    KHR_DF_CHANNEL_DXT1A_ALPHA = 1U,
    KHR_DF_CHANNEL_BC1A_ALPHAPRESENT  = 1U,
    KHR_DF_CHANNEL_BC1A_ALPHA  = 1U,
    /* MODEL_DXT2/3/MODEL_BC2 */
    KHR_DF_CHANNEL_DXT2_COLOR =  0U,
    KHR_DF_CHANNEL_DXT3_COLOR =  0U,
    KHR_DF_CHANNEL_BC2_COLOR  =  0U,
    KHR_DF_CHANNEL_DXT2_ALPHA = 15U,
    KHR_DF_CHANNEL_DXT3_ALPHA = 15U,
    KHR_DF_CHANNEL_BC2_ALPHA  = 15U,
    /* MODEL_DXT4/5/MODEL_BC3 */
    KHR_DF_CHANNEL_DXT4_COLOR =  0U,
    KHR_DF_CHANNEL_DXT5_COLOR =  0U,
    KHR_DF_CHANNEL_BC3_COLOR  =  0U,
    KHR_DF_CHANNEL_DXT4_ALPHA = 15U,
    KHR_DF_CHANNEL_DXT5_ALPHA = 15U,
    KHR_DF_CHANNEL_BC3_ALPHA  = 15U,
    /* MODEL_BC4 */
    KHR_DF_CHANNEL_BC4_DATA = 0U,
    /* MODEL_BC5 */
    KHR_DF_CHANNEL_BC5_RED   = 0U,
    KHR_DF_CHANNEL_BC5_R     = 0U,
    KHR_DF_CHANNEL_BC5_GREEN = 1U,
    KHR_DF_CHANNEL_BC5_G     = 1U,
    /* MODEL_BC6H */
    KHR_DF_CHANNEL_BC6H_COLOR = 0U,
    KHR_DF_CHANNEL_BC6H_DATA = 0U,
    /* MODEL_BC7 */
    KHR_DF_CHANNEL_BC7_DATA = 0U,
    KHR_DF_CHANNEL_BC7_COLOR = 0U,
    /* MODEL_ETC1 */
    KHR_DF_CHANNEL_ETC1_DATA  = 0U,
    KHR_DF_CHANNEL_ETC1_COLOR = 0U,
    /* MODEL_ETC2 */
    KHR_DF_CHANNEL_ETC2_RED   = 0U,
    KHR_DF_CHANNEL_ETC2_R     = 0U,
    KHR_DF_CHANNEL_ETC2_GREEN = 1U,
    KHR_DF_CHANNEL_ETC2_G     = 1U,
    KHR_DF_CHANNEL_ETC2_COLOR = 2U,
    KHR_DF_CHANNEL_ETC2_ALPHA = 15U,
    KHR_DF_CHANNEL_ETC2_A     = 15U,
    /* MODEL_ASTC */
    KHR_DF_CHANNEL_ASTC_DATA  = 0U,
    /* MODEL_ETC1S */
    KHR_DF_CHANNEL_ETC1S_RGB   = 0U,
    KHR_DF_CHANNEL_ETC1S_RRR   = 3U,
    KHR_DF_CHANNEL_ETC1S_GGG   = 4U,
    KHR_DF_CHANNEL_ETC1S_AAA   = 15U,
    /* MODEL_PVRTC */
    KHR_DF_CHANNEL_PVRTC_DATA  = 0U,
    KHR_DF_CHANNEL_PVRTC_COLOR = 0U,
    /* MODEL_PVRTC2 */
    KHR_DF_CHANNEL_PVRTC2_DATA  = 0U,
    KHR_DF_CHANNEL_PVRTC2_COLOR = 0U,
    /* MODEL UASTC */
    KHR_DF_CHANNEL_UASTC_RGB   = 0U,
    KHR_DF_CHANNEL_UASTC_RGBA  = 3U,
    KHR_DF_CHANNEL_UASTC_RRR   = 4U,
    KHR_DF_CHANNEL_UASTC_RRRG  = 5U,
    KHR_DF_CHANNEL_UASTC_RG    = 6U,

    /* Common channel names shared by multiple formats */
    KHR_DF_CHANNEL_COMMON_LUMA    =  0U,
    KHR_DF_CHANNEL_COMMON_L       =  0U,
    KHR_DF_CHANNEL_COMMON_STENCIL = 13U,
    KHR_DF_CHANNEL_COMMON_S       = 13U,
    KHR_DF_CHANNEL_COMMON_DEPTH   = 14U,
    KHR_DF_CHANNEL_COMMON_D       = 14U,
    KHR_DF_CHANNEL_COMMON_ALPHA   = 15U,
    KHR_DF_CHANNEL_COMMON_A       = 15U
} khr_df_model_channels_e;

/* Definition of the primary colors in color coordinates.
   This is implicitly responsible for defining the conversion
   between RGB an YUV color spaces.
   LAB and related absolute color models should use
   KHR_DF_PRIMARIES_CIEXYZ. */
typedef enum _khr_df_primaries_e {
    /* No color primaries defined */
    KHR_DF_PRIMARIES_UNSPECIFIED = 0U,
    /* Color primaries of ITU-R BT.709 and sRGB */
    KHR_DF_PRIMARIES_BT709       = 1U,
    /* Synonym for KHR_DF_PRIMARIES_BT709 */
    KHR_DF_PRIMARIES_SRGB        = 1U,
    /* Color primaries of ITU-R BT.601 (625-line EBU variant) */
    KHR_DF_PRIMARIES_BT601_EBU   = 2U,
    /* Color primaries of ITU-R BT.601 (525-line SMPTE C variant) */
    KHR_DF_PRIMARIES_BT601_SMPTE = 3U,
    /* Color primaries of ITU-R BT.2020 */
    KHR_DF_PRIMARIES_BT2020      = 4U,
    /* ITU-R BT.2100 uses the same primaries as BT.2020 */
    KHR_DF_PRIMARIES_BT2100      = 4U,
    /* CIE theoretical color coordinate space */
    KHR_DF_PRIMARIES_CIEXYZ      = 5U,
    /* Academy Color Encoding System primaries */
    KHR_DF_PRIMARIES_ACES        = 6U,
    /* Color primaries of ACEScc */
    KHR_DF_PRIMARIES_ACESCC      = 7U,
    /* Legacy NTSC 1953 primaries */
    KHR_DF_PRIMARIES_NTSC1953    = 8U,
    /* Legacy PAL 525-line primaries */
    KHR_DF_PRIMARIES_PAL525      = 9U,
    /* Color primaries of Display P3 */
    KHR_DF_PRIMARIES_DISPLAYP3   = 10U,
    /* Color primaries of Adobe RGB (1998) */
    KHR_DF_PRIMARIES_ADOBERGB    = 11U,
    KHR_DF_PRIMARIES_MAX         = 0xFFU
} khr_df_primaries_e;

/* Definition of the optical to digital transfer function
   ("gamma correction"). Most transfer functions are not a pure
   power function and also include a linear element.
   LAB and related absolute color representations should use
   KHR_DF_TRANSFER_UNSPECIFIED.
   These encodings indicate that the representation has had
   the corresponding transfer function applied relative to a
   linear representation; hence to process the linear intensity
   represented by the value, a corresponding inverse transform
   must be applied. */
typedef enum _khr_df_transfer_e {
    /* No transfer function defined */
    KHR_DF_TRANSFER_UNSPECIFIED = 0U,
    /* Linear transfer function (value proportional to intensity) */
    KHR_DF_TRANSFER_LINEAR      = 1U,
    /* Perceptually-linear transfer function of sRGB (~2.2); also used for scRGB */
    KHR_DF_TRANSFER_SRGB        = 2U,
    KHR_DF_TRANSFER_SRGB_EOTF   = 2U,
    KHR_DF_TRANSFER_SCRGB       = 2U,
    KHR_DF_TRANSFER_SCRGB_EOTF  = 2U,
    /* Perceptually-linear transfer function of ITU BT.601, BT.709 and BT.2020 (~1/.45) */
    KHR_DF_TRANSFER_ITU         = 3U,
    KHR_DF_TRANSFER_ITU_OETF    = 3U,
    KHR_DF_TRANSFER_BT601       = 3U,
    KHR_DF_TRANSFER_BT601_OETF  = 3U,
    KHR_DF_TRANSFER_BT709       = 3U,
    KHR_DF_TRANSFER_BT709_OETF  = 3U,
    KHR_DF_TRANSFER_BT2020      = 3U,
    KHR_DF_TRANSFER_BT2020_OETF = 3U,
    /* SMTPE170M (digital NTSC) defines an alias for the ITU transfer function (~1/.45) and a linear OOTF */
    KHR_DF_TRANSFER_SMTPE170M      = 3U,
    KHR_DF_TRANSFER_SMTPE170M_OETF = 3U,
    KHR_DF_TRANSFER_SMTPE170M_EOTF = 3U,
    /* Perceptually-linear gamma function of original NTSC (simple 2.2 gamma) */
    KHR_DF_TRANSFER_NTSC        = 4U,
    KHR_DF_TRANSFER_NTSC_EOTF   = 4U,
    /* Sony S-log used by Sony video cameras */
    KHR_DF_TRANSFER_SLOG        = 5U,
    KHR_DF_TRANSFER_SLOG_OETF   = 5U,
    /* Sony S-log 2 used by Sony video cameras */
    KHR_DF_TRANSFER_SLOG2       = 6U,
    KHR_DF_TRANSFER_SLOG2_OETF  = 6U,
    /* ITU BT.1886 EOTF */
    KHR_DF_TRANSFER_BT1886      = 7U,
    KHR_DF_TRANSFER_BT1886_EOTF = 7U,
    /* ITU BT.2100 HLG OETF (typical scene-referred content), linear light normalized 0..1 */
    KHR_DF_TRANSFER_HLG_OETF    = 8U,
    /* ITU BT.2100 HLG EOTF (nominal HDR display of HLG content), linear light normalized 0..1 */
    KHR_DF_TRANSFER_HLG_EOTF    = 9U,
    /* ITU BT.2100 PQ EOTF (typical HDR display-referred PQ content) */
    KHR_DF_TRANSFER_PQ_EOTF     = 10U,
    /* ITU BT.2100 PQ OETF (nominal scene described by PQ HDR content) */
    KHR_DF_TRANSFER_PQ_OETF     = 11U,
    /* DCI P3 transfer function */
    KHR_DF_TRANSFER_DCIP3       = 12U,
    KHR_DF_TRANSFER_DCIP3_EOTF  = 12U,
    /* Legacy PAL OETF */
    KHR_DF_TRANSFER_PAL_OETF    = 13U,
    /* Legacy PAL 625-line EOTF */
    KHR_DF_TRANSFER_PAL625_EOTF = 14U,
    /* Legacy ST240 transfer function */
    KHR_DF_TRANSFER_ST240       = 15U,
    KHR_DF_TRANSFER_ST240_OETF  = 15U,
    KHR_DF_TRANSFER_ST240_EOTF  = 15U,
    /* ACEScc transfer function */
    KHR_DF_TRANSFER_ACESCC      = 16U,
    KHR_DF_TRANSFER_ACESCC_OETF = 16U,
    /* ACEScct transfer function */
    KHR_DF_TRANSFER_ACESCCT      = 17U,
    KHR_DF_TRANSFER_ACESCCT_OETF = 17U,
    /* Adobe RGB (1998) transfer function */
    KHR_DF_TRANSFER_ADOBERGB      = 18U,
    KHR_DF_TRANSFER_ADOBERGB_EOTF = 18U,
    /* Legacy ITU BT.2100 HLG OETF (typical scene-referred content), linear light normalized 0..12 */
    KHR_DF_TRANSFER_HLG_UNNORMALIZED_OETF = 19U,
    KHR_DF_TRANSFER_MAX                   = 0xFFU
} khr_df_transfer_e;

typedef enum _khr_df_flags_e {
    KHR_DF_FLAG_ALPHA_STRAIGHT      = 0U,
    KHR_DF_FLAG_ALPHA_PREMULTIPLIED = 1U
} khr_df_flags_e;

typedef enum _khr_df_sample_datatype_qualifiers_e {
    KHR_DF_SAMPLE_DATATYPE_LINEAR = 1U << 4U,
    KHR_DF_SAMPLE_DATATYPE_EXPONENT = 1U << 5U,
    KHR_DF_SAMPLE_DATATYPE_SIGNED = 1U << 6U,
    KHR_DF_SAMPLE_DATATYPE_FLOAT = 1U << 7U
} khr_df_sample_datatype_qualifiers_e;

#endif
