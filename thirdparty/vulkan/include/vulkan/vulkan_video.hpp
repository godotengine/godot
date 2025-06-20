// Copyright 2021-2025 The Khronos Group Inc.
// SPDX-License-Identifier: Apache-2.0 OR MIT
//

// This header is generated from the Khronos Vulkan XML API Registry.

#ifndef VULKAN_VIDEO_HPP
#define VULKAN_VIDEO_HPP

// clang-format off
#include <vulkan/vulkan.hpp>
// clang-format on

#include <vk_video/vulkan_video_codec_av1std.h>
#include <vk_video/vulkan_video_codec_av1std_decode.h>
#if ( 301 < VK_HEADER_VERSION )
#  include <vk_video/vulkan_video_codec_av1std_encode.h>
#endif
#include <vk_video/vulkan_video_codec_h264std.h>
#include <vk_video/vulkan_video_codec_h264std_decode.h>
#include <vk_video/vulkan_video_codec_h264std_encode.h>
#include <vk_video/vulkan_video_codec_h265std.h>
#include <vk_video/vulkan_video_codec_h265std_decode.h>
#include <vk_video/vulkan_video_codec_h265std_encode.h>
#include <vk_video/vulkan_video_codecs_common.h>

#if !defined( VULKAN_HPP_VIDEO_NAMESPACE )
#  define VULKAN_HPP_VIDEO_NAMESPACE video
#endif

namespace VULKAN_HPP_NAMESPACE
{
  namespace VULKAN_HPP_VIDEO_NAMESPACE
  {

    //=============
    //=== ENUMs ===
    //=============

    //=== vulkan_video_codec_h264std ===

    enum class H264ChromaFormatIdc
    {
      eMonochrome = STD_VIDEO_H264_CHROMA_FORMAT_IDC_MONOCHROME,
      e420        = STD_VIDEO_H264_CHROMA_FORMAT_IDC_420,
      e422        = STD_VIDEO_H264_CHROMA_FORMAT_IDC_422,
      e444        = STD_VIDEO_H264_CHROMA_FORMAT_IDC_444,
      eInvalid    = STD_VIDEO_H264_CHROMA_FORMAT_IDC_INVALID
    };

    enum class H264ProfileIdc
    {
      eBaseline          = STD_VIDEO_H264_PROFILE_IDC_BASELINE,
      eMain              = STD_VIDEO_H264_PROFILE_IDC_MAIN,
      eHigh              = STD_VIDEO_H264_PROFILE_IDC_HIGH,
      eHigh444Predictive = STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE,
      eInvalid           = STD_VIDEO_H264_PROFILE_IDC_INVALID
    };

    enum class H264LevelIdc
    {
      e1_0     = STD_VIDEO_H264_LEVEL_IDC_1_0,
      e1_1     = STD_VIDEO_H264_LEVEL_IDC_1_1,
      e1_2     = STD_VIDEO_H264_LEVEL_IDC_1_2,
      e1_3     = STD_VIDEO_H264_LEVEL_IDC_1_3,
      e2_0     = STD_VIDEO_H264_LEVEL_IDC_2_0,
      e2_1     = STD_VIDEO_H264_LEVEL_IDC_2_1,
      e2_2     = STD_VIDEO_H264_LEVEL_IDC_2_2,
      e3_0     = STD_VIDEO_H264_LEVEL_IDC_3_0,
      e3_1     = STD_VIDEO_H264_LEVEL_IDC_3_1,
      e3_2     = STD_VIDEO_H264_LEVEL_IDC_3_2,
      e4_0     = STD_VIDEO_H264_LEVEL_IDC_4_0,
      e4_1     = STD_VIDEO_H264_LEVEL_IDC_4_1,
      e4_2     = STD_VIDEO_H264_LEVEL_IDC_4_2,
      e5_0     = STD_VIDEO_H264_LEVEL_IDC_5_0,
      e5_1     = STD_VIDEO_H264_LEVEL_IDC_5_1,
      e5_2     = STD_VIDEO_H264_LEVEL_IDC_5_2,
      e6_0     = STD_VIDEO_H264_LEVEL_IDC_6_0,
      e6_1     = STD_VIDEO_H264_LEVEL_IDC_6_1,
      e6_2     = STD_VIDEO_H264_LEVEL_IDC_6_2,
      eInvalid = STD_VIDEO_H264_LEVEL_IDC_INVALID
    };

    enum class H264PocType
    {
      e0       = STD_VIDEO_H264_POC_TYPE_0,
      e1       = STD_VIDEO_H264_POC_TYPE_1,
      e2       = STD_VIDEO_H264_POC_TYPE_2,
      eInvalid = STD_VIDEO_H264_POC_TYPE_INVALID
    };

    enum class H264AspectRatioIdc
    {
      eUnspecified = STD_VIDEO_H264_ASPECT_RATIO_IDC_UNSPECIFIED,
      eSquare      = STD_VIDEO_H264_ASPECT_RATIO_IDC_SQUARE,
      e12_11       = STD_VIDEO_H264_ASPECT_RATIO_IDC_12_11,
      e10_11       = STD_VIDEO_H264_ASPECT_RATIO_IDC_10_11,
      e16_11       = STD_VIDEO_H264_ASPECT_RATIO_IDC_16_11,
      e40_33       = STD_VIDEO_H264_ASPECT_RATIO_IDC_40_33,
      e24_11       = STD_VIDEO_H264_ASPECT_RATIO_IDC_24_11,
      e20_11       = STD_VIDEO_H264_ASPECT_RATIO_IDC_20_11,
      e32_11       = STD_VIDEO_H264_ASPECT_RATIO_IDC_32_11,
      e80_33       = STD_VIDEO_H264_ASPECT_RATIO_IDC_80_33,
      e18_11       = STD_VIDEO_H264_ASPECT_RATIO_IDC_18_11,
      e15_11       = STD_VIDEO_H264_ASPECT_RATIO_IDC_15_11,
      e64_33       = STD_VIDEO_H264_ASPECT_RATIO_IDC_64_33,
      e160_99      = STD_VIDEO_H264_ASPECT_RATIO_IDC_160_99,
      e4_3         = STD_VIDEO_H264_ASPECT_RATIO_IDC_4_3,
      e3_2         = STD_VIDEO_H264_ASPECT_RATIO_IDC_3_2,
      e2_1         = STD_VIDEO_H264_ASPECT_RATIO_IDC_2_1,
      eExtendedSar = STD_VIDEO_H264_ASPECT_RATIO_IDC_EXTENDED_SAR,
      eInvalid     = STD_VIDEO_H264_ASPECT_RATIO_IDC_INVALID
    };

    enum class H264WeightedBipredIdc
    {
      eDefault  = STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT,
      eExplicit = STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_EXPLICIT,
      eImplicit = STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_IMPLICIT,
      eInvalid  = STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_INVALID
    };

    enum class H264ModificationOfPicNumsIdc
    {
      eShortTermSubtract = STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_SHORT_TERM_SUBTRACT,
      eShortTermAdd      = STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_SHORT_TERM_ADD,
      eLongTerm          = STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_LONG_TERM,
      eEnd               = STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_END,
      eInvalid           = STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_INVALID
    };

    enum class H264MemMgmtControlOp
    {
      eEnd                   = STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_END,
      eUnmarkShortTerm       = STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_UNMARK_SHORT_TERM,
      eUnmarkLongTerm        = STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_UNMARK_LONG_TERM,
      eMarkLongTerm          = STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_MARK_LONG_TERM,
      eSetMaxLongTermIndex   = STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_SET_MAX_LONG_TERM_INDEX,
      eUnmarkAll             = STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_UNMARK_ALL,
      eMarkCurrentAsLongTerm = STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_MARK_CURRENT_AS_LONG_TERM,
      eInvalid               = STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_INVALID
    };

    enum class H264CabacInitIdc
    {
      e0       = STD_VIDEO_H264_CABAC_INIT_IDC_0,
      e1       = STD_VIDEO_H264_CABAC_INIT_IDC_1,
      e2       = STD_VIDEO_H264_CABAC_INIT_IDC_2,
      eInvalid = STD_VIDEO_H264_CABAC_INIT_IDC_INVALID
    };

    enum class H264DisableDeblockingFilterIdc
    {
      eDisabled = STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED,
      eEnabled  = STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_ENABLED,
      ePartial  = STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_PARTIAL,
      eInvalid  = STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_INVALID
    };

    enum class H264SliceType
    {
      eP       = STD_VIDEO_H264_SLICE_TYPE_P,
      eB       = STD_VIDEO_H264_SLICE_TYPE_B,
      eI       = STD_VIDEO_H264_SLICE_TYPE_I,
      eInvalid = STD_VIDEO_H264_SLICE_TYPE_INVALID
    };

    enum class H264PictureType
    {
      eP       = STD_VIDEO_H264_PICTURE_TYPE_P,
      eB       = STD_VIDEO_H264_PICTURE_TYPE_B,
      eI       = STD_VIDEO_H264_PICTURE_TYPE_I,
      eIdr     = STD_VIDEO_H264_PICTURE_TYPE_IDR,
      eInvalid = STD_VIDEO_H264_PICTURE_TYPE_INVALID
    };

    enum class H264NonVclNaluType
    {
      eSps           = STD_VIDEO_H264_NON_VCL_NALU_TYPE_SPS,
      ePps           = STD_VIDEO_H264_NON_VCL_NALU_TYPE_PPS,
      eAud           = STD_VIDEO_H264_NON_VCL_NALU_TYPE_AUD,
      ePrefix        = STD_VIDEO_H264_NON_VCL_NALU_TYPE_PREFIX,
      eEndOfSequence = STD_VIDEO_H264_NON_VCL_NALU_TYPE_END_OF_SEQUENCE,
      eEndOfStream   = STD_VIDEO_H264_NON_VCL_NALU_TYPE_END_OF_STREAM,
      ePrecoded      = STD_VIDEO_H264_NON_VCL_NALU_TYPE_PRECODED,
      eInvalid       = STD_VIDEO_H264_NON_VCL_NALU_TYPE_INVALID
    };

    //=== vulkan_video_codec_h264std_decode ===

    enum class DecodeH264FieldOrderCount
    {
      eTop     = STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_TOP,
      eBottom  = STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_BOTTOM,
      eInvalid = STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_INVALID
    };

    //=== vulkan_video_codec_h265std ===

    enum class H265ChromaFormatIdc
    {
      eMonochrome = STD_VIDEO_H265_CHROMA_FORMAT_IDC_MONOCHROME,
      e420        = STD_VIDEO_H265_CHROMA_FORMAT_IDC_420,
      e422        = STD_VIDEO_H265_CHROMA_FORMAT_IDC_422,
      e444        = STD_VIDEO_H265_CHROMA_FORMAT_IDC_444,
      eInvalid    = STD_VIDEO_H265_CHROMA_FORMAT_IDC_INVALID
    };

    enum class H265ProfileIdc
    {
      eMain                  = STD_VIDEO_H265_PROFILE_IDC_MAIN,
      eMain10                = STD_VIDEO_H265_PROFILE_IDC_MAIN_10,
      eMainStillPicture      = STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE,
      eFormatRangeExtensions = STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS,
      eSccExtensions         = STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS,
      eInvalid               = STD_VIDEO_H265_PROFILE_IDC_INVALID
    };

    enum class H265LevelIdc
    {
      e1_0     = STD_VIDEO_H265_LEVEL_IDC_1_0,
      e2_0     = STD_VIDEO_H265_LEVEL_IDC_2_0,
      e2_1     = STD_VIDEO_H265_LEVEL_IDC_2_1,
      e3_0     = STD_VIDEO_H265_LEVEL_IDC_3_0,
      e3_1     = STD_VIDEO_H265_LEVEL_IDC_3_1,
      e4_0     = STD_VIDEO_H265_LEVEL_IDC_4_0,
      e4_1     = STD_VIDEO_H265_LEVEL_IDC_4_1,
      e5_0     = STD_VIDEO_H265_LEVEL_IDC_5_0,
      e5_1     = STD_VIDEO_H265_LEVEL_IDC_5_1,
      e5_2     = STD_VIDEO_H265_LEVEL_IDC_5_2,
      e6_0     = STD_VIDEO_H265_LEVEL_IDC_6_0,
      e6_1     = STD_VIDEO_H265_LEVEL_IDC_6_1,
      e6_2     = STD_VIDEO_H265_LEVEL_IDC_6_2,
      eInvalid = STD_VIDEO_H265_LEVEL_IDC_INVALID
    };

    enum class H265SliceType
    {
      eB       = STD_VIDEO_H265_SLICE_TYPE_B,
      eP       = STD_VIDEO_H265_SLICE_TYPE_P,
      eI       = STD_VIDEO_H265_SLICE_TYPE_I,
      eInvalid = STD_VIDEO_H265_SLICE_TYPE_INVALID
    };

    enum class H265PictureType
    {
      eP       = STD_VIDEO_H265_PICTURE_TYPE_P,
      eB       = STD_VIDEO_H265_PICTURE_TYPE_B,
      eI       = STD_VIDEO_H265_PICTURE_TYPE_I,
      eIdr     = STD_VIDEO_H265_PICTURE_TYPE_IDR,
      eInvalid = STD_VIDEO_H265_PICTURE_TYPE_INVALID
    };

    enum class H265AspectRatioIdc
    {
      eUnspecified = STD_VIDEO_H265_ASPECT_RATIO_IDC_UNSPECIFIED,
      eSquare      = STD_VIDEO_H265_ASPECT_RATIO_IDC_SQUARE,
      e12_11       = STD_VIDEO_H265_ASPECT_RATIO_IDC_12_11,
      e10_11       = STD_VIDEO_H265_ASPECT_RATIO_IDC_10_11,
      e16_11       = STD_VIDEO_H265_ASPECT_RATIO_IDC_16_11,
      e40_33       = STD_VIDEO_H265_ASPECT_RATIO_IDC_40_33,
      e24_11       = STD_VIDEO_H265_ASPECT_RATIO_IDC_24_11,
      e20_11       = STD_VIDEO_H265_ASPECT_RATIO_IDC_20_11,
      e32_11       = STD_VIDEO_H265_ASPECT_RATIO_IDC_32_11,
      e80_33       = STD_VIDEO_H265_ASPECT_RATIO_IDC_80_33,
      e18_11       = STD_VIDEO_H265_ASPECT_RATIO_IDC_18_11,
      e15_11       = STD_VIDEO_H265_ASPECT_RATIO_IDC_15_11,
      e64_33       = STD_VIDEO_H265_ASPECT_RATIO_IDC_64_33,
      e160_99      = STD_VIDEO_H265_ASPECT_RATIO_IDC_160_99,
      e4_3         = STD_VIDEO_H265_ASPECT_RATIO_IDC_4_3,
      e3_2         = STD_VIDEO_H265_ASPECT_RATIO_IDC_3_2,
      e2_1         = STD_VIDEO_H265_ASPECT_RATIO_IDC_2_1,
      eExtendedSar = STD_VIDEO_H265_ASPECT_RATIO_IDC_EXTENDED_SAR,
      eInvalid     = STD_VIDEO_H265_ASPECT_RATIO_IDC_INVALID
    };

    //=== vulkan_video_codec_av1std ===

    enum class AV1Profile
    {
      eMain         = STD_VIDEO_AV1_PROFILE_MAIN,
      eHigh         = STD_VIDEO_AV1_PROFILE_HIGH,
      eProfessional = STD_VIDEO_AV1_PROFILE_PROFESSIONAL,
      eInvalid      = STD_VIDEO_AV1_PROFILE_INVALID
    };

    enum class AV1Level
    {
      e2_0     = STD_VIDEO_AV1_LEVEL_2_0,
      e2_1     = STD_VIDEO_AV1_LEVEL_2_1,
      e2_2     = STD_VIDEO_AV1_LEVEL_2_2,
      e2_3     = STD_VIDEO_AV1_LEVEL_2_3,
      e3_0     = STD_VIDEO_AV1_LEVEL_3_0,
      e3_1     = STD_VIDEO_AV1_LEVEL_3_1,
      e3_2     = STD_VIDEO_AV1_LEVEL_3_2,
      e3_3     = STD_VIDEO_AV1_LEVEL_3_3,
      e4_0     = STD_VIDEO_AV1_LEVEL_4_0,
      e4_1     = STD_VIDEO_AV1_LEVEL_4_1,
      e4_2     = STD_VIDEO_AV1_LEVEL_4_2,
      e4_3     = STD_VIDEO_AV1_LEVEL_4_3,
      e5_0     = STD_VIDEO_AV1_LEVEL_5_0,
      e5_1     = STD_VIDEO_AV1_LEVEL_5_1,
      e5_2     = STD_VIDEO_AV1_LEVEL_5_2,
      e5_3     = STD_VIDEO_AV1_LEVEL_5_3,
      e6_0     = STD_VIDEO_AV1_LEVEL_6_0,
      e6_1     = STD_VIDEO_AV1_LEVEL_6_1,
      e6_2     = STD_VIDEO_AV1_LEVEL_6_2,
      e6_3     = STD_VIDEO_AV1_LEVEL_6_3,
      e7_0     = STD_VIDEO_AV1_LEVEL_7_0,
      e7_1     = STD_VIDEO_AV1_LEVEL_7_1,
      e7_2     = STD_VIDEO_AV1_LEVEL_7_2,
      e7_3     = STD_VIDEO_AV1_LEVEL_7_3,
      eInvalid = STD_VIDEO_AV1_LEVEL_INVALID
    };

    enum class AV1FrameType
    {
      eKey       = STD_VIDEO_AV1_FRAME_TYPE_KEY,
      eInter     = STD_VIDEO_AV1_FRAME_TYPE_INTER,
      eIntraOnly = STD_VIDEO_AV1_FRAME_TYPE_INTRA_ONLY,
      eSwitch    = STD_VIDEO_AV1_FRAME_TYPE_SWITCH,
      eInvalid   = STD_VIDEO_AV1_FRAME_TYPE_INVALID
    };

    enum class AV1ReferenceName
    {
      eIntraFrame   = STD_VIDEO_AV1_REFERENCE_NAME_INTRA_FRAME,
      eLastFrame    = STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME,
      eLast2Frame   = STD_VIDEO_AV1_REFERENCE_NAME_LAST2_FRAME,
      eLast3Frame   = STD_VIDEO_AV1_REFERENCE_NAME_LAST3_FRAME,
      eGoldenFrame  = STD_VIDEO_AV1_REFERENCE_NAME_GOLDEN_FRAME,
      eBwdrefFrame  = STD_VIDEO_AV1_REFERENCE_NAME_BWDREF_FRAME,
      eAltref2Frame = STD_VIDEO_AV1_REFERENCE_NAME_ALTREF2_FRAME,
      eAltrefFrame  = STD_VIDEO_AV1_REFERENCE_NAME_ALTREF_FRAME,
      eInvalid      = STD_VIDEO_AV1_REFERENCE_NAME_INVALID
    };

    enum class AV1InterpolationFilter
    {
      eEighttap       = STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP,
      eEighttapSmooth = STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP_SMOOTH,
      eEighttapSharp  = STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP_SHARP,
      eBilinear       = STD_VIDEO_AV1_INTERPOLATION_FILTER_BILINEAR,
      eSwitchable     = STD_VIDEO_AV1_INTERPOLATION_FILTER_SWITCHABLE,
      eInvalid        = STD_VIDEO_AV1_INTERPOLATION_FILTER_INVALID
    };

    enum class AV1TxMode
    {
      eOnly4X4 = STD_VIDEO_AV1_TX_MODE_ONLY_4X4,
      eLargest = STD_VIDEO_AV1_TX_MODE_LARGEST,
      eSelect  = STD_VIDEO_AV1_TX_MODE_SELECT,
      eInvalid = STD_VIDEO_AV1_TX_MODE_INVALID
    };

    enum class AV1FrameRestorationType
    {
      eNone       = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
      eWiener     = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_WIENER,
      eSgrproj    = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SGRPROJ,
      eSwitchable = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE,
      eInvalid    = STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_INVALID
    };

    enum class AV1ColorPrimaries
    {
      eBt709       = STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709,
      eUnspecified = STD_VIDEO_AV1_COLOR_PRIMARIES_UNSPECIFIED,
      eBtUnspecified VULKAN_HPP_DEPRECATED_17( "eBtUnspecified is deprecated, eUnspecified should be used instead." ) =
        STD_VIDEO_AV1_COLOR_PRIMARIES_BT_UNSPECIFIED,
      eBt470M      = STD_VIDEO_AV1_COLOR_PRIMARIES_BT_470_M,
      eBt470BG     = STD_VIDEO_AV1_COLOR_PRIMARIES_BT_470_B_G,
      eBt601       = STD_VIDEO_AV1_COLOR_PRIMARIES_BT_601,
      eSmpte240    = STD_VIDEO_AV1_COLOR_PRIMARIES_SMPTE_240,
      eGenericFilm = STD_VIDEO_AV1_COLOR_PRIMARIES_GENERIC_FILM,
      eBt2020      = STD_VIDEO_AV1_COLOR_PRIMARIES_BT_2020,
      eXyz         = STD_VIDEO_AV1_COLOR_PRIMARIES_XYZ,
      eSmpte431    = STD_VIDEO_AV1_COLOR_PRIMARIES_SMPTE_431,
      eSmpte432    = STD_VIDEO_AV1_COLOR_PRIMARIES_SMPTE_432,
      eEbu3213     = STD_VIDEO_AV1_COLOR_PRIMARIES_EBU_3213,
      eInvalid     = STD_VIDEO_AV1_COLOR_PRIMARIES_INVALID
    };

    enum class AV1TransferCharacteristics
    {
      eReserved0    = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_RESERVED_0,
      eBt709        = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_709,
      eUnspecified  = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED,
      eReserved3    = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_RESERVED_3,
      eBt470M       = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_470_M,
      eBt470BG      = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_470_B_G,
      eBt601        = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_601,
      eSmpte240     = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SMPTE_240,
      eLinear       = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_LINEAR,
      eLog100       = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_LOG_100,
      eLog100Sqrt10 = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_LOG_100_SQRT10,
      eIec61966     = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_IEC_61966,
      eBt1361       = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_1361,
      eSrgb         = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SRGB,
      eBt2020_10Bit = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_2020_10_BIT,
      eBt2020_12Bit = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_2020_12_BIT,
      eSmpte2084    = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SMPTE_2084,
      eSmpte428     = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SMPTE_428,
      eHlg          = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_HLG,
      eInvalid      = STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_INVALID
    };

    enum class AV1MatrixCoefficients
    {
      eIdentity    = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_IDENTITY,
      eBt709       = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_709,
      eUnspecified = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED,
      eReserved3   = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_RESERVED_3,
      eFcc         = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_FCC,
      eBt470BG     = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_470_B_G,
      eBt601       = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_601,
      eSmpte240    = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_SMPTE_240,
      eSmpteYcgco  = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_SMPTE_YCGCO,
      eBt2020Ncl   = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_2020_NCL,
      eBt2020Cl    = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_2020_CL,
      eSmpte2085   = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_SMPTE_2085,
      eChromatNcl  = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_CHROMAT_NCL,
      eChromatCl   = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_CHROMAT_CL,
      eIctcp       = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_ICTCP,
      eInvalid     = STD_VIDEO_AV1_MATRIX_COEFFICIENTS_INVALID
    };

    enum class AV1ChromaSamplePosition
    {
      eUnknown   = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN,
      eVertical  = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_VERTICAL,
      eColocated = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_COLOCATED,
      eReserved  = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_RESERVED,
      eInvalid   = STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_INVALID
    };

    //===============
    //=== STRUCTS ===
    //===============

    //=== vulkan_video_codec_h264std ===

    struct H264SpsVuiFlags
    {
      using NativeType = StdVideoH264SpsVuiFlags;

      operator StdVideoH264SpsVuiFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH264SpsVuiFlags *>( this );
      }

      operator StdVideoH264SpsVuiFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH264SpsVuiFlags *>( this );
      }

      operator StdVideoH264SpsVuiFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH264SpsVuiFlags *>( this );
      }

      operator StdVideoH264SpsVuiFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH264SpsVuiFlags *>( this );
      }

      bool operator==( H264SpsVuiFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( aspect_ratio_info_present_flag == rhs.aspect_ratio_info_present_flag ) && ( overscan_info_present_flag == rhs.overscan_info_present_flag ) &&
               ( overscan_appropriate_flag == rhs.overscan_appropriate_flag ) && ( video_signal_type_present_flag == rhs.video_signal_type_present_flag ) &&
               ( video_full_range_flag == rhs.video_full_range_flag ) && ( color_description_present_flag == rhs.color_description_present_flag ) &&
               ( chroma_loc_info_present_flag == rhs.chroma_loc_info_present_flag ) && ( timing_info_present_flag == rhs.timing_info_present_flag ) &&
               ( fixed_frame_rate_flag == rhs.fixed_frame_rate_flag ) && ( bitstream_restriction_flag == rhs.bitstream_restriction_flag ) &&
               ( nal_hrd_parameters_present_flag == rhs.nal_hrd_parameters_present_flag ) &&
               ( vcl_hrd_parameters_present_flag == rhs.vcl_hrd_parameters_present_flag );
      }

      bool operator!=( H264SpsVuiFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t aspect_ratio_info_present_flag  : 1;
      uint32_t overscan_info_present_flag      : 1;
      uint32_t overscan_appropriate_flag       : 1;
      uint32_t video_signal_type_present_flag  : 1;
      uint32_t video_full_range_flag           : 1;
      uint32_t color_description_present_flag  : 1;
      uint32_t chroma_loc_info_present_flag    : 1;
      uint32_t timing_info_present_flag        : 1;
      uint32_t fixed_frame_rate_flag           : 1;
      uint32_t bitstream_restriction_flag      : 1;
      uint32_t nal_hrd_parameters_present_flag : 1;
      uint32_t vcl_hrd_parameters_present_flag : 1;
    };

    struct H264HrdParameters
    {
      using NativeType = StdVideoH264HrdParameters;

      operator StdVideoH264HrdParameters const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH264HrdParameters *>( this );
      }

      operator StdVideoH264HrdParameters &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH264HrdParameters *>( this );
      }

      operator StdVideoH264HrdParameters const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH264HrdParameters *>( this );
      }

      operator StdVideoH264HrdParameters *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH264HrdParameters *>( this );
      }

      bool operator==( H264HrdParameters const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( cpb_cnt_minus1 == rhs.cpb_cnt_minus1 ) && ( bit_rate_scale == rhs.bit_rate_scale ) && ( cpb_size_scale == rhs.cpb_size_scale ) &&
               ( reserved1 == rhs.reserved1 ) && ( bit_rate_value_minus1 == rhs.bit_rate_value_minus1 ) &&
               ( cpb_size_value_minus1 == rhs.cpb_size_value_minus1 ) && ( cbr_flag == rhs.cbr_flag ) &&
               ( initial_cpb_removal_delay_length_minus1 == rhs.initial_cpb_removal_delay_length_minus1 ) &&
               ( cpb_removal_delay_length_minus1 == rhs.cpb_removal_delay_length_minus1 ) &&
               ( dpb_output_delay_length_minus1 == rhs.dpb_output_delay_length_minus1 ) && ( time_offset_length == rhs.time_offset_length );
      }

      bool operator!=( H264HrdParameters const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint8_t                                                                          cpb_cnt_minus1                          = {};
      uint8_t                                                                          bit_rate_scale                          = {};
      uint8_t                                                                          cpb_size_scale                          = {};
      uint8_t                                                                          reserved1                               = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_H264_CPB_CNT_LIST_SIZE> bit_rate_value_minus1                   = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_H264_CPB_CNT_LIST_SIZE> cpb_size_value_minus1                   = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H264_CPB_CNT_LIST_SIZE>  cbr_flag                                = {};
      uint32_t                                                                         initial_cpb_removal_delay_length_minus1 = {};
      uint32_t                                                                         cpb_removal_delay_length_minus1         = {};
      uint32_t                                                                         dpb_output_delay_length_minus1          = {};
      uint32_t                                                                         time_offset_length                      = {};
    };

    struct H264SequenceParameterSetVui
    {
      using NativeType = StdVideoH264SequenceParameterSetVui;

      operator StdVideoH264SequenceParameterSetVui const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH264SequenceParameterSetVui *>( this );
      }

      operator StdVideoH264SequenceParameterSetVui &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH264SequenceParameterSetVui *>( this );
      }

      operator StdVideoH264SequenceParameterSetVui const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH264SequenceParameterSetVui *>( this );
      }

      operator StdVideoH264SequenceParameterSetVui *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH264SequenceParameterSetVui *>( this );
      }

      bool operator==( H264SequenceParameterSetVui const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( aspect_ratio_idc == rhs.aspect_ratio_idc ) && ( sar_width == rhs.sar_width ) && ( sar_height == rhs.sar_height ) &&
               ( video_format == rhs.video_format ) && ( colour_primaries == rhs.colour_primaries ) &&
               ( transfer_characteristics == rhs.transfer_characteristics ) && ( matrix_coefficients == rhs.matrix_coefficients ) &&
               ( num_units_in_tick == rhs.num_units_in_tick ) && ( time_scale == rhs.time_scale ) && ( max_num_reorder_frames == rhs.max_num_reorder_frames ) &&
               ( max_dec_frame_buffering == rhs.max_dec_frame_buffering ) && ( chroma_sample_loc_type_top_field == rhs.chroma_sample_loc_type_top_field ) &&
               ( chroma_sample_loc_type_bottom_field == rhs.chroma_sample_loc_type_bottom_field ) && ( reserved1 == rhs.reserved1 ) &&
               ( pHrdParameters == rhs.pHrdParameters );
      }

      bool operator!=( H264SequenceParameterSetVui const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264SpsVuiFlags    flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264AspectRatioIdc aspect_ratio_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264AspectRatioIdc::eUnspecified;
      uint16_t                                                                    sar_width                           = {};
      uint16_t                                                                    sar_height                          = {};
      uint8_t                                                                     video_format                        = {};
      uint8_t                                                                     colour_primaries                    = {};
      uint8_t                                                                     transfer_characteristics            = {};
      uint8_t                                                                     matrix_coefficients                 = {};
      uint32_t                                                                    num_units_in_tick                   = {};
      uint32_t                                                                    time_scale                          = {};
      uint8_t                                                                     max_num_reorder_frames              = {};
      uint8_t                                                                     max_dec_frame_buffering             = {};
      uint8_t                                                                     chroma_sample_loc_type_top_field    = {};
      uint8_t                                                                     chroma_sample_loc_type_bottom_field = {};
      uint32_t                                                                    reserved1                           = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264HrdParameters * pHrdParameters                      = {};
    };

    struct H264SpsFlags
    {
      using NativeType = StdVideoH264SpsFlags;

      operator StdVideoH264SpsFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH264SpsFlags *>( this );
      }

      operator StdVideoH264SpsFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH264SpsFlags *>( this );
      }

      operator StdVideoH264SpsFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH264SpsFlags *>( this );
      }

      operator StdVideoH264SpsFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH264SpsFlags *>( this );
      }

      bool operator==( H264SpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( constraint_set0_flag == rhs.constraint_set0_flag ) && ( constraint_set1_flag == rhs.constraint_set1_flag ) &&
               ( constraint_set2_flag == rhs.constraint_set2_flag ) && ( constraint_set3_flag == rhs.constraint_set3_flag ) &&
               ( constraint_set4_flag == rhs.constraint_set4_flag ) && ( constraint_set5_flag == rhs.constraint_set5_flag ) &&
               ( direct_8x8_inference_flag == rhs.direct_8x8_inference_flag ) && ( mb_adaptive_frame_field_flag == rhs.mb_adaptive_frame_field_flag ) &&
               ( frame_mbs_only_flag == rhs.frame_mbs_only_flag ) && ( delta_pic_order_always_zero_flag == rhs.delta_pic_order_always_zero_flag ) &&
               ( separate_colour_plane_flag == rhs.separate_colour_plane_flag ) &&
               ( gaps_in_frame_num_value_allowed_flag == rhs.gaps_in_frame_num_value_allowed_flag ) &&
               ( qpprime_y_zero_transform_bypass_flag == rhs.qpprime_y_zero_transform_bypass_flag ) && ( frame_cropping_flag == rhs.frame_cropping_flag ) &&
               ( seq_scaling_matrix_present_flag == rhs.seq_scaling_matrix_present_flag ) && ( vui_parameters_present_flag == rhs.vui_parameters_present_flag );
      }

      bool operator!=( H264SpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t constraint_set0_flag                 : 1;
      uint32_t constraint_set1_flag                 : 1;
      uint32_t constraint_set2_flag                 : 1;
      uint32_t constraint_set3_flag                 : 1;
      uint32_t constraint_set4_flag                 : 1;
      uint32_t constraint_set5_flag                 : 1;
      uint32_t direct_8x8_inference_flag            : 1;
      uint32_t mb_adaptive_frame_field_flag         : 1;
      uint32_t frame_mbs_only_flag                  : 1;
      uint32_t delta_pic_order_always_zero_flag     : 1;
      uint32_t separate_colour_plane_flag           : 1;
      uint32_t gaps_in_frame_num_value_allowed_flag : 1;
      uint32_t qpprime_y_zero_transform_bypass_flag : 1;
      uint32_t frame_cropping_flag                  : 1;
      uint32_t seq_scaling_matrix_present_flag      : 1;
      uint32_t vui_parameters_present_flag          : 1;
    };

    struct H264ScalingLists
    {
      using NativeType = StdVideoH264ScalingLists;

      operator StdVideoH264ScalingLists const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH264ScalingLists *>( this );
      }

      operator StdVideoH264ScalingLists &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH264ScalingLists *>( this );
      }

      operator StdVideoH264ScalingLists const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH264ScalingLists *>( this );
      }

      operator StdVideoH264ScalingLists *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH264ScalingLists *>( this );
      }

      bool operator==( H264ScalingLists const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( scaling_list_present_mask == rhs.scaling_list_present_mask ) && ( use_default_scaling_matrix_mask == rhs.use_default_scaling_matrix_mask ) &&
               ( ScalingList4x4 == rhs.ScalingList4x4 ) && ( ScalingList8x8 == rhs.ScalingList8x8 );
      }

      bool operator!=( H264ScalingLists const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint16_t scaling_list_present_mask       = {};
      uint16_t use_default_scaling_matrix_mask = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<uint8_t, STD_VIDEO_H264_SCALING_LIST_4X4_NUM_LISTS, STD_VIDEO_H264_SCALING_LIST_4X4_NUM_ELEMENTS>
        ScalingList4x4 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<uint8_t, STD_VIDEO_H264_SCALING_LIST_8X8_NUM_LISTS, STD_VIDEO_H264_SCALING_LIST_8X8_NUM_ELEMENTS>
        ScalingList8x8 = {};
    };

    struct H264SequenceParameterSet
    {
      using NativeType = StdVideoH264SequenceParameterSet;

      operator StdVideoH264SequenceParameterSet const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH264SequenceParameterSet *>( this );
      }

      operator StdVideoH264SequenceParameterSet &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH264SequenceParameterSet *>( this );
      }

      operator StdVideoH264SequenceParameterSet const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH264SequenceParameterSet *>( this );
      }

      operator StdVideoH264SequenceParameterSet *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH264SequenceParameterSet *>( this );
      }

      bool operator==( H264SequenceParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( profile_idc == rhs.profile_idc ) && ( level_idc == rhs.level_idc ) &&
               ( chroma_format_idc == rhs.chroma_format_idc ) && ( seq_parameter_set_id == rhs.seq_parameter_set_id ) &&
               ( bit_depth_luma_minus8 == rhs.bit_depth_luma_minus8 ) && ( bit_depth_chroma_minus8 == rhs.bit_depth_chroma_minus8 ) &&
               ( log2_max_frame_num_minus4 == rhs.log2_max_frame_num_minus4 ) && ( pic_order_cnt_type == rhs.pic_order_cnt_type ) &&
               ( offset_for_non_ref_pic == rhs.offset_for_non_ref_pic ) && ( offset_for_top_to_bottom_field == rhs.offset_for_top_to_bottom_field ) &&
               ( log2_max_pic_order_cnt_lsb_minus4 == rhs.log2_max_pic_order_cnt_lsb_minus4 ) &&
               ( num_ref_frames_in_pic_order_cnt_cycle == rhs.num_ref_frames_in_pic_order_cnt_cycle ) && ( max_num_ref_frames == rhs.max_num_ref_frames ) &&
               ( reserved1 == rhs.reserved1 ) && ( pic_width_in_mbs_minus1 == rhs.pic_width_in_mbs_minus1 ) &&
               ( pic_height_in_map_units_minus1 == rhs.pic_height_in_map_units_minus1 ) && ( frame_crop_left_offset == rhs.frame_crop_left_offset ) &&
               ( frame_crop_right_offset == rhs.frame_crop_right_offset ) && ( frame_crop_top_offset == rhs.frame_crop_top_offset ) &&
               ( frame_crop_bottom_offset == rhs.frame_crop_bottom_offset ) && ( reserved2 == rhs.reserved2 ) &&
               ( pOffsetForRefFrame == rhs.pOffsetForRefFrame ) && ( pScalingLists == rhs.pScalingLists ) &&
               ( pSequenceParameterSetVui == rhs.pSequenceParameterSetVui );
      }

      bool operator!=( H264SequenceParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264SpsFlags   flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264ProfileIdc profile_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264ProfileIdc::eBaseline;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264LevelIdc        level_idc = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264LevelIdc::e1_0;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264ChromaFormatIdc chroma_format_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264ChromaFormatIdc::eMonochrome;
      uint8_t                                                       seq_parameter_set_id      = {};
      uint8_t                                                       bit_depth_luma_minus8     = {};
      uint8_t                                                       bit_depth_chroma_minus8   = {};
      uint8_t                                                       log2_max_frame_num_minus4 = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264PocType pic_order_cnt_type     = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264PocType::e0;
      int32_t                                                       offset_for_non_ref_pic = {};
      int32_t                                                       offset_for_top_to_bottom_field                   = {};
      uint8_t                                                       log2_max_pic_order_cnt_lsb_minus4                = {};
      uint8_t                                                       num_ref_frames_in_pic_order_cnt_cycle            = {};
      uint8_t                                                       max_num_ref_frames                               = {};
      uint8_t                                                       reserved1                                        = {};
      uint32_t                                                      pic_width_in_mbs_minus1                          = {};
      uint32_t                                                      pic_height_in_map_units_minus1                   = {};
      uint32_t                                                      frame_crop_left_offset                           = {};
      uint32_t                                                      frame_crop_right_offset                          = {};
      uint32_t                                                      frame_crop_top_offset                            = {};
      uint32_t                                                      frame_crop_bottom_offset                         = {};
      uint32_t                                                      reserved2                                        = {};
      const int32_t *                                               pOffsetForRefFrame                               = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264ScalingLists *            pScalingLists            = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264SequenceParameterSetVui * pSequenceParameterSetVui = {};
    };

    struct H264PpsFlags
    {
      using NativeType = StdVideoH264PpsFlags;

      operator StdVideoH264PpsFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH264PpsFlags *>( this );
      }

      operator StdVideoH264PpsFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH264PpsFlags *>( this );
      }

      operator StdVideoH264PpsFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH264PpsFlags *>( this );
      }

      operator StdVideoH264PpsFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH264PpsFlags *>( this );
      }

      bool operator==( H264PpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( transform_8x8_mode_flag == rhs.transform_8x8_mode_flag ) && ( redundant_pic_cnt_present_flag == rhs.redundant_pic_cnt_present_flag ) &&
               ( constrained_intra_pred_flag == rhs.constrained_intra_pred_flag ) &&
               ( deblocking_filter_control_present_flag == rhs.deblocking_filter_control_present_flag ) && ( weighted_pred_flag == rhs.weighted_pred_flag ) &&
               ( bottom_field_pic_order_in_frame_present_flag == rhs.bottom_field_pic_order_in_frame_present_flag ) &&
               ( entropy_coding_mode_flag == rhs.entropy_coding_mode_flag ) && ( pic_scaling_matrix_present_flag == rhs.pic_scaling_matrix_present_flag );
      }

      bool operator!=( H264PpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t transform_8x8_mode_flag                      : 1;
      uint32_t redundant_pic_cnt_present_flag               : 1;
      uint32_t constrained_intra_pred_flag                  : 1;
      uint32_t deblocking_filter_control_present_flag       : 1;
      uint32_t weighted_pred_flag                           : 1;
      uint32_t bottom_field_pic_order_in_frame_present_flag : 1;
      uint32_t entropy_coding_mode_flag                     : 1;
      uint32_t pic_scaling_matrix_present_flag              : 1;
    };

    struct H264PictureParameterSet
    {
      using NativeType = StdVideoH264PictureParameterSet;

      operator StdVideoH264PictureParameterSet const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH264PictureParameterSet *>( this );
      }

      operator StdVideoH264PictureParameterSet &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH264PictureParameterSet *>( this );
      }

      operator StdVideoH264PictureParameterSet const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH264PictureParameterSet *>( this );
      }

      operator StdVideoH264PictureParameterSet *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH264PictureParameterSet *>( this );
      }

      bool operator==( H264PictureParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( seq_parameter_set_id == rhs.seq_parameter_set_id ) && ( pic_parameter_set_id == rhs.pic_parameter_set_id ) &&
               ( num_ref_idx_l0_default_active_minus1 == rhs.num_ref_idx_l0_default_active_minus1 ) &&
               ( num_ref_idx_l1_default_active_minus1 == rhs.num_ref_idx_l1_default_active_minus1 ) && ( weighted_bipred_idc == rhs.weighted_bipred_idc ) &&
               ( pic_init_qp_minus26 == rhs.pic_init_qp_minus26 ) && ( pic_init_qs_minus26 == rhs.pic_init_qs_minus26 ) &&
               ( chroma_qp_index_offset == rhs.chroma_qp_index_offset ) && ( second_chroma_qp_index_offset == rhs.second_chroma_qp_index_offset ) &&
               ( pScalingLists == rhs.pScalingLists );
      }

      bool operator!=( H264PictureParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264PpsFlags          flags                                = {};
      uint8_t                                                                 seq_parameter_set_id                 = {};
      uint8_t                                                                 pic_parameter_set_id                 = {};
      uint8_t                                                                 num_ref_idx_l0_default_active_minus1 = {};
      uint8_t                                                                 num_ref_idx_l1_default_active_minus1 = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264WeightedBipredIdc weighted_bipred_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264WeightedBipredIdc::eDefault;
      int8_t                                                                     pic_init_qp_minus26           = {};
      int8_t                                                                     pic_init_qs_minus26           = {};
      int8_t                                                                     chroma_qp_index_offset        = {};
      int8_t                                                                     second_chroma_qp_index_offset = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264ScalingLists * pScalingLists                 = {};
    };

    //=== vulkan_video_codec_h264std_decode ===

    struct DecodeH264PictureInfoFlags
    {
      using NativeType = StdVideoDecodeH264PictureInfoFlags;

      operator StdVideoDecodeH264PictureInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeH264PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeH264PictureInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeH264PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeH264PictureInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeH264PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeH264PictureInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeH264PictureInfoFlags *>( this );
      }

      bool operator==( DecodeH264PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( field_pic_flag == rhs.field_pic_flag ) && ( is_intra == rhs.is_intra ) && ( IdrPicFlag == rhs.IdrPicFlag ) &&
               ( bottom_field_flag == rhs.bottom_field_flag ) && ( is_reference == rhs.is_reference ) &&
               ( complementary_field_pair == rhs.complementary_field_pair );
      }

      bool operator!=( DecodeH264PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t field_pic_flag           : 1;
      uint32_t is_intra                 : 1;
      uint32_t IdrPicFlag               : 1;
      uint32_t bottom_field_flag        : 1;
      uint32_t is_reference             : 1;
      uint32_t complementary_field_pair : 1;
    };

    struct DecodeH264PictureInfo
    {
      using NativeType = StdVideoDecodeH264PictureInfo;

      operator StdVideoDecodeH264PictureInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeH264PictureInfo *>( this );
      }

      operator StdVideoDecodeH264PictureInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeH264PictureInfo *>( this );
      }

      operator StdVideoDecodeH264PictureInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeH264PictureInfo *>( this );
      }

      operator StdVideoDecodeH264PictureInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeH264PictureInfo *>( this );
      }

      bool operator==( DecodeH264PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( seq_parameter_set_id == rhs.seq_parameter_set_id ) && ( pic_parameter_set_id == rhs.pic_parameter_set_id ) &&
               ( reserved1 == rhs.reserved1 ) && ( reserved2 == rhs.reserved2 ) && ( frame_num == rhs.frame_num ) && ( idr_pic_id == rhs.idr_pic_id ) &&
               ( PicOrderCnt == rhs.PicOrderCnt );
      }

      bool operator!=( DecodeH264PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::DecodeH264PictureInfoFlags                     flags                = {};
      uint8_t                                                                                          seq_parameter_set_id = {};
      uint8_t                                                                                          pic_parameter_set_id = {};
      uint8_t                                                                                          reserved1            = {};
      uint8_t                                                                                          reserved2            = {};
      uint16_t                                                                                         frame_num            = {};
      uint16_t                                                                                         idr_pic_id           = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int32_t, STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_LIST_SIZE> PicOrderCnt          = {};
    };

    struct DecodeH264ReferenceInfoFlags
    {
      using NativeType = StdVideoDecodeH264ReferenceInfoFlags;

      operator StdVideoDecodeH264ReferenceInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeH264ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeH264ReferenceInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeH264ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeH264ReferenceInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeH264ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeH264ReferenceInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeH264ReferenceInfoFlags *>( this );
      }

      bool operator==( DecodeH264ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( top_field_flag == rhs.top_field_flag ) && ( bottom_field_flag == rhs.bottom_field_flag ) &&
               ( used_for_long_term_reference == rhs.used_for_long_term_reference ) && ( is_non_existing == rhs.is_non_existing );
      }

      bool operator!=( DecodeH264ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t top_field_flag               : 1;
      uint32_t bottom_field_flag            : 1;
      uint32_t used_for_long_term_reference : 1;
      uint32_t is_non_existing              : 1;
    };

    struct DecodeH264ReferenceInfo
    {
      using NativeType = StdVideoDecodeH264ReferenceInfo;

      operator StdVideoDecodeH264ReferenceInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeH264ReferenceInfo *>( this );
      }

      operator StdVideoDecodeH264ReferenceInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeH264ReferenceInfo *>( this );
      }

      operator StdVideoDecodeH264ReferenceInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeH264ReferenceInfo *>( this );
      }

      operator StdVideoDecodeH264ReferenceInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeH264ReferenceInfo *>( this );
      }

      bool operator==( DecodeH264ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( FrameNum == rhs.FrameNum ) && ( reserved == rhs.reserved ) && ( PicOrderCnt == rhs.PicOrderCnt );
      }

      bool operator!=( DecodeH264ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::DecodeH264ReferenceInfoFlags                   flags       = {};
      uint16_t                                                                                         FrameNum    = {};
      uint16_t                                                                                         reserved    = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int32_t, STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_LIST_SIZE> PicOrderCnt = {};
    };

    //=== vulkan_video_codec_h264std_encode ===

    struct EncodeH264WeightTableFlags
    {
      using NativeType = StdVideoEncodeH264WeightTableFlags;

      operator StdVideoEncodeH264WeightTableFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264WeightTableFlags *>( this );
      }

      operator StdVideoEncodeH264WeightTableFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264WeightTableFlags *>( this );
      }

      operator StdVideoEncodeH264WeightTableFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264WeightTableFlags *>( this );
      }

      operator StdVideoEncodeH264WeightTableFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264WeightTableFlags *>( this );
      }

      bool operator==( EncodeH264WeightTableFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( luma_weight_l0_flag == rhs.luma_weight_l0_flag ) && ( chroma_weight_l0_flag == rhs.chroma_weight_l0_flag ) &&
               ( luma_weight_l1_flag == rhs.luma_weight_l1_flag ) && ( chroma_weight_l1_flag == rhs.chroma_weight_l1_flag );
      }

      bool operator!=( EncodeH264WeightTableFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t luma_weight_l0_flag   = {};
      uint32_t chroma_weight_l0_flag = {};
      uint32_t luma_weight_l1_flag   = {};
      uint32_t chroma_weight_l1_flag = {};
    };

    struct EncodeH264WeightTable
    {
      using NativeType = StdVideoEncodeH264WeightTable;

      operator StdVideoEncodeH264WeightTable const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264WeightTable *>( this );
      }

      operator StdVideoEncodeH264WeightTable &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264WeightTable *>( this );
      }

      operator StdVideoEncodeH264WeightTable const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264WeightTable *>( this );
      }

      operator StdVideoEncodeH264WeightTable *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264WeightTable *>( this );
      }

      bool operator==( EncodeH264WeightTable const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( luma_log2_weight_denom == rhs.luma_log2_weight_denom ) &&
               ( chroma_log2_weight_denom == rhs.chroma_log2_weight_denom ) && ( luma_weight_l0 == rhs.luma_weight_l0 ) &&
               ( luma_offset_l0 == rhs.luma_offset_l0 ) && ( chroma_weight_l0 == rhs.chroma_weight_l0 ) && ( chroma_offset_l0 == rhs.chroma_offset_l0 ) &&
               ( luma_weight_l1 == rhs.luma_weight_l1 ) && ( luma_offset_l1 == rhs.luma_offset_l1 ) && ( chroma_weight_l1 == rhs.chroma_weight_l1 ) &&
               ( chroma_offset_l1 == rhs.chroma_offset_l1 );
      }

      bool operator!=( EncodeH264WeightTable const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264WeightTableFlags                                    flags                    = {};
      uint8_t                                                                                                         luma_log2_weight_denom   = {};
      uint8_t                                                                                                         chroma_log2_weight_denom = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF>                                   luma_weight_l0           = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF>                                   luma_offset_l0           = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF, STD_VIDEO_H264_MAX_CHROMA_PLANES> chroma_weight_l0         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF, STD_VIDEO_H264_MAX_CHROMA_PLANES> chroma_offset_l0         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF>                                   luma_weight_l1           = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF>                                   luma_offset_l1           = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF, STD_VIDEO_H264_MAX_CHROMA_PLANES> chroma_weight_l1         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF, STD_VIDEO_H264_MAX_CHROMA_PLANES> chroma_offset_l1         = {};
    };

    struct EncodeH264SliceHeaderFlags
    {
      using NativeType = StdVideoEncodeH264SliceHeaderFlags;

      operator StdVideoEncodeH264SliceHeaderFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264SliceHeaderFlags *>( this );
      }

      operator StdVideoEncodeH264SliceHeaderFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264SliceHeaderFlags *>( this );
      }

      operator StdVideoEncodeH264SliceHeaderFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264SliceHeaderFlags *>( this );
      }

      operator StdVideoEncodeH264SliceHeaderFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264SliceHeaderFlags *>( this );
      }

      bool operator==( EncodeH264SliceHeaderFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( direct_spatial_mv_pred_flag == rhs.direct_spatial_mv_pred_flag ) &&
               ( num_ref_idx_active_override_flag == rhs.num_ref_idx_active_override_flag ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeH264SliceHeaderFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t direct_spatial_mv_pred_flag      : 1;
      uint32_t num_ref_idx_active_override_flag : 1;
      uint32_t reserved                         : 30;
    };

    struct EncodeH264PictureInfoFlags
    {
      using NativeType = StdVideoEncodeH264PictureInfoFlags;

      operator StdVideoEncodeH264PictureInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeH264PictureInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeH264PictureInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeH264PictureInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264PictureInfoFlags *>( this );
      }

      bool operator==( EncodeH264PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( IdrPicFlag == rhs.IdrPicFlag ) && ( is_reference == rhs.is_reference ) &&
               ( no_output_of_prior_pics_flag == rhs.no_output_of_prior_pics_flag ) && ( long_term_reference_flag == rhs.long_term_reference_flag ) &&
               ( adaptive_ref_pic_marking_mode_flag == rhs.adaptive_ref_pic_marking_mode_flag ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeH264PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t IdrPicFlag                         : 1;
      uint32_t is_reference                       : 1;
      uint32_t no_output_of_prior_pics_flag       : 1;
      uint32_t long_term_reference_flag           : 1;
      uint32_t adaptive_ref_pic_marking_mode_flag : 1;
      uint32_t reserved                           : 27;
    };

    struct EncodeH264ReferenceInfoFlags
    {
      using NativeType = StdVideoEncodeH264ReferenceInfoFlags;

      operator StdVideoEncodeH264ReferenceInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeH264ReferenceInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeH264ReferenceInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeH264ReferenceInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264ReferenceInfoFlags *>( this );
      }

      bool operator==( EncodeH264ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( used_for_long_term_reference == rhs.used_for_long_term_reference ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeH264ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t used_for_long_term_reference : 1;
      uint32_t reserved                     : 31;
    };

    struct EncodeH264ReferenceListsInfoFlags
    {
      using NativeType = StdVideoEncodeH264ReferenceListsInfoFlags;

      operator StdVideoEncodeH264ReferenceListsInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264ReferenceListsInfoFlags *>( this );
      }

      operator StdVideoEncodeH264ReferenceListsInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264ReferenceListsInfoFlags *>( this );
      }

      operator StdVideoEncodeH264ReferenceListsInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264ReferenceListsInfoFlags *>( this );
      }

      operator StdVideoEncodeH264ReferenceListsInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264ReferenceListsInfoFlags *>( this );
      }

      bool operator==( EncodeH264ReferenceListsInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( ref_pic_list_modification_flag_l0 == rhs.ref_pic_list_modification_flag_l0 ) &&
               ( ref_pic_list_modification_flag_l1 == rhs.ref_pic_list_modification_flag_l1 ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeH264ReferenceListsInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t ref_pic_list_modification_flag_l0 : 1;
      uint32_t ref_pic_list_modification_flag_l1 : 1;
      uint32_t reserved                          : 30;
    };

    struct EncodeH264RefListModEntry
    {
      using NativeType = StdVideoEncodeH264RefListModEntry;

      operator StdVideoEncodeH264RefListModEntry const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264RefListModEntry *>( this );
      }

      operator StdVideoEncodeH264RefListModEntry &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264RefListModEntry *>( this );
      }

      operator StdVideoEncodeH264RefListModEntry const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264RefListModEntry *>( this );
      }

      operator StdVideoEncodeH264RefListModEntry *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264RefListModEntry *>( this );
      }

      bool operator==( EncodeH264RefListModEntry const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( modification_of_pic_nums_idc == rhs.modification_of_pic_nums_idc ) && ( abs_diff_pic_num_minus1 == rhs.abs_diff_pic_num_minus1 ) &&
               ( long_term_pic_num == rhs.long_term_pic_num );
      }

      bool operator!=( EncodeH264RefListModEntry const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264ModificationOfPicNumsIdc modification_of_pic_nums_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264ModificationOfPicNumsIdc::eShortTermSubtract;
      uint16_t abs_diff_pic_num_minus1 = {};
      uint16_t long_term_pic_num       = {};
    };

    struct EncodeH264RefPicMarkingEntry
    {
      using NativeType = StdVideoEncodeH264RefPicMarkingEntry;

      operator StdVideoEncodeH264RefPicMarkingEntry const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264RefPicMarkingEntry *>( this );
      }

      operator StdVideoEncodeH264RefPicMarkingEntry &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264RefPicMarkingEntry *>( this );
      }

      operator StdVideoEncodeH264RefPicMarkingEntry const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264RefPicMarkingEntry *>( this );
      }

      operator StdVideoEncodeH264RefPicMarkingEntry *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264RefPicMarkingEntry *>( this );
      }

      bool operator==( EncodeH264RefPicMarkingEntry const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( memory_management_control_operation == rhs.memory_management_control_operation ) &&
               ( difference_of_pic_nums_minus1 == rhs.difference_of_pic_nums_minus1 ) && ( long_term_pic_num == rhs.long_term_pic_num ) &&
               ( long_term_frame_idx == rhs.long_term_frame_idx ) && ( max_long_term_frame_idx_plus1 == rhs.max_long_term_frame_idx_plus1 );
      }

      bool operator!=( EncodeH264RefPicMarkingEntry const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264MemMgmtControlOp memory_management_control_operation =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264MemMgmtControlOp::eEnd;
      uint16_t difference_of_pic_nums_minus1 = {};
      uint16_t long_term_pic_num             = {};
      uint16_t long_term_frame_idx           = {};
      uint16_t max_long_term_frame_idx_plus1 = {};
    };

    struct EncodeH264ReferenceListsInfo
    {
      using NativeType = StdVideoEncodeH264ReferenceListsInfo;

      operator StdVideoEncodeH264ReferenceListsInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264ReferenceListsInfo *>( this );
      }

      operator StdVideoEncodeH264ReferenceListsInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264ReferenceListsInfo *>( this );
      }

      operator StdVideoEncodeH264ReferenceListsInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264ReferenceListsInfo *>( this );
      }

      operator StdVideoEncodeH264ReferenceListsInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264ReferenceListsInfo *>( this );
      }

      bool operator==( EncodeH264ReferenceListsInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( num_ref_idx_l0_active_minus1 == rhs.num_ref_idx_l0_active_minus1 ) &&
               ( num_ref_idx_l1_active_minus1 == rhs.num_ref_idx_l1_active_minus1 ) && ( RefPicList0 == rhs.RefPicList0 ) &&
               ( RefPicList1 == rhs.RefPicList1 ) && ( refList0ModOpCount == rhs.refList0ModOpCount ) && ( refList1ModOpCount == rhs.refList1ModOpCount ) &&
               ( refPicMarkingOpCount == rhs.refPicMarkingOpCount ) && ( reserved1 == rhs.reserved1 ) &&
               ( pRefList0ModOperations == rhs.pRefList0ModOperations ) && ( pRefList1ModOperations == rhs.pRefList1ModOperations ) &&
               ( pRefPicMarkingOperations == rhs.pRefPicMarkingOperations );
      }

      bool operator!=( EncodeH264ReferenceListsInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264ReferenceListsInfoFlags    flags                        = {};
      uint8_t                                                                                num_ref_idx_l0_active_minus1 = {};
      uint8_t                                                                                num_ref_idx_l1_active_minus1 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF>         RefPicList0                  = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H264_MAX_NUM_LIST_REF>         RefPicList1                  = {};
      uint8_t                                                                                refList0ModOpCount           = {};
      uint8_t                                                                                refList1ModOpCount           = {};
      uint8_t                                                                                refPicMarkingOpCount         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, 7>                                       reserved1                    = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264RefListModEntry *    pRefList0ModOperations       = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264RefListModEntry *    pRefList1ModOperations       = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264RefPicMarkingEntry * pRefPicMarkingOperations     = {};
    };

    struct EncodeH264PictureInfo
    {
      using NativeType = StdVideoEncodeH264PictureInfo;

      operator StdVideoEncodeH264PictureInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264PictureInfo *>( this );
      }

      operator StdVideoEncodeH264PictureInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264PictureInfo *>( this );
      }

      operator StdVideoEncodeH264PictureInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264PictureInfo *>( this );
      }

      operator StdVideoEncodeH264PictureInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264PictureInfo *>( this );
      }

      bool operator==( EncodeH264PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( seq_parameter_set_id == rhs.seq_parameter_set_id ) && ( pic_parameter_set_id == rhs.pic_parameter_set_id ) &&
               ( idr_pic_id == rhs.idr_pic_id ) && ( primary_pic_type == rhs.primary_pic_type ) && ( frame_num == rhs.frame_num ) &&
               ( PicOrderCnt == rhs.PicOrderCnt ) && ( temporal_id == rhs.temporal_id ) && ( reserved1 == rhs.reserved1 ) && ( pRefLists == rhs.pRefLists );
      }

      bool operator!=( EncodeH264PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264PictureInfoFlags flags                = {};
      uint8_t                                                                      seq_parameter_set_id = {};
      uint8_t                                                                      pic_parameter_set_id = {};
      uint16_t                                                                     idr_pic_id           = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264PictureType            primary_pic_type =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264PictureType::eP;
      uint32_t                                                                               frame_num   = {};
      int32_t                                                                                PicOrderCnt = {};
      uint8_t                                                                                temporal_id = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, 3>                                       reserved1   = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264ReferenceListsInfo * pRefLists   = {};
    };

    struct EncodeH264ReferenceInfo
    {
      using NativeType = StdVideoEncodeH264ReferenceInfo;

      operator StdVideoEncodeH264ReferenceInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264ReferenceInfo *>( this );
      }

      operator StdVideoEncodeH264ReferenceInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264ReferenceInfo *>( this );
      }

      operator StdVideoEncodeH264ReferenceInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264ReferenceInfo *>( this );
      }

      operator StdVideoEncodeH264ReferenceInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264ReferenceInfo *>( this );
      }

      bool operator==( EncodeH264ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( primary_pic_type == rhs.primary_pic_type ) && ( FrameNum == rhs.FrameNum ) && ( PicOrderCnt == rhs.PicOrderCnt ) &&
               ( long_term_pic_num == rhs.long_term_pic_num ) && ( long_term_frame_idx == rhs.long_term_frame_idx ) && ( temporal_id == rhs.temporal_id );
      }

      bool operator!=( EncodeH264ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264ReferenceInfoFlags flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264PictureType              primary_pic_type =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264PictureType::eP;
      uint32_t FrameNum            = {};
      int32_t  PicOrderCnt         = {};
      uint16_t long_term_pic_num   = {};
      uint16_t long_term_frame_idx = {};
      uint8_t  temporal_id         = {};
    };

    struct EncodeH264SliceHeader
    {
      using NativeType = StdVideoEncodeH264SliceHeader;

      operator StdVideoEncodeH264SliceHeader const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH264SliceHeader *>( this );
      }

      operator StdVideoEncodeH264SliceHeader &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH264SliceHeader *>( this );
      }

      operator StdVideoEncodeH264SliceHeader const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH264SliceHeader *>( this );
      }

      operator StdVideoEncodeH264SliceHeader *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH264SliceHeader *>( this );
      }

      bool operator==( EncodeH264SliceHeader const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( first_mb_in_slice == rhs.first_mb_in_slice ) && ( slice_type == rhs.slice_type ) &&
               ( slice_alpha_c0_offset_div2 == rhs.slice_alpha_c0_offset_div2 ) && ( slice_beta_offset_div2 == rhs.slice_beta_offset_div2 ) &&
               ( slice_qp_delta == rhs.slice_qp_delta ) && ( reserved1 == rhs.reserved1 ) && ( cabac_init_idc == rhs.cabac_init_idc ) &&
               ( disable_deblocking_filter_idc == rhs.disable_deblocking_filter_idc ) && ( pWeightTable == rhs.pWeightTable );
      }

      bool operator!=( EncodeH264SliceHeader const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264SliceHeaderFlags flags             = {};
      uint32_t                                                                     first_mb_in_slice = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264SliceType    slice_type = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264SliceType::eP;
      int8_t                                                             slice_alpha_c0_offset_div2 = {};
      int8_t                                                             slice_beta_offset_div2     = {};
      int8_t                                                             slice_qp_delta             = {};
      uint8_t                                                            reserved1                  = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264CabacInitIdc cabac_init_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264CabacInitIdc::e0;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264DisableDeblockingFilterIdc disable_deblocking_filter_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H264DisableDeblockingFilterIdc::eDisabled;
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH264WeightTable * pWeightTable = {};
    };

    //=== vulkan_video_codec_h265std ===

    struct H265DecPicBufMgr
    {
      using NativeType = StdVideoH265DecPicBufMgr;

      operator StdVideoH265DecPicBufMgr const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265DecPicBufMgr *>( this );
      }

      operator StdVideoH265DecPicBufMgr &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265DecPicBufMgr *>( this );
      }

      operator StdVideoH265DecPicBufMgr const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265DecPicBufMgr *>( this );
      }

      operator StdVideoH265DecPicBufMgr *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265DecPicBufMgr *>( this );
      }

      bool operator==( H265DecPicBufMgr const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( max_latency_increase_plus1 == rhs.max_latency_increase_plus1 ) && ( max_dec_pic_buffering_minus1 == rhs.max_dec_pic_buffering_minus1 ) &&
               ( max_num_reorder_pics == rhs.max_num_reorder_pics );
      }

      bool operator!=( H265DecPicBufMgr const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_H265_SUBLAYERS_LIST_SIZE> max_latency_increase_plus1   = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_SUBLAYERS_LIST_SIZE>  max_dec_pic_buffering_minus1 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_SUBLAYERS_LIST_SIZE>  max_num_reorder_pics         = {};
    };

    struct H265SubLayerHrdParameters
    {
      using NativeType = StdVideoH265SubLayerHrdParameters;

      operator StdVideoH265SubLayerHrdParameters const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265SubLayerHrdParameters *>( this );
      }

      operator StdVideoH265SubLayerHrdParameters &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265SubLayerHrdParameters *>( this );
      }

      operator StdVideoH265SubLayerHrdParameters const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265SubLayerHrdParameters *>( this );
      }

      operator StdVideoH265SubLayerHrdParameters *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265SubLayerHrdParameters *>( this );
      }

      bool operator==( H265SubLayerHrdParameters const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( bit_rate_value_minus1 == rhs.bit_rate_value_minus1 ) && ( cpb_size_value_minus1 == rhs.cpb_size_value_minus1 ) &&
               ( cpb_size_du_value_minus1 == rhs.cpb_size_du_value_minus1 ) && ( bit_rate_du_value_minus1 == rhs.bit_rate_du_value_minus1 ) &&
               ( cbr_flag == rhs.cbr_flag );
      }

      bool operator!=( H265SubLayerHrdParameters const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_H265_CPB_CNT_LIST_SIZE> bit_rate_value_minus1    = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_H265_CPB_CNT_LIST_SIZE> cpb_size_value_minus1    = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_H265_CPB_CNT_LIST_SIZE> cpb_size_du_value_minus1 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_H265_CPB_CNT_LIST_SIZE> bit_rate_du_value_minus1 = {};
      uint32_t                                                                         cbr_flag                 = {};
    };

    struct H265HrdFlags
    {
      using NativeType = StdVideoH265HrdFlags;

      operator StdVideoH265HrdFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265HrdFlags *>( this );
      }

      operator StdVideoH265HrdFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265HrdFlags *>( this );
      }

      operator StdVideoH265HrdFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265HrdFlags *>( this );
      }

      operator StdVideoH265HrdFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265HrdFlags *>( this );
      }

      bool operator==( H265HrdFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( nal_hrd_parameters_present_flag == rhs.nal_hrd_parameters_present_flag ) &&
               ( vcl_hrd_parameters_present_flag == rhs.vcl_hrd_parameters_present_flag ) &&
               ( sub_pic_hrd_params_present_flag == rhs.sub_pic_hrd_params_present_flag ) &&
               ( sub_pic_cpb_params_in_pic_timing_sei_flag == rhs.sub_pic_cpb_params_in_pic_timing_sei_flag ) &&
               ( fixed_pic_rate_general_flag == rhs.fixed_pic_rate_general_flag ) && ( fixed_pic_rate_within_cvs_flag == rhs.fixed_pic_rate_within_cvs_flag ) &&
               ( low_delay_hrd_flag == rhs.low_delay_hrd_flag );
      }

      bool operator!=( H265HrdFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t nal_hrd_parameters_present_flag           : 1;
      uint32_t vcl_hrd_parameters_present_flag           : 1;
      uint32_t sub_pic_hrd_params_present_flag           : 1;
      uint32_t sub_pic_cpb_params_in_pic_timing_sei_flag : 1;
      uint32_t fixed_pic_rate_general_flag               : 8;
      uint32_t fixed_pic_rate_within_cvs_flag            : 8;
      uint32_t low_delay_hrd_flag                        : 8;
    };

    struct H265HrdParameters
    {
      using NativeType = StdVideoH265HrdParameters;

      operator StdVideoH265HrdParameters const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265HrdParameters *>( this );
      }

      operator StdVideoH265HrdParameters &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265HrdParameters *>( this );
      }

      operator StdVideoH265HrdParameters const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265HrdParameters *>( this );
      }

      operator StdVideoH265HrdParameters *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265HrdParameters *>( this );
      }

      bool operator==( H265HrdParameters const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( tick_divisor_minus2 == rhs.tick_divisor_minus2 ) &&
               ( du_cpb_removal_delay_increment_length_minus1 == rhs.du_cpb_removal_delay_increment_length_minus1 ) &&
               ( dpb_output_delay_du_length_minus1 == rhs.dpb_output_delay_du_length_minus1 ) && ( bit_rate_scale == rhs.bit_rate_scale ) &&
               ( cpb_size_scale == rhs.cpb_size_scale ) && ( cpb_size_du_scale == rhs.cpb_size_du_scale ) &&
               ( initial_cpb_removal_delay_length_minus1 == rhs.initial_cpb_removal_delay_length_minus1 ) &&
               ( au_cpb_removal_delay_length_minus1 == rhs.au_cpb_removal_delay_length_minus1 ) &&
               ( dpb_output_delay_length_minus1 == rhs.dpb_output_delay_length_minus1 ) && ( cpb_cnt_minus1 == rhs.cpb_cnt_minus1 ) &&
               ( elemental_duration_in_tc_minus1 == rhs.elemental_duration_in_tc_minus1 ) && ( reserved == rhs.reserved ) &&
               ( pSubLayerHrdParametersNal == rhs.pSubLayerHrdParametersNal ) && ( pSubLayerHrdParametersVcl == rhs.pSubLayerHrdParametersVcl );
      }

      bool operator!=( H265HrdParameters const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265HrdFlags                      flags                                        = {};
      uint8_t                                                                             tick_divisor_minus2                          = {};
      uint8_t                                                                             du_cpb_removal_delay_increment_length_minus1 = {};
      uint8_t                                                                             dpb_output_delay_du_length_minus1            = {};
      uint8_t                                                                             bit_rate_scale                               = {};
      uint8_t                                                                             cpb_size_scale                               = {};
      uint8_t                                                                             cpb_size_du_scale                            = {};
      uint8_t                                                                             initial_cpb_removal_delay_length_minus1      = {};
      uint8_t                                                                             au_cpb_removal_delay_length_minus1           = {};
      uint8_t                                                                             dpb_output_delay_length_minus1               = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_SUBLAYERS_LIST_SIZE>   cpb_cnt_minus1                               = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint16_t, STD_VIDEO_H265_SUBLAYERS_LIST_SIZE>  elemental_duration_in_tc_minus1              = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint16_t, 3>                                   reserved                                     = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265SubLayerHrdParameters * pSubLayerHrdParametersNal                    = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265SubLayerHrdParameters * pSubLayerHrdParametersVcl                    = {};
    };

    struct H265VpsFlags
    {
      using NativeType = StdVideoH265VpsFlags;

      operator StdVideoH265VpsFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265VpsFlags *>( this );
      }

      operator StdVideoH265VpsFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265VpsFlags *>( this );
      }

      operator StdVideoH265VpsFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265VpsFlags *>( this );
      }

      operator StdVideoH265VpsFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265VpsFlags *>( this );
      }

      bool operator==( H265VpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( vps_temporal_id_nesting_flag == rhs.vps_temporal_id_nesting_flag ) &&
               ( vps_sub_layer_ordering_info_present_flag == rhs.vps_sub_layer_ordering_info_present_flag ) &&
               ( vps_timing_info_present_flag == rhs.vps_timing_info_present_flag ) &&
               ( vps_poc_proportional_to_timing_flag == rhs.vps_poc_proportional_to_timing_flag );
      }

      bool operator!=( H265VpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t vps_temporal_id_nesting_flag             : 1;
      uint32_t vps_sub_layer_ordering_info_present_flag : 1;
      uint32_t vps_timing_info_present_flag             : 1;
      uint32_t vps_poc_proportional_to_timing_flag      : 1;
    };

    struct H265ProfileTierLevelFlags
    {
      using NativeType = StdVideoH265ProfileTierLevelFlags;

      operator StdVideoH265ProfileTierLevelFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265ProfileTierLevelFlags *>( this );
      }

      operator StdVideoH265ProfileTierLevelFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265ProfileTierLevelFlags *>( this );
      }

      operator StdVideoH265ProfileTierLevelFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265ProfileTierLevelFlags *>( this );
      }

      operator StdVideoH265ProfileTierLevelFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265ProfileTierLevelFlags *>( this );
      }

      bool operator==( H265ProfileTierLevelFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( general_tier_flag == rhs.general_tier_flag ) && ( general_progressive_source_flag == rhs.general_progressive_source_flag ) &&
               ( general_interlaced_source_flag == rhs.general_interlaced_source_flag ) &&
               ( general_non_packed_constraint_flag == rhs.general_non_packed_constraint_flag ) &&
               ( general_frame_only_constraint_flag == rhs.general_frame_only_constraint_flag );
      }

      bool operator!=( H265ProfileTierLevelFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t general_tier_flag                  : 1;
      uint32_t general_progressive_source_flag    : 1;
      uint32_t general_interlaced_source_flag     : 1;
      uint32_t general_non_packed_constraint_flag : 1;
      uint32_t general_frame_only_constraint_flag : 1;
    };

    struct H265ProfileTierLevel
    {
      using NativeType = StdVideoH265ProfileTierLevel;

      operator StdVideoH265ProfileTierLevel const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265ProfileTierLevel *>( this );
      }

      operator StdVideoH265ProfileTierLevel &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265ProfileTierLevel *>( this );
      }

      operator StdVideoH265ProfileTierLevel const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265ProfileTierLevel *>( this );
      }

      operator StdVideoH265ProfileTierLevel *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265ProfileTierLevel *>( this );
      }

      bool operator==( H265ProfileTierLevel const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( general_profile_idc == rhs.general_profile_idc ) && ( general_level_idc == rhs.general_level_idc );
      }

      bool operator!=( H265ProfileTierLevel const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ProfileTierLevelFlags flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ProfileIdc            general_profile_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ProfileIdc::eMain;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265LevelIdc general_level_idc = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265LevelIdc::e1_0;
    };

    struct H265VideoParameterSet
    {
      using NativeType = StdVideoH265VideoParameterSet;

      operator StdVideoH265VideoParameterSet const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265VideoParameterSet *>( this );
      }

      operator StdVideoH265VideoParameterSet &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265VideoParameterSet *>( this );
      }

      operator StdVideoH265VideoParameterSet const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265VideoParameterSet *>( this );
      }

      operator StdVideoH265VideoParameterSet *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265VideoParameterSet *>( this );
      }

      bool operator==( H265VideoParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( vps_video_parameter_set_id == rhs.vps_video_parameter_set_id ) &&
               ( vps_max_sub_layers_minus1 == rhs.vps_max_sub_layers_minus1 ) && ( reserved1 == rhs.reserved1 ) && ( reserved2 == rhs.reserved2 ) &&
               ( vps_num_units_in_tick == rhs.vps_num_units_in_tick ) && ( vps_time_scale == rhs.vps_time_scale ) &&
               ( vps_num_ticks_poc_diff_one_minus1 == rhs.vps_num_ticks_poc_diff_one_minus1 ) && ( reserved3 == rhs.reserved3 ) &&
               ( pDecPicBufMgr == rhs.pDecPicBufMgr ) && ( pHrdParameters == rhs.pHrdParameters ) && ( pProfileTierLevel == rhs.pProfileTierLevel );
      }

      bool operator!=( H265VideoParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265VpsFlags                 flags                             = {};
      uint8_t                                                                        vps_video_parameter_set_id        = {};
      uint8_t                                                                        vps_max_sub_layers_minus1         = {};
      uint8_t                                                                        reserved1                         = {};
      uint8_t                                                                        reserved2                         = {};
      uint32_t                                                                       vps_num_units_in_tick             = {};
      uint32_t                                                                       vps_time_scale                    = {};
      uint32_t                                                                       vps_num_ticks_poc_diff_one_minus1 = {};
      uint32_t                                                                       reserved3                         = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265DecPicBufMgr *     pDecPicBufMgr                     = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265HrdParameters *    pHrdParameters                    = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ProfileTierLevel * pProfileTierLevel                 = {};
    };

    struct H265ScalingLists
    {
      using NativeType = StdVideoH265ScalingLists;

      operator StdVideoH265ScalingLists const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265ScalingLists *>( this );
      }

      operator StdVideoH265ScalingLists &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265ScalingLists *>( this );
      }

      operator StdVideoH265ScalingLists const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265ScalingLists *>( this );
      }

      operator StdVideoH265ScalingLists *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265ScalingLists *>( this );
      }

      bool operator==( H265ScalingLists const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( ScalingList4x4 == rhs.ScalingList4x4 ) && ( ScalingList8x8 == rhs.ScalingList8x8 ) && ( ScalingList16x16 == rhs.ScalingList16x16 ) &&
               ( ScalingList32x32 == rhs.ScalingList32x32 ) && ( ScalingListDCCoef16x16 == rhs.ScalingListDCCoef16x16 ) &&
               ( ScalingListDCCoef32x32 == rhs.ScalingListDCCoef32x32 );
      }

      bool operator!=( H265ScalingLists const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<uint8_t, STD_VIDEO_H265_SCALING_LIST_4X4_NUM_LISTS, STD_VIDEO_H265_SCALING_LIST_4X4_NUM_ELEMENTS>
        ScalingList4x4 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<uint8_t, STD_VIDEO_H265_SCALING_LIST_8X8_NUM_LISTS, STD_VIDEO_H265_SCALING_LIST_8X8_NUM_ELEMENTS>
        ScalingList8x8 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<uint8_t, STD_VIDEO_H265_SCALING_LIST_16X16_NUM_LISTS, STD_VIDEO_H265_SCALING_LIST_16X16_NUM_ELEMENTS>
        ScalingList16x16 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<uint8_t, STD_VIDEO_H265_SCALING_LIST_32X32_NUM_LISTS, STD_VIDEO_H265_SCALING_LIST_32X32_NUM_ELEMENTS>
                                                                                                 ScalingList32x32       = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_SCALING_LIST_16X16_NUM_LISTS> ScalingListDCCoef16x16 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_SCALING_LIST_32X32_NUM_LISTS> ScalingListDCCoef32x32 = {};
    };

    struct H265SpsVuiFlags
    {
      using NativeType = StdVideoH265SpsVuiFlags;

      operator StdVideoH265SpsVuiFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265SpsVuiFlags *>( this );
      }

      operator StdVideoH265SpsVuiFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265SpsVuiFlags *>( this );
      }

      operator StdVideoH265SpsVuiFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265SpsVuiFlags *>( this );
      }

      operator StdVideoH265SpsVuiFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265SpsVuiFlags *>( this );
      }

      bool operator==( H265SpsVuiFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( aspect_ratio_info_present_flag == rhs.aspect_ratio_info_present_flag ) && ( overscan_info_present_flag == rhs.overscan_info_present_flag ) &&
               ( overscan_appropriate_flag == rhs.overscan_appropriate_flag ) && ( video_signal_type_present_flag == rhs.video_signal_type_present_flag ) &&
               ( video_full_range_flag == rhs.video_full_range_flag ) && ( colour_description_present_flag == rhs.colour_description_present_flag ) &&
               ( chroma_loc_info_present_flag == rhs.chroma_loc_info_present_flag ) &&
               ( neutral_chroma_indication_flag == rhs.neutral_chroma_indication_flag ) && ( field_seq_flag == rhs.field_seq_flag ) &&
               ( frame_field_info_present_flag == rhs.frame_field_info_present_flag ) && ( default_display_window_flag == rhs.default_display_window_flag ) &&
               ( vui_timing_info_present_flag == rhs.vui_timing_info_present_flag ) &&
               ( vui_poc_proportional_to_timing_flag == rhs.vui_poc_proportional_to_timing_flag ) &&
               ( vui_hrd_parameters_present_flag == rhs.vui_hrd_parameters_present_flag ) && ( bitstream_restriction_flag == rhs.bitstream_restriction_flag ) &&
               ( tiles_fixed_structure_flag == rhs.tiles_fixed_structure_flag ) &&
               ( motion_vectors_over_pic_boundaries_flag == rhs.motion_vectors_over_pic_boundaries_flag ) &&
               ( restricted_ref_pic_lists_flag == rhs.restricted_ref_pic_lists_flag );
      }

      bool operator!=( H265SpsVuiFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t aspect_ratio_info_present_flag          : 1;
      uint32_t overscan_info_present_flag              : 1;
      uint32_t overscan_appropriate_flag               : 1;
      uint32_t video_signal_type_present_flag          : 1;
      uint32_t video_full_range_flag                   : 1;
      uint32_t colour_description_present_flag         : 1;
      uint32_t chroma_loc_info_present_flag            : 1;
      uint32_t neutral_chroma_indication_flag          : 1;
      uint32_t field_seq_flag                          : 1;
      uint32_t frame_field_info_present_flag           : 1;
      uint32_t default_display_window_flag             : 1;
      uint32_t vui_timing_info_present_flag            : 1;
      uint32_t vui_poc_proportional_to_timing_flag     : 1;
      uint32_t vui_hrd_parameters_present_flag         : 1;
      uint32_t bitstream_restriction_flag              : 1;
      uint32_t tiles_fixed_structure_flag              : 1;
      uint32_t motion_vectors_over_pic_boundaries_flag : 1;
      uint32_t restricted_ref_pic_lists_flag           : 1;
    };

    struct H265SequenceParameterSetVui
    {
      using NativeType = StdVideoH265SequenceParameterSetVui;

      operator StdVideoH265SequenceParameterSetVui const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265SequenceParameterSetVui *>( this );
      }

      operator StdVideoH265SequenceParameterSetVui &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265SequenceParameterSetVui *>( this );
      }

      operator StdVideoH265SequenceParameterSetVui const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265SequenceParameterSetVui *>( this );
      }

      operator StdVideoH265SequenceParameterSetVui *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265SequenceParameterSetVui *>( this );
      }

      bool operator==( H265SequenceParameterSetVui const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( aspect_ratio_idc == rhs.aspect_ratio_idc ) && ( sar_width == rhs.sar_width ) && ( sar_height == rhs.sar_height ) &&
               ( video_format == rhs.video_format ) && ( colour_primaries == rhs.colour_primaries ) &&
               ( transfer_characteristics == rhs.transfer_characteristics ) && ( matrix_coeffs == rhs.matrix_coeffs ) &&
               ( chroma_sample_loc_type_top_field == rhs.chroma_sample_loc_type_top_field ) &&
               ( chroma_sample_loc_type_bottom_field == rhs.chroma_sample_loc_type_bottom_field ) && ( reserved1 == rhs.reserved1 ) &&
               ( reserved2 == rhs.reserved2 ) && ( def_disp_win_left_offset == rhs.def_disp_win_left_offset ) &&
               ( def_disp_win_right_offset == rhs.def_disp_win_right_offset ) && ( def_disp_win_top_offset == rhs.def_disp_win_top_offset ) &&
               ( def_disp_win_bottom_offset == rhs.def_disp_win_bottom_offset ) && ( vui_num_units_in_tick == rhs.vui_num_units_in_tick ) &&
               ( vui_time_scale == rhs.vui_time_scale ) && ( vui_num_ticks_poc_diff_one_minus1 == rhs.vui_num_ticks_poc_diff_one_minus1 ) &&
               ( min_spatial_segmentation_idc == rhs.min_spatial_segmentation_idc ) && ( reserved3 == rhs.reserved3 ) &&
               ( max_bytes_per_pic_denom == rhs.max_bytes_per_pic_denom ) && ( max_bits_per_min_cu_denom == rhs.max_bits_per_min_cu_denom ) &&
               ( log2_max_mv_length_horizontal == rhs.log2_max_mv_length_horizontal ) && ( log2_max_mv_length_vertical == rhs.log2_max_mv_length_vertical ) &&
               ( pHrdParameters == rhs.pHrdParameters );
      }

      bool operator!=( H265SequenceParameterSetVui const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265SpsVuiFlags    flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265AspectRatioIdc aspect_ratio_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265AspectRatioIdc::eUnspecified;
      uint16_t                                                                    sar_width                           = {};
      uint16_t                                                                    sar_height                          = {};
      uint8_t                                                                     video_format                        = {};
      uint8_t                                                                     colour_primaries                    = {};
      uint8_t                                                                     transfer_characteristics            = {};
      uint8_t                                                                     matrix_coeffs                       = {};
      uint8_t                                                                     chroma_sample_loc_type_top_field    = {};
      uint8_t                                                                     chroma_sample_loc_type_bottom_field = {};
      uint8_t                                                                     reserved1                           = {};
      uint8_t                                                                     reserved2                           = {};
      uint16_t                                                                    def_disp_win_left_offset            = {};
      uint16_t                                                                    def_disp_win_right_offset           = {};
      uint16_t                                                                    def_disp_win_top_offset             = {};
      uint16_t                                                                    def_disp_win_bottom_offset          = {};
      uint32_t                                                                    vui_num_units_in_tick               = {};
      uint32_t                                                                    vui_time_scale                      = {};
      uint32_t                                                                    vui_num_ticks_poc_diff_one_minus1   = {};
      uint16_t                                                                    min_spatial_segmentation_idc        = {};
      uint16_t                                                                    reserved3                           = {};
      uint8_t                                                                     max_bytes_per_pic_denom             = {};
      uint8_t                                                                     max_bits_per_min_cu_denom           = {};
      uint8_t                                                                     log2_max_mv_length_horizontal       = {};
      uint8_t                                                                     log2_max_mv_length_vertical         = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265HrdParameters * pHrdParameters                      = {};
    };

    struct H265PredictorPaletteEntries
    {
      using NativeType = StdVideoH265PredictorPaletteEntries;

      operator StdVideoH265PredictorPaletteEntries const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265PredictorPaletteEntries *>( this );
      }

      operator StdVideoH265PredictorPaletteEntries &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265PredictorPaletteEntries *>( this );
      }

      operator StdVideoH265PredictorPaletteEntries const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265PredictorPaletteEntries *>( this );
      }

      operator StdVideoH265PredictorPaletteEntries *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265PredictorPaletteEntries *>( this );
      }

      bool operator==( H265PredictorPaletteEntries const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( PredictorPaletteEntries == rhs.PredictorPaletteEntries );
      }

      bool operator!=( H265PredictorPaletteEntries const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::
        ArrayWrapper2D<uint16_t, STD_VIDEO_H265_PREDICTOR_PALETTE_COMPONENTS_LIST_SIZE, STD_VIDEO_H265_PREDICTOR_PALETTE_COMP_ENTRIES_LIST_SIZE>
          PredictorPaletteEntries = {};
    };

    struct H265SpsFlags
    {
      using NativeType = StdVideoH265SpsFlags;

      operator StdVideoH265SpsFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265SpsFlags *>( this );
      }

      operator StdVideoH265SpsFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265SpsFlags *>( this );
      }

      operator StdVideoH265SpsFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265SpsFlags *>( this );
      }

      operator StdVideoH265SpsFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265SpsFlags *>( this );
      }

      bool operator==( H265SpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( sps_temporal_id_nesting_flag == rhs.sps_temporal_id_nesting_flag ) && ( separate_colour_plane_flag == rhs.separate_colour_plane_flag ) &&
               ( conformance_window_flag == rhs.conformance_window_flag ) &&
               ( sps_sub_layer_ordering_info_present_flag == rhs.sps_sub_layer_ordering_info_present_flag ) &&
               ( scaling_list_enabled_flag == rhs.scaling_list_enabled_flag ) &&
               ( sps_scaling_list_data_present_flag == rhs.sps_scaling_list_data_present_flag ) && ( amp_enabled_flag == rhs.amp_enabled_flag ) &&
               ( sample_adaptive_offset_enabled_flag == rhs.sample_adaptive_offset_enabled_flag ) && ( pcm_enabled_flag == rhs.pcm_enabled_flag ) &&
               ( pcm_loop_filter_disabled_flag == rhs.pcm_loop_filter_disabled_flag ) &&
               ( long_term_ref_pics_present_flag == rhs.long_term_ref_pics_present_flag ) &&
               ( sps_temporal_mvp_enabled_flag == rhs.sps_temporal_mvp_enabled_flag ) &&
               ( strong_intra_smoothing_enabled_flag == rhs.strong_intra_smoothing_enabled_flag ) &&
               ( vui_parameters_present_flag == rhs.vui_parameters_present_flag ) && ( sps_extension_present_flag == rhs.sps_extension_present_flag ) &&
               ( sps_range_extension_flag == rhs.sps_range_extension_flag ) &&
               ( transform_skip_rotation_enabled_flag == rhs.transform_skip_rotation_enabled_flag ) &&
               ( transform_skip_context_enabled_flag == rhs.transform_skip_context_enabled_flag ) &&
               ( implicit_rdpcm_enabled_flag == rhs.implicit_rdpcm_enabled_flag ) && ( explicit_rdpcm_enabled_flag == rhs.explicit_rdpcm_enabled_flag ) &&
               ( extended_precision_processing_flag == rhs.extended_precision_processing_flag ) &&
               ( intra_smoothing_disabled_flag == rhs.intra_smoothing_disabled_flag ) &&
               ( high_precision_offsets_enabled_flag == rhs.high_precision_offsets_enabled_flag ) &&
               ( persistent_rice_adaptation_enabled_flag == rhs.persistent_rice_adaptation_enabled_flag ) &&
               ( cabac_bypass_alignment_enabled_flag == rhs.cabac_bypass_alignment_enabled_flag ) && ( sps_scc_extension_flag == rhs.sps_scc_extension_flag ) &&
               ( sps_curr_pic_ref_enabled_flag == rhs.sps_curr_pic_ref_enabled_flag ) && ( palette_mode_enabled_flag == rhs.palette_mode_enabled_flag ) &&
               ( sps_palette_predictor_initializers_present_flag == rhs.sps_palette_predictor_initializers_present_flag ) &&
               ( intra_boundary_filtering_disabled_flag == rhs.intra_boundary_filtering_disabled_flag );
      }

      bool operator!=( H265SpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t sps_temporal_id_nesting_flag                    : 1;
      uint32_t separate_colour_plane_flag                      : 1;
      uint32_t conformance_window_flag                         : 1;
      uint32_t sps_sub_layer_ordering_info_present_flag        : 1;
      uint32_t scaling_list_enabled_flag                       : 1;
      uint32_t sps_scaling_list_data_present_flag              : 1;
      uint32_t amp_enabled_flag                                : 1;
      uint32_t sample_adaptive_offset_enabled_flag             : 1;
      uint32_t pcm_enabled_flag                                : 1;
      uint32_t pcm_loop_filter_disabled_flag                   : 1;
      uint32_t long_term_ref_pics_present_flag                 : 1;
      uint32_t sps_temporal_mvp_enabled_flag                   : 1;
      uint32_t strong_intra_smoothing_enabled_flag             : 1;
      uint32_t vui_parameters_present_flag                     : 1;
      uint32_t sps_extension_present_flag                      : 1;
      uint32_t sps_range_extension_flag                        : 1;
      uint32_t transform_skip_rotation_enabled_flag            : 1;
      uint32_t transform_skip_context_enabled_flag             : 1;
      uint32_t implicit_rdpcm_enabled_flag                     : 1;
      uint32_t explicit_rdpcm_enabled_flag                     : 1;
      uint32_t extended_precision_processing_flag              : 1;
      uint32_t intra_smoothing_disabled_flag                   : 1;
      uint32_t high_precision_offsets_enabled_flag             : 1;
      uint32_t persistent_rice_adaptation_enabled_flag         : 1;
      uint32_t cabac_bypass_alignment_enabled_flag             : 1;
      uint32_t sps_scc_extension_flag                          : 1;
      uint32_t sps_curr_pic_ref_enabled_flag                   : 1;
      uint32_t palette_mode_enabled_flag                       : 1;
      uint32_t sps_palette_predictor_initializers_present_flag : 1;
      uint32_t intra_boundary_filtering_disabled_flag          : 1;
    };

    struct H265ShortTermRefPicSetFlags
    {
      using NativeType = StdVideoH265ShortTermRefPicSetFlags;

      operator StdVideoH265ShortTermRefPicSetFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265ShortTermRefPicSetFlags *>( this );
      }

      operator StdVideoH265ShortTermRefPicSetFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265ShortTermRefPicSetFlags *>( this );
      }

      operator StdVideoH265ShortTermRefPicSetFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265ShortTermRefPicSetFlags *>( this );
      }

      operator StdVideoH265ShortTermRefPicSetFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265ShortTermRefPicSetFlags *>( this );
      }

      bool operator==( H265ShortTermRefPicSetFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( inter_ref_pic_set_prediction_flag == rhs.inter_ref_pic_set_prediction_flag ) && ( delta_rps_sign == rhs.delta_rps_sign );
      }

      bool operator!=( H265ShortTermRefPicSetFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t inter_ref_pic_set_prediction_flag : 1;
      uint32_t delta_rps_sign                    : 1;
    };

    struct H265ShortTermRefPicSet
    {
      using NativeType = StdVideoH265ShortTermRefPicSet;

      operator StdVideoH265ShortTermRefPicSet const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265ShortTermRefPicSet *>( this );
      }

      operator StdVideoH265ShortTermRefPicSet &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265ShortTermRefPicSet *>( this );
      }

      operator StdVideoH265ShortTermRefPicSet const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265ShortTermRefPicSet *>( this );
      }

      operator StdVideoH265ShortTermRefPicSet *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265ShortTermRefPicSet *>( this );
      }

      bool operator==( H265ShortTermRefPicSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( delta_idx_minus1 == rhs.delta_idx_minus1 ) && ( use_delta_flag == rhs.use_delta_flag ) &&
               ( abs_delta_rps_minus1 == rhs.abs_delta_rps_minus1 ) && ( used_by_curr_pic_flag == rhs.used_by_curr_pic_flag ) &&
               ( used_by_curr_pic_s0_flag == rhs.used_by_curr_pic_s0_flag ) && ( used_by_curr_pic_s1_flag == rhs.used_by_curr_pic_s1_flag ) &&
               ( reserved1 == rhs.reserved1 ) && ( reserved2 == rhs.reserved2 ) && ( reserved3 == rhs.reserved3 ) &&
               ( num_negative_pics == rhs.num_negative_pics ) && ( num_positive_pics == rhs.num_positive_pics ) &&
               ( delta_poc_s0_minus1 == rhs.delta_poc_s0_minus1 ) && ( delta_poc_s1_minus1 == rhs.delta_poc_s1_minus1 );
      }

      bool operator!=( H265ShortTermRefPicSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ShortTermRefPicSetFlags flags                    = {};
      uint32_t                                                                      delta_idx_minus1         = {};
      uint16_t                                                                      use_delta_flag           = {};
      uint16_t                                                                      abs_delta_rps_minus1     = {};
      uint16_t                                                                      used_by_curr_pic_flag    = {};
      uint16_t                                                                      used_by_curr_pic_s0_flag = {};
      uint16_t                                                                      used_by_curr_pic_s1_flag = {};
      uint16_t                                                                      reserved1                = {};
      uint8_t                                                                       reserved2                = {};
      uint8_t                                                                       reserved3                = {};
      uint8_t                                                                       num_negative_pics        = {};
      uint8_t                                                                       num_positive_pics        = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint16_t, STD_VIDEO_H265_MAX_DPB_SIZE>   delta_poc_s0_minus1      = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint16_t, STD_VIDEO_H265_MAX_DPB_SIZE>   delta_poc_s1_minus1      = {};
    };

    struct H265LongTermRefPicsSps
    {
      using NativeType = StdVideoH265LongTermRefPicsSps;

      operator StdVideoH265LongTermRefPicsSps const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265LongTermRefPicsSps *>( this );
      }

      operator StdVideoH265LongTermRefPicsSps &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265LongTermRefPicsSps *>( this );
      }

      operator StdVideoH265LongTermRefPicsSps const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265LongTermRefPicsSps *>( this );
      }

      operator StdVideoH265LongTermRefPicsSps *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265LongTermRefPicsSps *>( this );
      }

      bool operator==( H265LongTermRefPicsSps const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( used_by_curr_pic_lt_sps_flag == rhs.used_by_curr_pic_lt_sps_flag ) && ( lt_ref_pic_poc_lsb_sps == rhs.lt_ref_pic_poc_lsb_sps );
      }

      bool operator!=( H265LongTermRefPicsSps const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t                                                                                  used_by_curr_pic_lt_sps_flag = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_H265_MAX_LONG_TERM_REF_PICS_SPS> lt_ref_pic_poc_lsb_sps       = {};
    };

    struct H265SequenceParameterSet
    {
      using NativeType = StdVideoH265SequenceParameterSet;

      operator StdVideoH265SequenceParameterSet const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265SequenceParameterSet *>( this );
      }

      operator StdVideoH265SequenceParameterSet &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265SequenceParameterSet *>( this );
      }

      operator StdVideoH265SequenceParameterSet const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265SequenceParameterSet *>( this );
      }

      operator StdVideoH265SequenceParameterSet *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265SequenceParameterSet *>( this );
      }

      bool operator==( H265SequenceParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( chroma_format_idc == rhs.chroma_format_idc ) && ( pic_width_in_luma_samples == rhs.pic_width_in_luma_samples ) &&
               ( pic_height_in_luma_samples == rhs.pic_height_in_luma_samples ) && ( sps_video_parameter_set_id == rhs.sps_video_parameter_set_id ) &&
               ( sps_max_sub_layers_minus1 == rhs.sps_max_sub_layers_minus1 ) && ( sps_seq_parameter_set_id == rhs.sps_seq_parameter_set_id ) &&
               ( bit_depth_luma_minus8 == rhs.bit_depth_luma_minus8 ) && ( bit_depth_chroma_minus8 == rhs.bit_depth_chroma_minus8 ) &&
               ( log2_max_pic_order_cnt_lsb_minus4 == rhs.log2_max_pic_order_cnt_lsb_minus4 ) &&
               ( log2_min_luma_coding_block_size_minus3 == rhs.log2_min_luma_coding_block_size_minus3 ) &&
               ( log2_diff_max_min_luma_coding_block_size == rhs.log2_diff_max_min_luma_coding_block_size ) &&
               ( log2_min_luma_transform_block_size_minus2 == rhs.log2_min_luma_transform_block_size_minus2 ) &&
               ( log2_diff_max_min_luma_transform_block_size == rhs.log2_diff_max_min_luma_transform_block_size ) &&
               ( max_transform_hierarchy_depth_inter == rhs.max_transform_hierarchy_depth_inter ) &&
               ( max_transform_hierarchy_depth_intra == rhs.max_transform_hierarchy_depth_intra ) &&
               ( num_short_term_ref_pic_sets == rhs.num_short_term_ref_pic_sets ) && ( num_long_term_ref_pics_sps == rhs.num_long_term_ref_pics_sps ) &&
               ( pcm_sample_bit_depth_luma_minus1 == rhs.pcm_sample_bit_depth_luma_minus1 ) &&
               ( pcm_sample_bit_depth_chroma_minus1 == rhs.pcm_sample_bit_depth_chroma_minus1 ) &&
               ( log2_min_pcm_luma_coding_block_size_minus3 == rhs.log2_min_pcm_luma_coding_block_size_minus3 ) &&
               ( log2_diff_max_min_pcm_luma_coding_block_size == rhs.log2_diff_max_min_pcm_luma_coding_block_size ) && ( reserved1 == rhs.reserved1 ) &&
               ( reserved2 == rhs.reserved2 ) && ( palette_max_size == rhs.palette_max_size ) &&
               ( delta_palette_max_predictor_size == rhs.delta_palette_max_predictor_size ) &&
               ( motion_vector_resolution_control_idc == rhs.motion_vector_resolution_control_idc ) &&
               ( sps_num_palette_predictor_initializers_minus1 == rhs.sps_num_palette_predictor_initializers_minus1 ) &&
               ( conf_win_left_offset == rhs.conf_win_left_offset ) && ( conf_win_right_offset == rhs.conf_win_right_offset ) &&
               ( conf_win_top_offset == rhs.conf_win_top_offset ) && ( conf_win_bottom_offset == rhs.conf_win_bottom_offset ) &&
               ( pProfileTierLevel == rhs.pProfileTierLevel ) && ( pDecPicBufMgr == rhs.pDecPicBufMgr ) && ( pScalingLists == rhs.pScalingLists ) &&
               ( pShortTermRefPicSet == rhs.pShortTermRefPicSet ) && ( pLongTermRefPicsSps == rhs.pLongTermRefPicsSps ) &&
               ( pSequenceParameterSetVui == rhs.pSequenceParameterSetVui ) && ( pPredictorPaletteEntries == rhs.pPredictorPaletteEntries );
      }

      bool operator!=( H265SequenceParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265SpsFlags        flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ChromaFormatIdc chroma_format_idc =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ChromaFormatIdc::eMonochrome;
      uint32_t                                                                              pic_width_in_luma_samples                     = {};
      uint32_t                                                                              pic_height_in_luma_samples                    = {};
      uint8_t                                                                               sps_video_parameter_set_id                    = {};
      uint8_t                                                                               sps_max_sub_layers_minus1                     = {};
      uint8_t                                                                               sps_seq_parameter_set_id                      = {};
      uint8_t                                                                               bit_depth_luma_minus8                         = {};
      uint8_t                                                                               bit_depth_chroma_minus8                       = {};
      uint8_t                                                                               log2_max_pic_order_cnt_lsb_minus4             = {};
      uint8_t                                                                               log2_min_luma_coding_block_size_minus3        = {};
      uint8_t                                                                               log2_diff_max_min_luma_coding_block_size      = {};
      uint8_t                                                                               log2_min_luma_transform_block_size_minus2     = {};
      uint8_t                                                                               log2_diff_max_min_luma_transform_block_size   = {};
      uint8_t                                                                               max_transform_hierarchy_depth_inter           = {};
      uint8_t                                                                               max_transform_hierarchy_depth_intra           = {};
      uint8_t                                                                               num_short_term_ref_pic_sets                   = {};
      uint8_t                                                                               num_long_term_ref_pics_sps                    = {};
      uint8_t                                                                               pcm_sample_bit_depth_luma_minus1              = {};
      uint8_t                                                                               pcm_sample_bit_depth_chroma_minus1            = {};
      uint8_t                                                                               log2_min_pcm_luma_coding_block_size_minus3    = {};
      uint8_t                                                                               log2_diff_max_min_pcm_luma_coding_block_size  = {};
      uint8_t                                                                               reserved1                                     = {};
      uint8_t                                                                               reserved2                                     = {};
      uint8_t                                                                               palette_max_size                              = {};
      uint8_t                                                                               delta_palette_max_predictor_size              = {};
      uint8_t                                                                               motion_vector_resolution_control_idc          = {};
      uint8_t                                                                               sps_num_palette_predictor_initializers_minus1 = {};
      uint32_t                                                                              conf_win_left_offset                          = {};
      uint32_t                                                                              conf_win_right_offset                         = {};
      uint32_t                                                                              conf_win_top_offset                           = {};
      uint32_t                                                                              conf_win_bottom_offset                        = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ProfileTierLevel *        pProfileTierLevel                             = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265DecPicBufMgr *            pDecPicBufMgr                                 = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ScalingLists *            pScalingLists                                 = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ShortTermRefPicSet *      pShortTermRefPicSet                           = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265LongTermRefPicsSps *      pLongTermRefPicsSps                           = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265SequenceParameterSetVui * pSequenceParameterSetVui                      = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265PredictorPaletteEntries * pPredictorPaletteEntries                      = {};
    };

    struct H265PpsFlags
    {
      using NativeType = StdVideoH265PpsFlags;

      operator StdVideoH265PpsFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265PpsFlags *>( this );
      }

      operator StdVideoH265PpsFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265PpsFlags *>( this );
      }

      operator StdVideoH265PpsFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265PpsFlags *>( this );
      }

      operator StdVideoH265PpsFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265PpsFlags *>( this );
      }

      bool operator==( H265PpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( dependent_slice_segments_enabled_flag == rhs.dependent_slice_segments_enabled_flag ) &&
               ( output_flag_present_flag == rhs.output_flag_present_flag ) && ( sign_data_hiding_enabled_flag == rhs.sign_data_hiding_enabled_flag ) &&
               ( cabac_init_present_flag == rhs.cabac_init_present_flag ) && ( constrained_intra_pred_flag == rhs.constrained_intra_pred_flag ) &&
               ( transform_skip_enabled_flag == rhs.transform_skip_enabled_flag ) && ( cu_qp_delta_enabled_flag == rhs.cu_qp_delta_enabled_flag ) &&
               ( pps_slice_chroma_qp_offsets_present_flag == rhs.pps_slice_chroma_qp_offsets_present_flag ) &&
               ( weighted_pred_flag == rhs.weighted_pred_flag ) && ( weighted_bipred_flag == rhs.weighted_bipred_flag ) &&
               ( transquant_bypass_enabled_flag == rhs.transquant_bypass_enabled_flag ) && ( tiles_enabled_flag == rhs.tiles_enabled_flag ) &&
               ( entropy_coding_sync_enabled_flag == rhs.entropy_coding_sync_enabled_flag ) && ( uniform_spacing_flag == rhs.uniform_spacing_flag ) &&
               ( loop_filter_across_tiles_enabled_flag == rhs.loop_filter_across_tiles_enabled_flag ) &&
               ( pps_loop_filter_across_slices_enabled_flag == rhs.pps_loop_filter_across_slices_enabled_flag ) &&
               ( deblocking_filter_control_present_flag == rhs.deblocking_filter_control_present_flag ) &&
               ( deblocking_filter_override_enabled_flag == rhs.deblocking_filter_override_enabled_flag ) &&
               ( pps_deblocking_filter_disabled_flag == rhs.pps_deblocking_filter_disabled_flag ) &&
               ( pps_scaling_list_data_present_flag == rhs.pps_scaling_list_data_present_flag ) &&
               ( lists_modification_present_flag == rhs.lists_modification_present_flag ) &&
               ( slice_segment_header_extension_present_flag == rhs.slice_segment_header_extension_present_flag ) &&
               ( pps_extension_present_flag == rhs.pps_extension_present_flag ) &&
               ( cross_component_prediction_enabled_flag == rhs.cross_component_prediction_enabled_flag ) &&
               ( chroma_qp_offset_list_enabled_flag == rhs.chroma_qp_offset_list_enabled_flag ) &&
               ( pps_curr_pic_ref_enabled_flag == rhs.pps_curr_pic_ref_enabled_flag ) &&
               ( residual_adaptive_colour_transform_enabled_flag == rhs.residual_adaptive_colour_transform_enabled_flag ) &&
               ( pps_slice_act_qp_offsets_present_flag == rhs.pps_slice_act_qp_offsets_present_flag ) &&
               ( pps_palette_predictor_initializers_present_flag == rhs.pps_palette_predictor_initializers_present_flag ) &&
               ( monochrome_palette_flag == rhs.monochrome_palette_flag ) && ( pps_range_extension_flag == rhs.pps_range_extension_flag );
      }

      bool operator!=( H265PpsFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t dependent_slice_segments_enabled_flag           : 1;
      uint32_t output_flag_present_flag                        : 1;
      uint32_t sign_data_hiding_enabled_flag                   : 1;
      uint32_t cabac_init_present_flag                         : 1;
      uint32_t constrained_intra_pred_flag                     : 1;
      uint32_t transform_skip_enabled_flag                     : 1;
      uint32_t cu_qp_delta_enabled_flag                        : 1;
      uint32_t pps_slice_chroma_qp_offsets_present_flag        : 1;
      uint32_t weighted_pred_flag                              : 1;
      uint32_t weighted_bipred_flag                            : 1;
      uint32_t transquant_bypass_enabled_flag                  : 1;
      uint32_t tiles_enabled_flag                              : 1;
      uint32_t entropy_coding_sync_enabled_flag                : 1;
      uint32_t uniform_spacing_flag                            : 1;
      uint32_t loop_filter_across_tiles_enabled_flag           : 1;
      uint32_t pps_loop_filter_across_slices_enabled_flag      : 1;
      uint32_t deblocking_filter_control_present_flag          : 1;
      uint32_t deblocking_filter_override_enabled_flag         : 1;
      uint32_t pps_deblocking_filter_disabled_flag             : 1;
      uint32_t pps_scaling_list_data_present_flag              : 1;
      uint32_t lists_modification_present_flag                 : 1;
      uint32_t slice_segment_header_extension_present_flag     : 1;
      uint32_t pps_extension_present_flag                      : 1;
      uint32_t cross_component_prediction_enabled_flag         : 1;
      uint32_t chroma_qp_offset_list_enabled_flag              : 1;
      uint32_t pps_curr_pic_ref_enabled_flag                   : 1;
      uint32_t residual_adaptive_colour_transform_enabled_flag : 1;
      uint32_t pps_slice_act_qp_offsets_present_flag           : 1;
      uint32_t pps_palette_predictor_initializers_present_flag : 1;
      uint32_t monochrome_palette_flag                         : 1;
      uint32_t pps_range_extension_flag                        : 1;
    };

    struct H265PictureParameterSet
    {
      using NativeType = StdVideoH265PictureParameterSet;

      operator StdVideoH265PictureParameterSet const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoH265PictureParameterSet *>( this );
      }

      operator StdVideoH265PictureParameterSet &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoH265PictureParameterSet *>( this );
      }

      operator StdVideoH265PictureParameterSet const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoH265PictureParameterSet *>( this );
      }

      operator StdVideoH265PictureParameterSet *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoH265PictureParameterSet *>( this );
      }

      bool operator==( H265PictureParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( pps_pic_parameter_set_id == rhs.pps_pic_parameter_set_id ) &&
               ( pps_seq_parameter_set_id == rhs.pps_seq_parameter_set_id ) && ( sps_video_parameter_set_id == rhs.sps_video_parameter_set_id ) &&
               ( num_extra_slice_header_bits == rhs.num_extra_slice_header_bits ) &&
               ( num_ref_idx_l0_default_active_minus1 == rhs.num_ref_idx_l0_default_active_minus1 ) &&
               ( num_ref_idx_l1_default_active_minus1 == rhs.num_ref_idx_l1_default_active_minus1 ) && ( init_qp_minus26 == rhs.init_qp_minus26 ) &&
               ( diff_cu_qp_delta_depth == rhs.diff_cu_qp_delta_depth ) && ( pps_cb_qp_offset == rhs.pps_cb_qp_offset ) &&
               ( pps_cr_qp_offset == rhs.pps_cr_qp_offset ) && ( pps_beta_offset_div2 == rhs.pps_beta_offset_div2 ) &&
               ( pps_tc_offset_div2 == rhs.pps_tc_offset_div2 ) && ( log2_parallel_merge_level_minus2 == rhs.log2_parallel_merge_level_minus2 ) &&
               ( log2_max_transform_skip_block_size_minus2 == rhs.log2_max_transform_skip_block_size_minus2 ) &&
               ( diff_cu_chroma_qp_offset_depth == rhs.diff_cu_chroma_qp_offset_depth ) &&
               ( chroma_qp_offset_list_len_minus1 == rhs.chroma_qp_offset_list_len_minus1 ) && ( cb_qp_offset_list == rhs.cb_qp_offset_list ) &&
               ( cr_qp_offset_list == rhs.cr_qp_offset_list ) && ( log2_sao_offset_scale_luma == rhs.log2_sao_offset_scale_luma ) &&
               ( log2_sao_offset_scale_chroma == rhs.log2_sao_offset_scale_chroma ) && ( pps_act_y_qp_offset_plus5 == rhs.pps_act_y_qp_offset_plus5 ) &&
               ( pps_act_cb_qp_offset_plus5 == rhs.pps_act_cb_qp_offset_plus5 ) && ( pps_act_cr_qp_offset_plus3 == rhs.pps_act_cr_qp_offset_plus3 ) &&
               ( pps_num_palette_predictor_initializers == rhs.pps_num_palette_predictor_initializers ) &&
               ( luma_bit_depth_entry_minus8 == rhs.luma_bit_depth_entry_minus8 ) && ( chroma_bit_depth_entry_minus8 == rhs.chroma_bit_depth_entry_minus8 ) &&
               ( num_tile_columns_minus1 == rhs.num_tile_columns_minus1 ) && ( num_tile_rows_minus1 == rhs.num_tile_rows_minus1 ) &&
               ( reserved1 == rhs.reserved1 ) && ( reserved2 == rhs.reserved2 ) && ( column_width_minus1 == rhs.column_width_minus1 ) &&
               ( row_height_minus1 == rhs.row_height_minus1 ) && ( reserved3 == rhs.reserved3 ) && ( pScalingLists == rhs.pScalingLists ) &&
               ( pPredictorPaletteEntries == rhs.pPredictorPaletteEntries );
      }

      bool operator!=( H265PictureParameterSet const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265PpsFlags                                      flags                                     = {};
      uint8_t                                                                                             pps_pic_parameter_set_id                  = {};
      uint8_t                                                                                             pps_seq_parameter_set_id                  = {};
      uint8_t                                                                                             sps_video_parameter_set_id                = {};
      uint8_t                                                                                             num_extra_slice_header_bits               = {};
      uint8_t                                                                                             num_ref_idx_l0_default_active_minus1      = {};
      uint8_t                                                                                             num_ref_idx_l1_default_active_minus1      = {};
      int8_t                                                                                              init_qp_minus26                           = {};
      uint8_t                                                                                             diff_cu_qp_delta_depth                    = {};
      int8_t                                                                                              pps_cb_qp_offset                          = {};
      int8_t                                                                                              pps_cr_qp_offset                          = {};
      int8_t                                                                                              pps_beta_offset_div2                      = {};
      int8_t                                                                                              pps_tc_offset_div2                        = {};
      uint8_t                                                                                             log2_parallel_merge_level_minus2          = {};
      uint8_t                                                                                             log2_max_transform_skip_block_size_minus2 = {};
      uint8_t                                                                                             diff_cu_chroma_qp_offset_depth            = {};
      uint8_t                                                                                             chroma_qp_offset_list_len_minus1          = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H265_CHROMA_QP_OFFSET_LIST_SIZE>             cb_qp_offset_list                         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H265_CHROMA_QP_OFFSET_LIST_SIZE>             cr_qp_offset_list                         = {};
      uint8_t                                                                                             log2_sao_offset_scale_luma                = {};
      uint8_t                                                                                             log2_sao_offset_scale_chroma              = {};
      int8_t                                                                                              pps_act_y_qp_offset_plus5                 = {};
      int8_t                                                                                              pps_act_cb_qp_offset_plus5                = {};
      int8_t                                                                                              pps_act_cr_qp_offset_plus3                = {};
      uint8_t                                                                                             pps_num_palette_predictor_initializers    = {};
      uint8_t                                                                                             luma_bit_depth_entry_minus8               = {};
      uint8_t                                                                                             chroma_bit_depth_entry_minus8             = {};
      uint8_t                                                                                             num_tile_columns_minus1                   = {};
      uint8_t                                                                                             num_tile_rows_minus1                      = {};
      uint8_t                                                                                             reserved1                                 = {};
      uint8_t                                                                                             reserved2                                 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint16_t, STD_VIDEO_H265_CHROMA_QP_OFFSET_TILE_COLS_LIST_SIZE> column_width_minus1                       = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint16_t, STD_VIDEO_H265_CHROMA_QP_OFFSET_TILE_ROWS_LIST_SIZE> row_height_minus1                         = {};
      uint32_t                                                                                            reserved3                                 = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ScalingLists *                          pScalingLists                             = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265PredictorPaletteEntries *               pPredictorPaletteEntries                  = {};
    };

    //=== vulkan_video_codec_h265std_decode ===

    struct DecodeH265PictureInfoFlags
    {
      using NativeType = StdVideoDecodeH265PictureInfoFlags;

      operator StdVideoDecodeH265PictureInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeH265PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeH265PictureInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeH265PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeH265PictureInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeH265PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeH265PictureInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeH265PictureInfoFlags *>( this );
      }

      bool operator==( DecodeH265PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( IrapPicFlag == rhs.IrapPicFlag ) && ( IdrPicFlag == rhs.IdrPicFlag ) && ( IsReference == rhs.IsReference ) &&
               ( short_term_ref_pic_set_sps_flag == rhs.short_term_ref_pic_set_sps_flag );
      }

      bool operator!=( DecodeH265PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t IrapPicFlag                     : 1;
      uint32_t IdrPicFlag                      : 1;
      uint32_t IsReference                     : 1;
      uint32_t short_term_ref_pic_set_sps_flag : 1;
    };

    struct DecodeH265PictureInfo
    {
      using NativeType = StdVideoDecodeH265PictureInfo;

      operator StdVideoDecodeH265PictureInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeH265PictureInfo *>( this );
      }

      operator StdVideoDecodeH265PictureInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeH265PictureInfo *>( this );
      }

      operator StdVideoDecodeH265PictureInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeH265PictureInfo *>( this );
      }

      operator StdVideoDecodeH265PictureInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeH265PictureInfo *>( this );
      }

      bool operator==( DecodeH265PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( sps_video_parameter_set_id == rhs.sps_video_parameter_set_id ) &&
               ( pps_seq_parameter_set_id == rhs.pps_seq_parameter_set_id ) && ( pps_pic_parameter_set_id == rhs.pps_pic_parameter_set_id ) &&
               ( NumDeltaPocsOfRefRpsIdx == rhs.NumDeltaPocsOfRefRpsIdx ) && ( PicOrderCntVal == rhs.PicOrderCntVal ) &&
               ( NumBitsForSTRefPicSetInSlice == rhs.NumBitsForSTRefPicSetInSlice ) && ( reserved == rhs.reserved ) &&
               ( RefPicSetStCurrBefore == rhs.RefPicSetStCurrBefore ) && ( RefPicSetStCurrAfter == rhs.RefPicSetStCurrAfter ) &&
               ( RefPicSetLtCurr == rhs.RefPicSetLtCurr );
      }

      bool operator!=( DecodeH265PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::DecodeH265PictureInfoFlags               flags                        = {};
      uint8_t                                                                                    sps_video_parameter_set_id   = {};
      uint8_t                                                                                    pps_seq_parameter_set_id     = {};
      uint8_t                                                                                    pps_pic_parameter_set_id     = {};
      uint8_t                                                                                    NumDeltaPocsOfRefRpsIdx      = {};
      int32_t                                                                                    PicOrderCntVal               = {};
      uint16_t                                                                                   NumBitsForSTRefPicSetInSlice = {};
      uint16_t                                                                                   reserved                     = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE> RefPicSetStCurrBefore        = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE> RefPicSetStCurrAfter         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE> RefPicSetLtCurr              = {};
    };

    struct DecodeH265ReferenceInfoFlags
    {
      using NativeType = StdVideoDecodeH265ReferenceInfoFlags;

      operator StdVideoDecodeH265ReferenceInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeH265ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeH265ReferenceInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeH265ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeH265ReferenceInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeH265ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeH265ReferenceInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeH265ReferenceInfoFlags *>( this );
      }

      bool operator==( DecodeH265ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( used_for_long_term_reference == rhs.used_for_long_term_reference ) && ( unused_for_reference == rhs.unused_for_reference );
      }

      bool operator!=( DecodeH265ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t used_for_long_term_reference : 1;
      uint32_t unused_for_reference         : 1;
    };

    struct DecodeH265ReferenceInfo
    {
      using NativeType = StdVideoDecodeH265ReferenceInfo;

      operator StdVideoDecodeH265ReferenceInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeH265ReferenceInfo *>( this );
      }

      operator StdVideoDecodeH265ReferenceInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeH265ReferenceInfo *>( this );
      }

      operator StdVideoDecodeH265ReferenceInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeH265ReferenceInfo *>( this );
      }

      operator StdVideoDecodeH265ReferenceInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeH265ReferenceInfo *>( this );
      }

      bool operator==( DecodeH265ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( PicOrderCntVal == rhs.PicOrderCntVal );
      }

      bool operator!=( DecodeH265ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::DecodeH265ReferenceInfoFlags flags          = {};
      int32_t                                                                        PicOrderCntVal = {};
    };

    //=== vulkan_video_codec_h265std_encode ===

    struct EncodeH265WeightTableFlags
    {
      using NativeType = StdVideoEncodeH265WeightTableFlags;

      operator StdVideoEncodeH265WeightTableFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265WeightTableFlags *>( this );
      }

      operator StdVideoEncodeH265WeightTableFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265WeightTableFlags *>( this );
      }

      operator StdVideoEncodeH265WeightTableFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265WeightTableFlags *>( this );
      }

      operator StdVideoEncodeH265WeightTableFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265WeightTableFlags *>( this );
      }

      bool operator==( EncodeH265WeightTableFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( luma_weight_l0_flag == rhs.luma_weight_l0_flag ) && ( chroma_weight_l0_flag == rhs.chroma_weight_l0_flag ) &&
               ( luma_weight_l1_flag == rhs.luma_weight_l1_flag ) && ( chroma_weight_l1_flag == rhs.chroma_weight_l1_flag );
      }

      bool operator!=( EncodeH265WeightTableFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint16_t luma_weight_l0_flag   = {};
      uint16_t chroma_weight_l0_flag = {};
      uint16_t luma_weight_l1_flag   = {};
      uint16_t chroma_weight_l1_flag = {};
    };

    struct EncodeH265WeightTable
    {
      using NativeType = StdVideoEncodeH265WeightTable;

      operator StdVideoEncodeH265WeightTable const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265WeightTable *>( this );
      }

      operator StdVideoEncodeH265WeightTable &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265WeightTable *>( this );
      }

      operator StdVideoEncodeH265WeightTable const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265WeightTable *>( this );
      }

      operator StdVideoEncodeH265WeightTable *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265WeightTable *>( this );
      }

      bool operator==( EncodeH265WeightTable const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( luma_log2_weight_denom == rhs.luma_log2_weight_denom ) &&
               ( delta_chroma_log2_weight_denom == rhs.delta_chroma_log2_weight_denom ) && ( delta_luma_weight_l0 == rhs.delta_luma_weight_l0 ) &&
               ( luma_offset_l0 == rhs.luma_offset_l0 ) && ( delta_chroma_weight_l0 == rhs.delta_chroma_weight_l0 ) &&
               ( delta_chroma_offset_l0 == rhs.delta_chroma_offset_l0 ) && ( delta_luma_weight_l1 == rhs.delta_luma_weight_l1 ) &&
               ( luma_offset_l1 == rhs.luma_offset_l1 ) && ( delta_chroma_weight_l1 == rhs.delta_chroma_weight_l1 ) &&
               ( delta_chroma_offset_l1 == rhs.delta_chroma_offset_l1 );
      }

      bool operator!=( EncodeH265WeightTable const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH265WeightTableFlags                                    flags                          = {};
      uint8_t                                                                                                         luma_log2_weight_denom         = {};
      int8_t                                                                                                          delta_chroma_log2_weight_denom = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF>                                   delta_luma_weight_l0           = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF>                                   luma_offset_l0                 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF, STD_VIDEO_H265_MAX_CHROMA_PLANES> delta_chroma_weight_l0         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF, STD_VIDEO_H265_MAX_CHROMA_PLANES> delta_chroma_offset_l0         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF>                                   delta_luma_weight_l1           = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF>                                   luma_offset_l1                 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF, STD_VIDEO_H265_MAX_CHROMA_PLANES> delta_chroma_weight_l1         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF, STD_VIDEO_H265_MAX_CHROMA_PLANES> delta_chroma_offset_l1         = {};
    };

    struct EncodeH265SliceSegmentHeaderFlags
    {
      using NativeType = StdVideoEncodeH265SliceSegmentHeaderFlags;

      operator StdVideoEncodeH265SliceSegmentHeaderFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265SliceSegmentHeaderFlags *>( this );
      }

      operator StdVideoEncodeH265SliceSegmentHeaderFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265SliceSegmentHeaderFlags *>( this );
      }

      operator StdVideoEncodeH265SliceSegmentHeaderFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265SliceSegmentHeaderFlags *>( this );
      }

      operator StdVideoEncodeH265SliceSegmentHeaderFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265SliceSegmentHeaderFlags *>( this );
      }

      bool operator==( EncodeH265SliceSegmentHeaderFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( first_slice_segment_in_pic_flag == rhs.first_slice_segment_in_pic_flag ) &&
               ( dependent_slice_segment_flag == rhs.dependent_slice_segment_flag ) && ( slice_sao_luma_flag == rhs.slice_sao_luma_flag ) &&
               ( slice_sao_chroma_flag == rhs.slice_sao_chroma_flag ) && ( num_ref_idx_active_override_flag == rhs.num_ref_idx_active_override_flag ) &&
               ( mvd_l1_zero_flag == rhs.mvd_l1_zero_flag ) && ( cabac_init_flag == rhs.cabac_init_flag ) &&
               ( cu_chroma_qp_offset_enabled_flag == rhs.cu_chroma_qp_offset_enabled_flag ) &&
               ( deblocking_filter_override_flag == rhs.deblocking_filter_override_flag ) &&
               ( slice_deblocking_filter_disabled_flag == rhs.slice_deblocking_filter_disabled_flag ) &&
               ( collocated_from_l0_flag == rhs.collocated_from_l0_flag ) &&
               ( slice_loop_filter_across_slices_enabled_flag == rhs.slice_loop_filter_across_slices_enabled_flag ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeH265SliceSegmentHeaderFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t first_slice_segment_in_pic_flag              : 1;
      uint32_t dependent_slice_segment_flag                 : 1;
      uint32_t slice_sao_luma_flag                          : 1;
      uint32_t slice_sao_chroma_flag                        : 1;
      uint32_t num_ref_idx_active_override_flag             : 1;
      uint32_t mvd_l1_zero_flag                             : 1;
      uint32_t cabac_init_flag                              : 1;
      uint32_t cu_chroma_qp_offset_enabled_flag             : 1;
      uint32_t deblocking_filter_override_flag              : 1;
      uint32_t slice_deblocking_filter_disabled_flag        : 1;
      uint32_t collocated_from_l0_flag                      : 1;
      uint32_t slice_loop_filter_across_slices_enabled_flag : 1;
      uint32_t reserved                                     : 20;
    };

    struct EncodeH265SliceSegmentHeader
    {
      using NativeType = StdVideoEncodeH265SliceSegmentHeader;

      operator StdVideoEncodeH265SliceSegmentHeader const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265SliceSegmentHeader *>( this );
      }

      operator StdVideoEncodeH265SliceSegmentHeader &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265SliceSegmentHeader *>( this );
      }

      operator StdVideoEncodeH265SliceSegmentHeader const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265SliceSegmentHeader *>( this );
      }

      operator StdVideoEncodeH265SliceSegmentHeader *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265SliceSegmentHeader *>( this );
      }

      bool operator==( EncodeH265SliceSegmentHeader const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( slice_type == rhs.slice_type ) && ( slice_segment_address == rhs.slice_segment_address ) &&
               ( collocated_ref_idx == rhs.collocated_ref_idx ) && ( MaxNumMergeCand == rhs.MaxNumMergeCand ) &&
               ( slice_cb_qp_offset == rhs.slice_cb_qp_offset ) && ( slice_cr_qp_offset == rhs.slice_cr_qp_offset ) &&
               ( slice_beta_offset_div2 == rhs.slice_beta_offset_div2 ) && ( slice_tc_offset_div2 == rhs.slice_tc_offset_div2 ) &&
               ( slice_act_y_qp_offset == rhs.slice_act_y_qp_offset ) && ( slice_act_cb_qp_offset == rhs.slice_act_cb_qp_offset ) &&
               ( slice_act_cr_qp_offset == rhs.slice_act_cr_qp_offset ) && ( slice_qp_delta == rhs.slice_qp_delta ) && ( reserved1 == rhs.reserved1 ) &&
               ( pWeightTable == rhs.pWeightTable );
      }

      bool operator!=( EncodeH265SliceSegmentHeader const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH265SliceSegmentHeaderFlags flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265SliceType slice_type = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265SliceType::eB;
      uint32_t                                                        slice_segment_address        = {};
      uint8_t                                                         collocated_ref_idx           = {};
      uint8_t                                                         MaxNumMergeCand              = {};
      int8_t                                                          slice_cb_qp_offset           = {};
      int8_t                                                          slice_cr_qp_offset           = {};
      int8_t                                                          slice_beta_offset_div2       = {};
      int8_t                                                          slice_tc_offset_div2         = {};
      int8_t                                                          slice_act_y_qp_offset        = {};
      int8_t                                                          slice_act_cb_qp_offset       = {};
      int8_t                                                          slice_act_cr_qp_offset       = {};
      int8_t                                                          slice_qp_delta               = {};
      uint16_t                                                        reserved1                    = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH265WeightTable * pWeightTable = {};
    };

    struct EncodeH265ReferenceListsInfoFlags
    {
      using NativeType = StdVideoEncodeH265ReferenceListsInfoFlags;

      operator StdVideoEncodeH265ReferenceListsInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265ReferenceListsInfoFlags *>( this );
      }

      operator StdVideoEncodeH265ReferenceListsInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265ReferenceListsInfoFlags *>( this );
      }

      operator StdVideoEncodeH265ReferenceListsInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265ReferenceListsInfoFlags *>( this );
      }

      operator StdVideoEncodeH265ReferenceListsInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265ReferenceListsInfoFlags *>( this );
      }

      bool operator==( EncodeH265ReferenceListsInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( ref_pic_list_modification_flag_l0 == rhs.ref_pic_list_modification_flag_l0 ) &&
               ( ref_pic_list_modification_flag_l1 == rhs.ref_pic_list_modification_flag_l1 ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeH265ReferenceListsInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t ref_pic_list_modification_flag_l0 : 1;
      uint32_t ref_pic_list_modification_flag_l1 : 1;
      uint32_t reserved                          : 30;
    };

    struct EncodeH265ReferenceListsInfo
    {
      using NativeType = StdVideoEncodeH265ReferenceListsInfo;

      operator StdVideoEncodeH265ReferenceListsInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265ReferenceListsInfo *>( this );
      }

      operator StdVideoEncodeH265ReferenceListsInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265ReferenceListsInfo *>( this );
      }

      operator StdVideoEncodeH265ReferenceListsInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265ReferenceListsInfo *>( this );
      }

      operator StdVideoEncodeH265ReferenceListsInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265ReferenceListsInfo *>( this );
      }

      bool operator==( EncodeH265ReferenceListsInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( num_ref_idx_l0_active_minus1 == rhs.num_ref_idx_l0_active_minus1 ) &&
               ( num_ref_idx_l1_active_minus1 == rhs.num_ref_idx_l1_active_minus1 ) && ( RefPicList0 == rhs.RefPicList0 ) &&
               ( RefPicList1 == rhs.RefPicList1 ) && ( list_entry_l0 == rhs.list_entry_l0 ) && ( list_entry_l1 == rhs.list_entry_l1 );
      }

      bool operator!=( EncodeH265ReferenceListsInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH265ReferenceListsInfoFlags flags                        = {};
      uint8_t                                                                             num_ref_idx_l0_active_minus1 = {};
      uint8_t                                                                             num_ref_idx_l1_active_minus1 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF>      RefPicList0                  = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF>      RefPicList1                  = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF>      list_entry_l0                = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_MAX_NUM_LIST_REF>      list_entry_l1                = {};
    };

    struct EncodeH265PictureInfoFlags
    {
      using NativeType = StdVideoEncodeH265PictureInfoFlags;

      operator StdVideoEncodeH265PictureInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeH265PictureInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeH265PictureInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeH265PictureInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265PictureInfoFlags *>( this );
      }

      bool operator==( EncodeH265PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( is_reference == rhs.is_reference ) && ( IrapPicFlag == rhs.IrapPicFlag ) &&
               ( used_for_long_term_reference == rhs.used_for_long_term_reference ) && ( discardable_flag == rhs.discardable_flag ) &&
               ( cross_layer_bla_flag == rhs.cross_layer_bla_flag ) && ( pic_output_flag == rhs.pic_output_flag ) &&
               ( no_output_of_prior_pics_flag == rhs.no_output_of_prior_pics_flag ) &&
               ( short_term_ref_pic_set_sps_flag == rhs.short_term_ref_pic_set_sps_flag ) &&
               ( slice_temporal_mvp_enabled_flag == rhs.slice_temporal_mvp_enabled_flag ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeH265PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t is_reference                    : 1;
      uint32_t IrapPicFlag                     : 1;
      uint32_t used_for_long_term_reference    : 1;
      uint32_t discardable_flag                : 1;
      uint32_t cross_layer_bla_flag            : 1;
      uint32_t pic_output_flag                 : 1;
      uint32_t no_output_of_prior_pics_flag    : 1;
      uint32_t short_term_ref_pic_set_sps_flag : 1;
      uint32_t slice_temporal_mvp_enabled_flag : 1;
      uint32_t reserved                        : 23;
    };

    struct EncodeH265LongTermRefPics
    {
      using NativeType = StdVideoEncodeH265LongTermRefPics;

      operator StdVideoEncodeH265LongTermRefPics const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265LongTermRefPics *>( this );
      }

      operator StdVideoEncodeH265LongTermRefPics &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265LongTermRefPics *>( this );
      }

      operator StdVideoEncodeH265LongTermRefPics const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265LongTermRefPics *>( this );
      }

      operator StdVideoEncodeH265LongTermRefPics *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265LongTermRefPics *>( this );
      }

      bool operator==( EncodeH265LongTermRefPics const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( num_long_term_sps == rhs.num_long_term_sps ) && ( num_long_term_pics == rhs.num_long_term_pics ) && ( lt_idx_sps == rhs.lt_idx_sps ) &&
               ( poc_lsb_lt == rhs.poc_lsb_lt ) && ( used_by_curr_pic_lt_flag == rhs.used_by_curr_pic_lt_flag ) &&
               ( delta_poc_msb_present_flag == rhs.delta_poc_msb_present_flag ) && ( delta_poc_msb_cycle_lt == rhs.delta_poc_msb_cycle_lt );
      }

      bool operator!=( EncodeH265LongTermRefPics const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint8_t                                                                                  num_long_term_sps          = {};
      uint8_t                                                                                  num_long_term_pics         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_MAX_LONG_TERM_REF_PICS_SPS> lt_idx_sps                 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_MAX_LONG_TERM_PICS>         poc_lsb_lt                 = {};
      uint16_t                                                                                 used_by_curr_pic_lt_flag   = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_MAX_DELTA_POC>              delta_poc_msb_present_flag = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_H265_MAX_DELTA_POC>              delta_poc_msb_cycle_lt     = {};
    };

    struct EncodeH265PictureInfo
    {
      using NativeType = StdVideoEncodeH265PictureInfo;

      operator StdVideoEncodeH265PictureInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265PictureInfo *>( this );
      }

      operator StdVideoEncodeH265PictureInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265PictureInfo *>( this );
      }

      operator StdVideoEncodeH265PictureInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265PictureInfo *>( this );
      }

      operator StdVideoEncodeH265PictureInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265PictureInfo *>( this );
      }

      bool operator==( EncodeH265PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( pic_type == rhs.pic_type ) && ( sps_video_parameter_set_id == rhs.sps_video_parameter_set_id ) &&
               ( pps_seq_parameter_set_id == rhs.pps_seq_parameter_set_id ) && ( pps_pic_parameter_set_id == rhs.pps_pic_parameter_set_id ) &&
               ( short_term_ref_pic_set_idx == rhs.short_term_ref_pic_set_idx ) && ( PicOrderCntVal == rhs.PicOrderCntVal ) &&
               ( TemporalId == rhs.TemporalId ) && ( reserved1 == rhs.reserved1 ) && ( pRefLists == rhs.pRefLists ) &&
               ( pShortTermRefPicSet == rhs.pShortTermRefPicSet ) && ( pLongTermRefPics == rhs.pLongTermRefPics );
      }

      bool operator!=( EncodeH265PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH265PictureInfoFlags flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265PictureType pic_type = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265PictureType::eP;
      uint8_t                                                           sps_video_parameter_set_id               = {};
      uint8_t                                                           pps_seq_parameter_set_id                 = {};
      uint8_t                                                           pps_pic_parameter_set_id                 = {};
      uint8_t                                                           short_term_ref_pic_set_idx               = {};
      int32_t                                                           PicOrderCntVal                           = {};
      uint8_t                                                           TemporalId                               = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, 7>                  reserved1                                = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH265ReferenceListsInfo * pRefLists           = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265ShortTermRefPicSet *       pShortTermRefPicSet = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH265LongTermRefPics *    pLongTermRefPics    = {};
    };

    struct EncodeH265ReferenceInfoFlags
    {
      using NativeType = StdVideoEncodeH265ReferenceInfoFlags;

      operator StdVideoEncodeH265ReferenceInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeH265ReferenceInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeH265ReferenceInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeH265ReferenceInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265ReferenceInfoFlags *>( this );
      }

      bool operator==( EncodeH265ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( used_for_long_term_reference == rhs.used_for_long_term_reference ) && ( unused_for_reference == rhs.unused_for_reference ) &&
               ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeH265ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t used_for_long_term_reference : 1;
      uint32_t unused_for_reference         : 1;
      uint32_t reserved                     : 30;
    };

    struct EncodeH265ReferenceInfo
    {
      using NativeType = StdVideoEncodeH265ReferenceInfo;

      operator StdVideoEncodeH265ReferenceInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeH265ReferenceInfo *>( this );
      }

      operator StdVideoEncodeH265ReferenceInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeH265ReferenceInfo *>( this );
      }

      operator StdVideoEncodeH265ReferenceInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeH265ReferenceInfo *>( this );
      }

      operator StdVideoEncodeH265ReferenceInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeH265ReferenceInfo *>( this );
      }

      bool operator==( EncodeH265ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( pic_type == rhs.pic_type ) && ( PicOrderCntVal == rhs.PicOrderCntVal ) && ( TemporalId == rhs.TemporalId );
      }

      bool operator!=( EncodeH265ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeH265ReferenceInfoFlags flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265PictureType pic_type       = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::H265PictureType::eP;
      int32_t                                                           PicOrderCntVal = {};
      uint8_t                                                           TemporalId     = {};
    };

    //=== vulkan_video_codec_av1std ===

    struct AV1ColorConfigFlags
    {
      using NativeType = StdVideoAV1ColorConfigFlags;

      operator StdVideoAV1ColorConfigFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1ColorConfigFlags *>( this );
      }

      operator StdVideoAV1ColorConfigFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1ColorConfigFlags *>( this );
      }

      operator StdVideoAV1ColorConfigFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1ColorConfigFlags *>( this );
      }

      operator StdVideoAV1ColorConfigFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1ColorConfigFlags *>( this );
      }

      bool operator==( AV1ColorConfigFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( mono_chrome == rhs.mono_chrome ) && ( color_range == rhs.color_range ) && ( separate_uv_delta_q == rhs.separate_uv_delta_q ) &&
               ( color_description_present_flag == rhs.color_description_present_flag ) && ( reserved == rhs.reserved );
      }

      bool operator!=( AV1ColorConfigFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t mono_chrome                    : 1;
      uint32_t color_range                    : 1;
      uint32_t separate_uv_delta_q            : 1;
      uint32_t color_description_present_flag : 1;
      uint32_t reserved                       : 28;
    };

    struct AV1ColorConfig
    {
      using NativeType = StdVideoAV1ColorConfig;

      operator StdVideoAV1ColorConfig const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1ColorConfig *>( this );
      }

      operator StdVideoAV1ColorConfig &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1ColorConfig *>( this );
      }

      operator StdVideoAV1ColorConfig const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1ColorConfig *>( this );
      }

      operator StdVideoAV1ColorConfig *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1ColorConfig *>( this );
      }

      bool operator==( AV1ColorConfig const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( BitDepth == rhs.BitDepth ) && ( subsampling_x == rhs.subsampling_x ) && ( subsampling_y == rhs.subsampling_y ) &&
               ( reserved1 == rhs.reserved1 ) && ( color_primaries == rhs.color_primaries ) && ( transfer_characteristics == rhs.transfer_characteristics ) &&
               ( matrix_coefficients == rhs.matrix_coefficients ) && ( chroma_sample_position == rhs.chroma_sample_position );
      }

      bool operator!=( AV1ColorConfig const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1ColorConfigFlags flags         = {};
      uint8_t                                                               BitDepth      = {};
      uint8_t                                                               subsampling_x = {};
      uint8_t                                                               subsampling_y = {};
      uint8_t                                                               reserved1     = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1ColorPrimaries   color_primaries =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1ColorPrimaries::eBt709;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TransferCharacteristics transfer_characteristics =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TransferCharacteristics::eReserved0;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1MatrixCoefficients matrix_coefficients =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1MatrixCoefficients::eIdentity;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1ChromaSamplePosition chroma_sample_position =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1ChromaSamplePosition::eUnknown;
    };

    struct AV1TimingInfoFlags
    {
      using NativeType = StdVideoAV1TimingInfoFlags;

      operator StdVideoAV1TimingInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1TimingInfoFlags *>( this );
      }

      operator StdVideoAV1TimingInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1TimingInfoFlags *>( this );
      }

      operator StdVideoAV1TimingInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1TimingInfoFlags *>( this );
      }

      operator StdVideoAV1TimingInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1TimingInfoFlags *>( this );
      }

      bool operator==( AV1TimingInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( equal_picture_interval == rhs.equal_picture_interval ) && ( reserved == rhs.reserved );
      }

      bool operator!=( AV1TimingInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t equal_picture_interval : 1;
      uint32_t reserved               : 31;
    };

    struct AV1TimingInfo
    {
      using NativeType = StdVideoAV1TimingInfo;

      operator StdVideoAV1TimingInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1TimingInfo *>( this );
      }

      operator StdVideoAV1TimingInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1TimingInfo *>( this );
      }

      operator StdVideoAV1TimingInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1TimingInfo *>( this );
      }

      operator StdVideoAV1TimingInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1TimingInfo *>( this );
      }

      bool operator==( AV1TimingInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( num_units_in_display_tick == rhs.num_units_in_display_tick ) && ( time_scale == rhs.time_scale ) &&
               ( num_ticks_per_picture_minus_1 == rhs.num_ticks_per_picture_minus_1 );
      }

      bool operator!=( AV1TimingInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TimingInfoFlags flags                         = {};
      uint32_t                                                             num_units_in_display_tick     = {};
      uint32_t                                                             time_scale                    = {};
      uint32_t                                                             num_ticks_per_picture_minus_1 = {};
    };

    struct AV1LoopFilterFlags
    {
      using NativeType = StdVideoAV1LoopFilterFlags;

      operator StdVideoAV1LoopFilterFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1LoopFilterFlags *>( this );
      }

      operator StdVideoAV1LoopFilterFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1LoopFilterFlags *>( this );
      }

      operator StdVideoAV1LoopFilterFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1LoopFilterFlags *>( this );
      }

      operator StdVideoAV1LoopFilterFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1LoopFilterFlags *>( this );
      }

      bool operator==( AV1LoopFilterFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( loop_filter_delta_enabled == rhs.loop_filter_delta_enabled ) && ( loop_filter_delta_update == rhs.loop_filter_delta_update ) &&
               ( reserved == rhs.reserved );
      }

      bool operator!=( AV1LoopFilterFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t loop_filter_delta_enabled : 1;
      uint32_t loop_filter_delta_update  : 1;
      uint32_t reserved                  : 30;
    };

    struct AV1LoopFilter
    {
      using NativeType = StdVideoAV1LoopFilter;

      operator StdVideoAV1LoopFilter const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1LoopFilter *>( this );
      }

      operator StdVideoAV1LoopFilter &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1LoopFilter *>( this );
      }

      operator StdVideoAV1LoopFilter const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1LoopFilter *>( this );
      }

      operator StdVideoAV1LoopFilter *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1LoopFilter *>( this );
      }

      bool operator==( AV1LoopFilter const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( loop_filter_level == rhs.loop_filter_level ) && ( loop_filter_sharpness == rhs.loop_filter_sharpness ) &&
               ( update_ref_delta == rhs.update_ref_delta ) && ( loop_filter_ref_deltas == rhs.loop_filter_ref_deltas ) &&
               ( update_mode_delta == rhs.update_mode_delta ) && ( loop_filter_mode_deltas == rhs.loop_filter_mode_deltas );
      }

      bool operator!=( AV1LoopFilter const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1LoopFilterFlags                   flags                   = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_LOOP_FILTER_STRENGTHS> loop_filter_level       = {};
      uint8_t                                                                                loop_filter_sharpness   = {};
      uint8_t                                                                                update_ref_delta        = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME>       loop_filter_ref_deltas  = {};
      uint8_t                                                                                update_mode_delta       = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS>    loop_filter_mode_deltas = {};
    };

    struct AV1QuantizationFlags
    {
      using NativeType = StdVideoAV1QuantizationFlags;

      operator StdVideoAV1QuantizationFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1QuantizationFlags *>( this );
      }

      operator StdVideoAV1QuantizationFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1QuantizationFlags *>( this );
      }

      operator StdVideoAV1QuantizationFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1QuantizationFlags *>( this );
      }

      operator StdVideoAV1QuantizationFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1QuantizationFlags *>( this );
      }

      bool operator==( AV1QuantizationFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( using_qmatrix == rhs.using_qmatrix ) && ( diff_uv_delta == rhs.diff_uv_delta ) && ( reserved == rhs.reserved );
      }

      bool operator!=( AV1QuantizationFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t using_qmatrix : 1;
      uint32_t diff_uv_delta : 1;
      uint32_t reserved      : 30;
    };

    struct AV1Quantization
    {
      using NativeType = StdVideoAV1Quantization;

      operator StdVideoAV1Quantization const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1Quantization *>( this );
      }

      operator StdVideoAV1Quantization &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1Quantization *>( this );
      }

      operator StdVideoAV1Quantization const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1Quantization *>( this );
      }

      operator StdVideoAV1Quantization *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1Quantization *>( this );
      }

      bool operator==( AV1Quantization const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( base_q_idx == rhs.base_q_idx ) && ( DeltaQYDc == rhs.DeltaQYDc ) && ( DeltaQUDc == rhs.DeltaQUDc ) &&
               ( DeltaQUAc == rhs.DeltaQUAc ) && ( DeltaQVDc == rhs.DeltaQVDc ) && ( DeltaQVAc == rhs.DeltaQVAc ) && ( qm_y == rhs.qm_y ) &&
               ( qm_u == rhs.qm_u ) && ( qm_v == rhs.qm_v );
      }

      bool operator!=( AV1Quantization const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1QuantizationFlags flags      = {};
      uint8_t                                                                base_q_idx = {};
      int8_t                                                                 DeltaQYDc  = {};
      int8_t                                                                 DeltaQUDc  = {};
      int8_t                                                                 DeltaQUAc  = {};
      int8_t                                                                 DeltaQVDc  = {};
      int8_t                                                                 DeltaQVAc  = {};
      uint8_t                                                                qm_y       = {};
      uint8_t                                                                qm_u       = {};
      uint8_t                                                                qm_v       = {};
    };

    struct AV1Segmentation
    {
      using NativeType = StdVideoAV1Segmentation;

      operator StdVideoAV1Segmentation const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1Segmentation *>( this );
      }

      operator StdVideoAV1Segmentation &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1Segmentation *>( this );
      }

      operator StdVideoAV1Segmentation const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1Segmentation *>( this );
      }

      operator StdVideoAV1Segmentation *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1Segmentation *>( this );
      }

      bool operator==( AV1Segmentation const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( FeatureEnabled == rhs.FeatureEnabled ) && ( FeatureData == rhs.FeatureData );
      }

      bool operator!=( AV1Segmentation const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_SEGMENTS>                            FeatureEnabled = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int16_t, STD_VIDEO_AV1_MAX_SEGMENTS, STD_VIDEO_AV1_SEG_LVL_MAX> FeatureData    = {};
    };

    struct AV1TileInfoFlags
    {
      using NativeType = StdVideoAV1TileInfoFlags;

      operator StdVideoAV1TileInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1TileInfoFlags *>( this );
      }

      operator StdVideoAV1TileInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1TileInfoFlags *>( this );
      }

      operator StdVideoAV1TileInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1TileInfoFlags *>( this );
      }

      operator StdVideoAV1TileInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1TileInfoFlags *>( this );
      }

      bool operator==( AV1TileInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( uniform_tile_spacing_flag == rhs.uniform_tile_spacing_flag ) && ( reserved == rhs.reserved );
      }

      bool operator!=( AV1TileInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t uniform_tile_spacing_flag : 1;
      uint32_t reserved                  : 31;
    };

    struct AV1TileInfo
    {
      using NativeType = StdVideoAV1TileInfo;

      operator StdVideoAV1TileInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1TileInfo *>( this );
      }

      operator StdVideoAV1TileInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1TileInfo *>( this );
      }

      operator StdVideoAV1TileInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1TileInfo *>( this );
      }

      operator StdVideoAV1TileInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1TileInfo *>( this );
      }

      bool operator==( AV1TileInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( TileCols == rhs.TileCols ) && ( TileRows == rhs.TileRows ) &&
               ( context_update_tile_id == rhs.context_update_tile_id ) && ( tile_size_bytes_minus_1 == rhs.tile_size_bytes_minus_1 ) &&
               ( reserved1 == rhs.reserved1 ) && ( pMiColStarts == rhs.pMiColStarts ) && ( pMiRowStarts == rhs.pMiRowStarts ) &&
               ( pWidthInSbsMinus1 == rhs.pWidthInSbsMinus1 ) && ( pHeightInSbsMinus1 == rhs.pHeightInSbsMinus1 );
      }

      bool operator!=( AV1TileInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TileInfoFlags flags                   = {};
      uint8_t                                                            TileCols                = {};
      uint8_t                                                            TileRows                = {};
      uint16_t                                                           context_update_tile_id  = {};
      uint8_t                                                            tile_size_bytes_minus_1 = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, 7>                   reserved1               = {};
      const uint16_t *                                                   pMiColStarts            = {};
      const uint16_t *                                                   pMiRowStarts            = {};
      const uint16_t *                                                   pWidthInSbsMinus1       = {};
      const uint16_t *                                                   pHeightInSbsMinus1      = {};
    };

    struct AV1CDEF
    {
      using NativeType = StdVideoAV1CDEF;

      operator StdVideoAV1CDEF const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1CDEF *>( this );
      }

      operator StdVideoAV1CDEF &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1CDEF *>( this );
      }

      operator StdVideoAV1CDEF const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1CDEF *>( this );
      }

      operator StdVideoAV1CDEF *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1CDEF *>( this );
      }

      bool operator==( AV1CDEF const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( cdef_damping_minus_3 == rhs.cdef_damping_minus_3 ) && ( cdef_bits == rhs.cdef_bits ) && ( cdef_y_pri_strength == rhs.cdef_y_pri_strength ) &&
               ( cdef_y_sec_strength == rhs.cdef_y_sec_strength ) && ( cdef_uv_pri_strength == rhs.cdef_uv_pri_strength ) &&
               ( cdef_uv_sec_strength == rhs.cdef_uv_sec_strength );
      }

      bool operator!=( AV1CDEF const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint8_t                                                                                cdef_damping_minus_3 = {};
      uint8_t                                                                                cdef_bits            = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS> cdef_y_pri_strength  = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS> cdef_y_sec_strength  = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS> cdef_uv_pri_strength = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS> cdef_uv_sec_strength = {};
    };

    struct AV1LoopRestoration
    {
      using NativeType = StdVideoAV1LoopRestoration;

      operator StdVideoAV1LoopRestoration const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1LoopRestoration *>( this );
      }

      operator StdVideoAV1LoopRestoration &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1LoopRestoration *>( this );
      }

      operator StdVideoAV1LoopRestoration const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1LoopRestoration *>( this );
      }

      operator StdVideoAV1LoopRestoration *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1LoopRestoration *>( this );
      }

      bool operator==( AV1LoopRestoration const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( FrameRestorationType == rhs.FrameRestorationType ) && ( LoopRestorationSize == rhs.LoopRestorationSize );
      }

      bool operator!=( AV1LoopRestoration const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<StdVideoAV1FrameRestorationType, STD_VIDEO_AV1_MAX_NUM_PLANES> FrameRestorationType = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint16_t, STD_VIDEO_AV1_MAX_NUM_PLANES>                        LoopRestorationSize  = {};
    };

    struct AV1GlobalMotion
    {
      using NativeType = StdVideoAV1GlobalMotion;

      operator StdVideoAV1GlobalMotion const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1GlobalMotion *>( this );
      }

      operator StdVideoAV1GlobalMotion &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1GlobalMotion *>( this );
      }

      operator StdVideoAV1GlobalMotion const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1GlobalMotion *>( this );
      }

      operator StdVideoAV1GlobalMotion *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1GlobalMotion *>( this );
      }

      bool operator==( AV1GlobalMotion const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( GmType == rhs.GmType ) && ( gm_params == rhs.gm_params );
      }

      bool operator!=( AV1GlobalMotion const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_NUM_REF_FRAMES>                                     GmType    = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper2D<int32_t, STD_VIDEO_AV1_NUM_REF_FRAMES, STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS> gm_params = {};
    };

    struct AV1FilmGrainFlags
    {
      using NativeType = StdVideoAV1FilmGrainFlags;

      operator StdVideoAV1FilmGrainFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1FilmGrainFlags *>( this );
      }

      operator StdVideoAV1FilmGrainFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1FilmGrainFlags *>( this );
      }

      operator StdVideoAV1FilmGrainFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1FilmGrainFlags *>( this );
      }

      operator StdVideoAV1FilmGrainFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1FilmGrainFlags *>( this );
      }

      bool operator==( AV1FilmGrainFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( chroma_scaling_from_luma == rhs.chroma_scaling_from_luma ) && ( overlap_flag == rhs.overlap_flag ) &&
               ( clip_to_restricted_range == rhs.clip_to_restricted_range ) && ( update_grain == rhs.update_grain ) && ( reserved == rhs.reserved );
      }

      bool operator!=( AV1FilmGrainFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t chroma_scaling_from_luma : 1;
      uint32_t overlap_flag             : 1;
      uint32_t clip_to_restricted_range : 1;
      uint32_t update_grain             : 1;
      uint32_t reserved                 : 28;
    };

    struct AV1FilmGrain
    {
      using NativeType = StdVideoAV1FilmGrain;

      operator StdVideoAV1FilmGrain const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1FilmGrain *>( this );
      }

      operator StdVideoAV1FilmGrain &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1FilmGrain *>( this );
      }

      operator StdVideoAV1FilmGrain const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1FilmGrain *>( this );
      }

      operator StdVideoAV1FilmGrain *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1FilmGrain *>( this );
      }

      bool operator==( AV1FilmGrain const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( grain_scaling_minus_8 == rhs.grain_scaling_minus_8 ) && ( ar_coeff_lag == rhs.ar_coeff_lag ) &&
               ( ar_coeff_shift_minus_6 == rhs.ar_coeff_shift_minus_6 ) && ( grain_scale_shift == rhs.grain_scale_shift ) && ( grain_seed == rhs.grain_seed ) &&
               ( film_grain_params_ref_idx == rhs.film_grain_params_ref_idx ) && ( num_y_points == rhs.num_y_points ) &&
               ( point_y_value == rhs.point_y_value ) && ( point_y_scaling == rhs.point_y_scaling ) && ( num_cb_points == rhs.num_cb_points ) &&
               ( point_cb_value == rhs.point_cb_value ) && ( point_cb_scaling == rhs.point_cb_scaling ) && ( num_cr_points == rhs.num_cr_points ) &&
               ( point_cr_value == rhs.point_cr_value ) && ( point_cr_scaling == rhs.point_cr_scaling ) &&
               ( ar_coeffs_y_plus_128 == rhs.ar_coeffs_y_plus_128 ) && ( ar_coeffs_cb_plus_128 == rhs.ar_coeffs_cb_plus_128 ) &&
               ( ar_coeffs_cr_plus_128 == rhs.ar_coeffs_cr_plus_128 ) && ( cb_mult == rhs.cb_mult ) && ( cb_luma_mult == rhs.cb_luma_mult ) &&
               ( cb_offset == rhs.cb_offset ) && ( cr_mult == rhs.cr_mult ) && ( cr_luma_mult == rhs.cr_luma_mult ) && ( cr_offset == rhs.cr_offset );
      }

      bool operator!=( AV1FilmGrain const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1FilmGrainFlags            flags                     = {};
      uint8_t                                                                        grain_scaling_minus_8     = {};
      uint8_t                                                                        ar_coeff_lag              = {};
      uint8_t                                                                        ar_coeff_shift_minus_6    = {};
      uint8_t                                                                        grain_scale_shift         = {};
      uint16_t                                                                       grain_seed                = {};
      uint8_t                                                                        film_grain_params_ref_idx = {};
      uint8_t                                                                        num_y_points              = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_NUM_Y_POINTS>  point_y_value             = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_NUM_Y_POINTS>  point_y_scaling           = {};
      uint8_t                                                                        num_cb_points             = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_NUM_CB_POINTS> point_cb_value            = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_NUM_CB_POINTS> point_cb_scaling          = {};
      uint8_t                                                                        num_cr_points             = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_NUM_CR_POINTS> point_cr_value            = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_MAX_NUM_CR_POINTS> point_cr_scaling          = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_AV1_MAX_NUM_POS_LUMA>   ar_coeffs_y_plus_128      = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_AV1_MAX_NUM_POS_CHROMA> ar_coeffs_cb_plus_128     = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_AV1_MAX_NUM_POS_CHROMA> ar_coeffs_cr_plus_128     = {};
      uint8_t                                                                        cb_mult                   = {};
      uint8_t                                                                        cb_luma_mult              = {};
      uint16_t                                                                       cb_offset                 = {};
      uint8_t                                                                        cr_mult                   = {};
      uint8_t                                                                        cr_luma_mult              = {};
      uint16_t                                                                       cr_offset                 = {};
    };

    struct AV1SequenceHeaderFlags
    {
      using NativeType = StdVideoAV1SequenceHeaderFlags;

      operator StdVideoAV1SequenceHeaderFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1SequenceHeaderFlags *>( this );
      }

      operator StdVideoAV1SequenceHeaderFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1SequenceHeaderFlags *>( this );
      }

      operator StdVideoAV1SequenceHeaderFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1SequenceHeaderFlags *>( this );
      }

      operator StdVideoAV1SequenceHeaderFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1SequenceHeaderFlags *>( this );
      }

      bool operator==( AV1SequenceHeaderFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( still_picture == rhs.still_picture ) && ( reduced_still_picture_header == rhs.reduced_still_picture_header ) &&
               ( use_128x128_superblock == rhs.use_128x128_superblock ) && ( enable_filter_intra == rhs.enable_filter_intra ) &&
               ( enable_intra_edge_filter == rhs.enable_intra_edge_filter ) && ( enable_interintra_compound == rhs.enable_interintra_compound ) &&
               ( enable_masked_compound == rhs.enable_masked_compound ) && ( enable_warped_motion == rhs.enable_warped_motion ) &&
               ( enable_dual_filter == rhs.enable_dual_filter ) && ( enable_order_hint == rhs.enable_order_hint ) &&
               ( enable_jnt_comp == rhs.enable_jnt_comp ) && ( enable_ref_frame_mvs == rhs.enable_ref_frame_mvs ) &&
               ( frame_id_numbers_present_flag == rhs.frame_id_numbers_present_flag ) && ( enable_superres == rhs.enable_superres ) &&
               ( enable_cdef == rhs.enable_cdef ) && ( enable_restoration == rhs.enable_restoration ) &&
               ( film_grain_params_present == rhs.film_grain_params_present ) && ( timing_info_present_flag == rhs.timing_info_present_flag ) &&
               ( initial_display_delay_present_flag == rhs.initial_display_delay_present_flag ) && ( reserved == rhs.reserved );
      }

      bool operator!=( AV1SequenceHeaderFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t still_picture                      : 1;
      uint32_t reduced_still_picture_header       : 1;
      uint32_t use_128x128_superblock             : 1;
      uint32_t enable_filter_intra                : 1;
      uint32_t enable_intra_edge_filter           : 1;
      uint32_t enable_interintra_compound         : 1;
      uint32_t enable_masked_compound             : 1;
      uint32_t enable_warped_motion               : 1;
      uint32_t enable_dual_filter                 : 1;
      uint32_t enable_order_hint                  : 1;
      uint32_t enable_jnt_comp                    : 1;
      uint32_t enable_ref_frame_mvs               : 1;
      uint32_t frame_id_numbers_present_flag      : 1;
      uint32_t enable_superres                    : 1;
      uint32_t enable_cdef                        : 1;
      uint32_t enable_restoration                 : 1;
      uint32_t film_grain_params_present          : 1;
      uint32_t timing_info_present_flag           : 1;
      uint32_t initial_display_delay_present_flag : 1;
      uint32_t reserved                           : 13;
    };

    struct AV1SequenceHeader
    {
      using NativeType = StdVideoAV1SequenceHeader;

      operator StdVideoAV1SequenceHeader const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoAV1SequenceHeader *>( this );
      }

      operator StdVideoAV1SequenceHeader &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoAV1SequenceHeader *>( this );
      }

      operator StdVideoAV1SequenceHeader const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoAV1SequenceHeader *>( this );
      }

      operator StdVideoAV1SequenceHeader *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoAV1SequenceHeader *>( this );
      }

      bool operator==( AV1SequenceHeader const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( seq_profile == rhs.seq_profile ) && ( frame_width_bits_minus_1 == rhs.frame_width_bits_minus_1 ) &&
               ( frame_height_bits_minus_1 == rhs.frame_height_bits_minus_1 ) && ( max_frame_width_minus_1 == rhs.max_frame_width_minus_1 ) &&
               ( max_frame_height_minus_1 == rhs.max_frame_height_minus_1 ) && ( delta_frame_id_length_minus_2 == rhs.delta_frame_id_length_minus_2 ) &&
               ( additional_frame_id_length_minus_1 == rhs.additional_frame_id_length_minus_1 ) && ( order_hint_bits_minus_1 == rhs.order_hint_bits_minus_1 ) &&
               ( seq_force_integer_mv == rhs.seq_force_integer_mv ) && ( seq_force_screen_content_tools == rhs.seq_force_screen_content_tools ) &&
               ( reserved1 == rhs.reserved1 ) && ( pColorConfig == rhs.pColorConfig ) && ( pTimingInfo == rhs.pTimingInfo );
      }

      bool operator!=( AV1SequenceHeader const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1SequenceHeaderFlags flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1Profile seq_profile       = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1Profile::eMain;
      uint8_t                                                      frame_width_bits_minus_1           = {};
      uint8_t                                                      frame_height_bits_minus_1          = {};
      uint16_t                                                     max_frame_width_minus_1            = {};
      uint16_t                                                     max_frame_height_minus_1           = {};
      uint8_t                                                      delta_frame_id_length_minus_2      = {};
      uint8_t                                                      additional_frame_id_length_minus_1 = {};
      uint8_t                                                      order_hint_bits_minus_1            = {};
      uint8_t                                                      seq_force_integer_mv               = {};
      uint8_t                                                      seq_force_screen_content_tools     = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, 5>             reserved1                          = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1ColorConfig * pColorConfig           = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TimingInfo *  pTimingInfo            = {};
    };

    //=== vulkan_video_codec_av1std_decode ===

    struct DecodeAV1PictureInfoFlags
    {
      using NativeType = StdVideoDecodeAV1PictureInfoFlags;

      operator StdVideoDecodeAV1PictureInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeAV1PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeAV1PictureInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeAV1PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeAV1PictureInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeAV1PictureInfoFlags *>( this );
      }

      operator StdVideoDecodeAV1PictureInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeAV1PictureInfoFlags *>( this );
      }

      bool operator==( DecodeAV1PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( error_resilient_mode == rhs.error_resilient_mode ) && ( disable_cdf_update == rhs.disable_cdf_update ) &&
               ( use_superres == rhs.use_superres ) && ( render_and_frame_size_different == rhs.render_and_frame_size_different ) &&
               ( allow_screen_content_tools == rhs.allow_screen_content_tools ) && ( is_filter_switchable == rhs.is_filter_switchable ) &&
               ( force_integer_mv == rhs.force_integer_mv ) && ( frame_size_override_flag == rhs.frame_size_override_flag ) &&
               ( buffer_removal_time_present_flag == rhs.buffer_removal_time_present_flag ) && ( allow_intrabc == rhs.allow_intrabc ) &&
               ( frame_refs_short_signaling == rhs.frame_refs_short_signaling ) && ( allow_high_precision_mv == rhs.allow_high_precision_mv ) &&
               ( is_motion_mode_switchable == rhs.is_motion_mode_switchable ) && ( use_ref_frame_mvs == rhs.use_ref_frame_mvs ) &&
               ( disable_frame_end_update_cdf == rhs.disable_frame_end_update_cdf ) && ( allow_warped_motion == rhs.allow_warped_motion ) &&
               ( reduced_tx_set == rhs.reduced_tx_set ) && ( reference_select == rhs.reference_select ) && ( skip_mode_present == rhs.skip_mode_present ) &&
               ( delta_q_present == rhs.delta_q_present ) && ( delta_lf_present == rhs.delta_lf_present ) && ( delta_lf_multi == rhs.delta_lf_multi ) &&
               ( segmentation_enabled == rhs.segmentation_enabled ) && ( segmentation_update_map == rhs.segmentation_update_map ) &&
               ( segmentation_temporal_update == rhs.segmentation_temporal_update ) && ( segmentation_update_data == rhs.segmentation_update_data ) &&
               ( UsesLr == rhs.UsesLr ) && ( usesChromaLr == rhs.usesChromaLr ) && ( apply_grain == rhs.apply_grain ) && ( reserved == rhs.reserved );
      }

      bool operator!=( DecodeAV1PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t error_resilient_mode             : 1;
      uint32_t disable_cdf_update               : 1;
      uint32_t use_superres                     : 1;
      uint32_t render_and_frame_size_different  : 1;
      uint32_t allow_screen_content_tools       : 1;
      uint32_t is_filter_switchable             : 1;
      uint32_t force_integer_mv                 : 1;
      uint32_t frame_size_override_flag         : 1;
      uint32_t buffer_removal_time_present_flag : 1;
      uint32_t allow_intrabc                    : 1;
      uint32_t frame_refs_short_signaling       : 1;
      uint32_t allow_high_precision_mv          : 1;
      uint32_t is_motion_mode_switchable        : 1;
      uint32_t use_ref_frame_mvs                : 1;
      uint32_t disable_frame_end_update_cdf     : 1;
      uint32_t allow_warped_motion              : 1;
      uint32_t reduced_tx_set                   : 1;
      uint32_t reference_select                 : 1;
      uint32_t skip_mode_present                : 1;
      uint32_t delta_q_present                  : 1;
      uint32_t delta_lf_present                 : 1;
      uint32_t delta_lf_multi                   : 1;
      uint32_t segmentation_enabled             : 1;
      uint32_t segmentation_update_map          : 1;
      uint32_t segmentation_temporal_update     : 1;
      uint32_t segmentation_update_data         : 1;
      uint32_t UsesLr                           : 1;
      uint32_t usesChromaLr                     : 1;
      uint32_t apply_grain                      : 1;
      uint32_t reserved                         : 3;
    };

    struct DecodeAV1PictureInfo
    {
      using NativeType = StdVideoDecodeAV1PictureInfo;

      operator StdVideoDecodeAV1PictureInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeAV1PictureInfo *>( this );
      }

      operator StdVideoDecodeAV1PictureInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeAV1PictureInfo *>( this );
      }

      operator StdVideoDecodeAV1PictureInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeAV1PictureInfo *>( this );
      }

      operator StdVideoDecodeAV1PictureInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeAV1PictureInfo *>( this );
      }

      bool operator==( DecodeAV1PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( frame_type == rhs.frame_type ) && ( current_frame_id == rhs.current_frame_id ) && ( OrderHint == rhs.OrderHint ) &&
               ( primary_ref_frame == rhs.primary_ref_frame ) && ( refresh_frame_flags == rhs.refresh_frame_flags ) && ( reserved1 == rhs.reserved1 ) &&
               ( interpolation_filter == rhs.interpolation_filter ) && ( TxMode == rhs.TxMode ) && ( delta_q_res == rhs.delta_q_res ) &&
               ( delta_lf_res == rhs.delta_lf_res ) && ( SkipModeFrame == rhs.SkipModeFrame ) && ( coded_denom == rhs.coded_denom ) &&
               ( reserved2 == rhs.reserved2 ) && ( OrderHints == rhs.OrderHints ) && ( expectedFrameId == rhs.expectedFrameId ) &&
               ( pTileInfo == rhs.pTileInfo ) && ( pQuantization == rhs.pQuantization ) && ( pSegmentation == rhs.pSegmentation ) &&
               ( pLoopFilter == rhs.pLoopFilter ) && ( pCDEF == rhs.pCDEF ) && ( pLoopRestoration == rhs.pLoopRestoration ) &&
               ( pGlobalMotion == rhs.pGlobalMotion ) && ( pFilmGrain == rhs.pFilmGrain );
      }

      bool operator!=( DecodeAV1PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::DecodeAV1PictureInfoFlags flags  = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1FrameType frame_type          = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1FrameType::eKey;
      uint32_t                                                       current_frame_id    = {};
      uint8_t                                                        OrderHint           = {};
      uint8_t                                                        primary_ref_frame   = {};
      uint8_t                                                        refresh_frame_flags = {};
      uint8_t                                                        reserved1           = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1InterpolationFilter interpolation_filter =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1InterpolationFilter::eEighttap;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TxMode TxMode       = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TxMode::eOnly4X4;
      uint8_t                                                     delta_q_res  = {};
      uint8_t                                                     delta_lf_res = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_SKIP_MODE_FRAMES> SkipModeFrame    = {};
      uint8_t                                                                       coded_denom      = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, 3>                              reserved2        = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_NUM_REF_FRAMES>   OrderHints       = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_AV1_NUM_REF_FRAMES>  expectedFrameId  = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TileInfo *         pTileInfo        = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1Quantization *     pQuantization    = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1Segmentation *     pSegmentation    = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1LoopFilter *       pLoopFilter      = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1CDEF *             pCDEF            = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1LoopRestoration *  pLoopRestoration = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1GlobalMotion *     pGlobalMotion    = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1FilmGrain *        pFilmGrain       = {};
    };

    struct DecodeAV1ReferenceInfoFlags
    {
      using NativeType = StdVideoDecodeAV1ReferenceInfoFlags;

      operator StdVideoDecodeAV1ReferenceInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeAV1ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeAV1ReferenceInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeAV1ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeAV1ReferenceInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeAV1ReferenceInfoFlags *>( this );
      }

      operator StdVideoDecodeAV1ReferenceInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeAV1ReferenceInfoFlags *>( this );
      }

      bool operator==( DecodeAV1ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( disable_frame_end_update_cdf == rhs.disable_frame_end_update_cdf ) && ( segmentation_enabled == rhs.segmentation_enabled ) &&
               ( reserved == rhs.reserved );
      }

      bool operator!=( DecodeAV1ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t disable_frame_end_update_cdf : 1;
      uint32_t segmentation_enabled         : 1;
      uint32_t reserved                     : 30;
    };

    struct DecodeAV1ReferenceInfo
    {
      using NativeType = StdVideoDecodeAV1ReferenceInfo;

      operator StdVideoDecodeAV1ReferenceInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoDecodeAV1ReferenceInfo *>( this );
      }

      operator StdVideoDecodeAV1ReferenceInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoDecodeAV1ReferenceInfo *>( this );
      }

      operator StdVideoDecodeAV1ReferenceInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoDecodeAV1ReferenceInfo *>( this );
      }

      operator StdVideoDecodeAV1ReferenceInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoDecodeAV1ReferenceInfo *>( this );
      }

      bool operator==( DecodeAV1ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( frame_type == rhs.frame_type ) && ( RefFrameSignBias == rhs.RefFrameSignBias ) && ( OrderHint == rhs.OrderHint ) &&
               ( SavedOrderHints == rhs.SavedOrderHints );
      }

      bool operator!=( DecodeAV1ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::DecodeAV1ReferenceInfoFlags flags            = {};
      uint8_t                                                                       frame_type       = {};
      uint8_t                                                                       RefFrameSignBias = {};
      uint8_t                                                                       OrderHint        = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_NUM_REF_FRAMES>   SavedOrderHints  = {};
    };

    //=== vulkan_video_codec_av1std_encode ===

    struct EncodeAV1DecoderModelInfo
    {
      using NativeType = StdVideoEncodeAV1DecoderModelInfo;

      operator StdVideoEncodeAV1DecoderModelInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeAV1DecoderModelInfo *>( this );
      }

      operator StdVideoEncodeAV1DecoderModelInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeAV1DecoderModelInfo *>( this );
      }

      operator StdVideoEncodeAV1DecoderModelInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeAV1DecoderModelInfo *>( this );
      }

      operator StdVideoEncodeAV1DecoderModelInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeAV1DecoderModelInfo *>( this );
      }

      bool operator==( EncodeAV1DecoderModelInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( buffer_delay_length_minus_1 == rhs.buffer_delay_length_minus_1 ) &&
               ( buffer_removal_time_length_minus_1 == rhs.buffer_removal_time_length_minus_1 ) &&
               ( frame_presentation_time_length_minus_1 == rhs.frame_presentation_time_length_minus_1 ) && ( reserved1 == rhs.reserved1 ) &&
               ( num_units_in_decoding_tick == rhs.num_units_in_decoding_tick );
      }

      bool operator!=( EncodeAV1DecoderModelInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint8_t  buffer_delay_length_minus_1            = {};
      uint8_t  buffer_removal_time_length_minus_1     = {};
      uint8_t  frame_presentation_time_length_minus_1 = {};
      uint8_t  reserved1                              = {};
      uint32_t num_units_in_decoding_tick             = {};
    };

    struct EncodeAV1ExtensionHeader
    {
      using NativeType = StdVideoEncodeAV1ExtensionHeader;

      operator StdVideoEncodeAV1ExtensionHeader const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeAV1ExtensionHeader *>( this );
      }

      operator StdVideoEncodeAV1ExtensionHeader &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeAV1ExtensionHeader *>( this );
      }

      operator StdVideoEncodeAV1ExtensionHeader const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeAV1ExtensionHeader *>( this );
      }

      operator StdVideoEncodeAV1ExtensionHeader *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeAV1ExtensionHeader *>( this );
      }

      bool operator==( EncodeAV1ExtensionHeader const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( temporal_id == rhs.temporal_id ) && ( spatial_id == rhs.spatial_id );
      }

      bool operator!=( EncodeAV1ExtensionHeader const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint8_t temporal_id = {};
      uint8_t spatial_id  = {};
    };

    struct EncodeAV1OperatingPointInfoFlags
    {
      using NativeType = StdVideoEncodeAV1OperatingPointInfoFlags;

      operator StdVideoEncodeAV1OperatingPointInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeAV1OperatingPointInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1OperatingPointInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeAV1OperatingPointInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1OperatingPointInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeAV1OperatingPointInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1OperatingPointInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeAV1OperatingPointInfoFlags *>( this );
      }

      bool operator==( EncodeAV1OperatingPointInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( decoder_model_present_for_this_op == rhs.decoder_model_present_for_this_op ) && ( low_delay_mode_flag == rhs.low_delay_mode_flag ) &&
               ( initial_display_delay_present_for_this_op == rhs.initial_display_delay_present_for_this_op ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeAV1OperatingPointInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t decoder_model_present_for_this_op         : 1;
      uint32_t low_delay_mode_flag                       : 1;
      uint32_t initial_display_delay_present_for_this_op : 1;
      uint32_t reserved                                  : 29;
    };

    struct EncodeAV1OperatingPointInfo
    {
      using NativeType = StdVideoEncodeAV1OperatingPointInfo;

      operator StdVideoEncodeAV1OperatingPointInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeAV1OperatingPointInfo *>( this );
      }

      operator StdVideoEncodeAV1OperatingPointInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeAV1OperatingPointInfo *>( this );
      }

      operator StdVideoEncodeAV1OperatingPointInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeAV1OperatingPointInfo *>( this );
      }

      operator StdVideoEncodeAV1OperatingPointInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeAV1OperatingPointInfo *>( this );
      }

      bool operator==( EncodeAV1OperatingPointInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( operating_point_idc == rhs.operating_point_idc ) && ( seq_level_idx == rhs.seq_level_idx ) &&
               ( seq_tier == rhs.seq_tier ) && ( decoder_buffer_delay == rhs.decoder_buffer_delay ) && ( encoder_buffer_delay == rhs.encoder_buffer_delay ) &&
               ( initial_display_delay_minus_1 == rhs.initial_display_delay_minus_1 );
      }

      bool operator!=( EncodeAV1OperatingPointInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeAV1OperatingPointInfoFlags flags                         = {};
      uint16_t                                                                           operating_point_idc           = {};
      uint8_t                                                                            seq_level_idx                 = {};
      uint8_t                                                                            seq_tier                      = {};
      uint32_t                                                                           decoder_buffer_delay          = {};
      uint32_t                                                                           encoder_buffer_delay          = {};
      uint8_t                                                                            initial_display_delay_minus_1 = {};
    };

    struct EncodeAV1PictureInfoFlags
    {
      using NativeType = StdVideoEncodeAV1PictureInfoFlags;

      operator StdVideoEncodeAV1PictureInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeAV1PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1PictureInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeAV1PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1PictureInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeAV1PictureInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1PictureInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeAV1PictureInfoFlags *>( this );
      }

      bool operator==( EncodeAV1PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( error_resilient_mode == rhs.error_resilient_mode ) && ( disable_cdf_update == rhs.disable_cdf_update ) &&
               ( use_superres == rhs.use_superres ) && ( render_and_frame_size_different == rhs.render_and_frame_size_different ) &&
               ( allow_screen_content_tools == rhs.allow_screen_content_tools ) && ( is_filter_switchable == rhs.is_filter_switchable ) &&
               ( force_integer_mv == rhs.force_integer_mv ) && ( frame_size_override_flag == rhs.frame_size_override_flag ) &&
               ( buffer_removal_time_present_flag == rhs.buffer_removal_time_present_flag ) && ( allow_intrabc == rhs.allow_intrabc ) &&
               ( frame_refs_short_signaling == rhs.frame_refs_short_signaling ) && ( allow_high_precision_mv == rhs.allow_high_precision_mv ) &&
               ( is_motion_mode_switchable == rhs.is_motion_mode_switchable ) && ( use_ref_frame_mvs == rhs.use_ref_frame_mvs ) &&
               ( disable_frame_end_update_cdf == rhs.disable_frame_end_update_cdf ) && ( allow_warped_motion == rhs.allow_warped_motion ) &&
               ( reduced_tx_set == rhs.reduced_tx_set ) && ( skip_mode_present == rhs.skip_mode_present ) && ( delta_q_present == rhs.delta_q_present ) &&
               ( delta_lf_present == rhs.delta_lf_present ) && ( delta_lf_multi == rhs.delta_lf_multi ) &&
               ( segmentation_enabled == rhs.segmentation_enabled ) && ( segmentation_update_map == rhs.segmentation_update_map ) &&
               ( segmentation_temporal_update == rhs.segmentation_temporal_update ) && ( segmentation_update_data == rhs.segmentation_update_data ) &&
               ( UsesLr == rhs.UsesLr ) && ( usesChromaLr == rhs.usesChromaLr ) && ( show_frame == rhs.show_frame ) &&
               ( showable_frame == rhs.showable_frame ) && ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeAV1PictureInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t error_resilient_mode             : 1;
      uint32_t disable_cdf_update               : 1;
      uint32_t use_superres                     : 1;
      uint32_t render_and_frame_size_different  : 1;
      uint32_t allow_screen_content_tools       : 1;
      uint32_t is_filter_switchable             : 1;
      uint32_t force_integer_mv                 : 1;
      uint32_t frame_size_override_flag         : 1;
      uint32_t buffer_removal_time_present_flag : 1;
      uint32_t allow_intrabc                    : 1;
      uint32_t frame_refs_short_signaling       : 1;
      uint32_t allow_high_precision_mv          : 1;
      uint32_t is_motion_mode_switchable        : 1;
      uint32_t use_ref_frame_mvs                : 1;
      uint32_t disable_frame_end_update_cdf     : 1;
      uint32_t allow_warped_motion              : 1;
      uint32_t reduced_tx_set                   : 1;
      uint32_t skip_mode_present                : 1;
      uint32_t delta_q_present                  : 1;
      uint32_t delta_lf_present                 : 1;
      uint32_t delta_lf_multi                   : 1;
      uint32_t segmentation_enabled             : 1;
      uint32_t segmentation_update_map          : 1;
      uint32_t segmentation_temporal_update     : 1;
      uint32_t segmentation_update_data         : 1;
      uint32_t UsesLr                           : 1;
      uint32_t usesChromaLr                     : 1;
      uint32_t show_frame                       : 1;
      uint32_t showable_frame                   : 1;
      uint32_t reserved                         : 3;
    };

    struct EncodeAV1PictureInfo
    {
      using NativeType = StdVideoEncodeAV1PictureInfo;

      operator StdVideoEncodeAV1PictureInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeAV1PictureInfo *>( this );
      }

      operator StdVideoEncodeAV1PictureInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeAV1PictureInfo *>( this );
      }

      operator StdVideoEncodeAV1PictureInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeAV1PictureInfo *>( this );
      }

      operator StdVideoEncodeAV1PictureInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeAV1PictureInfo *>( this );
      }

      bool operator==( EncodeAV1PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( frame_type == rhs.frame_type ) && ( frame_presentation_time == rhs.frame_presentation_time ) &&
               ( current_frame_id == rhs.current_frame_id ) && ( order_hint == rhs.order_hint ) && ( primary_ref_frame == rhs.primary_ref_frame ) &&
               ( refresh_frame_flags == rhs.refresh_frame_flags ) && ( coded_denom == rhs.coded_denom ) &&
               ( render_width_minus_1 == rhs.render_width_minus_1 ) && ( render_height_minus_1 == rhs.render_height_minus_1 ) &&
               ( interpolation_filter == rhs.interpolation_filter ) && ( TxMode == rhs.TxMode ) && ( delta_q_res == rhs.delta_q_res ) &&
               ( delta_lf_res == rhs.delta_lf_res ) && ( ref_order_hint == rhs.ref_order_hint ) && ( ref_frame_idx == rhs.ref_frame_idx ) &&
               ( reserved1 == rhs.reserved1 ) && ( delta_frame_id_minus_1 == rhs.delta_frame_id_minus_1 ) && ( pTileInfo == rhs.pTileInfo ) &&
               ( pQuantization == rhs.pQuantization ) && ( pSegmentation == rhs.pSegmentation ) && ( pLoopFilter == rhs.pLoopFilter ) &&
               ( pCDEF == rhs.pCDEF ) && ( pLoopRestoration == rhs.pLoopRestoration ) && ( pGlobalMotion == rhs.pGlobalMotion ) &&
               ( pExtensionHeader == rhs.pExtensionHeader ) && ( pBufferRemovalTimes == rhs.pBufferRemovalTimes );
      }

      bool operator!=( EncodeAV1PictureInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeAV1PictureInfoFlags flags = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1FrameType frame_type         = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1FrameType::eKey;
      uint32_t                                                       frame_presentation_time = {};
      uint32_t                                                       current_frame_id        = {};
      uint8_t                                                        order_hint              = {};
      uint8_t                                                        primary_ref_frame       = {};
      uint8_t                                                        refresh_frame_flags     = {};
      uint8_t                                                        coded_denom             = {};
      uint16_t                                                       render_width_minus_1    = {};
      uint16_t                                                       render_height_minus_1   = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1InterpolationFilter interpolation_filter =
        VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1InterpolationFilter::eEighttap;
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TxMode TxMode       = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TxMode::eOnly4X4;
      uint8_t                                                     delta_q_res  = {};
      uint8_t                                                     delta_lf_res = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, STD_VIDEO_AV1_NUM_REF_FRAMES>        ref_order_hint         = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<int8_t, STD_VIDEO_AV1_REFS_PER_FRAME>         ref_frame_idx          = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, 3>                                   reserved1              = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint32_t, STD_VIDEO_AV1_REFS_PER_FRAME>       delta_frame_id_minus_1 = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1TileInfo *              pTileInfo              = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1Quantization *          pQuantization          = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1Segmentation *          pSegmentation          = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1LoopFilter *            pLoopFilter            = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1CDEF *                  pCDEF                  = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1LoopRestoration *       pLoopRestoration       = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1GlobalMotion *          pGlobalMotion          = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeAV1ExtensionHeader * pExtensionHeader       = {};
      const uint32_t *                                                                   pBufferRemovalTimes    = {};
    };

    struct EncodeAV1ReferenceInfoFlags
    {
      using NativeType = StdVideoEncodeAV1ReferenceInfoFlags;

      operator StdVideoEncodeAV1ReferenceInfoFlags const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeAV1ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1ReferenceInfoFlags &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeAV1ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1ReferenceInfoFlags const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeAV1ReferenceInfoFlags *>( this );
      }

      operator StdVideoEncodeAV1ReferenceInfoFlags *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeAV1ReferenceInfoFlags *>( this );
      }

      bool operator==( EncodeAV1ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( disable_frame_end_update_cdf == rhs.disable_frame_end_update_cdf ) && ( segmentation_enabled == rhs.segmentation_enabled ) &&
               ( reserved == rhs.reserved );
      }

      bool operator!=( EncodeAV1ReferenceInfoFlags const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      uint32_t disable_frame_end_update_cdf : 1;
      uint32_t segmentation_enabled         : 1;
      uint32_t reserved                     : 30;
    };

    struct EncodeAV1ReferenceInfo
    {
      using NativeType = StdVideoEncodeAV1ReferenceInfo;

      operator StdVideoEncodeAV1ReferenceInfo const &() const VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<const StdVideoEncodeAV1ReferenceInfo *>( this );
      }

      operator StdVideoEncodeAV1ReferenceInfo &() VULKAN_HPP_NOEXCEPT
      {
        return *reinterpret_cast<StdVideoEncodeAV1ReferenceInfo *>( this );
      }

      operator StdVideoEncodeAV1ReferenceInfo const *() const VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<const StdVideoEncodeAV1ReferenceInfo *>( this );
      }

      operator StdVideoEncodeAV1ReferenceInfo *() VULKAN_HPP_NOEXCEPT
      {
        return reinterpret_cast<StdVideoEncodeAV1ReferenceInfo *>( this );
      }

      bool operator==( EncodeAV1ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return ( flags == rhs.flags ) && ( RefFrameId == rhs.RefFrameId ) && ( frame_type == rhs.frame_type ) && ( OrderHint == rhs.OrderHint ) &&
               ( reserved1 == rhs.reserved1 ) && ( pExtensionHeader == rhs.pExtensionHeader );
      }

      bool operator!=( EncodeAV1ReferenceInfo const & rhs ) const VULKAN_HPP_NOEXCEPT
      {
        return !operator==( rhs );
      }

    public:
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeAV1ReferenceInfoFlags flags      = {};
      uint32_t                                                                      RefFrameId = {};
      VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1FrameType frame_type = VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::AV1FrameType::eKey;
      uint8_t                                                        OrderHint  = {};
      VULKAN_HPP_NAMESPACE::ArrayWrapper1D<uint8_t, 3>               reserved1  = {};
      const VULKAN_HPP_NAMESPACE::VULKAN_HPP_VIDEO_NAMESPACE::EncodeAV1ExtensionHeader * pExtensionHeader = {};
    };

  }  // namespace VULKAN_HPP_VIDEO_NAMESPACE
}  // namespace VULKAN_HPP_NAMESPACE
#endif
