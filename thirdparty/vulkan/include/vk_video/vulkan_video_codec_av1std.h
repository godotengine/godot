#ifndef VULKAN_VIDEO_CODEC_AV1STD_H_
#define VULKAN_VIDEO_CODEC_AV1STD_H_ 1

/*
** Copyright 2015-2025 The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

/*
** This header is generated from the Khronos Vulkan XML API Registry.
**
*/


#ifdef __cplusplus
extern "C" {
#endif



// vulkan_video_codec_av1std is a preprocessor guard. Do not pass it to API calls.
#define vulkan_video_codec_av1std 1
#include "vulkan_video_codecs_common.h"
#define STD_VIDEO_AV1_NUM_REF_FRAMES      8U
#define STD_VIDEO_AV1_REFS_PER_FRAME      7U
#define STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME 8U
#define STD_VIDEO_AV1_MAX_TILE_COLS       64U
#define STD_VIDEO_AV1_MAX_TILE_ROWS       64U
#define STD_VIDEO_AV1_MAX_SEGMENTS        8U
#define STD_VIDEO_AV1_SEG_LVL_MAX         8U
#define STD_VIDEO_AV1_PRIMARY_REF_NONE    7U
#define STD_VIDEO_AV1_SELECT_INTEGER_MV   2U
#define STD_VIDEO_AV1_SELECT_SCREEN_CONTENT_TOOLS 2U
#define STD_VIDEO_AV1_SKIP_MODE_FRAMES    2U
#define STD_VIDEO_AV1_MAX_LOOP_FILTER_STRENGTHS 4U
#define STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS 2U
#define STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS 8U
#define STD_VIDEO_AV1_MAX_NUM_PLANES      3U
#define STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS 6U
#define STD_VIDEO_AV1_MAX_NUM_Y_POINTS    14U
#define STD_VIDEO_AV1_MAX_NUM_CB_POINTS   10U
#define STD_VIDEO_AV1_MAX_NUM_CR_POINTS   10U
#define STD_VIDEO_AV1_MAX_NUM_POS_LUMA    24U
#define STD_VIDEO_AV1_MAX_NUM_POS_CHROMA  25U

typedef enum StdVideoAV1Profile {
    STD_VIDEO_AV1_PROFILE_MAIN = 0,
    STD_VIDEO_AV1_PROFILE_HIGH = 1,
    STD_VIDEO_AV1_PROFILE_PROFESSIONAL = 2,
    STD_VIDEO_AV1_PROFILE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_PROFILE_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1Profile;

typedef enum StdVideoAV1Level {
    STD_VIDEO_AV1_LEVEL_2_0 = 0,
    STD_VIDEO_AV1_LEVEL_2_1 = 1,
    STD_VIDEO_AV1_LEVEL_2_2 = 2,
    STD_VIDEO_AV1_LEVEL_2_3 = 3,
    STD_VIDEO_AV1_LEVEL_3_0 = 4,
    STD_VIDEO_AV1_LEVEL_3_1 = 5,
    STD_VIDEO_AV1_LEVEL_3_2 = 6,
    STD_VIDEO_AV1_LEVEL_3_3 = 7,
    STD_VIDEO_AV1_LEVEL_4_0 = 8,
    STD_VIDEO_AV1_LEVEL_4_1 = 9,
    STD_VIDEO_AV1_LEVEL_4_2 = 10,
    STD_VIDEO_AV1_LEVEL_4_3 = 11,
    STD_VIDEO_AV1_LEVEL_5_0 = 12,
    STD_VIDEO_AV1_LEVEL_5_1 = 13,
    STD_VIDEO_AV1_LEVEL_5_2 = 14,
    STD_VIDEO_AV1_LEVEL_5_3 = 15,
    STD_VIDEO_AV1_LEVEL_6_0 = 16,
    STD_VIDEO_AV1_LEVEL_6_1 = 17,
    STD_VIDEO_AV1_LEVEL_6_2 = 18,
    STD_VIDEO_AV1_LEVEL_6_3 = 19,
    STD_VIDEO_AV1_LEVEL_7_0 = 20,
    STD_VIDEO_AV1_LEVEL_7_1 = 21,
    STD_VIDEO_AV1_LEVEL_7_2 = 22,
    STD_VIDEO_AV1_LEVEL_7_3 = 23,
    STD_VIDEO_AV1_LEVEL_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_LEVEL_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1Level;

typedef enum StdVideoAV1FrameType {
    STD_VIDEO_AV1_FRAME_TYPE_KEY = 0,
    STD_VIDEO_AV1_FRAME_TYPE_INTER = 1,
    STD_VIDEO_AV1_FRAME_TYPE_INTRA_ONLY = 2,
    STD_VIDEO_AV1_FRAME_TYPE_SWITCH = 3,
    STD_VIDEO_AV1_FRAME_TYPE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_FRAME_TYPE_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1FrameType;

typedef enum StdVideoAV1ReferenceName {
    STD_VIDEO_AV1_REFERENCE_NAME_INTRA_FRAME = 0,
    STD_VIDEO_AV1_REFERENCE_NAME_LAST_FRAME = 1,
    STD_VIDEO_AV1_REFERENCE_NAME_LAST2_FRAME = 2,
    STD_VIDEO_AV1_REFERENCE_NAME_LAST3_FRAME = 3,
    STD_VIDEO_AV1_REFERENCE_NAME_GOLDEN_FRAME = 4,
    STD_VIDEO_AV1_REFERENCE_NAME_BWDREF_FRAME = 5,
    STD_VIDEO_AV1_REFERENCE_NAME_ALTREF2_FRAME = 6,
    STD_VIDEO_AV1_REFERENCE_NAME_ALTREF_FRAME = 7,
    STD_VIDEO_AV1_REFERENCE_NAME_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_REFERENCE_NAME_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1ReferenceName;

typedef enum StdVideoAV1InterpolationFilter {
    STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP = 0,
    STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP_SMOOTH = 1,
    STD_VIDEO_AV1_INTERPOLATION_FILTER_EIGHTTAP_SHARP = 2,
    STD_VIDEO_AV1_INTERPOLATION_FILTER_BILINEAR = 3,
    STD_VIDEO_AV1_INTERPOLATION_FILTER_SWITCHABLE = 4,
    STD_VIDEO_AV1_INTERPOLATION_FILTER_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_INTERPOLATION_FILTER_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1InterpolationFilter;

typedef enum StdVideoAV1TxMode {
    STD_VIDEO_AV1_TX_MODE_ONLY_4X4 = 0,
    STD_VIDEO_AV1_TX_MODE_LARGEST = 1,
    STD_VIDEO_AV1_TX_MODE_SELECT = 2,
    STD_VIDEO_AV1_TX_MODE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_TX_MODE_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1TxMode;

typedef enum StdVideoAV1FrameRestorationType {
    STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE = 0,
    STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_WIENER = 1,
    STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SGRPROJ = 2,
    STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE = 3,
    STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1FrameRestorationType;

typedef enum StdVideoAV1ColorPrimaries {
    STD_VIDEO_AV1_COLOR_PRIMARIES_BT_709 = 1,
    STD_VIDEO_AV1_COLOR_PRIMARIES_UNSPECIFIED = 2,
    STD_VIDEO_AV1_COLOR_PRIMARIES_BT_470_M = 4,
    STD_VIDEO_AV1_COLOR_PRIMARIES_BT_470_B_G = 5,
    STD_VIDEO_AV1_COLOR_PRIMARIES_BT_601 = 6,
    STD_VIDEO_AV1_COLOR_PRIMARIES_SMPTE_240 = 7,
    STD_VIDEO_AV1_COLOR_PRIMARIES_GENERIC_FILM = 8,
    STD_VIDEO_AV1_COLOR_PRIMARIES_BT_2020 = 9,
    STD_VIDEO_AV1_COLOR_PRIMARIES_XYZ = 10,
    STD_VIDEO_AV1_COLOR_PRIMARIES_SMPTE_431 = 11,
    STD_VIDEO_AV1_COLOR_PRIMARIES_SMPTE_432 = 12,
    STD_VIDEO_AV1_COLOR_PRIMARIES_EBU_3213 = 22,
    STD_VIDEO_AV1_COLOR_PRIMARIES_INVALID = 0x7FFFFFFF,
  // STD_VIDEO_AV1_COLOR_PRIMARIES_BT_UNSPECIFIED is a legacy alias
    STD_VIDEO_AV1_COLOR_PRIMARIES_BT_UNSPECIFIED = STD_VIDEO_AV1_COLOR_PRIMARIES_UNSPECIFIED,
    STD_VIDEO_AV1_COLOR_PRIMARIES_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1ColorPrimaries;

typedef enum StdVideoAV1TransferCharacteristics {
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_RESERVED_0 = 0,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_709 = 1,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_UNSPECIFIED = 2,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_RESERVED_3 = 3,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_470_M = 4,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_470_B_G = 5,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_601 = 6,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SMPTE_240 = 7,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_LINEAR = 8,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_LOG_100 = 9,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_LOG_100_SQRT10 = 10,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_IEC_61966 = 11,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_1361 = 12,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SRGB = 13,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_2020_10_BIT = 14,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_BT_2020_12_BIT = 15,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SMPTE_2084 = 16,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_SMPTE_428 = 17,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_HLG = 18,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_TRANSFER_CHARACTERISTICS_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1TransferCharacteristics;

typedef enum StdVideoAV1MatrixCoefficients {
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_IDENTITY = 0,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_709 = 1,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_UNSPECIFIED = 2,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_RESERVED_3 = 3,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_FCC = 4,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_470_B_G = 5,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_601 = 6,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_SMPTE_240 = 7,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_SMPTE_YCGCO = 8,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_2020_NCL = 9,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_BT_2020_CL = 10,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_SMPTE_2085 = 11,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_CHROMAT_NCL = 12,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_CHROMAT_CL = 13,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_ICTCP = 14,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_MATRIX_COEFFICIENTS_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1MatrixCoefficients;

typedef enum StdVideoAV1ChromaSamplePosition {
    STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_UNKNOWN = 0,
    STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_VERTICAL = 1,
    STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_COLOCATED = 2,
    STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_RESERVED = 3,
    STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_INVALID = 0x7FFFFFFF,
    STD_VIDEO_AV1_CHROMA_SAMPLE_POSITION_MAX_ENUM = 0x7FFFFFFF
} StdVideoAV1ChromaSamplePosition;
typedef struct StdVideoAV1ColorConfigFlags {
    uint32_t    mono_chrome : 1;
    uint32_t    color_range : 1;
    uint32_t    separate_uv_delta_q : 1;
    uint32_t    color_description_present_flag : 1;
    uint32_t    reserved : 28;
} StdVideoAV1ColorConfigFlags;

typedef struct StdVideoAV1ColorConfig {
    StdVideoAV1ColorConfigFlags           flags;
    uint8_t                               BitDepth;
    uint8_t                               subsampling_x;
    uint8_t                               subsampling_y;
    uint8_t                               reserved1;
    StdVideoAV1ColorPrimaries             color_primaries;
    StdVideoAV1TransferCharacteristics    transfer_characteristics;
    StdVideoAV1MatrixCoefficients         matrix_coefficients;
    StdVideoAV1ChromaSamplePosition       chroma_sample_position;
} StdVideoAV1ColorConfig;

typedef struct StdVideoAV1TimingInfoFlags {
    uint32_t    equal_picture_interval : 1;
    uint32_t    reserved : 31;
} StdVideoAV1TimingInfoFlags;

typedef struct StdVideoAV1TimingInfo {
    StdVideoAV1TimingInfoFlags    flags;
    uint32_t                      num_units_in_display_tick;
    uint32_t                      time_scale;
    uint32_t                      num_ticks_per_picture_minus_1;
} StdVideoAV1TimingInfo;

typedef struct StdVideoAV1LoopFilterFlags {
    uint32_t    loop_filter_delta_enabled : 1;
    uint32_t    loop_filter_delta_update : 1;
    uint32_t    reserved : 30;
} StdVideoAV1LoopFilterFlags;

typedef struct StdVideoAV1LoopFilter {
    StdVideoAV1LoopFilterFlags    flags;
    uint8_t                       loop_filter_level[STD_VIDEO_AV1_MAX_LOOP_FILTER_STRENGTHS];
    uint8_t                       loop_filter_sharpness;
    uint8_t                       update_ref_delta;
    int8_t                        loop_filter_ref_deltas[STD_VIDEO_AV1_TOTAL_REFS_PER_FRAME];
    uint8_t                       update_mode_delta;
    int8_t                        loop_filter_mode_deltas[STD_VIDEO_AV1_LOOP_FILTER_ADJUSTMENTS];
} StdVideoAV1LoopFilter;

typedef struct StdVideoAV1QuantizationFlags {
    uint32_t    using_qmatrix : 1;
    uint32_t    diff_uv_delta : 1;
    uint32_t    reserved : 30;
} StdVideoAV1QuantizationFlags;

typedef struct StdVideoAV1Quantization {
    StdVideoAV1QuantizationFlags    flags;
    uint8_t                         base_q_idx;
    int8_t                          DeltaQYDc;
    int8_t                          DeltaQUDc;
    int8_t                          DeltaQUAc;
    int8_t                          DeltaQVDc;
    int8_t                          DeltaQVAc;
    uint8_t                         qm_y;
    uint8_t                         qm_u;
    uint8_t                         qm_v;
} StdVideoAV1Quantization;

typedef struct StdVideoAV1Segmentation {
    uint8_t    FeatureEnabled[STD_VIDEO_AV1_MAX_SEGMENTS];
    int16_t    FeatureData[STD_VIDEO_AV1_MAX_SEGMENTS][STD_VIDEO_AV1_SEG_LVL_MAX];
} StdVideoAV1Segmentation;

typedef struct StdVideoAV1TileInfoFlags {
    uint32_t    uniform_tile_spacing_flag : 1;
    uint32_t    reserved : 31;
} StdVideoAV1TileInfoFlags;

typedef struct StdVideoAV1TileInfo {
    StdVideoAV1TileInfoFlags    flags;
    uint8_t                     TileCols;
    uint8_t                     TileRows;
    uint16_t                    context_update_tile_id;
    uint8_t                     tile_size_bytes_minus_1;
    uint8_t                     reserved1[7];
    const uint16_t*             pMiColStarts;
    const uint16_t*             pMiRowStarts;
    const uint16_t*             pWidthInSbsMinus1;
    const uint16_t*             pHeightInSbsMinus1;
} StdVideoAV1TileInfo;

typedef struct StdVideoAV1CDEF {
    uint8_t    cdef_damping_minus_3;
    uint8_t    cdef_bits;
    uint8_t    cdef_y_pri_strength[STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS];
    uint8_t    cdef_y_sec_strength[STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS];
    uint8_t    cdef_uv_pri_strength[STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS];
    uint8_t    cdef_uv_sec_strength[STD_VIDEO_AV1_MAX_CDEF_FILTER_STRENGTHS];
} StdVideoAV1CDEF;

typedef struct StdVideoAV1LoopRestoration {
    StdVideoAV1FrameRestorationType    FrameRestorationType[STD_VIDEO_AV1_MAX_NUM_PLANES];
    uint16_t                           LoopRestorationSize[STD_VIDEO_AV1_MAX_NUM_PLANES];
} StdVideoAV1LoopRestoration;

typedef struct StdVideoAV1GlobalMotion {
    uint8_t    GmType[STD_VIDEO_AV1_NUM_REF_FRAMES];
    int32_t    gm_params[STD_VIDEO_AV1_NUM_REF_FRAMES][STD_VIDEO_AV1_GLOBAL_MOTION_PARAMS];
} StdVideoAV1GlobalMotion;

typedef struct StdVideoAV1FilmGrainFlags {
    uint32_t    chroma_scaling_from_luma : 1;
    uint32_t    overlap_flag : 1;
    uint32_t    clip_to_restricted_range : 1;
    uint32_t    update_grain : 1;
    uint32_t    reserved : 28;
} StdVideoAV1FilmGrainFlags;

typedef struct StdVideoAV1FilmGrain {
    StdVideoAV1FilmGrainFlags    flags;
    uint8_t                      grain_scaling_minus_8;
    uint8_t                      ar_coeff_lag;
    uint8_t                      ar_coeff_shift_minus_6;
    uint8_t                      grain_scale_shift;
    uint16_t                     grain_seed;
    uint8_t                      film_grain_params_ref_idx;
    uint8_t                      num_y_points;
    uint8_t                      point_y_value[STD_VIDEO_AV1_MAX_NUM_Y_POINTS];
    uint8_t                      point_y_scaling[STD_VIDEO_AV1_MAX_NUM_Y_POINTS];
    uint8_t                      num_cb_points;
    uint8_t                      point_cb_value[STD_VIDEO_AV1_MAX_NUM_CB_POINTS];
    uint8_t                      point_cb_scaling[STD_VIDEO_AV1_MAX_NUM_CB_POINTS];
    uint8_t                      num_cr_points;
    uint8_t                      point_cr_value[STD_VIDEO_AV1_MAX_NUM_CR_POINTS];
    uint8_t                      point_cr_scaling[STD_VIDEO_AV1_MAX_NUM_CR_POINTS];
    int8_t                       ar_coeffs_y_plus_128[STD_VIDEO_AV1_MAX_NUM_POS_LUMA];
    int8_t                       ar_coeffs_cb_plus_128[STD_VIDEO_AV1_MAX_NUM_POS_CHROMA];
    int8_t                       ar_coeffs_cr_plus_128[STD_VIDEO_AV1_MAX_NUM_POS_CHROMA];
    uint8_t                      cb_mult;
    uint8_t                      cb_luma_mult;
    uint16_t                     cb_offset;
    uint8_t                      cr_mult;
    uint8_t                      cr_luma_mult;
    uint16_t                     cr_offset;
} StdVideoAV1FilmGrain;

typedef struct StdVideoAV1SequenceHeaderFlags {
    uint32_t    still_picture : 1;
    uint32_t    reduced_still_picture_header : 1;
    uint32_t    use_128x128_superblock : 1;
    uint32_t    enable_filter_intra : 1;
    uint32_t    enable_intra_edge_filter : 1;
    uint32_t    enable_interintra_compound : 1;
    uint32_t    enable_masked_compound : 1;
    uint32_t    enable_warped_motion : 1;
    uint32_t    enable_dual_filter : 1;
    uint32_t    enable_order_hint : 1;
    uint32_t    enable_jnt_comp : 1;
    uint32_t    enable_ref_frame_mvs : 1;
    uint32_t    frame_id_numbers_present_flag : 1;
    uint32_t    enable_superres : 1;
    uint32_t    enable_cdef : 1;
    uint32_t    enable_restoration : 1;
    uint32_t    film_grain_params_present : 1;
    uint32_t    timing_info_present_flag : 1;
    uint32_t    initial_display_delay_present_flag : 1;
    uint32_t    reserved : 13;
} StdVideoAV1SequenceHeaderFlags;

typedef struct StdVideoAV1SequenceHeader {
    StdVideoAV1SequenceHeaderFlags    flags;
    StdVideoAV1Profile                seq_profile;
    uint8_t                           frame_width_bits_minus_1;
    uint8_t                           frame_height_bits_minus_1;
    uint16_t                          max_frame_width_minus_1;
    uint16_t                          max_frame_height_minus_1;
    uint8_t                           delta_frame_id_length_minus_2;
    uint8_t                           additional_frame_id_length_minus_1;
    uint8_t                           order_hint_bits_minus_1;
    uint8_t                           seq_force_integer_mv;
    uint8_t                           seq_force_screen_content_tools;
    uint8_t                           reserved1[5];
    const StdVideoAV1ColorConfig*     pColorConfig;
    const StdVideoAV1TimingInfo*      pTimingInfo;
} StdVideoAV1SequenceHeader;


#ifdef __cplusplus
}
#endif

#endif
