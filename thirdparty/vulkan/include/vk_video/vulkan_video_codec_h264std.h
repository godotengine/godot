#ifndef VULKAN_VIDEO_CODEC_H264STD_H_
#define VULKAN_VIDEO_CODEC_H264STD_H_ 1

/*
** Copyright 2015-2024 The Khronos Group Inc.
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



// vulkan_video_codec_h264std is a preprocessor guard. Do not pass it to API calls.
#define vulkan_video_codec_h264std 1
#include "vulkan_video_codecs_common.h"
#define STD_VIDEO_H264_CPB_CNT_LIST_SIZE  32
#define STD_VIDEO_H264_SCALING_LIST_4X4_NUM_LISTS 6
#define STD_VIDEO_H264_SCALING_LIST_4X4_NUM_ELEMENTS 16
#define STD_VIDEO_H264_SCALING_LIST_8X8_NUM_LISTS 6
#define STD_VIDEO_H264_SCALING_LIST_8X8_NUM_ELEMENTS 64
#define STD_VIDEO_H264_MAX_NUM_LIST_REF   32
#define STD_VIDEO_H264_MAX_CHROMA_PLANES  2
#define STD_VIDEO_H264_NO_REFERENCE_PICTURE 0xFF

typedef enum StdVideoH264ChromaFormatIdc {
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_MONOCHROME = 0,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_420 = 1,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_422 = 2,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_444 = 3,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_CHROMA_FORMAT_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264ChromaFormatIdc;

typedef enum StdVideoH264ProfileIdc {
    STD_VIDEO_H264_PROFILE_IDC_BASELINE = 66,
    STD_VIDEO_H264_PROFILE_IDC_MAIN = 77,
    STD_VIDEO_H264_PROFILE_IDC_HIGH = 100,
    STD_VIDEO_H264_PROFILE_IDC_HIGH_444_PREDICTIVE = 244,
    STD_VIDEO_H264_PROFILE_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_PROFILE_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264ProfileIdc;

typedef enum StdVideoH264LevelIdc {
    STD_VIDEO_H264_LEVEL_IDC_1_0 = 0,
    STD_VIDEO_H264_LEVEL_IDC_1_1 = 1,
    STD_VIDEO_H264_LEVEL_IDC_1_2 = 2,
    STD_VIDEO_H264_LEVEL_IDC_1_3 = 3,
    STD_VIDEO_H264_LEVEL_IDC_2_0 = 4,
    STD_VIDEO_H264_LEVEL_IDC_2_1 = 5,
    STD_VIDEO_H264_LEVEL_IDC_2_2 = 6,
    STD_VIDEO_H264_LEVEL_IDC_3_0 = 7,
    STD_VIDEO_H264_LEVEL_IDC_3_1 = 8,
    STD_VIDEO_H264_LEVEL_IDC_3_2 = 9,
    STD_VIDEO_H264_LEVEL_IDC_4_0 = 10,
    STD_VIDEO_H264_LEVEL_IDC_4_1 = 11,
    STD_VIDEO_H264_LEVEL_IDC_4_2 = 12,
    STD_VIDEO_H264_LEVEL_IDC_5_0 = 13,
    STD_VIDEO_H264_LEVEL_IDC_5_1 = 14,
    STD_VIDEO_H264_LEVEL_IDC_5_2 = 15,
    STD_VIDEO_H264_LEVEL_IDC_6_0 = 16,
    STD_VIDEO_H264_LEVEL_IDC_6_1 = 17,
    STD_VIDEO_H264_LEVEL_IDC_6_2 = 18,
    STD_VIDEO_H264_LEVEL_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_LEVEL_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264LevelIdc;

typedef enum StdVideoH264PocType {
    STD_VIDEO_H264_POC_TYPE_0 = 0,
    STD_VIDEO_H264_POC_TYPE_1 = 1,
    STD_VIDEO_H264_POC_TYPE_2 = 2,
    STD_VIDEO_H264_POC_TYPE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_POC_TYPE_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264PocType;

typedef enum StdVideoH264AspectRatioIdc {
    STD_VIDEO_H264_ASPECT_RATIO_IDC_UNSPECIFIED = 0,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_SQUARE = 1,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_12_11 = 2,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_10_11 = 3,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_16_11 = 4,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_40_33 = 5,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_24_11 = 6,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_20_11 = 7,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_32_11 = 8,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_80_33 = 9,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_18_11 = 10,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_15_11 = 11,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_64_33 = 12,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_160_99 = 13,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_4_3 = 14,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_3_2 = 15,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_2_1 = 16,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_EXTENDED_SAR = 255,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_ASPECT_RATIO_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264AspectRatioIdc;

typedef enum StdVideoH264WeightedBipredIdc {
    STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_DEFAULT = 0,
    STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_EXPLICIT = 1,
    STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_IMPLICIT = 2,
    STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_WEIGHTED_BIPRED_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264WeightedBipredIdc;

typedef enum StdVideoH264ModificationOfPicNumsIdc {
    STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_SHORT_TERM_SUBTRACT = 0,
    STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_SHORT_TERM_ADD = 1,
    STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_LONG_TERM = 2,
    STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_END = 3,
    STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_MODIFICATION_OF_PIC_NUMS_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264ModificationOfPicNumsIdc;

typedef enum StdVideoH264MemMgmtControlOp {
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_END = 0,
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_UNMARK_SHORT_TERM = 1,
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_UNMARK_LONG_TERM = 2,
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_MARK_LONG_TERM = 3,
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_SET_MAX_LONG_TERM_INDEX = 4,
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_UNMARK_ALL = 5,
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_MARK_CURRENT_AS_LONG_TERM = 6,
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_MEM_MGMT_CONTROL_OP_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264MemMgmtControlOp;

typedef enum StdVideoH264CabacInitIdc {
    STD_VIDEO_H264_CABAC_INIT_IDC_0 = 0,
    STD_VIDEO_H264_CABAC_INIT_IDC_1 = 1,
    STD_VIDEO_H264_CABAC_INIT_IDC_2 = 2,
    STD_VIDEO_H264_CABAC_INIT_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_CABAC_INIT_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264CabacInitIdc;

typedef enum StdVideoH264DisableDeblockingFilterIdc {
    STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_DISABLED = 0,
    STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_ENABLED = 1,
    STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_PARTIAL = 2,
    STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_DISABLE_DEBLOCKING_FILTER_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264DisableDeblockingFilterIdc;

typedef enum StdVideoH264SliceType {
    STD_VIDEO_H264_SLICE_TYPE_P = 0,
    STD_VIDEO_H264_SLICE_TYPE_B = 1,
    STD_VIDEO_H264_SLICE_TYPE_I = 2,
    STD_VIDEO_H264_SLICE_TYPE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_SLICE_TYPE_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264SliceType;

typedef enum StdVideoH264PictureType {
    STD_VIDEO_H264_PICTURE_TYPE_P = 0,
    STD_VIDEO_H264_PICTURE_TYPE_B = 1,
    STD_VIDEO_H264_PICTURE_TYPE_I = 2,
    STD_VIDEO_H264_PICTURE_TYPE_IDR = 5,
    STD_VIDEO_H264_PICTURE_TYPE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_PICTURE_TYPE_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264PictureType;

typedef enum StdVideoH264NonVclNaluType {
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_SPS = 0,
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_PPS = 1,
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_AUD = 2,
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_PREFIX = 3,
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_END_OF_SEQUENCE = 4,
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_END_OF_STREAM = 5,
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_PRECODED = 6,
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H264_NON_VCL_NALU_TYPE_MAX_ENUM = 0x7FFFFFFF
} StdVideoH264NonVclNaluType;
typedef struct StdVideoH264SpsVuiFlags {
    uint32_t    aspect_ratio_info_present_flag : 1;
    uint32_t    overscan_info_present_flag : 1;
    uint32_t    overscan_appropriate_flag : 1;
    uint32_t    video_signal_type_present_flag : 1;
    uint32_t    video_full_range_flag : 1;
    uint32_t    color_description_present_flag : 1;
    uint32_t    chroma_loc_info_present_flag : 1;
    uint32_t    timing_info_present_flag : 1;
    uint32_t    fixed_frame_rate_flag : 1;
    uint32_t    bitstream_restriction_flag : 1;
    uint32_t    nal_hrd_parameters_present_flag : 1;
    uint32_t    vcl_hrd_parameters_present_flag : 1;
} StdVideoH264SpsVuiFlags;

typedef struct StdVideoH264HrdParameters {
    uint8_t     cpb_cnt_minus1;
    uint8_t     bit_rate_scale;
    uint8_t     cpb_size_scale;
    uint8_t     reserved1;
    uint32_t    bit_rate_value_minus1[STD_VIDEO_H264_CPB_CNT_LIST_SIZE];
    uint32_t    cpb_size_value_minus1[STD_VIDEO_H264_CPB_CNT_LIST_SIZE];
    uint8_t     cbr_flag[STD_VIDEO_H264_CPB_CNT_LIST_SIZE];
    uint32_t    initial_cpb_removal_delay_length_minus1;
    uint32_t    cpb_removal_delay_length_minus1;
    uint32_t    dpb_output_delay_length_minus1;
    uint32_t    time_offset_length;
} StdVideoH264HrdParameters;

typedef struct StdVideoH264SequenceParameterSetVui {
    StdVideoH264SpsVuiFlags             flags;
    StdVideoH264AspectRatioIdc          aspect_ratio_idc;
    uint16_t                            sar_width;
    uint16_t                            sar_height;
    uint8_t                             video_format;
    uint8_t                             colour_primaries;
    uint8_t                             transfer_characteristics;
    uint8_t                             matrix_coefficients;
    uint32_t                            num_units_in_tick;
    uint32_t                            time_scale;
    uint8_t                             max_num_reorder_frames;
    uint8_t                             max_dec_frame_buffering;
    uint8_t                             chroma_sample_loc_type_top_field;
    uint8_t                             chroma_sample_loc_type_bottom_field;
    uint32_t                            reserved1;
    const StdVideoH264HrdParameters*    pHrdParameters;
} StdVideoH264SequenceParameterSetVui;

typedef struct StdVideoH264SpsFlags {
    uint32_t    constraint_set0_flag : 1;
    uint32_t    constraint_set1_flag : 1;
    uint32_t    constraint_set2_flag : 1;
    uint32_t    constraint_set3_flag : 1;
    uint32_t    constraint_set4_flag : 1;
    uint32_t    constraint_set5_flag : 1;
    uint32_t    direct_8x8_inference_flag : 1;
    uint32_t    mb_adaptive_frame_field_flag : 1;
    uint32_t    frame_mbs_only_flag : 1;
    uint32_t    delta_pic_order_always_zero_flag : 1;
    uint32_t    separate_colour_plane_flag : 1;
    uint32_t    gaps_in_frame_num_value_allowed_flag : 1;
    uint32_t    qpprime_y_zero_transform_bypass_flag : 1;
    uint32_t    frame_cropping_flag : 1;
    uint32_t    seq_scaling_matrix_present_flag : 1;
    uint32_t    vui_parameters_present_flag : 1;
} StdVideoH264SpsFlags;

typedef struct StdVideoH264ScalingLists {
    uint16_t    scaling_list_present_mask;
    uint16_t    use_default_scaling_matrix_mask;
    uint8_t     ScalingList4x4[STD_VIDEO_H264_SCALING_LIST_4X4_NUM_LISTS][STD_VIDEO_H264_SCALING_LIST_4X4_NUM_ELEMENTS];
    uint8_t     ScalingList8x8[STD_VIDEO_H264_SCALING_LIST_8X8_NUM_LISTS][STD_VIDEO_H264_SCALING_LIST_8X8_NUM_ELEMENTS];
} StdVideoH264ScalingLists;

typedef struct StdVideoH264SequenceParameterSet {
    StdVideoH264SpsFlags                          flags;
    StdVideoH264ProfileIdc                        profile_idc;
    StdVideoH264LevelIdc                          level_idc;
    StdVideoH264ChromaFormatIdc                   chroma_format_idc;
    uint8_t                                       seq_parameter_set_id;
    uint8_t                                       bit_depth_luma_minus8;
    uint8_t                                       bit_depth_chroma_minus8;
    uint8_t                                       log2_max_frame_num_minus4;
    StdVideoH264PocType                           pic_order_cnt_type;
    int32_t                                       offset_for_non_ref_pic;
    int32_t                                       offset_for_top_to_bottom_field;
    uint8_t                                       log2_max_pic_order_cnt_lsb_minus4;
    uint8_t                                       num_ref_frames_in_pic_order_cnt_cycle;
    uint8_t                                       max_num_ref_frames;
    uint8_t                                       reserved1;
    uint32_t                                      pic_width_in_mbs_minus1;
    uint32_t                                      pic_height_in_map_units_minus1;
    uint32_t                                      frame_crop_left_offset;
    uint32_t                                      frame_crop_right_offset;
    uint32_t                                      frame_crop_top_offset;
    uint32_t                                      frame_crop_bottom_offset;
    uint32_t                                      reserved2;
    const int32_t*                                pOffsetForRefFrame;
    const StdVideoH264ScalingLists*               pScalingLists;
    const StdVideoH264SequenceParameterSetVui*    pSequenceParameterSetVui;
} StdVideoH264SequenceParameterSet;

typedef struct StdVideoH264PpsFlags {
    uint32_t    transform_8x8_mode_flag : 1;
    uint32_t    redundant_pic_cnt_present_flag : 1;
    uint32_t    constrained_intra_pred_flag : 1;
    uint32_t    deblocking_filter_control_present_flag : 1;
    uint32_t    weighted_pred_flag : 1;
    uint32_t    bottom_field_pic_order_in_frame_present_flag : 1;
    uint32_t    entropy_coding_mode_flag : 1;
    uint32_t    pic_scaling_matrix_present_flag : 1;
} StdVideoH264PpsFlags;

typedef struct StdVideoH264PictureParameterSet {
    StdVideoH264PpsFlags               flags;
    uint8_t                            seq_parameter_set_id;
    uint8_t                            pic_parameter_set_id;
    uint8_t                            num_ref_idx_l0_default_active_minus1;
    uint8_t                            num_ref_idx_l1_default_active_minus1;
    StdVideoH264WeightedBipredIdc      weighted_bipred_idc;
    int8_t                             pic_init_qp_minus26;
    int8_t                             pic_init_qs_minus26;
    int8_t                             chroma_qp_index_offset;
    int8_t                             second_chroma_qp_index_offset;
    const StdVideoH264ScalingLists*    pScalingLists;
} StdVideoH264PictureParameterSet;


#ifdef __cplusplus
}
#endif

#endif
