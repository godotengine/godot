/*
** Copyright (c) 2019-2021 The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

#ifndef VULKAN_VIDEO_CODEC_H264STD_H_
#define VULKAN_VIDEO_CODEC_H264STD_H_ 1

#ifdef __cplusplus
extern "C" {
#endif

#include "vk_video/vulkan_video_codecs_common.h"

// Vulkan 0.9 provisional Vulkan video H.264 encode and decode std specification version number
#define VK_STD_VULKAN_VIDEO_CODEC_H264_API_VERSION_0_9 VK_MAKE_VIDEO_STD_VERSION(0, 9, 0) // Patch version should always be set to 0

// Format must be in the form XX.XX where the first two digits are the major and the second two, the minor.
#define VK_STD_VULKAN_VIDEO_CODEC_H264_SPEC_VERSION   VK_STD_VULKAN_VIDEO_CODEC_H264_API_VERSION_0_9
#define VK_STD_VULKAN_VIDEO_CODEC_H264_EXTENSION_NAME "VK_STD_vulkan_video_codec_h264"

// *************************************************
// Video H.264 common definitions:
// *************************************************

typedef enum StdVideoH264ChromaFormatIdc {
    std_video_h264_chroma_format_idc_monochrome  = 0,
    std_video_h264_chroma_format_idc_420         = 1,
    std_video_h264_chroma_format_idc_422         = 2,
    std_video_h264_chroma_format_idc_444         = 3,
} StdVideoH264ChromaFormatIdc;

typedef enum StdVideoH264ProfileIdc {
    std_video_h264_profile_idc_baseline             = 66, /* Only constrained baseline is supported */
    std_video_h264_profile_idc_main                 = 77,
    std_video_h264_profile_idc_high                 = 100,
    std_video_h264_profile_idc_high_444_predictive  = 244,
    std_video_h264_profile_idc_invalid              = 0x7FFFFFFF
} StdVideoH264ProfileIdc;

typedef enum StdVideoH264Level {
    std_video_h264_level_1_0 = 0,
    std_video_h264_level_1_1 = 1,
    std_video_h264_level_1_2 = 2,
    std_video_h264_level_1_3 = 3,
    std_video_h264_level_2_0 = 4,
    std_video_h264_level_2_1 = 5,
    std_video_h264_level_2_2 = 6,
    std_video_h264_level_3_0 = 7,
    std_video_h264_level_3_1 = 8,
    std_video_h264_level_3_2 = 9,
    std_video_h264_level_4_0 = 10,
    std_video_h264_level_4_1 = 11,
    std_video_h264_level_4_2 = 12,
    std_video_h264_level_5_0 = 13,
    std_video_h264_level_5_1 = 14,
    std_video_h264_level_5_2 = 15,
    std_video_h264_level_6_0 = 16,
    std_video_h264_level_6_1 = 17,
    std_video_h264_level_6_2 = 18,
    std_video_h264_level_invalid = 0x7FFFFFFF
} StdVideoH264Level;

typedef enum StdVideoH264PocType {
    std_video_h264_poc_type_0 = 0,
    std_video_h264_poc_type_1 = 1,
    std_video_h264_poc_type_2 = 2,
    std_video_h264_poc_type_invalid = 0x7FFFFFFF
} StdVideoH264PocType;

typedef enum StdVideoH264AspectRatioIdc {
    std_video_h264_aspect_ratio_idc_unspecified = 0,
    std_video_h264_aspect_ratio_idc_square = 1,
    std_video_h264_aspect_ratio_idc_12_11 = 2,
    std_video_h264_aspect_ratio_idc_10_11 = 3,
    std_video_h264_aspect_ratio_idc_16_11 = 4,
    std_video_h264_aspect_ratio_idc_40_33 = 5,
    std_video_h264_aspect_ratio_idc_24_11 = 6,
    std_video_h264_aspect_ratio_idc_20_11 = 7,
    std_video_h264_aspect_ratio_idc_32_11 = 8,
    std_video_h264_aspect_ratio_idc_80_33 = 9,
    std_video_h264_aspect_ratio_idc_18_11 = 10,
    std_video_h264_aspect_ratio_idc_15_11 = 11,
    std_video_h264_aspect_ratio_idc_64_33 = 12,
    std_video_h264_aspect_ratio_idc_160_99 = 13,
    std_video_h264_aspect_ratio_idc_4_3 = 14,
    std_video_h264_aspect_ratio_idc_3_2 = 15,
    std_video_h264_aspect_ratio_idc_2_1 = 16,
    std_video_h264_aspect_ratio_idc_extended_sar = 255,
    std_video_h264_aspect_ratio_idc_invalid = 0x7FFFFFFF
} StdVideoH264AspectRatioIdc;

typedef enum StdVideoH264WeightedBiPredIdc {
    std_video_h264_default_weighted_b_slices_prediction_idc = 0,
    std_video_h264_explicit_weighted_b_slices_prediction_idc = 1,
    std_video_h264_implicit_weighted_b_slices_prediction_idc = 2,
    std_video_h264_invalid_weighted_b_slices_prediction_idc = 0x7FFFFFFF
} StdVideoH264WeightedBiPredIdc;

typedef enum StdVideoH264ModificationOfPicNumsIdc {
    std_video_h264_modification_of_pic_nums_idc_short_term_subtract = 0,
    std_video_h264_modification_of_pic_nums_idc_short_term_add = 1,
    std_video_h264_modification_of_pic_nums_idc_long_term = 2,
    std_video_h264_modification_of_pic_nums_idc_end = 3,
    std_video_h264_modification_of_pic_nums_idc_invalid = 0x7FFFFFFF
} StdVideoH264ModificationOfPicNumsIdc;

typedef enum StdVideoH264MemMgmtControlOp {
    std_video_h264_mem_mgmt_control_op_end = 0,
    std_video_h264_mem_mgmt_control_op_unmark_short_term = 1,
    std_video_h264_mem_mgmt_control_op_unmark_long_term = 2,
    std_video_h264_mem_mgmt_control_op_mark_long_term = 3,
    std_video_h264_mem_mgmt_control_op_set_max_long_term_index = 4,
    std_video_h264_mem_mgmt_control_op_unmark_all = 5,
    std_video_h264_mem_mgmt_control_op_mark_current_as_long_term = 6,
    std_video_h264_mem_mgmt_control_op_invalid = 0x7FFFFFFF
} StdVideoH264MemMgmtControlOp;

typedef enum StdVideoH264CabacInitIdc {
    std_video_h264_cabac_init_idc_0 = 0,
    std_video_h264_cabac_init_idc_1 = 1,
    std_video_h264_cabac_init_idc_2 = 2,
    std_video_h264_cabac_init_idc_invalid = 0x7FFFFFFF
} StdVideoH264CabacInitIdc;

typedef enum StdVideoH264DisableDeblockingFilterIdc {
    std_video_h264_disable_deblocking_filter_idc_disabled = 0,
    std_video_h264_disable_deblocking_filter_idc_enabled = 1,
    std_video_h264_disable_deblocking_filter_idc_partial = 2,
    std_video_h264_disable_deblocking_filter_idc_invalid = 0x7FFFFFFF
} StdVideoH264DisableDeblockingFilterIdc;

typedef enum StdVideoH264PictureType {
    std_video_h264_picture_type_i = 0,
    std_video_h264_picture_type_p = 1,
    std_video_h264_picture_type_b = 2,
    std_video_h264_picture_type_invalid = 0x7FFFFFFF
} StdVideoH264PictureType;

typedef enum StdVideoH264SliceType {
    std_video_h264_slice_type_i = 0,
    std_video_h264_slice_type_p = 1,
    std_video_h264_slice_type_b = 2,
    std_video_h264_slice_type_invalid = 0x7FFFFFFF
} StdVideoH264SliceType;

typedef enum StdVideoH264NonVclNaluType {
    std_video_h264_non_vcl_nalu_type_sps = 0,
    std_video_h264_non_vcl_nalu_type_pps = 1,
    std_video_h264_non_vcl_nalu_type_aud = 2,
    std_video_h264_non_vcl_nalu_type_prefix = 3,
    std_video_h264_non_vcl_nalu_type_end_of_sequence = 4,
    std_video_h264_non_vcl_nalu_type_end_of_stream = 5,
    std_video_h264_non_vcl_nalu_type_precoded = 6,
    std_video_h264_non_vcl_nalu_type_invalid = 0x7FFFFFFF
} StdVideoH264NonVclNaluType;

typedef struct StdVideoH264SpsVuiFlags {
    uint32_t aspect_ratio_info_present_flag:1;
    uint32_t overscan_info_present_flag:1;
    uint32_t overscan_appropriate_flag:1;
    uint32_t video_signal_type_present_flag:1;
    uint32_t video_full_range_flag:1;
    uint32_t color_description_present_flag:1;
    uint32_t chroma_loc_info_present_flag:1;
    uint32_t timing_info_present_flag:1;
    uint32_t fixed_frame_rate_flag:1;
    uint32_t bitstream_restriction_flag:1;
    uint32_t nal_hrd_parameters_present_flag:1;
    uint32_t vcl_hrd_parameters_present_flag:1;
} StdVideoH264SpsVuiFlags;

typedef struct StdVideoH264HrdParameters {
    uint8_t                    cpb_cnt_minus1;
    uint8_t                    bit_rate_scale;
    uint8_t                    cpb_size_scale;
    uint32_t                   bit_rate_value_minus1[32];
    uint32_t                   cpb_size_value_minus1[32];
    uint8_t                    cbr_flag[32];
    uint32_t                   initial_cpb_removal_delay_length_minus1;
    uint32_t                   cpb_removal_delay_length_minus1;
    uint32_t                   dpb_output_delay_length_minus1;
    uint32_t                   time_offset_length;
} StdVideoH264HrdParameters;

typedef struct StdVideoH264SequenceParameterSetVui {
    StdVideoH264AspectRatioIdc  aspect_ratio_idc;
    uint16_t                    sar_width;
    uint16_t                    sar_height;
    uint8_t                     video_format;
    uint8_t                     color_primaries;
    uint8_t                     transfer_characteristics;
    uint8_t                     matrix_coefficients;
    uint32_t                    num_units_in_tick;
    uint32_t                    time_scale;
    StdVideoH264HrdParameters   hrd_parameters;
    uint8_t                     num_reorder_frames;
    uint8_t                     max_dec_frame_buffering;
    StdVideoH264SpsVuiFlags     flags;
} StdVideoH264SequenceParameterSetVui;

typedef struct StdVideoH264SpsFlags {
    uint32_t constraint_set0_flag:1;
    uint32_t constraint_set1_flag:1;
    uint32_t constraint_set2_flag:1;
    uint32_t constraint_set3_flag:1;
    uint32_t constraint_set4_flag:1;
    uint32_t constraint_set5_flag:1;
    uint32_t direct_8x8_inference_flag:1;
    uint32_t mb_adaptive_frame_field_flag:1;
    uint32_t frame_mbs_only_flag:1;
    uint32_t delta_pic_order_always_zero_flag:1;
    uint32_t residual_colour_transform_flag:1;
    uint32_t gaps_in_frame_num_value_allowed_flag:1;
    uint32_t first_picture_after_seek_flag:1; // where is this being documented?
    uint32_t qpprime_y_zero_transform_bypass_flag:1;
    uint32_t frame_cropping_flag:1;
    uint32_t scaling_matrix_present_flag:1;
    uint32_t vui_parameters_present_flag:1;
} StdVideoH264SpsFlags;

typedef struct StdVideoH264ScalingLists
{
    // scaling_list_present_mask has one bit for each
    // seq_scaling_list_present_flag[i] for SPS OR
    // pic_scaling_list_present_flag[i] for PPS,
    // bit 0 - 5 are for each entry of ScalingList4x4
    // bit 6 - 7 are for each entry plus 6 for ScalingList8x8
    uint8_t scaling_list_present_mask;
    // use_default_scaling_matrix_mask has one bit for each
    // UseDefaultScalingMatrix4x4Flag[ i ] and
    // UseDefaultScalingMatrix8x8Flag[ i - 6 ] for SPS OR PPS
    // bit 0 - 5 are for each entry of ScalingList4x4
    // bit 6 - 7 are for each entry plus 6 for ScalingList8x8
    uint8_t use_default_scaling_matrix_mask;
    uint8_t ScalingList4x4[6][16];
    uint8_t ScalingList8x8[2][64];
} StdVideoH264ScalingLists;

typedef struct StdVideoH264SequenceParameterSet
{
    StdVideoH264ProfileIdc               profile_idc;
    StdVideoH264Level                    level_idc;
    uint8_t                              seq_parameter_set_id;
    StdVideoH264ChromaFormatIdc          chroma_format_idc;
    uint8_t                              bit_depth_luma_minus8;
    uint8_t                              bit_depth_chroma_minus8;
    uint8_t                              log2_max_frame_num_minus4;
    StdVideoH264PocType                  pic_order_cnt_type;
    uint8_t                              log2_max_pic_order_cnt_lsb_minus4;
    int32_t                              offset_for_non_ref_pic;
    int32_t                              offset_for_top_to_bottom_field;
    uint8_t                              num_ref_frames_in_pic_order_cnt_cycle;
    uint8_t                              max_num_ref_frames;
    uint32_t                             pic_width_in_mbs_minus1;
    uint32_t                             pic_height_in_map_units_minus1;
    uint32_t                             frame_crop_left_offset;
    uint32_t                             frame_crop_right_offset;
    uint32_t                             frame_crop_top_offset;
    uint32_t                             frame_crop_bottom_offset;
    StdVideoH264SpsFlags                 flags;
    int32_t                              offset_for_ref_frame[255]; // The number of valid values are defined by the num_ref_frames_in_pic_order_cnt_cycle
    StdVideoH264ScalingLists*            pScalingLists;             // Must be a valid pointer if scaling_matrix_present_flag is set
    StdVideoH264SequenceParameterSetVui* pSequenceParameterSetVui;  // Must be a valid pointer if StdVideoH264SpsFlags:vui_parameters_present_flag is set
} StdVideoH264SequenceParameterSet;

typedef struct StdVideoH264PpsFlags {
    uint32_t transform_8x8_mode_flag:1;
    uint32_t redundant_pic_cnt_present_flag:1;
    uint32_t constrained_intra_pred_flag:1;
    uint32_t deblocking_filter_control_present_flag:1;
    uint32_t weighted_bipred_idc_flag:1;
    uint32_t weighted_pred_flag:1;
    uint32_t pic_order_present_flag:1;
    uint32_t entropy_coding_mode_flag:1;
    uint32_t scaling_matrix_present_flag:1;
} StdVideoH264PpsFlags;

typedef struct StdVideoH264PictureParameterSet
{
    uint8_t                       seq_parameter_set_id;
    uint8_t                       pic_parameter_set_id;
    uint8_t                       num_ref_idx_l0_default_active_minus1;
    uint8_t                       num_ref_idx_l1_default_active_minus1;
    StdVideoH264WeightedBiPredIdc weighted_bipred_idc;
    int8_t                        pic_init_qp_minus26;
    int8_t                        pic_init_qs_minus26;
    int8_t                        chroma_qp_index_offset;
    int8_t                        second_chroma_qp_index_offset;
    StdVideoH264PpsFlags          flags;
    StdVideoH264ScalingLists*     pScalingLists; // Must be a valid pointer if  StdVideoH264PpsFlags::scaling_matrix_present_flag is set.
} StdVideoH264PictureParameterSet;

#ifdef __cplusplus
}
#endif

#endif // VULKAN_VIDEO_CODEC_H264STD_H_
