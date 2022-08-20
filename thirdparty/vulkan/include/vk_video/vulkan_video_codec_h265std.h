#ifndef VULKAN_VIDEO_CODEC_H265STD_H_
#define VULKAN_VIDEO_CODEC_H265STD_H_ 1

/*
** Copyright 2015-2022 The Khronos Group Inc.
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



#define vulkan_video_codec_h265std 1
// Vulkan 0.5 version number WIP
#define VK_STD_VULKAN_VIDEO_CODEC_H265_API_VERSION_0_9_5 VK_MAKE_VIDEO_STD_VERSION(0, 9, 5) // Patch version should always be set to 0

#define STD_VIDEO_H265_SUBLAYERS_MINUS1_LIST_SIZE 7
#define STD_VIDEO_H265_CPB_CNT_LIST_SIZE  32
#define STD_VIDEO_H265_SCALING_LIST_4X4_NUM_LISTS 6
#define STD_VIDEO_H265_SCALING_LIST_4X4_NUM_ELEMENTS 16
#define STD_VIDEO_H265_SCALING_LIST_8X8_NUM_LISTS 6
#define STD_VIDEO_H265_SCALING_LIST_8X8_NUM_ELEMENTS 64
#define STD_VIDEO_H265_SCALING_LIST_16X16_NUM_LISTS 6
#define STD_VIDEO_H265_SCALING_LIST_16X16_NUM_ELEMENTS 64
#define STD_VIDEO_H265_SCALING_LIST_32X32_NUM_LISTS 2
#define STD_VIDEO_H265_SCALING_LIST_32X32_NUM_ELEMENTS 64
#define STD_VIDEO_H265_PREDICTOR_PALETTE_COMPONENTS_LIST_SIZE 3
#define STD_VIDEO_H265_PREDICTOR_PALETTE_COMP_ENTRIES_LIST_SIZE 128
#define STD_VIDEO_H265_CHROMA_QP_OFFSET_TILE_COLS_LIST_SIZE 19
#define STD_VIDEO_H265_CHROMA_QP_OFFSET_TILE_ROWS_LIST_SIZE 21
#define STD_VIDEO_H265_CHROMA_QP_OFFSET_LIST_SIZE 6
#define VK_STD_VULKAN_VIDEO_CODEC_H265_SPEC_VERSION VK_STD_VULKAN_VIDEO_CODEC_H265_API_VERSION_0_9_5
#define VK_STD_VULKAN_VIDEO_CODEC_H265_EXTENSION_NAME "VK_STD_vulkan_video_codec_h265"

typedef enum StdVideoH265ChromaFormatIdc {
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_MONOCHROME = 0,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_420 = 1,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_422 = 2,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_444 = 3,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H265_CHROMA_FORMAT_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH265ChromaFormatIdc;

typedef enum StdVideoH265ProfileIdc {
    STD_VIDEO_H265_PROFILE_IDC_MAIN = 1,
    STD_VIDEO_H265_PROFILE_IDC_MAIN_10 = 2,
    STD_VIDEO_H265_PROFILE_IDC_MAIN_STILL_PICTURE = 3,
    STD_VIDEO_H265_PROFILE_IDC_FORMAT_RANGE_EXTENSIONS = 4,
    STD_VIDEO_H265_PROFILE_IDC_SCC_EXTENSIONS = 9,
    STD_VIDEO_H265_PROFILE_IDC_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H265_PROFILE_IDC_MAX_ENUM = 0x7FFFFFFF
} StdVideoH265ProfileIdc;

typedef enum StdVideoH265Level {
    STD_VIDEO_H265_LEVEL_1_0 = 0,
    STD_VIDEO_H265_LEVEL_2_0 = 1,
    STD_VIDEO_H265_LEVEL_2_1 = 2,
    STD_VIDEO_H265_LEVEL_3_0 = 3,
    STD_VIDEO_H265_LEVEL_3_1 = 4,
    STD_VIDEO_H265_LEVEL_4_0 = 5,
    STD_VIDEO_H265_LEVEL_4_1 = 6,
    STD_VIDEO_H265_LEVEL_5_0 = 7,
    STD_VIDEO_H265_LEVEL_5_1 = 8,
    STD_VIDEO_H265_LEVEL_5_2 = 9,
    STD_VIDEO_H265_LEVEL_6_0 = 10,
    STD_VIDEO_H265_LEVEL_6_1 = 11,
    STD_VIDEO_H265_LEVEL_6_2 = 12,
    STD_VIDEO_H265_LEVEL_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H265_LEVEL_MAX_ENUM = 0x7FFFFFFF
} StdVideoH265Level;

typedef enum StdVideoH265SliceType {
    STD_VIDEO_H265_SLICE_TYPE_B = 0,
    STD_VIDEO_H265_SLICE_TYPE_P = 1,
    STD_VIDEO_H265_SLICE_TYPE_I = 2,
    STD_VIDEO_H265_SLICE_TYPE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H265_SLICE_TYPE_MAX_ENUM = 0x7FFFFFFF
} StdVideoH265SliceType;

typedef enum StdVideoH265PictureType {
    STD_VIDEO_H265_PICTURE_TYPE_P = 0,
    STD_VIDEO_H265_PICTURE_TYPE_B = 1,
    STD_VIDEO_H265_PICTURE_TYPE_I = 2,
    STD_VIDEO_H265_PICTURE_TYPE_IDR = 3,
    STD_VIDEO_H265_PICTURE_TYPE_INVALID = 0x7FFFFFFF,
    STD_VIDEO_H265_PICTURE_TYPE_MAX_ENUM = 0x7FFFFFFF
} StdVideoH265PictureType;
typedef struct StdVideoH265DecPicBufMgr {
    uint32_t    max_latency_increase_plus1[STD_VIDEO_H265_SUBLAYERS_MINUS1_LIST_SIZE];
    uint8_t     max_dec_pic_buffering_minus1[STD_VIDEO_H265_SUBLAYERS_MINUS1_LIST_SIZE];
    uint8_t     max_num_reorder_pics[STD_VIDEO_H265_SUBLAYERS_MINUS1_LIST_SIZE];
} StdVideoH265DecPicBufMgr;

typedef struct StdVideoH265SubLayerHrdParameters {
    uint32_t    bit_rate_value_minus1[STD_VIDEO_H265_CPB_CNT_LIST_SIZE];
    uint32_t    cpb_size_value_minus1[STD_VIDEO_H265_CPB_CNT_LIST_SIZE];
    uint32_t    cpb_size_du_value_minus1[STD_VIDEO_H265_CPB_CNT_LIST_SIZE];
    uint32_t    bit_rate_du_value_minus1[STD_VIDEO_H265_CPB_CNT_LIST_SIZE];
    uint32_t    cbr_flag;
} StdVideoH265SubLayerHrdParameters;

typedef struct StdVideoH265HrdFlags {
    uint32_t    nal_hrd_parameters_present_flag : 1;
    uint32_t    vcl_hrd_parameters_present_flag : 1;
    uint32_t    sub_pic_hrd_params_present_flag : 1;
    uint32_t    sub_pic_cpb_params_in_pic_timing_sei_flag : 1;
    uint32_t    fixed_pic_rate_general_flag : 8;
    uint32_t    fixed_pic_rate_within_cvs_flag : 8;
    uint32_t    low_delay_hrd_flag : 8;
} StdVideoH265HrdFlags;

typedef struct StdVideoH265HrdParameters {
    uint8_t                               tick_divisor_minus2;
    uint8_t                               du_cpb_removal_delay_increment_length_minus1;
    uint8_t                               dpb_output_delay_du_length_minus1;
    uint8_t                               bit_rate_scale;
    uint8_t                               cpb_size_scale;
    uint8_t                               cpb_size_du_scale;
    uint8_t                               initial_cpb_removal_delay_length_minus1;
    uint8_t                               au_cpb_removal_delay_length_minus1;
    uint8_t                               dpb_output_delay_length_minus1;
    uint8_t                               cpb_cnt_minus1[STD_VIDEO_H265_SUBLAYERS_MINUS1_LIST_SIZE];
    uint16_t                              elemental_duration_in_tc_minus1[STD_VIDEO_H265_SUBLAYERS_MINUS1_LIST_SIZE];
    StdVideoH265SubLayerHrdParameters*    pSubLayerHrdParametersNal[STD_VIDEO_H265_SUBLAYERS_MINUS1_LIST_SIZE];
    StdVideoH265SubLayerHrdParameters*    pSubLayerHrdParametersVcl[STD_VIDEO_H265_SUBLAYERS_MINUS1_LIST_SIZE];
    StdVideoH265HrdFlags                  flags;
} StdVideoH265HrdParameters;

typedef struct StdVideoH265VpsFlags {
    uint32_t    vps_temporal_id_nesting_flag : 1;
    uint32_t    vps_sub_layer_ordering_info_present_flag : 1;
    uint32_t    vps_timing_info_present_flag : 1;
    uint32_t    vps_poc_proportional_to_timing_flag : 1;
} StdVideoH265VpsFlags;

typedef struct StdVideoH265VideoParameterSet {
    uint8_t                       vps_video_parameter_set_id;
    uint8_t                       vps_max_sub_layers_minus1;
    uint32_t                      vps_num_units_in_tick;
    uint32_t                      vps_time_scale;
    uint32_t                      vps_num_ticks_poc_diff_one_minus1;
    StdVideoH265DecPicBufMgr*     pDecPicBufMgr;
    StdVideoH265HrdParameters*    pHrdParameters;
    StdVideoH265VpsFlags          flags;
} StdVideoH265VideoParameterSet;

typedef struct StdVideoH265ScalingLists {
    uint8_t    ScalingList4x4[STD_VIDEO_H265_SCALING_LIST_4X4_NUM_LISTS][STD_VIDEO_H265_SCALING_LIST_4X4_NUM_ELEMENTS];
    uint8_t    ScalingList8x8[STD_VIDEO_H265_SCALING_LIST_8X8_NUM_LISTS][STD_VIDEO_H265_SCALING_LIST_8X8_NUM_ELEMENTS];
    uint8_t    ScalingList16x16[STD_VIDEO_H265_SCALING_LIST_16X16_NUM_LISTS][STD_VIDEO_H265_SCALING_LIST_16X16_NUM_ELEMENTS];
    uint8_t    ScalingList32x32[STD_VIDEO_H265_SCALING_LIST_32X32_NUM_LISTS][STD_VIDEO_H265_SCALING_LIST_32X32_NUM_ELEMENTS];
    uint8_t    ScalingListDCCoef16x16[STD_VIDEO_H265_SCALING_LIST_16X16_NUM_LISTS];
    uint8_t    ScalingListDCCoef32x32[STD_VIDEO_H265_SCALING_LIST_32X32_NUM_LISTS];
} StdVideoH265ScalingLists;

typedef struct StdVideoH265SpsVuiFlags {
    uint32_t    aspect_ratio_info_present_flag : 1;
    uint32_t    overscan_info_present_flag : 1;
    uint32_t    overscan_appropriate_flag : 1;
    uint32_t    video_signal_type_present_flag : 1;
    uint32_t    video_full_range_flag : 1;
    uint32_t    colour_description_present_flag : 1;
    uint32_t    chroma_loc_info_present_flag : 1;
    uint32_t    neutral_chroma_indication_flag : 1;
    uint32_t    field_seq_flag : 1;
    uint32_t    frame_field_info_present_flag : 1;
    uint32_t    default_display_window_flag : 1;
    uint32_t    vui_timing_info_present_flag : 1;
    uint32_t    vui_poc_proportional_to_timing_flag : 1;
    uint32_t    vui_hrd_parameters_present_flag : 1;
    uint32_t    bitstream_restriction_flag : 1;
    uint32_t    tiles_fixed_structure_flag : 1;
    uint32_t    motion_vectors_over_pic_boundaries_flag : 1;
    uint32_t    restricted_ref_pic_lists_flag : 1;
} StdVideoH265SpsVuiFlags;

typedef struct StdVideoH265SequenceParameterSetVui {
    uint8_t                       aspect_ratio_idc;
    uint16_t                      sar_width;
    uint16_t                      sar_height;
    uint8_t                       video_format;
    uint8_t                       colour_primaries;
    uint8_t                       transfer_characteristics;
    uint8_t                       matrix_coeffs;
    uint8_t                       chroma_sample_loc_type_top_field;
    uint8_t                       chroma_sample_loc_type_bottom_field;
    uint16_t                      def_disp_win_left_offset;
    uint16_t                      def_disp_win_right_offset;
    uint16_t                      def_disp_win_top_offset;
    uint16_t                      def_disp_win_bottom_offset;
    uint32_t                      vui_num_units_in_tick;
    uint32_t                      vui_time_scale;
    uint32_t                      vui_num_ticks_poc_diff_one_minus1;
    StdVideoH265HrdParameters*    pHrdParameters;
    uint16_t                      min_spatial_segmentation_idc;
    uint8_t                       max_bytes_per_pic_denom;
    uint8_t                       max_bits_per_min_cu_denom;
    uint8_t                       log2_max_mv_length_horizontal;
    uint8_t                       log2_max_mv_length_vertical;
    StdVideoH265SpsVuiFlags       flags;
} StdVideoH265SequenceParameterSetVui;

typedef struct StdVideoH265PredictorPaletteEntries {
    uint16_t    PredictorPaletteEntries[STD_VIDEO_H265_PREDICTOR_PALETTE_COMPONENTS_LIST_SIZE][STD_VIDEO_H265_PREDICTOR_PALETTE_COMP_ENTRIES_LIST_SIZE];
} StdVideoH265PredictorPaletteEntries;

typedef struct StdVideoH265SpsFlags {
    uint32_t    sps_temporal_id_nesting_flag : 1;
    uint32_t    separate_colour_plane_flag : 1;
    uint32_t    scaling_list_enabled_flag : 1;
    uint32_t    sps_scaling_list_data_present_flag : 1;
    uint32_t    amp_enabled_flag : 1;
    uint32_t    sample_adaptive_offset_enabled_flag : 1;
    uint32_t    pcm_enabled_flag : 1;
    uint32_t    pcm_loop_filter_disabled_flag : 1;
    uint32_t    long_term_ref_pics_present_flag : 1;
    uint32_t    sps_temporal_mvp_enabled_flag : 1;
    uint32_t    strong_intra_smoothing_enabled_flag : 1;
    uint32_t    vui_parameters_present_flag : 1;
    uint32_t    sps_extension_present_flag : 1;
    uint32_t    sps_range_extension_flag : 1;
    uint32_t    transform_skip_rotation_enabled_flag : 1;
    uint32_t    transform_skip_context_enabled_flag : 1;
    uint32_t    implicit_rdpcm_enabled_flag : 1;
    uint32_t    explicit_rdpcm_enabled_flag : 1;
    uint32_t    extended_precision_processing_flag : 1;
    uint32_t    intra_smoothing_disabled_flag : 1;
    uint32_t    high_precision_offsets_enabled_flag : 1;
    uint32_t    persistent_rice_adaptation_enabled_flag : 1;
    uint32_t    cabac_bypass_alignment_enabled_flag : 1;
    uint32_t    sps_curr_pic_ref_enabled_flag : 1;
    uint32_t    palette_mode_enabled_flag : 1;
    uint32_t    sps_palette_predictor_initializer_present_flag : 1;
    uint32_t    intra_boundary_filtering_disabled_flag : 1;
} StdVideoH265SpsFlags;

typedef struct StdVideoH265SequenceParameterSet {
    StdVideoH265ProfileIdc                  profile_idc;
    StdVideoH265Level                       level_idc;
    uint32_t                                pic_width_in_luma_samples;
    uint32_t                                pic_height_in_luma_samples;
    uint8_t                                 sps_video_parameter_set_id;
    uint8_t                                 sps_max_sub_layers_minus1;
    uint8_t                                 sps_seq_parameter_set_id;
    uint8_t                                 chroma_format_idc;
    uint8_t                                 bit_depth_luma_minus8;
    uint8_t                                 bit_depth_chroma_minus8;
    uint8_t                                 log2_max_pic_order_cnt_lsb_minus4;
    uint8_t                                 sps_max_dec_pic_buffering_minus1;
    uint8_t                                 log2_min_luma_coding_block_size_minus3;
    uint8_t                                 log2_diff_max_min_luma_coding_block_size;
    uint8_t                                 log2_min_luma_transform_block_size_minus2;
    uint8_t                                 log2_diff_max_min_luma_transform_block_size;
    uint8_t                                 max_transform_hierarchy_depth_inter;
    uint8_t                                 max_transform_hierarchy_depth_intra;
    uint8_t                                 num_short_term_ref_pic_sets;
    uint8_t                                 num_long_term_ref_pics_sps;
    uint8_t                                 pcm_sample_bit_depth_luma_minus1;
    uint8_t                                 pcm_sample_bit_depth_chroma_minus1;
    uint8_t                                 log2_min_pcm_luma_coding_block_size_minus3;
    uint8_t                                 log2_diff_max_min_pcm_luma_coding_block_size;
    uint32_t                                conf_win_left_offset;
    uint32_t                                conf_win_right_offset;
    uint32_t                                conf_win_top_offset;
    uint32_t                                conf_win_bottom_offset;
    StdVideoH265DecPicBufMgr*               pDecPicBufMgr;
    StdVideoH265SpsFlags                    flags;
    StdVideoH265ScalingLists*               pScalingLists;
    StdVideoH265SequenceParameterSetVui*    pSequenceParameterSetVui;
    uint8_t                                 palette_max_size;
    uint8_t                                 delta_palette_max_predictor_size;
    uint8_t                                 motion_vector_resolution_control_idc;
    uint8_t                                 sps_num_palette_predictor_initializer_minus1;
    StdVideoH265PredictorPaletteEntries*    pPredictorPaletteEntries;
} StdVideoH265SequenceParameterSet;

typedef struct StdVideoH265PpsFlags {
    uint32_t    dependent_slice_segments_enabled_flag : 1;
    uint32_t    output_flag_present_flag : 1;
    uint32_t    sign_data_hiding_enabled_flag : 1;
    uint32_t    cabac_init_present_flag : 1;
    uint32_t    constrained_intra_pred_flag : 1;
    uint32_t    transform_skip_enabled_flag : 1;
    uint32_t    cu_qp_delta_enabled_flag : 1;
    uint32_t    pps_slice_chroma_qp_offsets_present_flag : 1;
    uint32_t    weighted_pred_flag : 1;
    uint32_t    weighted_bipred_flag : 1;
    uint32_t    transquant_bypass_enabled_flag : 1;
    uint32_t    tiles_enabled_flag : 1;
    uint32_t    entropy_coding_sync_enabled_flag : 1;
    uint32_t    uniform_spacing_flag : 1;
    uint32_t    loop_filter_across_tiles_enabled_flag : 1;
    uint32_t    pps_loop_filter_across_slices_enabled_flag : 1;
    uint32_t    deblocking_filter_control_present_flag : 1;
    uint32_t    deblocking_filter_override_enabled_flag : 1;
    uint32_t    pps_deblocking_filter_disabled_flag : 1;
    uint32_t    pps_scaling_list_data_present_flag : 1;
    uint32_t    lists_modification_present_flag : 1;
    uint32_t    slice_segment_header_extension_present_flag : 1;
    uint32_t    pps_extension_present_flag : 1;
    uint32_t    cross_component_prediction_enabled_flag : 1;
    uint32_t    chroma_qp_offset_list_enabled_flag : 1;
    uint32_t    pps_curr_pic_ref_enabled_flag : 1;
    uint32_t    residual_adaptive_colour_transform_enabled_flag : 1;
    uint32_t    pps_slice_act_qp_offsets_present_flag : 1;
    uint32_t    pps_palette_predictor_initializer_present_flag : 1;
    uint32_t    monochrome_palette_flag : 1;
    uint32_t    pps_range_extension_flag : 1;
} StdVideoH265PpsFlags;

typedef struct StdVideoH265PictureParameterSet {
    uint8_t                                 pps_pic_parameter_set_id;
    uint8_t                                 pps_seq_parameter_set_id;
    uint8_t                                 num_extra_slice_header_bits;
    uint8_t                                 num_ref_idx_l0_default_active_minus1;
    uint8_t                                 num_ref_idx_l1_default_active_minus1;
    int8_t                                  init_qp_minus26;
    uint8_t                                 diff_cu_qp_delta_depth;
    int8_t                                  pps_cb_qp_offset;
    int8_t                                  pps_cr_qp_offset;
    uint8_t                                 num_tile_columns_minus1;
    uint8_t                                 num_tile_rows_minus1;
    uint16_t                                column_width_minus1[STD_VIDEO_H265_CHROMA_QP_OFFSET_TILE_COLS_LIST_SIZE];
    uint16_t                                row_height_minus1[STD_VIDEO_H265_CHROMA_QP_OFFSET_TILE_ROWS_LIST_SIZE];
    int8_t                                  pps_beta_offset_div2;
    int8_t                                  pps_tc_offset_div2;
    uint8_t                                 log2_parallel_merge_level_minus2;
    StdVideoH265PpsFlags                    flags;
    StdVideoH265ScalingLists*               pScalingLists;
    uint8_t                                 log2_max_transform_skip_block_size_minus2;
    uint8_t                                 diff_cu_chroma_qp_offset_depth;
    uint8_t                                 chroma_qp_offset_list_len_minus1;
    int8_t                                  cb_qp_offset_list[STD_VIDEO_H265_CHROMA_QP_OFFSET_LIST_SIZE];
    int8_t                                  cr_qp_offset_list[STD_VIDEO_H265_CHROMA_QP_OFFSET_LIST_SIZE];
    uint8_t                                 log2_sao_offset_scale_luma;
    uint8_t                                 log2_sao_offset_scale_chroma;
    int8_t                                  pps_act_y_qp_offset_plus5;
    int8_t                                  pps_act_cb_qp_offset_plus5;
    int8_t                                  pps_act_cr_qp_offset_plus5;
    uint8_t                                 pps_num_palette_predictor_initializer;
    uint8_t                                 luma_bit_depth_entry_minus8;
    uint8_t                                 chroma_bit_depth_entry_minus8;
    StdVideoH265PredictorPaletteEntries*    pPredictorPaletteEntries;
} StdVideoH265PictureParameterSet;


#ifdef __cplusplus
}
#endif

#endif
