#ifndef VULKAN_VIDEO_CODEC_H265STD_ENCODE_H_
#define VULKAN_VIDEO_CODEC_H265STD_ENCODE_H_ 1

/*
** Copyright 2015-2023 The Khronos Group Inc.
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



#define vulkan_video_codec_h265std_encode 1
// Vulkan 0.9 provisional Vulkan video H.265 encode std specification version number
#define VK_STD_VULKAN_VIDEO_CODEC_H265_ENCODE_API_VERSION_0_9_10 VK_MAKE_VIDEO_STD_VERSION(0, 9, 10)

#define VK_STD_VULKAN_VIDEO_CODEC_H265_ENCODE_SPEC_VERSION VK_STD_VULKAN_VIDEO_CODEC_H265_ENCODE_API_VERSION_0_9_10
#define VK_STD_VULKAN_VIDEO_CODEC_H265_ENCODE_EXTENSION_NAME "VK_STD_vulkan_video_codec_h265_encode"
typedef struct StdVideoEncodeH265WeightTableFlags {
    uint16_t    luma_weight_l0_flag;
    uint16_t    chroma_weight_l0_flag;
    uint16_t    luma_weight_l1_flag;
    uint16_t    chroma_weight_l1_flag;
} StdVideoEncodeH265WeightTableFlags;

typedef struct StdVideoEncodeH265WeightTable {
    StdVideoEncodeH265WeightTableFlags    flags;
    uint8_t                               luma_log2_weight_denom;
    int8_t                                delta_chroma_log2_weight_denom;
    int8_t                                delta_luma_weight_l0[STD_VIDEO_H265_MAX_NUM_LIST_REF];
    int8_t                                luma_offset_l0[STD_VIDEO_H265_MAX_NUM_LIST_REF];
    int8_t                                delta_chroma_weight_l0[STD_VIDEO_H265_MAX_NUM_LIST_REF][STD_VIDEO_H265_MAX_CHROMA_PLANES];
    int8_t                                delta_chroma_offset_l0[STD_VIDEO_H265_MAX_NUM_LIST_REF][STD_VIDEO_H265_MAX_CHROMA_PLANES];
    int8_t                                delta_luma_weight_l1[STD_VIDEO_H265_MAX_NUM_LIST_REF];
    int8_t                                luma_offset_l1[STD_VIDEO_H265_MAX_NUM_LIST_REF];
    int8_t                                delta_chroma_weight_l1[STD_VIDEO_H265_MAX_NUM_LIST_REF][STD_VIDEO_H265_MAX_CHROMA_PLANES];
    int8_t                                delta_chroma_offset_l1[STD_VIDEO_H265_MAX_NUM_LIST_REF][STD_VIDEO_H265_MAX_CHROMA_PLANES];
} StdVideoEncodeH265WeightTable;

typedef struct StdVideoEncodeH265SliceSegmentHeaderFlags {
    uint32_t    first_slice_segment_in_pic_flag : 1;
    uint32_t    no_output_of_prior_pics_flag : 1;
    uint32_t    dependent_slice_segment_flag : 1;
    uint32_t    pic_output_flag : 1;
    uint32_t    short_term_ref_pic_set_sps_flag : 1;
    uint32_t    slice_temporal_mvp_enable_flag : 1;
    uint32_t    slice_sao_luma_flag : 1;
    uint32_t    slice_sao_chroma_flag : 1;
    uint32_t    num_ref_idx_active_override_flag : 1;
    uint32_t    mvd_l1_zero_flag : 1;
    uint32_t    cabac_init_flag : 1;
    uint32_t    cu_chroma_qp_offset_enabled_flag : 1;
    uint32_t    deblocking_filter_override_flag : 1;
    uint32_t    slice_deblocking_filter_disabled_flag : 1;
    uint32_t    collocated_from_l0_flag : 1;
    uint32_t    slice_loop_filter_across_slices_enabled_flag : 1;
} StdVideoEncodeH265SliceSegmentHeaderFlags;

typedef struct StdVideoEncodeH265SliceSegmentLongTermRefPics {
    uint8_t     num_long_term_sps;
    uint8_t     num_long_term_pics;
    uint8_t     lt_idx_sps[STD_VIDEO_H265_MAX_LONG_TERM_REF_PICS_SPS];
    uint8_t     poc_lsb_lt[STD_VIDEO_H265_MAX_LONG_TERM_PICS];
    uint16_t    used_by_curr_pic_lt_flag;
    uint8_t     delta_poc_msb_present_flag[STD_VIDEO_H265_MAX_DELTA_POC];
    uint8_t     delta_poc_msb_cycle_lt[STD_VIDEO_H265_MAX_DELTA_POC];
} StdVideoEncodeH265SliceSegmentLongTermRefPics;

typedef struct StdVideoEncodeH265SliceSegmentHeader {
    StdVideoEncodeH265SliceSegmentHeaderFlags               flags;
    StdVideoH265SliceType                                   slice_type;
    uint32_t                                                slice_segment_address;
    uint8_t                                                 short_term_ref_pic_set_idx;
    uint8_t                                                 collocated_ref_idx;
    uint8_t                                                 num_ref_idx_l0_active_minus1;
    uint8_t                                                 num_ref_idx_l1_active_minus1;
    uint8_t                                                 MaxNumMergeCand;
    int8_t                                                  slice_cb_qp_offset;
    int8_t                                                  slice_cr_qp_offset;
    int8_t                                                  slice_beta_offset_div2;
    int8_t                                                  slice_tc_offset_div2;
    int8_t                                                  slice_act_y_qp_offset;
    int8_t                                                  slice_act_cb_qp_offset;
    int8_t                                                  slice_act_cr_qp_offset;
    const StdVideoH265ShortTermRefPicSet*                   pShortTermRefPicSet;
    const StdVideoEncodeH265SliceSegmentLongTermRefPics*    pLongTermRefPics;
    const StdVideoEncodeH265WeightTable*                    pWeightTable;
} StdVideoEncodeH265SliceSegmentHeader;

typedef struct StdVideoEncodeH265ReferenceListsInfoFlags {
    uint32_t    ref_pic_list_modification_flag_l0 : 1;
    uint32_t    ref_pic_list_modification_flag_l1 : 1;
} StdVideoEncodeH265ReferenceListsInfoFlags;

typedef struct StdVideoEncodeH265ReferenceListsInfo {
    StdVideoEncodeH265ReferenceListsInfoFlags    flags;
    uint8_t                                      num_ref_idx_l0_active_minus1;
    uint8_t                                      num_ref_idx_l1_active_minus1;
    uint16_t                                     reserved1;
    const uint8_t*                               pRefPicList0Entries;
    const uint8_t*                               pRefPicList1Entries;
    const uint8_t*                               pRefList0Modifications;
    const uint8_t*                               pRefList1Modifications;
} StdVideoEncodeH265ReferenceListsInfo;

typedef struct StdVideoEncodeH265PictureInfoFlags {
    uint32_t    is_reference_flag : 1;
    uint32_t    IrapPicFlag : 1;
    uint32_t    long_term_flag : 1;
    uint32_t    discardable_flag : 1;
    uint32_t    cross_layer_bla_flag : 1;
} StdVideoEncodeH265PictureInfoFlags;

typedef struct StdVideoEncodeH265PictureInfo {
    StdVideoEncodeH265PictureInfoFlags    flags;
    StdVideoH265PictureType               PictureType;
    uint8_t                               sps_video_parameter_set_id;
    uint8_t                               pps_seq_parameter_set_id;
    uint8_t                               pps_pic_parameter_set_id;
    uint8_t                               TemporalId;
    int32_t                               PicOrderCntVal;
} StdVideoEncodeH265PictureInfo;

typedef struct StdVideoEncodeH265ReferenceInfoFlags {
    uint32_t    used_for_long_term_reference : 1;
    uint32_t    unused_for_reference : 1;
} StdVideoEncodeH265ReferenceInfoFlags;

typedef struct StdVideoEncodeH265ReferenceInfo {
    StdVideoEncodeH265ReferenceInfoFlags    flags;
    StdVideoH265PictureType                 PictureType;
    int32_t                                 PicOrderCntVal;
    uint8_t                                 TemporalId;
} StdVideoEncodeH265ReferenceInfo;


#ifdef __cplusplus
}
#endif

#endif
