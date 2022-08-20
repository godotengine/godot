#ifndef VULKAN_VIDEO_CODEC_H265STD_ENCODE_H_
#define VULKAN_VIDEO_CODEC_H265STD_ENCODE_H_ 1

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



#define vulkan_video_codec_h265std_encode 1
#define STD_VIDEO_ENCODE_H265_LUMA_LIST_SIZE 15
#define STD_VIDEO_ENCODE_H265_CHROMA_LIST_SIZE 15
#define STD_VIDEO_ENCODE_H265_CHROMA_LISTS_NUM 2
typedef struct StdVideoEncodeH265SliceSegmentHeaderFlags {
    uint32_t    first_slice_segment_in_pic_flag : 1;
    uint32_t    no_output_of_prior_pics_flag : 1;
    uint32_t    dependent_slice_segment_flag : 1;
    uint32_t    short_term_ref_pic_set_sps_flag : 1;
    uint32_t    slice_temporal_mvp_enable_flag : 1;
    uint32_t    slice_sao_luma_flag : 1;
    uint32_t    slice_sao_chroma_flag : 1;
    uint32_t    num_ref_idx_active_override_flag : 1;
    uint32_t    mvd_l1_zero_flag : 1;
    uint32_t    cabac_init_flag : 1;
    uint32_t    slice_deblocking_filter_disable_flag : 1;
    uint32_t    collocated_from_l0_flag : 1;
    uint32_t    slice_loop_filter_across_slices_enabled_flag : 1;
    uint32_t    bLastSliceInPic : 1;
    uint32_t    reservedBits : 18;
    uint16_t    luma_weight_l0_flag;
    uint16_t    chroma_weight_l0_flag;
    uint16_t    luma_weight_l1_flag;
    uint16_t    chroma_weight_l1_flag;
} StdVideoEncodeH265SliceSegmentHeaderFlags;

typedef struct StdVideoEncodeH265SliceSegmentHeader {
    StdVideoH265SliceType                        slice_type;
    uint8_t                                      slice_pic_parameter_set_id;
    uint8_t                                      num_short_term_ref_pic_sets;
    uint32_t                                     slice_segment_address;
    uint8_t                                      short_term_ref_pic_set_idx;
    uint8_t                                      num_long_term_sps;
    uint8_t                                      num_long_term_pics;
    uint8_t                                      collocated_ref_idx;
    uint8_t                                      num_ref_idx_l0_active_minus1;
    uint8_t                                      num_ref_idx_l1_active_minus1;
    uint8_t                                      luma_log2_weight_denom;
    int8_t                                       delta_chroma_log2_weight_denom;
    int8_t                                       delta_luma_weight_l0[STD_VIDEO_ENCODE_H265_LUMA_LIST_SIZE];
    int8_t                                       luma_offset_l0[STD_VIDEO_ENCODE_H265_LUMA_LIST_SIZE];
    int8_t                                       delta_chroma_weight_l0[STD_VIDEO_ENCODE_H265_CHROMA_LIST_SIZE][STD_VIDEO_ENCODE_H265_CHROMA_LISTS_NUM];
    int8_t                                       delta_chroma_offset_l0[STD_VIDEO_ENCODE_H265_CHROMA_LIST_SIZE][STD_VIDEO_ENCODE_H265_CHROMA_LISTS_NUM];
    int8_t                                       delta_luma_weight_l1[STD_VIDEO_ENCODE_H265_LUMA_LIST_SIZE];
    int8_t                                       luma_offset_l1[STD_VIDEO_ENCODE_H265_LUMA_LIST_SIZE];
    int8_t                                       delta_chroma_weight_l1[STD_VIDEO_ENCODE_H265_CHROMA_LIST_SIZE][STD_VIDEO_ENCODE_H265_CHROMA_LISTS_NUM];
    int8_t                                       delta_chroma_offset_l1[STD_VIDEO_ENCODE_H265_CHROMA_LIST_SIZE][STD_VIDEO_ENCODE_H265_CHROMA_LISTS_NUM];
    uint8_t                                      MaxNumMergeCand;
    int8_t                                       slice_qp_delta;
    int8_t                                       slice_cb_qp_offset;
    int8_t                                       slice_cr_qp_offset;
    int8_t                                       slice_beta_offset_div2;
    int8_t                                       slice_tc_offset_div2;
    int8_t                                       slice_act_y_qp_offset;
    int8_t                                       slice_act_cb_qp_offset;
    int8_t                                       slice_act_cr_qp_offset;
    StdVideoEncodeH265SliceSegmentHeaderFlags    flags;
} StdVideoEncodeH265SliceSegmentHeader;

typedef struct StdVideoEncodeH265ReferenceModificationFlags {
    uint32_t    ref_pic_list_modification_flag_l0 : 1;
    uint32_t    ref_pic_list_modification_flag_l1 : 1;
} StdVideoEncodeH265ReferenceModificationFlags;

typedef struct StdVideoEncodeH265ReferenceModifications {
    StdVideoEncodeH265ReferenceModificationFlags    flags;
    uint8_t                                         referenceList0ModificationsCount;
    uint8_t*                                        pReferenceList0Modifications;
    uint8_t                                         referenceList1ModificationsCount;
    uint8_t*                                        pReferenceList1Modifications;
} StdVideoEncodeH265ReferenceModifications;

typedef struct StdVideoEncodeH265PictureInfoFlags {
    uint32_t    is_reference_flag : 1;
    uint32_t    IrapPicFlag : 1;
    uint32_t    long_term_flag : 1;
} StdVideoEncodeH265PictureInfoFlags;

typedef struct StdVideoEncodeH265PictureInfo {
    StdVideoH265PictureType               PictureType;
    uint8_t                               sps_video_parameter_set_id;
    uint8_t                               pps_seq_parameter_set_id;
    int32_t                               PicOrderCntVal;
    uint8_t                               TemporalId;
    StdVideoEncodeH265PictureInfoFlags    flags;
} StdVideoEncodeH265PictureInfo;

typedef struct StdVideoEncodeH265ReferenceInfoFlags {
    uint32_t    is_long_term : 1;
    uint32_t    isUsedFlag : 1;
} StdVideoEncodeH265ReferenceInfoFlags;

typedef struct StdVideoEncodeH265ReferenceInfo {
    int32_t                                 PicOrderCntVal;
    uint8_t                                 TemporalId;
    StdVideoEncodeH265ReferenceInfoFlags    flags;
} StdVideoEncodeH265ReferenceInfo;


#ifdef __cplusplus
}
#endif

#endif
