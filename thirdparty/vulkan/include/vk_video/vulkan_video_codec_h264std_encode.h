#ifndef VULKAN_VIDEO_CODEC_H264STD_ENCODE_H_
#define VULKAN_VIDEO_CODEC_H264STD_ENCODE_H_ 1

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



#define vulkan_video_codec_h264std_encode 1
// Vulkan 0.9 provisional Vulkan video H.264 encode std specification version number
#define VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_API_VERSION_0_9_9 VK_MAKE_VIDEO_STD_VERSION(0, 9, 9)

#define VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_SPEC_VERSION VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_API_VERSION_0_9_9
#define VK_STD_VULKAN_VIDEO_CODEC_H264_ENCODE_EXTENSION_NAME "VK_STD_vulkan_video_codec_h264_encode"
typedef struct StdVideoEncodeH264WeightTableFlags {
    uint32_t    luma_weight_l0_flag;
    uint32_t    chroma_weight_l0_flag;
    uint32_t    luma_weight_l1_flag;
    uint32_t    chroma_weight_l1_flag;
} StdVideoEncodeH264WeightTableFlags;

typedef struct StdVideoEncodeH264WeightTable {
    StdVideoEncodeH264WeightTableFlags    flags;
    uint8_t                               luma_log2_weight_denom;
    uint8_t                               chroma_log2_weight_denom;
    int8_t                                luma_weight_l0[STD_VIDEO_H264_MAX_NUM_LIST_REF];
    int8_t                                luma_offset_l0[STD_VIDEO_H264_MAX_NUM_LIST_REF];
    int8_t                                chroma_weight_l0[STD_VIDEO_H264_MAX_NUM_LIST_REF][STD_VIDEO_H264_MAX_CHROMA_PLANES];
    int8_t                                chroma_offset_l0[STD_VIDEO_H264_MAX_NUM_LIST_REF][STD_VIDEO_H264_MAX_CHROMA_PLANES];
    int8_t                                luma_weight_l1[STD_VIDEO_H264_MAX_NUM_LIST_REF];
    int8_t                                luma_offset_l1[STD_VIDEO_H264_MAX_NUM_LIST_REF];
    int8_t                                chroma_weight_l1[STD_VIDEO_H264_MAX_NUM_LIST_REF][STD_VIDEO_H264_MAX_CHROMA_PLANES];
    int8_t                                chroma_offset_l1[STD_VIDEO_H264_MAX_NUM_LIST_REF][STD_VIDEO_H264_MAX_CHROMA_PLANES];
} StdVideoEncodeH264WeightTable;

typedef struct StdVideoEncodeH264SliceHeaderFlags {
    uint32_t    direct_spatial_mv_pred_flag : 1;
    uint32_t    num_ref_idx_active_override_flag : 1;
    uint32_t    no_output_of_prior_pics_flag : 1;
    uint32_t    adaptive_ref_pic_marking_mode_flag : 1;
    uint32_t    no_prior_references_available_flag : 1;
} StdVideoEncodeH264SliceHeaderFlags;

typedef struct StdVideoEncodeH264PictureInfoFlags {
    uint32_t    idr_flag : 1;
    uint32_t    is_reference_flag : 1;
    uint32_t    used_for_long_term_reference : 1;
} StdVideoEncodeH264PictureInfoFlags;

typedef struct StdVideoEncodeH264ReferenceInfoFlags {
    uint32_t    used_for_long_term_reference : 1;
} StdVideoEncodeH264ReferenceInfoFlags;

typedef struct StdVideoEncodeH264ReferenceListsInfoFlags {
    uint32_t    ref_pic_list_modification_flag_l0 : 1;
    uint32_t    ref_pic_list_modification_flag_l1 : 1;
} StdVideoEncodeH264ReferenceListsInfoFlags;

typedef struct StdVideoEncodeH264RefListModEntry {
    StdVideoH264ModificationOfPicNumsIdc    modification_of_pic_nums_idc;
    uint16_t                                abs_diff_pic_num_minus1;
    uint16_t                                long_term_pic_num;
} StdVideoEncodeH264RefListModEntry;

typedef struct StdVideoEncodeH264RefPicMarkingEntry {
    StdVideoH264MemMgmtControlOp    operation;
    uint16_t                        difference_of_pic_nums_minus1;
    uint16_t                        long_term_pic_num;
    uint16_t                        long_term_frame_idx;
    uint16_t                        max_long_term_frame_idx_plus1;
} StdVideoEncodeH264RefPicMarkingEntry;

typedef struct StdVideoEncodeH264ReferenceListsInfo {
    StdVideoEncodeH264ReferenceListsInfoFlags      flags;
    uint8_t                                        refPicList0EntryCount;
    uint8_t                                        refPicList1EntryCount;
    uint8_t                                        refList0ModOpCount;
    uint8_t                                        refList1ModOpCount;
    uint8_t                                        refPicMarkingOpCount;
    uint8_t                                        reserved1[7];
    const uint8_t*                                 pRefPicList0Entries;
    const uint8_t*                                 pRefPicList1Entries;
    const StdVideoEncodeH264RefListModEntry*       pRefList0ModOperations;
    const StdVideoEncodeH264RefListModEntry*       pRefList1ModOperations;
    const StdVideoEncodeH264RefPicMarkingEntry*    pRefPicMarkingOperations;
} StdVideoEncodeH264ReferenceListsInfo;

typedef struct StdVideoEncodeH264PictureInfo {
    StdVideoEncodeH264PictureInfoFlags    flags;
    uint8_t                               seq_parameter_set_id;
    uint8_t                               pic_parameter_set_id;
    uint16_t                              reserved1;
    StdVideoH264PictureType               pictureType;
    uint32_t                              frame_num;
    int32_t                               PicOrderCnt;
} StdVideoEncodeH264PictureInfo;

typedef struct StdVideoEncodeH264ReferenceInfo {
    StdVideoEncodeH264ReferenceInfoFlags    flags;
    StdVideoH264PictureType                 pictureType;
    uint32_t                                FrameNum;
    int32_t                                 PicOrderCnt;
    uint16_t                                long_term_pic_num;
    uint16_t                                long_term_frame_idx;
} StdVideoEncodeH264ReferenceInfo;

typedef struct StdVideoEncodeH264SliceHeader {
    StdVideoEncodeH264SliceHeaderFlags        flags;
    uint32_t                                  first_mb_in_slice;
    StdVideoH264SliceType                     slice_type;
    uint16_t                                  idr_pic_id;
    uint8_t                                   num_ref_idx_l0_active_minus1;
    uint8_t                                   num_ref_idx_l1_active_minus1;
    StdVideoH264CabacInitIdc                  cabac_init_idc;
    StdVideoH264DisableDeblockingFilterIdc    disable_deblocking_filter_idc;
    int8_t                                    slice_alpha_c0_offset_div2;
    int8_t                                    slice_beta_offset_div2;
    uint16_t                                  reserved1;
    uint32_t                                  reserved2;
    const StdVideoEncodeH264WeightTable*      pWeightTable;
} StdVideoEncodeH264SliceHeader;


#ifdef __cplusplus
}
#endif

#endif
