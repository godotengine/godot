#ifndef VULKAN_VIDEO_CODEC_H264STD_ENCODE_H_
#define VULKAN_VIDEO_CODEC_H264STD_ENCODE_H_ 1

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



#define vulkan_video_codec_h264std_encode 1
typedef struct StdVideoEncodeH264SliceHeaderFlags {
    uint32_t    idr_flag : 1;
    uint32_t    is_reference_flag : 1;
    uint32_t    num_ref_idx_active_override_flag : 1;
    uint32_t    no_output_of_prior_pics_flag : 1;
    uint32_t    long_term_reference_flag : 1;
    uint32_t    adaptive_ref_pic_marking_mode_flag : 1;
    uint32_t    no_prior_references_available_flag : 1;
} StdVideoEncodeH264SliceHeaderFlags;

typedef struct StdVideoEncodeH264PictureInfoFlags {
    uint32_t    idr_flag : 1;
    uint32_t    is_reference_flag : 1;
    uint32_t    long_term_reference_flag : 1;
} StdVideoEncodeH264PictureInfoFlags;

typedef struct StdVideoEncodeH264RefMgmtFlags {
    uint32_t    ref_pic_list_modification_l0_flag : 1;
    uint32_t    ref_pic_list_modification_l1_flag : 1;
} StdVideoEncodeH264RefMgmtFlags;

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

typedef struct StdVideoEncodeH264RefMemMgmtCtrlOperations {
    StdVideoEncodeH264RefMgmtFlags           flags;
    uint8_t                                  refList0ModOpCount;
    StdVideoEncodeH264RefListModEntry*       pRefList0ModOperations;
    uint8_t                                  refList1ModOpCount;
    StdVideoEncodeH264RefListModEntry*       pRefList1ModOperations;
    uint8_t                                  refPicMarkingOpCount;
    StdVideoEncodeH264RefPicMarkingEntry*    pRefPicMarkingOperations;
} StdVideoEncodeH264RefMemMgmtCtrlOperations;

typedef struct StdVideoEncodeH264PictureInfo {
    StdVideoEncodeH264PictureInfoFlags    flags;
    StdVideoH264PictureType               pictureType;
    uint32_t                              frameNum;
    uint32_t                              pictureOrderCount;
    uint16_t                              long_term_pic_num;
    uint16_t                              long_term_frame_idx;
} StdVideoEncodeH264PictureInfo;

typedef struct StdVideoEncodeH264SliceHeader {
    StdVideoEncodeH264SliceHeaderFlags             flags;
    StdVideoH264SliceType                          slice_type;
    uint8_t                                        seq_parameter_set_id;
    uint8_t                                        pic_parameter_set_id;
    uint16_t                                       idr_pic_id;
    uint8_t                                        num_ref_idx_l0_active_minus1;
    uint8_t                                        num_ref_idx_l1_active_minus1;
    StdVideoH264CabacInitIdc                       cabac_init_idc;
    StdVideoH264DisableDeblockingFilterIdc         disable_deblocking_filter_idc;
    int8_t                                         slice_alpha_c0_offset_div2;
    int8_t                                         slice_beta_offset_div2;
    StdVideoEncodeH264RefMemMgmtCtrlOperations*    pMemMgmtCtrlOperations;
} StdVideoEncodeH264SliceHeader;


#ifdef __cplusplus
}
#endif

#endif
