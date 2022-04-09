#ifndef VULKAN_VIDEO_CODEC_H265STD_DECODE_H_
#define VULKAN_VIDEO_CODEC_H265STD_DECODE_H_ 1

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



#define vulkan_video_codec_h265std_decode 1
#define STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE 8
typedef struct StdVideoDecodeH265PictureInfoFlags {
    uint32_t    IrapPicFlag : 1;
    uint32_t    IdrPicFlag  : 1;
    uint32_t    IsReference : 1;
    uint32_t    short_term_ref_pic_set_sps_flag : 1;
} StdVideoDecodeH265PictureInfoFlags;

typedef struct StdVideoDecodeH265PictureInfo {
    uint8_t                               vps_video_parameter_set_id;
    uint8_t                               sps_seq_parameter_set_id;
    uint8_t                               pps_pic_parameter_set_id;
    uint8_t                               num_short_term_ref_pic_sets;
    int32_t                               PicOrderCntVal;
    uint16_t                              NumBitsForSTRefPicSetInSlice;
    uint8_t                               NumDeltaPocsOfRefRpsIdx;
    uint8_t                               RefPicSetStCurrBefore[STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE];
    uint8_t                               RefPicSetStCurrAfter[STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE];
    uint8_t                               RefPicSetLtCurr[STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE];
    StdVideoDecodeH265PictureInfoFlags    flags;
} StdVideoDecodeH265PictureInfo;

typedef struct StdVideoDecodeH265ReferenceInfoFlags {
    uint32_t    is_long_term : 1;
    uint32_t    is_non_existing : 1;
} StdVideoDecodeH265ReferenceInfoFlags;

typedef struct StdVideoDecodeH265ReferenceInfo {
    int32_t                                 PicOrderCntVal;
    StdVideoDecodeH265ReferenceInfoFlags    flags;
} StdVideoDecodeH265ReferenceInfo;


#ifdef __cplusplus
}
#endif

#endif
