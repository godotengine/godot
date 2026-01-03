#ifndef VULKAN_VIDEO_CODEC_H265STD_DECODE_H_
#define VULKAN_VIDEO_CODEC_H265STD_DECODE_H_ 1

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



// vulkan_video_codec_h265std_decode is a preprocessor guard. Do not pass it to API calls.
#define vulkan_video_codec_h265std_decode 1
#include "vulkan_video_codec_h265std.h"

#define VK_STD_VULKAN_VIDEO_CODEC_H265_DECODE_API_VERSION_1_0_0 VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)

#define VK_STD_VULKAN_VIDEO_CODEC_H265_DECODE_SPEC_VERSION VK_STD_VULKAN_VIDEO_CODEC_H265_DECODE_API_VERSION_1_0_0
#define VK_STD_VULKAN_VIDEO_CODEC_H265_DECODE_EXTENSION_NAME "VK_STD_vulkan_video_codec_h265_decode"
#define STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE 8U
typedef struct StdVideoDecodeH265PictureInfoFlags {
    uint32_t    IrapPicFlag : 1;
    uint32_t    IdrPicFlag  : 1;
    uint32_t    IsReference : 1;
    uint32_t    short_term_ref_pic_set_sps_flag : 1;
} StdVideoDecodeH265PictureInfoFlags;

typedef struct StdVideoDecodeH265PictureInfo {
    StdVideoDecodeH265PictureInfoFlags    flags;
    uint8_t                               sps_video_parameter_set_id;
    uint8_t                               pps_seq_parameter_set_id;
    uint8_t                               pps_pic_parameter_set_id;
    uint8_t                               NumDeltaPocsOfRefRpsIdx;
    int32_t                               PicOrderCntVal;
    uint16_t                              NumBitsForSTRefPicSetInSlice;
    uint16_t                              reserved;
    uint8_t                               RefPicSetStCurrBefore[STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE];
    uint8_t                               RefPicSetStCurrAfter[STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE];
    uint8_t                               RefPicSetLtCurr[STD_VIDEO_DECODE_H265_REF_PIC_SET_LIST_SIZE];
} StdVideoDecodeH265PictureInfo;

typedef struct StdVideoDecodeH265ReferenceInfoFlags {
    uint32_t    used_for_long_term_reference : 1;
    uint32_t    unused_for_reference : 1;
} StdVideoDecodeH265ReferenceInfoFlags;

typedef struct StdVideoDecodeH265ReferenceInfo {
    StdVideoDecodeH265ReferenceInfoFlags    flags;
    int32_t                                 PicOrderCntVal;
} StdVideoDecodeH265ReferenceInfo;


#ifdef __cplusplus
}
#endif

#endif
