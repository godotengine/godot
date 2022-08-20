#ifndef VULKAN_VIDEO_CODEC_H264STD_DECODE_H_
#define VULKAN_VIDEO_CODEC_H264STD_DECODE_H_ 1

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



#define vulkan_video_codec_h264std_decode 1
#define STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_LIST_SIZE 2
#define STD_VIDEO_DECODE_H264_MVC_REF_LIST_SIZE 15

typedef enum StdVideoDecodeH264FieldOrderCount {
    STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_TOP = 0,
    STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_BOTTOM = 1,
    STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_INVALID = 0x7FFFFFFF,
    STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_MAX_ENUM = 0x7FFFFFFF
} StdVideoDecodeH264FieldOrderCount;
typedef struct StdVideoDecodeH264PictureInfoFlags {
    uint32_t    field_pic_flag : 1;
    uint32_t    is_intra : 1;
    uint32_t    IdrPicFlag : 1;
    uint32_t    bottom_field_flag : 1;
    uint32_t    is_reference : 1;
    uint32_t    complementary_field_pair : 1;
} StdVideoDecodeH264PictureInfoFlags;

typedef struct StdVideoDecodeH264PictureInfo {
    uint8_t                               seq_parameter_set_id;
    uint8_t                               pic_parameter_set_id;
    uint16_t                              reserved;
    uint16_t                              frame_num;
    uint16_t                              idr_pic_id;
    int32_t                               PicOrderCnt[STD_VIDEO_DECODE_H264_FIELD_ORDER_COUNT_LIST_SIZE];
    StdVideoDecodeH264PictureInfoFlags    flags;
} StdVideoDecodeH264PictureInfo;

typedef struct StdVideoDecodeH264ReferenceInfoFlags {
    uint32_t    top_field_flag : 1;
    uint32_t    bottom_field_flag : 1;
    uint32_t    is_long_term : 1;
    uint32_t    is_non_existing : 1;
} StdVideoDecodeH264ReferenceInfoFlags;

typedef struct StdVideoDecodeH264ReferenceInfo {
    uint16_t                                FrameNum;
    uint16_t                                reserved;
    int32_t                                 PicOrderCnt[2];
    StdVideoDecodeH264ReferenceInfoFlags    flags;
} StdVideoDecodeH264ReferenceInfo;

typedef struct StdVideoDecodeH264MvcElementFlags {
    uint32_t    non_idr : 1;
    uint32_t    anchor_pic : 1;
    uint32_t    inter_view : 1;
} StdVideoDecodeH264MvcElementFlags;

typedef struct StdVideoDecodeH264MvcElement {
    StdVideoDecodeH264MvcElementFlags    flags;
    uint16_t                             viewOrderIndex;
    uint16_t                             viewId;
    uint16_t                             temporalId;
    uint16_t                             priorityId;
    uint16_t                             numOfAnchorRefsInL0;
    uint16_t                             viewIdOfAnchorRefsInL0[STD_VIDEO_DECODE_H264_MVC_REF_LIST_SIZE];
    uint16_t                             numOfAnchorRefsInL1;
    uint16_t                             viewIdOfAnchorRefsInL1[STD_VIDEO_DECODE_H264_MVC_REF_LIST_SIZE];
    uint16_t                             numOfNonAnchorRefsInL0;
    uint16_t                             viewIdOfNonAnchorRefsInL0[STD_VIDEO_DECODE_H264_MVC_REF_LIST_SIZE];
    uint16_t                             numOfNonAnchorRefsInL1;
    uint16_t                             viewIdOfNonAnchorRefsInL1[STD_VIDEO_DECODE_H264_MVC_REF_LIST_SIZE];
} StdVideoDecodeH264MvcElement;

typedef struct StdVideoDecodeH264Mvc {
    uint32_t                         viewId0;
    uint32_t                         mvcElementCount;
    StdVideoDecodeH264MvcElement*    pMvcElements;
} StdVideoDecodeH264Mvc;


#ifdef __cplusplus
}
#endif

#endif
