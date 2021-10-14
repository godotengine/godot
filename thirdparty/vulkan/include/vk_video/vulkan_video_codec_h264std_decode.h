/*
** Copyright (c) 2019-2020 The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

#ifndef VULKAN_VIDEO_CODEC_H264STD_DECODE_H_
#define VULKAN_VIDEO_CODEC_H264STD_DECODE_H_ 1

#ifdef __cplusplus
extern "C" {
#endif

#include "vk_video/vulkan_video_codec_h264std.h"

// *************************************************
// Video H.264 Decode related parameters:
// *************************************************

typedef struct StdVideoDecodeH264PictureInfoFlags {
    uint32_t field_pic_flag:1;             // Is field picture
    uint32_t is_intra:1;                   // Is intra picture
    uint32_t bottom_field_flag:1;          // bottom (true) or top (false) field if field_pic_flag is set.
    uint32_t is_reference:1;               // This only applies to picture info, and not to the DPB lists.
    uint32_t complementary_field_pair:1;   // complementary field pair, complementary non-reference field pair, complementary reference field pair
} StdVideoDecodeH264PictureInfoFlags;

typedef struct StdVideoDecodeH264PictureInfo {
    uint8_t  seq_parameter_set_id;          // Selecting SPS from the Picture Parameters
    uint8_t  pic_parameter_set_id;          // Selecting PPS from the Picture Parameters and the SPS
    uint16_t reserved;                      // for structure members 32-bit packing/alignment
    uint16_t frame_num;                     // 7.4.3 Slice header semantics
    uint16_t idr_pic_id;                    // 7.4.3 Slice header semantics
    // PicOrderCnt is based on TopFieldOrderCnt and BottomFieldOrderCnt. See 8.2.1 Decoding process for picture order count type 0 - 2
    int32_t  PicOrderCnt[2];                // TopFieldOrderCnt and BottomFieldOrderCnt fields.
    StdVideoDecodeH264PictureInfoFlags flags;
} StdVideoDecodeH264PictureInfo;

typedef struct StdVideoDecodeH264ReferenceInfoFlags {
    uint32_t top_field_flag:1;             // Reference is used for top field reference.
    uint32_t bottom_field_flag:1;          // Reference is used for bottom field reference.
    uint32_t is_long_term:1;               // this is a long term reference
    uint32_t is_non_existing:1;            // Must be handled in accordance with 8.2.5.2: Decoding process for gaps in frame_num
} StdVideoDecodeH264ReferenceInfoFlags;

typedef struct StdVideoDecodeH264ReferenceInfo {
    // FrameNum = is_long_term ?  long_term_frame_idx : frame_num
    uint16_t FrameNum;                     // 7.4.3.3 Decoded reference picture marking semantics
    uint16_t reserved;                     // for structure members 32-bit packing/alignment
    int32_t  PicOrderCnt[2];               // TopFieldOrderCnt and BottomFieldOrderCnt fields.
    StdVideoDecodeH264ReferenceInfoFlags flags;
} StdVideoDecodeH264ReferenceInfo;

typedef struct StdVideoDecodeH264MvcElementFlags {
    uint32_t non_idr:1;
    uint32_t anchor_pic:1;
    uint32_t inter_view:1;
} StdVideoDecodeH264MvcElementFlags;

typedef struct StdVideoDecodeH264MvcElement {
    StdVideoDecodeH264MvcElementFlags flags;
    uint16_t viewOrderIndex;
    uint16_t viewId;
    uint16_t temporalId; // move out?
    uint16_t priorityId; // move out?
    uint16_t numOfAnchorRefsInL0;
    uint16_t viewIdOfAnchorRefsInL0[15];
    uint16_t numOfAnchorRefsInL1;
    uint16_t viewIdOfAnchorRefsInL1[15];
    uint16_t numOfNonAnchorRefsInL0;
    uint16_t viewIdOfNonAnchorRefsInL0[15];
    uint16_t numOfNonAnchorRefsInL1;
    uint16_t viewIdOfNonAnchorRefsInL1[15];
} StdVideoDecodeH264MvcElement;

typedef struct StdVideoDecodeH264Mvc {
    uint32_t viewId0;
    uint32_t mvcElementCount;
    StdVideoDecodeH264MvcElement* pMvcElements;
} StdVideoDecodeH264Mvc;


#ifdef __cplusplus
}
#endif

#endif // VULKAN_VIDEO_CODEC_H264STD_DECODE_H_
