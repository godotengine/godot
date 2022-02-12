/*
** Copyright (c) 2019-2021 The Khronos Group Inc.
**
** SPDX-License-Identifier: Apache-2.0
*/

#ifndef VULKAN_VIDEO_CODEC_H265STD_DECODE_H_
#define VULKAN_VIDEO_CODEC_H265STD_DECODE_H_ 1

#ifdef __cplusplus
extern "C" {
#endif

#include "vk_video/vulkan_video_codec_h265std.h"

// *************************************************
// Video h265 Decode related parameters:
// *************************************************

typedef struct StdVideoDecodeH265PictureInfoFlags {
    uint32_t IrapPicFlag : 1;
    uint32_t IdrPicFlag  : 1;
    uint32_t IsReference : 1;
    uint32_t short_term_ref_pic_set_sps_flag : 1;
} StdVideoDecodeH265PictureInfoFlags;

typedef struct StdVideoDecodeH265PictureInfo {
    uint8_t                            vps_video_parameter_set_id;
    uint8_t                            sps_seq_parameter_set_id;
    uint8_t                            pps_pic_parameter_set_id;
    uint8_t                            num_short_term_ref_pic_sets;
    int32_t                            PicOrderCntVal;
    uint16_t                           NumBitsForSTRefPicSetInSlice; // number of bits used in st_ref_pic_set()
                                                                     //when short_term_ref_pic_set_sps_flag is 0; otherwise set to 0.
    uint8_t                            NumDeltaPocsOfRefRpsIdx;      // NumDeltaPocs[ RefRpsIdx ] when short_term_ref_pic_set_sps_flag = 1, otherwise 0
    uint8_t                            RefPicSetStCurrBefore[8];     // slotIndex as used in VkVideoReferenceSlotKHR structures representing
                                                                     //pReferenceSlots in VkVideoDecodeInfoKHR, 0xff for invalid slotIndex
    uint8_t                            RefPicSetStCurrAfter[8];      // slotIndex as used in VkVideoReferenceSlotKHR structures representing
                                                                     //pReferenceSlots in VkVideoDecodeInfoKHR, 0xff for invalid slotIndex
    uint8_t                            RefPicSetLtCurr[8];           // slotIndex as used in VkVideoReferenceSlotKHR structures representing
                                                                     //pReferenceSlots in VkVideoDecodeInfoKHR, 0xff for invalid slotIndex
    StdVideoDecodeH265PictureInfoFlags flags;
} StdVideoDecodeH265PictureInfo;

typedef struct StdVideoDecodeH265ReferenceInfoFlags {
    uint32_t is_long_term : 1;
    uint32_t is_non_existing : 1;
} StdVideoDecodeH265ReferenceInfoFlags;

typedef struct StdVideoDecodeH265ReferenceInfo {
    int32_t                              PicOrderCntVal;
    StdVideoDecodeH265ReferenceInfoFlags flags;
} StdVideoDecodeH265ReferenceInfo;

#ifdef __cplusplus
}
#endif

#endif // VULKAN_VIDEO_CODEC_H265STD_DECODE_H_
