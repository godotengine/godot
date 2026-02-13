#ifndef VULKAN_VIDEO_CODEC_VP9STD_DECODE_H_
#define VULKAN_VIDEO_CODEC_VP9STD_DECODE_H_ 1

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



// vulkan_video_codec_vp9std_decode is a preprocessor guard. Do not pass it to API calls.
#define vulkan_video_codec_vp9std_decode 1
#include "vulkan_video_codec_vp9std.h"

#define VK_STD_VULKAN_VIDEO_CODEC_VP9_DECODE_API_VERSION_1_0_0 VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)

#define VK_STD_VULKAN_VIDEO_CODEC_VP9_DECODE_SPEC_VERSION VK_STD_VULKAN_VIDEO_CODEC_VP9_DECODE_API_VERSION_1_0_0
#define VK_STD_VULKAN_VIDEO_CODEC_VP9_DECODE_EXTENSION_NAME "VK_STD_vulkan_video_codec_vp9_decode"
typedef struct StdVideoDecodeVP9PictureInfoFlags {
    uint32_t    error_resilient_mode : 1;
    uint32_t    intra_only : 1;
    uint32_t    allow_high_precision_mv : 1;
    uint32_t    refresh_frame_context : 1;
    uint32_t    frame_parallel_decoding_mode : 1;
    uint32_t    segmentation_enabled : 1;
    uint32_t    show_frame : 1;
    uint32_t    UsePrevFrameMvs : 1;
    uint32_t    reserved : 24;
} StdVideoDecodeVP9PictureInfoFlags;

typedef struct StdVideoDecodeVP9PictureInfo {
    StdVideoDecodeVP9PictureInfoFlags    flags;
    StdVideoVP9Profile                   profile;
    StdVideoVP9FrameType                 frame_type;
    uint8_t                              frame_context_idx;
    uint8_t                              reset_frame_context;
    uint8_t                              refresh_frame_flags;
    uint8_t                              ref_frame_sign_bias_mask;
    StdVideoVP9InterpolationFilter       interpolation_filter;
    uint8_t                              base_q_idx;
    int8_t                               delta_q_y_dc;
    int8_t                               delta_q_uv_dc;
    int8_t                               delta_q_uv_ac;
    uint8_t                              tile_cols_log2;
    uint8_t                              tile_rows_log2;
    uint16_t                             reserved1[3];
    const StdVideoVP9ColorConfig*        pColorConfig;
    const StdVideoVP9LoopFilter*         pLoopFilter;
    const StdVideoVP9Segmentation*       pSegmentation;
} StdVideoDecodeVP9PictureInfo;


#ifdef __cplusplus
}
#endif

#endif
