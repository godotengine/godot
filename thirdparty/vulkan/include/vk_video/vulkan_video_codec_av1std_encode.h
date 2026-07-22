#ifndef VULKAN_VIDEO_CODEC_AV1STD_ENCODE_H_
#define VULKAN_VIDEO_CODEC_AV1STD_ENCODE_H_ 1

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



// vulkan_video_codec_av1std_encode is a preprocessor guard. Do not pass it to API calls.
#define vulkan_video_codec_av1std_encode 1
#include "vulkan_video_codec_av1std.h"

#define VK_STD_VULKAN_VIDEO_CODEC_AV1_ENCODE_API_VERSION_1_0_0 VK_MAKE_VIDEO_STD_VERSION(1, 0, 0)

#define VK_STD_VULKAN_VIDEO_CODEC_AV1_ENCODE_SPEC_VERSION VK_STD_VULKAN_VIDEO_CODEC_AV1_ENCODE_API_VERSION_1_0_0
#define VK_STD_VULKAN_VIDEO_CODEC_AV1_ENCODE_EXTENSION_NAME "VK_STD_vulkan_video_codec_av1_encode"
typedef struct StdVideoEncodeAV1DecoderModelInfo {
    uint8_t     buffer_delay_length_minus_1;
    uint8_t     buffer_removal_time_length_minus_1;
    uint8_t     frame_presentation_time_length_minus_1;
    uint8_t     reserved1;
    uint32_t    num_units_in_decoding_tick;
} StdVideoEncodeAV1DecoderModelInfo;

typedef struct StdVideoEncodeAV1ExtensionHeader {
    uint8_t    temporal_id;
    uint8_t    spatial_id;
} StdVideoEncodeAV1ExtensionHeader;

typedef struct StdVideoEncodeAV1OperatingPointInfoFlags {
    uint32_t    decoder_model_present_for_this_op : 1;
    uint32_t    low_delay_mode_flag : 1;
    uint32_t    initial_display_delay_present_for_this_op : 1;
    uint32_t    reserved : 29;
} StdVideoEncodeAV1OperatingPointInfoFlags;

typedef struct StdVideoEncodeAV1OperatingPointInfo {
    StdVideoEncodeAV1OperatingPointInfoFlags    flags;
    uint16_t                                    operating_point_idc;
    uint8_t                                     seq_level_idx;
    uint8_t                                     seq_tier;
    uint32_t                                    decoder_buffer_delay;
    uint32_t                                    encoder_buffer_delay;
    uint8_t                                     initial_display_delay_minus_1;
} StdVideoEncodeAV1OperatingPointInfo;

typedef struct StdVideoEncodeAV1PictureInfoFlags {
    uint32_t    error_resilient_mode : 1;
    uint32_t    disable_cdf_update : 1;
    uint32_t    use_superres : 1;
    uint32_t    render_and_frame_size_different : 1;
    uint32_t    allow_screen_content_tools : 1;
    uint32_t    is_filter_switchable : 1;
    uint32_t    force_integer_mv : 1;
    uint32_t    frame_size_override_flag : 1;
    uint32_t    buffer_removal_time_present_flag : 1;
    uint32_t    allow_intrabc : 1;
    uint32_t    frame_refs_short_signaling : 1;
    uint32_t    allow_high_precision_mv : 1;
    uint32_t    is_motion_mode_switchable : 1;
    uint32_t    use_ref_frame_mvs : 1;
    uint32_t    disable_frame_end_update_cdf : 1;
    uint32_t    allow_warped_motion : 1;
    uint32_t    reduced_tx_set : 1;
    uint32_t    skip_mode_present : 1;
    uint32_t    delta_q_present : 1;
    uint32_t    delta_lf_present : 1;
    uint32_t    delta_lf_multi : 1;
    uint32_t    segmentation_enabled : 1;
    uint32_t    segmentation_update_map : 1;
    uint32_t    segmentation_temporal_update : 1;
    uint32_t    segmentation_update_data : 1;
    uint32_t    UsesLr : 1;
    uint32_t    usesChromaLr : 1;
    uint32_t    show_frame : 1;
    uint32_t    showable_frame : 1;
    uint32_t    reserved : 3;
} StdVideoEncodeAV1PictureInfoFlags;

typedef struct StdVideoEncodeAV1PictureInfo {
    StdVideoEncodeAV1PictureInfoFlags          flags;
    StdVideoAV1FrameType                       frame_type;
    uint32_t                                   frame_presentation_time;
    uint32_t                                   current_frame_id;
    uint8_t                                    order_hint;
    uint8_t                                    primary_ref_frame;
    uint8_t                                    refresh_frame_flags;
    uint8_t                                    coded_denom;
    uint16_t                                   render_width_minus_1;
    uint16_t                                   render_height_minus_1;
    StdVideoAV1InterpolationFilter             interpolation_filter;
    StdVideoAV1TxMode                          TxMode;
    uint8_t                                    delta_q_res;
    uint8_t                                    delta_lf_res;
    uint8_t                                    ref_order_hint[STD_VIDEO_AV1_NUM_REF_FRAMES];
    int8_t                                     ref_frame_idx[STD_VIDEO_AV1_REFS_PER_FRAME];
    uint8_t                                    reserved1[3];
    uint32_t                                   delta_frame_id_minus_1[STD_VIDEO_AV1_REFS_PER_FRAME];
    const StdVideoAV1TileInfo*                 pTileInfo;
    const StdVideoAV1Quantization*             pQuantization;
    const StdVideoAV1Segmentation*             pSegmentation;
    const StdVideoAV1LoopFilter*               pLoopFilter;
    const StdVideoAV1CDEF*                     pCDEF;
    const StdVideoAV1LoopRestoration*          pLoopRestoration;
    const StdVideoAV1GlobalMotion*             pGlobalMotion;
    const StdVideoEncodeAV1ExtensionHeader*    pExtensionHeader;
    const uint32_t*                            pBufferRemovalTimes;
} StdVideoEncodeAV1PictureInfo;

typedef struct StdVideoEncodeAV1ReferenceInfoFlags {
    uint32_t    disable_frame_end_update_cdf : 1;
    uint32_t    segmentation_enabled : 1;
    uint32_t    reserved : 30;
} StdVideoEncodeAV1ReferenceInfoFlags;

typedef struct StdVideoEncodeAV1ReferenceInfo {
    StdVideoEncodeAV1ReferenceInfoFlags        flags;
    uint32_t                                   RefFrameId;
    StdVideoAV1FrameType                       frame_type;
    uint8_t                                    OrderHint;
    uint8_t                                    reserved1[3];
    const StdVideoEncodeAV1ExtensionHeader*    pExtensionHeader;
} StdVideoEncodeAV1ReferenceInfo;


#ifdef __cplusplus
}
#endif

#endif
