/**************************************************************************
 *
 * Copyright 2009 Younes Manton.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#ifndef PIPE_VIDEO_ENUMS_H
#define PIPE_VIDEO_ENUMS_H

#ifdef __cplusplus
extern "C" {
#endif

enum pipe_video_format
{
   PIPE_VIDEO_FORMAT_UNKNOWN = 0,
   PIPE_VIDEO_FORMAT_MPEG12,   /**< MPEG1, MPEG2 */
   PIPE_VIDEO_FORMAT_MPEG4,    /**< DIVX, XVID */
   PIPE_VIDEO_FORMAT_VC1,      /**< WMV */
   PIPE_VIDEO_FORMAT_MPEG4_AVC,/**< H.264 */
   PIPE_VIDEO_FORMAT_HEVC,     /**< H.265 */
   PIPE_VIDEO_FORMAT_JPEG,     /**< JPEG */
   PIPE_VIDEO_FORMAT_VP9,      /**< VP9 */
   PIPE_VIDEO_FORMAT_AV1       /**< AV1 */
};

enum pipe_video_profile
{
   PIPE_VIDEO_PROFILE_UNKNOWN,
   PIPE_VIDEO_PROFILE_MPEG1,
   PIPE_VIDEO_PROFILE_MPEG2_SIMPLE,
   PIPE_VIDEO_PROFILE_MPEG2_MAIN,
   PIPE_VIDEO_PROFILE_MPEG4_SIMPLE,
   PIPE_VIDEO_PROFILE_MPEG4_ADVANCED_SIMPLE,
   PIPE_VIDEO_PROFILE_VC1_SIMPLE,
   PIPE_VIDEO_PROFILE_VC1_MAIN,
   PIPE_VIDEO_PROFILE_VC1_ADVANCED,
   PIPE_VIDEO_PROFILE_MPEG4_AVC_BASELINE,
   PIPE_VIDEO_PROFILE_MPEG4_AVC_CONSTRAINED_BASELINE,
   PIPE_VIDEO_PROFILE_MPEG4_AVC_MAIN,
   PIPE_VIDEO_PROFILE_MPEG4_AVC_EXTENDED,
   PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH,
   PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH10,
   PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH422,
   PIPE_VIDEO_PROFILE_MPEG4_AVC_HIGH444,
   PIPE_VIDEO_PROFILE_HEVC_MAIN,
   PIPE_VIDEO_PROFILE_HEVC_MAIN_10,
   PIPE_VIDEO_PROFILE_HEVC_MAIN_STILL,
   PIPE_VIDEO_PROFILE_HEVC_MAIN_12,
   PIPE_VIDEO_PROFILE_HEVC_MAIN_444,
   PIPE_VIDEO_PROFILE_JPEG_BASELINE,
   PIPE_VIDEO_PROFILE_VP9_PROFILE0,
   PIPE_VIDEO_PROFILE_VP9_PROFILE2,
   PIPE_VIDEO_PROFILE_AV1_MAIN,
   PIPE_VIDEO_PROFILE_MAX
};

/* Video caps, can be different for each codec/profile */
enum pipe_video_cap
{
   PIPE_VIDEO_CAP_SUPPORTED = 0,
   PIPE_VIDEO_CAP_NPOT_TEXTURES = 1,
   PIPE_VIDEO_CAP_MAX_WIDTH = 2,
   PIPE_VIDEO_CAP_MAX_HEIGHT = 3,
   PIPE_VIDEO_CAP_PREFERED_FORMAT = 4,
   PIPE_VIDEO_CAP_PREFERS_INTERLACED = 5,
   PIPE_VIDEO_CAP_SUPPORTS_PROGRESSIVE = 6,
   PIPE_VIDEO_CAP_SUPPORTS_INTERLACED = 7,
   PIPE_VIDEO_CAP_MAX_LEVEL = 8,
   PIPE_VIDEO_CAP_STACKED_FRAMES = 9,
   PIPE_VIDEO_CAP_MAX_MACROBLOCKS = 10,
   PIPE_VIDEO_CAP_MAX_TEMPORAL_LAYERS = 11,
   PIPE_VIDEO_CAP_EFC_SUPPORTED = 12,
   PIPE_VIDEO_CAP_ENC_MAX_SLICES_PER_FRAME = 13,
   PIPE_VIDEO_CAP_ENC_SLICES_STRUCTURE = 14,
   PIPE_VIDEO_CAP_ENC_MAX_REFERENCES_PER_FRAME = 15,
   PIPE_VIDEO_CAP_VPP_ORIENTATION_MODES = 16,
   PIPE_VIDEO_CAP_VPP_BLEND_MODES = 17,
   PIPE_VIDEO_CAP_VPP_MAX_INPUT_WIDTH = 18,
   PIPE_VIDEO_CAP_VPP_MAX_INPUT_HEIGHT = 19,
   PIPE_VIDEO_CAP_VPP_MIN_INPUT_WIDTH = 20,
   PIPE_VIDEO_CAP_VPP_MIN_INPUT_HEIGHT = 21,
   PIPE_VIDEO_CAP_VPP_MAX_OUTPUT_WIDTH = 22,
   PIPE_VIDEO_CAP_VPP_MAX_OUTPUT_HEIGHT = 23,
   PIPE_VIDEO_CAP_VPP_MIN_OUTPUT_WIDTH = 24,
   PIPE_VIDEO_CAP_VPP_MIN_OUTPUT_HEIGHT = 25,
   PIPE_VIDEO_CAP_ENC_QUALITY_LEVEL = 26,
   /* If true, when mapping planar textures like NV12 or P016 the mapped buffer contains
   all the planes contiguously. This allows for use with some frontends functions that
   require this like vaDeriveImage */
   PIPE_VIDEO_CAP_SUPPORTS_CONTIGUOUS_PLANES_MAP = 27,
   PIPE_VIDEO_CAP_ENC_SUPPORTS_MAX_FRAME_SIZE = 28,
   PIPE_VIDEO_CAP_ENC_HEVC_BLOCK_SIZES = 29,
   PIPE_VIDEO_CAP_ENC_HEVC_FEATURE_FLAGS = 30,
   PIPE_VIDEO_CAP_ENC_HEVC_PREDICTION_DIRECTION = 31,
   /*
      If reported by the driver, then pipe_video_codec.flush(...)
      needs to be called after pipe_video_codec.end_frame(...)
      to kick off the work in the device
   */
   PIPE_VIDEO_CAP_REQUIRES_FLUSH_ON_END_FRAME = 32,

   /*
      If reported by the driver, then multiple p_video_codec encode
      operations can be asynchronously enqueued (and also flushed)
      with different feedback values in the device before get_feedback
      is called on them to synchronize. The device can block on begin_frame
      when it has reached its maximum async depth capacity
   */
   PIPE_VIDEO_CAP_ENC_SUPPORTS_ASYNC_OPERATION = 33,
};

/* To be used with PIPE_VIDEO_CAP_VPP_ORIENTATION_MODES and for VPP state*/
enum pipe_video_vpp_orientation
{
   PIPE_VIDEO_VPP_ORIENTATION_DEFAULT = 0x0,
   PIPE_VIDEO_VPP_ROTATION_90 = 0x01,
   PIPE_VIDEO_VPP_ROTATION_180 = 0x02,
   PIPE_VIDEO_VPP_ROTATION_270 = 0x04,
   PIPE_VIDEO_VPP_FLIP_HORIZONTAL = 0x08,
   PIPE_VIDEO_VPP_FLIP_VERTICAL = 0x10,
};

/* To be used with PIPE_VIDEO_CAP_VPP_BLEND_MODES and for VPP state*/
enum pipe_video_vpp_blend_mode
{
   PIPE_VIDEO_VPP_BLEND_MODE_NONE = 0x0,
   PIPE_VIDEO_VPP_BLEND_MODE_GLOBAL_ALPHA = 0x1,
};


/* To be used with cap PIPE_VIDEO_CAP_ENC_SLICES_STRUCTURE*/
/**
 * pipe_video_cap_slice_structure
 *
 * This attribute determines slice structures supported by the
 * driver for encoding. This attribute is a hint to the user so
 * that he can choose a suitable surface size and how to arrange
 * the encoding process of multiple slices per frame.
 *
 * More specifically, for H.264 encoding, this attribute
 * determines the range of accepted values to
 * h264_slice_descriptor::macroblock_address and
 * h264_slice_descriptor::num_macroblocks.
 */
enum pipe_video_cap_slice_structure
{
   /* Driver does not supports multiple slice per frame.*/
   PIPE_VIDEO_CAP_SLICE_STRUCTURE_NONE = 0x00000000,
   /* Driver supports a power-of-two number of rows per slice.*/
   PIPE_VIDEO_CAP_SLICE_STRUCTURE_POWER_OF_TWO_ROWS = 0x00000001,
   /* Driver supports an arbitrary number of macroblocks per slice.*/
   PIPE_VIDEO_CAP_SLICE_STRUCTURE_ARBITRARY_MACROBLOCKS = 0x00000002,
   /* Driver support 1 row per slice*/
   PIPE_VIDEO_CAP_SLICE_STRUCTURE_EQUAL_ROWS = 0x00000004,
   /* Driver support max encoded slice size per slice */
   PIPE_VIDEO_CAP_SLICE_STRUCTURE_MAX_SLICE_SIZE = 0x00000008,
   /* Driver supports an arbitrary number of rows per slice. */
   PIPE_VIDEO_CAP_SLICE_STRUCTURE_ARBITRARY_ROWS = 0x00000010,
   /* Driver supports any number of rows per slice but they must be the same
   *  for all slices except for the last one, which must be equal or smaller
   *  to the previous slices. */
   PIPE_VIDEO_CAP_SLICE_STRUCTURE_EQUAL_MULTI_ROWS = 0x00000020,
};


enum pipe_video_entrypoint
{
   PIPE_VIDEO_ENTRYPOINT_UNKNOWN,
   PIPE_VIDEO_ENTRYPOINT_BITSTREAM,
   PIPE_VIDEO_ENTRYPOINT_IDCT,
   PIPE_VIDEO_ENTRYPOINT_MC,
   PIPE_VIDEO_ENTRYPOINT_ENCODE,
   PIPE_VIDEO_ENTRYPOINT_PROCESSING,
};

#if defined(__cplusplus)
}
#endif

#endif /* PIPE_VIDEO_ENUMS_H */
