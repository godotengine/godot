/**************************************************************************/
/*  video_coding_av1_decode.h                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "video_coding_av1.h"

struct VideoDecodeAV1Frame {
	VideoCodingAV1FrameType frame_type;

	uint32_t current_frame_id;

	bool error_resilient_mode_flag;
	bool disable_cdf_update_flag;

	bool allow_screen_content_tools_flag;
	bool force_integer_mv_flag;

	bool frame_size_override_flag;

	uint8_t order_hint;

	uint8_t primary_ref_frame;

	bool buffer_removal_time_flag;

	bool allow_high_precision_mv;
	bool use_ref_frame_mvs;
	bool allow_intrabc;

	uint8_t refresh_frame_flags;

	uint8_t ref_order_hint[VIDEO_CODING_AV1_NUM_REF_FRAMES];

	bool use_superres;
	uint8_t coded_denom;
	bool render_and_frame_size_different;

	bool frame_refs_short_signaling;

	uint8_t expected_frame_id[VIDEO_CODING_AV1_NUM_REF_FRAMES];

	bool is_filter_switchable;
	VideoCodingAV1InterpolationFilter interpolation_filter;

	bool is_motion_mode_switchable;

	uint8_t order_hints[VIDEO_CODING_AV1_NUM_REF_FRAMES];

	bool disable_frame_end_update_cdf;

	VideoCodingAV1TileInfo tile_info;
	VideoCodingAV1Quantization quantization;

	bool segmentation_enabled;
	bool segmentation_update_map;
	bool segmentation_temporal_update;
	bool segmentation_update_data;
	VideoCodingAV1Segmentation segmentation;

	bool delta_q_present;
	uint8_t delta_q_res;

	bool delta_lf_present;
	uint8_t delta_lf_res;
	bool delta_lf_multi;

	VideoCodingAV1LoopFilter loop_filter;
	VideoCodingAV1CDEF cdef;
	VideoCodingAV1LoopRestoration loop_restoration;

	VideoCodingAV1TxMode tx_mode;

	bool allow_warped_motion;
	bool reduced_tx_set;

	VideoCodingAV1GlobalMotion global_motion;
	VideoCodingAV1FilmGrain film_grain;
};
