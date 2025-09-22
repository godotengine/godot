/**************************************************************************/
/*  video_stream_h264.h                                                   */
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

#include "scene/resources/video_stream_encoding.h"

#include <vk_video/vulkan_video_codec_h264std.h>
#include <vk_video/vulkan_video_codec_h264std_decode.h>

#include <cstdint>

class VideoStreamH264 : public VideoStreamEncoding {
	GDCLASS(VideoStreamH264, VideoStreamEncoding);

private:
	uint8_t *src = nullptr;
	uint8_t shift = 7;

	RD::VideoCodingH264ProfileIdc target_profile_idc = RenderingDeviceCommons::VIDEO_CODING_H264_PROFILE_IDC_MAIN;
	RD::VideoCodingH264ProfileIdc minimum_profile_idc;

	// TODO make an RD version
	uint32_t target_level_idc;

	// TODO make RD versions
	StdVideoH264SequenceParameterSet active_sps;
	StdVideoH264PictureParameterSet active_pps;
	Vector<StdVideoDecodeH264PictureInfo> slice_metadatas;
	Vector<Vector<uint8_t>> slice_spans;

	RID video_profile;

public:
	RID create_video_profile() final override;

	RID decode_cluster() final override;

	void parse_container_metadata(uint8_t *p_stream, uint64_t p_size) final override;
	void parse_container_block(uint8_t *p_stream, uint64_t p_size) final override;

	bool parse_nal_unit(uint64_t p_size);
	StdVideoH264SequenceParameterSet parse_sequence_parameter_set(uint64_t p_size);
	StdVideoH264PictureParameterSet parse_picture_parameter_set(uint64_t p_size);
	StdVideoDecodeH264PictureInfo parse_slice_header(uint64_t p_size, bool p_is_idr);

	uint64_t read_bits(uint8_t p_amount);
	uint64_t read_ue();
	int64_t read_se();
};
