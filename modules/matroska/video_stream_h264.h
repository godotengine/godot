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

#include "servers/rendering/rendering_device.h"
#include <vk_video/vulkan_video_codec_h264std_decode.h>

class VideoStreamH264 : public VideoStreamEncoding {
	GDCLASS(VideoStreamH264, VideoStreamEncoding);

private:
	const uint8_t *src = nullptr;
	uint8_t shift = 7;

	uint64_t prev_pic_order_cnt_lsb;
	uint64_t prev_pic_order_cnt_msb;

	RD::VideoCodingH264ProfileIdc target_profile_idc = RenderingDeviceCommons::VIDEO_CODING_H264_PROFILE_IDC_HIGH;
	RD::VideoCodingH264ProfileIdc minimum_profile_idc;

	// TODO make an RD version
	uint32_t target_level_idc;

	RD::VideoProfile video_profile = {};

	RD::VideoCodingH264SequenceParameterSet active_sps;
	Vector<RD::VideoCodingH264SequenceParameterSet> sps_sets;
	Vector<RD::VideoCodingH264PictureParameterSet> pps_sets;

	RID video_session;

	// TODO: use a pool of dst textures
	RID dst_texture;
	RID dpb_texture;

	uint8_t target_dpb_layer = 0;
	uint8_t target_dst_layer = 0;

public:
	RID create_video_session(uint32_t p_width, uint32_t p_height) final override;

	void parse_container_metadata(const uint8_t *p_stream, uint64_t p_size) final override;

	virtual void begin_cluster() final override;
	virtual void append_container_block(Vector<uint8_t> p_block) final override;
	virtual RID end_cluster() final override;

	void parse_nal_unit(uint64_t p_size, bool p_is_metadata);
	RD::VideoCodingH264SequenceParameterSet parse_sequence_parameter_set(uint64_t p_size);
	RD::VideoCodingH264PictureParameterSet parse_picture_parameter_set(uint64_t p_size);
	StdVideoDecodeH264PictureInfo parse_slice_header(uint64_t p_size, bool p_is_idr);

	uint64_t read_bits(uint8_t p_amount);
	uint64_t read_ue();
	int64_t read_se();
};
