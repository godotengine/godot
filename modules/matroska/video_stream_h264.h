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
#include "servers/rendering/video/video_coding_h264.h"

class VideoStreamH264 : public VideoStreamEncoding {
	GDCLASS(VideoStreamH264, VideoStreamEncoding);

private:
	const uint8_t *src = nullptr;
	uint8_t shift = 0;

	uint8_t length_size;

	VideoCodingH264ProfileIdc target_profile_idc = VIDEO_CODING_H264_PROFILE_IDC_HIGH;
	VideoCodingH264ProfileIdc minimum_profile_idc;

	uint32_t target_level_idc;

	Vector<VideoCodingH264SequenceParameterSet> sps_sets;
	Vector<VideoCodingH264PictureParameterSet> pps_sets;

	uint64_t prev_pic_order_cnt_lsb;
	uint64_t prev_pic_order_cnt_msb;
	uint64_t prev_frame_num_offset;
	uint64_t prev_frame_num;

	VideoProfile video_profile = {};
	RID video_session;

	RenderingDevice *coding_device;
	RID texture_sampler;

	uint64_t read_bits(uint8_t p_amount);
	uint64_t read_ue();
	int64_t read_se();

	VideoCodingH264NalUnitType parse_nal_unit(uint64_t p_size, VideoDecodeH264SliceHeader *r_h264_slice_header);
	VideoCodingH264SequenceParameterSet parse_sequence_parameter_set(uint64_t p_size);
	VideoCodingH264PictureParameterSet parse_picture_parameter_set(uint64_t p_size);
	VideoDecodeH264SliceHeader parse_slice_header(uint64_t p_size, bool p_is_reference, bool p_is_idr);

public:
	virtual void parse_container_metadata(const uint8_t *p_stream, uint64_t p_size) final override;

	virtual void set_rendering_device(RenderingDevice *p_coding_device) final override;
	virtual RID create_video_session(uint32_t p_width, uint32_t p_height) final override;
	virtual RID create_texture_sampler(RD::SamplerState &p_sampler_template) final override;
	virtual RID create_texture(RD::TextureFormat &p_texture_template) final override;

	virtual void parse_container_block(Vector<uint8_t> p_block, RID p_dst_texture) final override;

	VideoStreamH264();
};
