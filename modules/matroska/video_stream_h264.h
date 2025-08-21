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
	StdVideoH264SequenceParameterSet active_sps;
	StdVideoH264PictureParameterSet active_pps;
	Vector<StdVideoDecodeH264PictureInfo> slice_metadatas;
	Vector<Span<uint8_t>> slice_spans;

public:
	void parse_container_metadata(uint8_t *p_stream, uint64_t p_size) override;
	void parse_container_block(uint8_t *p_stream, uint64_t p_size) override;

	bool parse_nal_unit(uint8_t *p_stream);
	StdVideoH264SequenceParameterSet parse_sequence_parameter_set(uint8_t *p_stream);
	StdVideoH264PictureParameterSet parse_picture_parameter_set(uint8_t *p_stream);
	StdVideoDecodeH264PictureInfo parse_slice_header(uint8_t *p_stream);

	uint64_t parse_ue(uint8_t *p_stream, uint8_t *shift, uint8_t *read);
	int64_t parse_se(uint8_t *p_stream, uint8_t *shift, uint8_t *read);

	void cool_stuff();
};
