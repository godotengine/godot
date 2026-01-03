/**************************************************************************/
/*  video_stream_encoding.h                                               */
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

#include "core/io/resource.h"
#include "servers/rendering/rendering_device.h"

class VideoStreamEncoding : public Resource {
	GDCLASS(VideoStreamEncoding, Resource);

public:
	// Video Metadata.
	virtual void parse_container_metadata(const uint8_t *p_stream, uint64_t p_size) = 0;

	// Rendering Device API.
	virtual void set_rendering_device(RenderingDevice *p_coding_device) = 0;
	virtual RID create_video_session(uint32_t p_width, uint32_t p_height) = 0;
	virtual RID create_texture_sampler(RD::SamplerState &p_sampler_template) = 0;
	virtual RID create_texture(RD::TextureFormat &p_texture_template) = 0;

	// Block decoding.
	virtual void parse_container_block(Vector<uint8_t> p_block, RID p_dst_texture) = 0;
};
