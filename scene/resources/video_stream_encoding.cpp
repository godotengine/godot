/**************************************************************************/
/*  video_stream_encoding.cpp                                             */
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

#include "video_stream_encoding.h"
#include "modules/matroska/ycbcr_sampler.glsl.gen.h"
#include "servers/rendering/rendering_device_binds.h"

void VideoStreamEncoding::_yuv_to_rgba(RID p_src_yuv, RID p_dst_rgba) {
	Vector<RD::Uniform> uniforms;
	RD::Uniform src_texture = {};
	src_texture.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	src_texture.binding = 0;
	src_texture.append_id(yuv_sampler);
	src_texture.append_id(p_src_yuv);
	uniforms.push_back(src_texture);

	RD::Uniform dst_texture = {};
	dst_texture.uniform_type = RD::UNIFORM_TYPE_IMAGE;
	dst_texture.binding = 1;
	dst_texture.append_id(p_dst_rgba);
	uniforms.push_back(dst_texture);

	RID uniform_set = local_device->uniform_set_create(uniforms, yuv_shader, 0);

	// TODO: use work groups better
	RD::ComputeListID compute_list = local_device->compute_list_begin();
	local_device->compute_list_bind_compute_pipeline(compute_list, yuv_pipeline);
	local_device->compute_list_bind_uniform_set(compute_list, uniform_set, 0);
	local_device->compute_list_dispatch(compute_list, 1920, 1080, 1);
	local_device->compute_list_end();
	local_device->submit();
	local_device->sync();

	local_device->free_rid(uniform_set);
}

void VideoStreamEncoding::initialize(RD::VideoSessionInfo p_session_template, RD::SamplerState p_sampler_template, RD::TextureFormat p_texture_template) {
	const size_t yuv_pool_size = 15;
	const size_t rgba_pool_size = 15;

	local_device = RD::get_singleton()->create_local_device();

	yuv_sampler = _create_texture_sampler(p_sampler_template);

	RD::TextureFormat yuv_format;
	yuv_format.format = RD::DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	yuv_format.width = p_session_template.width;
	yuv_format.height = p_session_template.width;
	yuv_format.depth = 1;
	yuv_format.usage_bits = RD::TEXTURE_USAGE_VIDEO_DECODE_DST_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

	RD::TextureView yuv_view;
	yuv_view.ycbcr_sampler = yuv_sampler;

	for (size_t i = 0; i < yuv_pool_size; i++) {
		RID yuv_texture = _create_texture(yuv_format, yuv_view);
		yuv_pool.push_back(yuv_texture);
	}

	for (size_t i = 0; i < rgba_pool_size; i++) {
		RID rgba_texture = local_device->texture_create(p_texture_template, RD::TextureView());
		rgba_pool.push_back(rgba_texture);
	}

	p_session_template.dst_yuv_textures = yuv_pool;
	video_session = _create_video_session(p_session_template);

	Vector<RD::Uniform> uniforms;
	RD::Uniform immutable_sampler;
	immutable_sampler.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	immutable_sampler.binding = 0;
	immutable_sampler.immutable_sampler = true;
	immutable_sampler.append_id(yuv_sampler);
	immutable_sampler.append_id(rgba_pool[0]);
	uniforms.push_back(immutable_sampler);

	Ref<RDShaderFile> yuv_shader_src;
	yuv_shader_src.instantiate();
	yuv_shader_src->parse_versions_from_text(ycbcr_sampler_shader_glsl);

	Vector<RD::ShaderStageSPIRVData> yuv_spirv = yuv_shader_src->get_spirv_stages();
	Vector<uint8_t> yuv_bytecode = local_device->shader_compile_binary_from_spirv(yuv_spirv);

	yuv_shader = local_device->shader_create_placeholder();
	yuv_shader = local_device->shader_create_from_bytecode_with_samplers(yuv_bytecode, yuv_shader, uniforms);
	yuv_pipeline = local_device->compute_pipeline_create(yuv_shader);
}

VideoStreamEncoding::~VideoStreamEncoding() {
	if (local_device != nullptr) {
		if (yuv_shader.is_valid()) {
			local_device->free_rid(yuv_shader);
		}

		if (yuv_sampler.is_valid()) {
			local_device->free_rid(yuv_sampler);
		}

		for (RID yuv_texture : yuv_pool) {
			if (yuv_texture.is_valid()) {
				local_device->free_rid(yuv_texture);
			}
		}

		for (RID rgba_texture : rgba_pool) {
			if (rgba_texture.is_valid()) {
				local_device->free_rid(rgba_texture);
			}
		}

		if (video_session.is_valid()) {
			local_device->free_rid(video_session);
		}

		// TODO: why can we not delete this?
		//memdelete(local_device);
	}
}
