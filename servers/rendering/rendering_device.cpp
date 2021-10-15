/*************************************************************************/
/*  rendering_device.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "rendering_device.h"

#include "rendering_device_binds.h"

RenderingDevice *RenderingDevice::singleton = nullptr;

RenderingDevice *RenderingDevice::get_singleton() {
	return singleton;
}

RenderingDevice::ShaderCompileToSPIRVFunction RenderingDevice::compile_to_spirv_function = nullptr;
RenderingDevice::ShaderCacheFunction RenderingDevice::cache_function = nullptr;
RenderingDevice::ShaderSPIRVGetCacheKeyFunction RenderingDevice::get_spirv_cache_key_function = nullptr;

void RenderingDevice::shader_set_compile_to_spirv_function(ShaderCompileToSPIRVFunction p_function) {
	compile_to_spirv_function = p_function;
}

void RenderingDevice::shader_set_spirv_cache_function(ShaderCacheFunction p_function) {
	cache_function = p_function;
}

void RenderingDevice::shader_set_get_cache_key_function(ShaderSPIRVGetCacheKeyFunction p_function) {
	get_spirv_cache_key_function = p_function;
}

Vector<uint8_t> RenderingDevice::shader_compile_spirv_from_source(ShaderStage p_stage, const String &p_source_code, ShaderLanguage p_language, String *r_error, bool p_allow_cache) {
	if (p_allow_cache && cache_function) {
		Vector<uint8_t> cache = cache_function(p_stage, p_source_code, p_language);
		if (cache.size()) {
			return cache;
		}
	}

	ERR_FAIL_COND_V(!compile_to_spirv_function, Vector<uint8_t>());

	return compile_to_spirv_function(p_stage, p_source_code, p_language, r_error, &device_capabilities);
}

String RenderingDevice::shader_get_spirv_cache_key() const {
	if (get_spirv_cache_key_function) {
		return get_spirv_cache_key_function(&device_capabilities);
	}
	return String();
}

RID RenderingDevice::shader_create_from_spirv(const Vector<ShaderStageSPIRVData> &p_spirv, const String &p_shader_name) {
	Vector<uint8_t> bytecode = shader_compile_binary_from_spirv(p_spirv, p_shader_name);
	ERR_FAIL_COND_V(bytecode.size() == 0, RID());
	return shader_create_from_bytecode(bytecode);
}

RID RenderingDevice::_texture_create(const Ref<RDTextureFormat> &p_format, const Ref<RDTextureView> &p_view, const TypedArray<PackedByteArray> &p_data) {
	ERR_FAIL_COND_V(p_format.is_null(), RID());
	ERR_FAIL_COND_V(p_view.is_null(), RID());
	Vector<Vector<uint8_t>> data;
	for (int i = 0; i < p_data.size(); i++) {
		Vector<uint8_t> byte_slice = p_data[i];
		ERR_FAIL_COND_V(byte_slice.is_empty(), RID());
		data.push_back(byte_slice);
	}
	return texture_create(p_format->base, p_view->base, data);
}

RID RenderingDevice::_texture_create_shared(const Ref<RDTextureView> &p_view, RID p_with_texture) {
	ERR_FAIL_COND_V(p_view.is_null(), RID());

	return texture_create_shared(p_view->base, p_with_texture);
}

RID RenderingDevice::_texture_create_shared_from_slice(const Ref<RDTextureView> &p_view, RID p_with_texture, uint32_t p_layer, uint32_t p_mipmap, TextureSliceType p_slice_type) {
	ERR_FAIL_COND_V(p_view.is_null(), RID());

	return texture_create_shared_from_slice(p_view->base, p_with_texture, p_layer, p_mipmap, p_slice_type);
}

RenderingDevice::FramebufferFormatID RenderingDevice::_framebuffer_format_create(const TypedArray<RDAttachmentFormat> &p_attachments, uint32_t p_view_count) {
	Vector<AttachmentFormat> attachments;
	attachments.resize(p_attachments.size());

	for (int i = 0; i < p_attachments.size(); i++) {
		Ref<RDAttachmentFormat> af = p_attachments[i];
		ERR_FAIL_COND_V(af.is_null(), INVALID_FORMAT_ID);
		attachments.write[i] = af->base;
	}
	return framebuffer_format_create(attachments, p_view_count);
}

RenderingDevice::FramebufferFormatID RenderingDevice::_framebuffer_format_create_multipass(const TypedArray<RDAttachmentFormat> &p_attachments, const TypedArray<RDFramebufferPass> &p_passes, uint32_t p_view_count) {
	Vector<AttachmentFormat> attachments;
	attachments.resize(p_attachments.size());

	for (int i = 0; i < p_attachments.size(); i++) {
		Ref<RDAttachmentFormat> af = p_attachments[i];
		ERR_FAIL_COND_V(af.is_null(), INVALID_FORMAT_ID);
		attachments.write[i] = af->base;
	}

	Vector<FramebufferPass> passes;
	for (int i = 0; i < p_passes.size(); i++) {
		Ref<RDFramebufferPass> pass = p_passes[i];
		ERR_CONTINUE(pass.is_null());
		passes.push_back(pass->base);
	}

	return framebuffer_format_create_multipass(attachments, passes, p_view_count);
}

RID RenderingDevice::_framebuffer_create(const TypedArray<RID> &p_textures, FramebufferFormatID p_format_check, uint32_t p_view_count) {
	Vector<RID> textures = Variant(p_textures);
	return framebuffer_create(textures, p_format_check, p_view_count);
}

RID RenderingDevice::_framebuffer_create_multipass(const TypedArray<RID> &p_textures, const TypedArray<RDFramebufferPass> &p_passes, FramebufferFormatID p_format_check, uint32_t p_view_count) {
	Vector<RID> textures = Variant(p_textures);
	Vector<FramebufferPass> passes;
	for (int i = 0; i < p_passes.size(); i++) {
		Ref<RDFramebufferPass> pass = p_passes[i];
		ERR_CONTINUE(pass.is_null());
		passes.push_back(pass->base);
	}
	return framebuffer_create_multipass(textures, passes, p_format_check, p_view_count);
}

RID RenderingDevice::_sampler_create(const Ref<RDSamplerState> &p_state) {
	ERR_FAIL_COND_V(p_state.is_null(), RID());

	return sampler_create(p_state->base);
}

RenderingDevice::VertexFormatID RenderingDevice::_vertex_format_create(const TypedArray<RDVertexAttribute> &p_vertex_formats) {
	Vector<VertexAttribute> descriptions;
	descriptions.resize(p_vertex_formats.size());

	for (int i = 0; i < p_vertex_formats.size(); i++) {
		Ref<RDVertexAttribute> af = p_vertex_formats[i];
		ERR_FAIL_COND_V(af.is_null(), INVALID_FORMAT_ID);
		descriptions.write[i] = af->base;
	}
	return vertex_format_create(descriptions);
}

RID RenderingDevice::_vertex_array_create(uint32_t p_vertex_count, VertexFormatID p_vertex_format, const TypedArray<RID> &p_src_buffers) {
	Vector<RID> buffers = Variant(p_src_buffers);

	return vertex_array_create(p_vertex_count, p_vertex_format, buffers);
}

Ref<RDShaderSPIRV> RenderingDevice::_shader_compile_spirv_from_source(const Ref<RDShaderSource> &p_source, bool p_allow_cache) {
	ERR_FAIL_COND_V(p_source.is_null(), Ref<RDShaderSPIRV>());

	Ref<RDShaderSPIRV> bytecode;
	bytecode.instantiate();
	for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
		String error;

		ShaderStage stage = ShaderStage(i);
		Vector<uint8_t> spirv = shader_compile_spirv_from_source(stage, p_source->get_stage_source(stage), p_source->get_language(), &error, p_allow_cache);
		bytecode->set_stage_bytecode(stage, spirv);
		bytecode->set_stage_compile_error(stage, error);
	}
	return bytecode;
}

Vector<uint8_t> RenderingDevice::_shader_compile_binary_from_spirv(const Ref<RDShaderSPIRV> &p_spirv, const String &p_shader_name) {
	ERR_FAIL_COND_V(p_spirv.is_null(), Vector<uint8_t>());

	Vector<ShaderStageSPIRVData> stage_data;
	for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
		ShaderStage stage = ShaderStage(i);
		ShaderStageSPIRVData sd;
		sd.shader_stage = stage;
		String error = p_spirv->get_stage_compile_error(stage);
		ERR_FAIL_COND_V_MSG(error != String(), Vector<uint8_t>(), "Can't create a shader from an errored bytecode. Check errors in source bytecode.");
		sd.spir_v = p_spirv->get_stage_bytecode(stage);
		if (sd.spir_v.is_empty()) {
			continue;
		}
		stage_data.push_back(sd);
	}

	return shader_compile_binary_from_spirv(stage_data, p_shader_name);
}

RID RenderingDevice::_shader_create_from_spirv(const Ref<RDShaderSPIRV> &p_spirv, const String &p_shader_name) {
	ERR_FAIL_COND_V(p_spirv.is_null(), RID());

	Vector<ShaderStageSPIRVData> stage_data;
	for (int i = 0; i < RD::SHADER_STAGE_MAX; i++) {
		ShaderStage stage = ShaderStage(i);
		ShaderStageSPIRVData sd;
		sd.shader_stage = stage;
		String error = p_spirv->get_stage_compile_error(stage);
		ERR_FAIL_COND_V_MSG(error != String(), RID(), "Can't create a shader from an errored bytecode. Check errors in source bytecode.");
		sd.spir_v = p_spirv->get_stage_bytecode(stage);
		if (sd.spir_v.is_empty()) {
			continue;
		}
		stage_data.push_back(sd);
	}
	return shader_create_from_spirv(stage_data);
}

RID RenderingDevice::_uniform_set_create(const Array &p_uniforms, RID p_shader, uint32_t p_shader_set) {
	Vector<Uniform> uniforms;
	uniforms.resize(p_uniforms.size());
	for (int i = 0; i < p_uniforms.size(); i++) {
		Ref<RDUniform> uniform = p_uniforms[i];
		ERR_FAIL_COND_V(!uniform.is_valid(), RID());
		uniforms.write[i] = uniform->base;
	}
	return uniform_set_create(uniforms, p_shader, p_shader_set);
}

Error RenderingDevice::_buffer_update(RID p_buffer, uint32_t p_offset, uint32_t p_size, const Vector<uint8_t> &p_data, uint32_t p_post_barrier) {
	return buffer_update(p_buffer, p_offset, p_size, p_data.ptr(), p_post_barrier);
}

static Vector<RenderingDevice::PipelineSpecializationConstant> _get_spec_constants(const TypedArray<RDPipelineSpecializationConstant> &p_constants) {
	Vector<RenderingDevice::PipelineSpecializationConstant> ret;
	ret.resize(p_constants.size());
	for (int i = 0; i < p_constants.size(); i++) {
		Ref<RDPipelineSpecializationConstant> c = p_constants[i];
		ERR_CONTINUE(c.is_null());
		RenderingDevice::PipelineSpecializationConstant &sc = ret.write[i];
		Variant value = c->get_value();
		switch (value.get_type()) {
			case Variant::BOOL: {
				sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
				sc.bool_value = value;
			} break;
			case Variant::INT: {
				sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT;
				sc.int_value = value;
			} break;
			case Variant::FLOAT: {
				sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT;
				sc.float_value = value;
			} break;
			default: {
			}
		}

		sc.constant_id = c->get_constant_id();
	}
	return ret;
}
RID RenderingDevice::_render_pipeline_create(RID p_shader, FramebufferFormatID p_framebuffer_format, VertexFormatID p_vertex_format, RenderPrimitive p_render_primitive, const Ref<RDPipelineRasterizationState> &p_rasterization_state, const Ref<RDPipelineMultisampleState> &p_multisample_state, const Ref<RDPipelineDepthStencilState> &p_depth_stencil_state, const Ref<RDPipelineColorBlendState> &p_blend_state, int p_dynamic_state_flags, uint32_t p_for_render_pass, const TypedArray<RDPipelineSpecializationConstant> &p_specialization_constants) {
	PipelineRasterizationState rasterization_state;
	if (p_rasterization_state.is_valid()) {
		rasterization_state = p_rasterization_state->base;
	}

	PipelineMultisampleState multisample_state;
	if (p_multisample_state.is_valid()) {
		multisample_state = p_multisample_state->base;
		for (int i = 0; i < p_multisample_state->sample_masks.size(); i++) {
			int64_t mask = p_multisample_state->sample_masks[i];
			multisample_state.sample_mask.push_back(mask);
		}
	}

	PipelineDepthStencilState depth_stencil_state;
	if (p_depth_stencil_state.is_valid()) {
		depth_stencil_state = p_depth_stencil_state->base;
	}

	PipelineColorBlendState color_blend_state;
	if (p_blend_state.is_valid()) {
		color_blend_state = p_blend_state->base;
		for (int i = 0; i < p_blend_state->attachments.size(); i++) {
			Ref<RDPipelineColorBlendStateAttachment> attachment = p_blend_state->attachments[i];
			if (attachment.is_valid()) {
				color_blend_state.attachments.push_back(attachment->base);
			}
		}
	}

	return render_pipeline_create(p_shader, p_framebuffer_format, p_vertex_format, p_render_primitive, rasterization_state, multisample_state, depth_stencil_state, color_blend_state, p_dynamic_state_flags, p_for_render_pass, _get_spec_constants(p_specialization_constants));
}

RID RenderingDevice::_compute_pipeline_create(RID p_shader, const TypedArray<RDPipelineSpecializationConstant> &p_specialization_constants = TypedArray<RDPipelineSpecializationConstant>()) {
	return compute_pipeline_create(p_shader, _get_spec_constants(p_specialization_constants));
}

Vector<int64_t> RenderingDevice::_draw_list_begin_split(RID p_framebuffer, uint32_t p_splits, InitialAction p_initial_color_action, FinalAction p_final_color_action, InitialAction p_initial_depth_action, FinalAction p_final_depth_action, const Vector<Color> &p_clear_color_values, float p_clear_depth, uint32_t p_clear_stencil, const Rect2 &p_region, const TypedArray<RID> &p_storage_textures) {
	Vector<DrawListID> splits;
	splits.resize(p_splits);
	Vector<RID> stextures;
	for (int i = 0; i < p_storage_textures.size(); i++) {
		stextures.push_back(p_storage_textures[i]);
	}
	draw_list_begin_split(p_framebuffer, p_splits, splits.ptrw(), p_initial_color_action, p_final_color_action, p_initial_depth_action, p_final_depth_action, p_clear_color_values, p_clear_depth, p_clear_stencil, p_region, stextures);

	Vector<int64_t> split_ids;
	split_ids.resize(splits.size());
	for (int i = 0; i < splits.size(); i++) {
		split_ids.write[i] = splits[i];
	}

	return split_ids;
}

Vector<int64_t> RenderingDevice::_draw_list_switch_to_next_pass_split(uint32_t p_splits) {
	Vector<DrawListID> splits;
	splits.resize(p_splits);

	Error err = draw_list_switch_to_next_pass_split(p_splits, splits.ptrw());
	ERR_FAIL_COND_V(err != OK, Vector<int64_t>());

	Vector<int64_t> split_ids;
	split_ids.resize(splits.size());
	for (int i = 0; i < splits.size(); i++) {
		split_ids.write[i] = splits[i];
	}

	return split_ids;
}

void RenderingDevice::_draw_list_set_push_constant(DrawListID p_list, const Vector<uint8_t> &p_data, uint32_t p_data_size) {
	ERR_FAIL_COND((uint32_t)p_data.size() > p_data_size);
	draw_list_set_push_constant(p_list, p_data.ptr(), p_data_size);
}

void RenderingDevice::_compute_list_set_push_constant(ComputeListID p_list, const Vector<uint8_t> &p_data, uint32_t p_data_size) {
	ERR_FAIL_COND((uint32_t)p_data.size() > p_data_size);
	compute_list_set_push_constant(p_list, p_data.ptr(), p_data_size);
}

void RenderingDevice::_bind_methods() {
	ClassDB::bind_method(D_METHOD("texture_create", "format", "view", "data"), &RenderingDevice::_texture_create, DEFVAL(Array()));
	ClassDB::bind_method(D_METHOD("texture_create_shared", "view", "with_texture"), &RenderingDevice::_texture_create_shared);
	ClassDB::bind_method(D_METHOD("texture_create_shared_from_slice", "view", "with_texture", "layer", "mipmap", "slice_type"), &RenderingDevice::_texture_create_shared_from_slice, DEFVAL(TEXTURE_SLICE_2D));

	ClassDB::bind_method(D_METHOD("texture_update", "texture", "layer", "data", "post_barrier"), &RenderingDevice::texture_update, DEFVAL(BARRIER_MASK_ALL));
	ClassDB::bind_method(D_METHOD("texture_get_data", "texture", "layer"), &RenderingDevice::texture_get_data);

	ClassDB::bind_method(D_METHOD("texture_is_format_supported_for_usage", "format", "usage_flags"), &RenderingDevice::texture_is_format_supported_for_usage);

	ClassDB::bind_method(D_METHOD("texture_is_shared", "texture"), &RenderingDevice::texture_is_shared);
	ClassDB::bind_method(D_METHOD("texture_is_valid", "texture"), &RenderingDevice::texture_is_valid);

	ClassDB::bind_method(D_METHOD("texture_copy", "from_texture", "to_texture", "from_pos", "to_pos", "size", "src_mipmap", "dst_mipmap", "src_layer", "dst_layer", "post_barrier"), &RenderingDevice::texture_copy, DEFVAL(BARRIER_MASK_ALL));
	ClassDB::bind_method(D_METHOD("texture_clear", "texture", "color", "base_mipmap", "mipmap_count", "base_layer", "layer_count", "post_barrier"), &RenderingDevice::texture_clear, DEFVAL(BARRIER_MASK_ALL));
	ClassDB::bind_method(D_METHOD("texture_resolve_multisample", "from_texture", "to_texture", "post_barrier"), &RenderingDevice::texture_resolve_multisample, DEFVAL(BARRIER_MASK_ALL));

	ClassDB::bind_method(D_METHOD("framebuffer_format_create", "attachments", "view_count"), &RenderingDevice::_framebuffer_format_create, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("framebuffer_format_create_multipass", "attachments", "passes", "view_count"), &RenderingDevice::_framebuffer_format_create_multipass, DEFVAL(1));
	ClassDB::bind_method(D_METHOD("framebuffer_format_create_empty", "samples"), &RenderingDevice::framebuffer_format_create_empty, DEFVAL(TEXTURE_SAMPLES_1));
	ClassDB::bind_method(D_METHOD("framebuffer_format_get_texture_samples", "format", "render_pass"), &RenderingDevice::framebuffer_format_get_texture_samples, DEFVAL(0));
	ClassDB::bind_method(D_METHOD("framebuffer_create", "textures", "validate_with_format", "view_count"), &RenderingDevice::_framebuffer_create, DEFVAL(INVALID_FORMAT_ID), DEFVAL(1));
	ClassDB::bind_method(D_METHOD("framebuffer_create_multipass", "textures", "passes", "validate_with_format", "view_count"), &RenderingDevice::_framebuffer_create_multipass, DEFVAL(INVALID_FORMAT_ID), DEFVAL(1));
	ClassDB::bind_method(D_METHOD("framebuffer_create_empty", "size", "samples", "validate_with_format"), &RenderingDevice::framebuffer_create_empty, DEFVAL(TEXTURE_SAMPLES_1), DEFVAL(INVALID_FORMAT_ID));
	ClassDB::bind_method(D_METHOD("framebuffer_get_format", "framebuffer"), &RenderingDevice::framebuffer_get_format);

	ClassDB::bind_method(D_METHOD("sampler_create", "state"), &RenderingDevice::_sampler_create);

	ClassDB::bind_method(D_METHOD("vertex_buffer_create", "size_bytes", "data", "use_as_storage"), &RenderingDevice::vertex_buffer_create, DEFVAL(Vector<uint8_t>()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("vertex_format_create", "vertex_descriptions"), &RenderingDevice::_vertex_format_create);

	ClassDB::bind_method(D_METHOD("index_buffer_create", "size_indices", "format", "data", "use_restart_indices"), &RenderingDevice::index_buffer_create, DEFVAL(Vector<uint8_t>()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("index_array_create", "index_buffer", "index_offset", "index_count"), &RenderingDevice::index_array_create);

	ClassDB::bind_method(D_METHOD("shader_compile_spirv_from_source", "shader_source", "allow_cache"), &RenderingDevice::_shader_compile_spirv_from_source, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("shader_compile_binary_from_spirv", "spirv_data", "name"), &RenderingDevice::_shader_compile_binary_from_spirv, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("shader_create_from_spirv", "spirv_data", "name"), &RenderingDevice::_shader_create_from_spirv, DEFVAL(""));
	ClassDB::bind_method(D_METHOD("shader_create_from_bytecode", "binary_data"), &RenderingDevice::shader_create_from_bytecode);
	ClassDB::bind_method(D_METHOD("shader_get_vertex_input_attribute_mask", "shader"), &RenderingDevice::shader_get_vertex_input_attribute_mask);

	ClassDB::bind_method(D_METHOD("uniform_buffer_create", "size_bytes", "data"), &RenderingDevice::uniform_buffer_create, DEFVAL(Vector<uint8_t>()));
	ClassDB::bind_method(D_METHOD("storage_buffer_create", "size_bytes", "data", "usage"), &RenderingDevice::storage_buffer_create, DEFVAL(Vector<uint8_t>()), DEFVAL(0));
	ClassDB::bind_method(D_METHOD("texture_buffer_create", "size_bytes", "format", "data"), &RenderingDevice::texture_buffer_create, DEFVAL(Vector<uint8_t>()));

	ClassDB::bind_method(D_METHOD("uniform_set_create", "uniforms", "shader", "shader_set"), &RenderingDevice::_uniform_set_create);
	ClassDB::bind_method(D_METHOD("uniform_set_is_valid", "uniform_set"), &RenderingDevice::uniform_set_is_valid);

	ClassDB::bind_method(D_METHOD("buffer_update", "buffer", "offset", "size_bytes", "data", "post_barrier"), &RenderingDevice::_buffer_update, DEFVAL(BARRIER_MASK_ALL));
	ClassDB::bind_method(D_METHOD("buffer_clear", "buffer", "offset", "size_bytes", "post_barrier"), &RenderingDevice::buffer_clear, DEFVAL(BARRIER_MASK_ALL));
	ClassDB::bind_method(D_METHOD("buffer_get_data", "buffer"), &RenderingDevice::buffer_get_data);

	ClassDB::bind_method(D_METHOD("render_pipeline_create", "shader", "framebuffer_format", "vertex_format", "primitive", "rasterization_state", "multisample_state", "stencil_state", "color_blend_state", "dynamic_state_flags", "for_render_pass", "specialization_constants"), &RenderingDevice::_render_pipeline_create, DEFVAL(0), DEFVAL(0), DEFVAL(TypedArray<RDPipelineSpecializationConstant>()));
	ClassDB::bind_method(D_METHOD("render_pipeline_is_valid", "render_pipeline"), &RenderingDevice::render_pipeline_is_valid);

	ClassDB::bind_method(D_METHOD("compute_pipeline_create", "shader", "specialization_constants"), &RenderingDevice::_compute_pipeline_create, DEFVAL(TypedArray<RDPipelineSpecializationConstant>()));
	ClassDB::bind_method(D_METHOD("compute_pipeline_is_valid", "compute_pieline"), &RenderingDevice::compute_pipeline_is_valid);

	ClassDB::bind_method(D_METHOD("screen_get_width", "screen"), &RenderingDevice::screen_get_width, DEFVAL(DisplayServer::MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("screen_get_height", "screen"), &RenderingDevice::screen_get_height, DEFVAL(DisplayServer::MAIN_WINDOW_ID));
	ClassDB::bind_method(D_METHOD("screen_get_framebuffer_format"), &RenderingDevice::screen_get_framebuffer_format);

	ClassDB::bind_method(D_METHOD("draw_list_begin_for_screen", "screen", "clear_color"), &RenderingDevice::draw_list_begin_for_screen, DEFVAL(DisplayServer::MAIN_WINDOW_ID), DEFVAL(Color()));

	ClassDB::bind_method(D_METHOD("draw_list_begin", "framebuffer", "initial_color_action", "final_color_action", "initial_depth_action", "final_depth_action", "clear_color_values", "clear_depth", "clear_stencil", "region", "storage_textures"), &RenderingDevice::draw_list_begin, DEFVAL(Vector<Color>()), DEFVAL(1.0), DEFVAL(0), DEFVAL(Rect2()), DEFVAL(TypedArray<RID>()));
	ClassDB::bind_method(D_METHOD("draw_list_begin_split", "framebuffer", "splits", "initial_color_action", "final_color_action", "initial_depth_action", "final_depth_action", "clear_color_values", "clear_depth", "clear_stencil", "region", "storage_textures"), &RenderingDevice::_draw_list_begin_split, DEFVAL(Vector<Color>()), DEFVAL(1.0), DEFVAL(0), DEFVAL(Rect2()), DEFVAL(TypedArray<RID>()));

	ClassDB::bind_method(D_METHOD("draw_list_bind_render_pipeline", "draw_list", "render_pipeline"), &RenderingDevice::draw_list_bind_render_pipeline);
	ClassDB::bind_method(D_METHOD("draw_list_bind_uniform_set", "draw_list", "uniform_set", "set_index"), &RenderingDevice::draw_list_bind_uniform_set);
	ClassDB::bind_method(D_METHOD("draw_list_bind_vertex_array", "draw_list", "vertex_array"), &RenderingDevice::draw_list_bind_vertex_array);
	ClassDB::bind_method(D_METHOD("draw_list_bind_index_array", "draw_list", "index_array"), &RenderingDevice::draw_list_bind_index_array);
	ClassDB::bind_method(D_METHOD("draw_list_set_push_constant", "draw_list", "buffer", "size_bytes"), &RenderingDevice::_draw_list_set_push_constant);

	ClassDB::bind_method(D_METHOD("draw_list_draw", "draw_list", "use_indices", "instances", "procedural_vertex_count"), &RenderingDevice::draw_list_draw, DEFVAL(0));

	ClassDB::bind_method(D_METHOD("draw_list_enable_scissor", "draw_list", "rect"), &RenderingDevice::draw_list_enable_scissor, DEFVAL(Rect2()));
	ClassDB::bind_method(D_METHOD("draw_list_disable_scissor", "draw_list"), &RenderingDevice::draw_list_disable_scissor);

	ClassDB::bind_method(D_METHOD("draw_list_switch_to_next_pass"), &RenderingDevice::draw_list_switch_to_next_pass);
	ClassDB::bind_method(D_METHOD("draw_list_switch_to_next_pass_split", "splits"), &RenderingDevice::_draw_list_switch_to_next_pass_split);

	ClassDB::bind_method(D_METHOD("draw_list_end", "post_barrier"), &RenderingDevice::draw_list_end, DEFVAL(BARRIER_MASK_ALL));

	ClassDB::bind_method(D_METHOD("compute_list_begin", "allow_draw_overlap"), &RenderingDevice::compute_list_begin, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("compute_list_bind_compute_pipeline", "compute_list", "compute_pipeline"), &RenderingDevice::compute_list_bind_compute_pipeline);
	ClassDB::bind_method(D_METHOD("compute_list_set_push_constant", "compute_list", "buffer", "size_bytes"), &RenderingDevice::_compute_list_set_push_constant);
	ClassDB::bind_method(D_METHOD("compute_list_bind_uniform_set", "compute_list", "uniform_set", "set_index"), &RenderingDevice::compute_list_bind_uniform_set);
	ClassDB::bind_method(D_METHOD("compute_list_dispatch", "compute_list", "x_groups", "y_groups", "z_groups"), &RenderingDevice::compute_list_dispatch);
	ClassDB::bind_method(D_METHOD("compute_list_add_barrier", "compute_list"), &RenderingDevice::compute_list_add_barrier);
	ClassDB::bind_method(D_METHOD("compute_list_end", "post_barrier"), &RenderingDevice::compute_list_end, DEFVAL(BARRIER_MASK_ALL));

	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &RenderingDevice::free);

	ClassDB::bind_method(D_METHOD("capture_timestamp", "name"), &RenderingDevice::capture_timestamp);
	ClassDB::bind_method(D_METHOD("get_captured_timestamps_count"), &RenderingDevice::get_captured_timestamps_count);
	ClassDB::bind_method(D_METHOD("get_captured_timestamps_frame"), &RenderingDevice::get_captured_timestamps_frame);
	ClassDB::bind_method(D_METHOD("get_captured_timestamp_gpu_time", "index"), &RenderingDevice::get_captured_timestamp_gpu_time);
	ClassDB::bind_method(D_METHOD("get_captured_timestamp_cpu_time", "index"), &RenderingDevice::get_captured_timestamp_cpu_time);
	ClassDB::bind_method(D_METHOD("get_captured_timestamp_name", "index"), &RenderingDevice::get_captured_timestamp_name);

	ClassDB::bind_method(D_METHOD("limit_get", "limit"), &RenderingDevice::limit_get);
	ClassDB::bind_method(D_METHOD("get_frame_delay"), &RenderingDevice::get_frame_delay);
	ClassDB::bind_method(D_METHOD("submit"), &RenderingDevice::submit);
	ClassDB::bind_method(D_METHOD("sync"), &RenderingDevice::sync);

	ClassDB::bind_method(D_METHOD("barrier", "from", "to"), &RenderingDevice::barrier, DEFVAL(BARRIER_MASK_ALL), DEFVAL(BARRIER_MASK_ALL));
	ClassDB::bind_method(D_METHOD("full_barrier"), &RenderingDevice::full_barrier);

	ClassDB::bind_method(D_METHOD("create_local_device"), &RenderingDevice::create_local_device);

	ClassDB::bind_method(D_METHOD("set_resource_name", "id", "name"), &RenderingDevice::set_resource_name);

	ClassDB::bind_method(D_METHOD("draw_command_begin_label", "name", "color"), &RenderingDevice::draw_command_begin_label);
	ClassDB::bind_method(D_METHOD("draw_command_insert_label", "name", "color"), &RenderingDevice::draw_command_insert_label);
	ClassDB::bind_method(D_METHOD("draw_command_end_label"), &RenderingDevice::draw_command_end_label);

	ClassDB::bind_method(D_METHOD("get_device_vendor_name"), &RenderingDevice::get_device_vendor_name);
	ClassDB::bind_method(D_METHOD("get_device_name"), &RenderingDevice::get_device_name);
	ClassDB::bind_method(D_METHOD("get_device_pipeline_cache_uuid"), &RenderingDevice::get_device_pipeline_cache_uuid);

	ClassDB::bind_method(D_METHOD("get_memory_usage", "type"), &RenderingDevice::get_memory_usage);

	ClassDB::bind_method(D_METHOD("get_driver_resource", "resource", "rid", "index"), &RenderingDevice::get_driver_resource);

	BIND_CONSTANT(BARRIER_MASK_RASTER);
	BIND_CONSTANT(BARRIER_MASK_COMPUTE);
	BIND_CONSTANT(BARRIER_MASK_TRANSFER);
	BIND_CONSTANT(BARRIER_MASK_ALL);
	BIND_CONSTANT(BARRIER_MASK_NO_BARRIER);

	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_DEVICE);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_PHYSICAL_DEVICE);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_INSTANCE);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_QUEUE);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_QUEUE_FAMILY_INDEX);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_IMAGE);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_IMAGE_VIEW);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_IMAGE_NATIVE_TEXTURE_FORMAT);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_SAMPLER);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_DESCRIPTOR_SET);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_BUFFER);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_COMPUTE_PIPELINE);
	BIND_ENUM_CONSTANT(DRIVER_RESOURCE_VULKAN_RENDER_PIPELINE);

	BIND_ENUM_CONSTANT(DATA_FORMAT_R4G4_UNORM_PACK8);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R4G4B4A4_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B4G4R4A4_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R5G6B5_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B5G6R5_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R5G5B5A1_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B5G5R5A1_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A1R5G5B5_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8_SRGB);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8_SRGB);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8_SRGB);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8_SRGB);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8A8_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8A8_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8A8_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8A8_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8A8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8A8_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R8G8B8A8_SRGB);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8A8_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8A8_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8A8_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8A8_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8A8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8A8_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8A8_SRGB);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A8B8G8R8_UNORM_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A8B8G8R8_SNORM_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A8B8G8R8_USCALED_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A8B8G8R8_SSCALED_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A8B8G8R8_UINT_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A8B8G8R8_SINT_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A8B8G8R8_SRGB_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2R10G10B10_UNORM_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2R10G10B10_SNORM_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2R10G10B10_USCALED_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2R10G10B10_SSCALED_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2R10G10B10_UINT_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2R10G10B10_SINT_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2B10G10R10_UNORM_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2B10G10R10_SNORM_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2B10G10R10_USCALED_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2B10G10R10_SSCALED_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2B10G10R10_UINT_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_A2B10G10R10_SINT_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16A16_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16A16_SNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16A16_USCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16A16_SSCALED);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16A16_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16A16_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R16G16B16A16_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32B32_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32B32_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32B32_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32B32A32_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32B32A32_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R32G32B32A32_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64B64_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64B64_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64B64_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64B64A64_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64B64A64_SINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R64G64B64A64_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B10G11R11_UFLOAT_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_E5B9G9R9_UFLOAT_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_D16_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_X8_D24_UNORM_PACK32);
	BIND_ENUM_CONSTANT(DATA_FORMAT_D32_SFLOAT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_S8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_D16_UNORM_S8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_D24_UNORM_S8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_D32_SFLOAT_S8_UINT);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC1_RGB_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC1_RGB_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC1_RGBA_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC1_RGBA_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC2_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC2_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC3_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC3_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC4_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC4_SNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC5_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC5_SNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC6H_UFLOAT_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC6H_SFLOAT_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC7_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_BC7_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ETC2_R8G8B8_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ETC2_R8G8B8_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_EAC_R11_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_EAC_R11_SNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_EAC_R11G11_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_EAC_R11G11_SNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_4x4_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_4x4_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_5x4_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_5x4_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_5x5_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_5x5_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_6x5_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_6x5_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_6x6_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_6x6_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_8x5_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_8x5_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_8x6_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_8x6_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_8x8_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_8x8_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_10x5_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_10x5_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_10x6_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_10x6_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_10x8_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_10x8_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_10x10_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_10x10_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_12x10_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_12x10_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_12x12_UNORM_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_ASTC_12x12_SRGB_BLOCK);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G8B8G8R8_422_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B8G8R8G8_422_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G8_B8_R8_3PLANE_420_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G8_B8R8_2PLANE_420_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G8_B8_R8_3PLANE_422_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G8_B8R8_2PLANE_422_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G8_B8_R8_3PLANE_444_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R10X6_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R10X6G10X6_UNORM_2PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R12X4_UNORM_PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R12X4G12X4_UNORM_2PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G16B16G16R16_422_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_B16G16R16G16_422_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G16_B16_R16_3PLANE_420_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G16_B16R16_2PLANE_420_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G16_B16_R16_3PLANE_422_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G16_B16R16_2PLANE_422_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_G16_B16_R16_3PLANE_444_UNORM);
	BIND_ENUM_CONSTANT(DATA_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG);
	BIND_ENUM_CONSTANT(DATA_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG);
	BIND_ENUM_CONSTANT(DATA_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG);
	BIND_ENUM_CONSTANT(DATA_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG);
	BIND_ENUM_CONSTANT(DATA_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG);
	BIND_ENUM_CONSTANT(DATA_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG);
	BIND_ENUM_CONSTANT(DATA_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG);
	BIND_ENUM_CONSTANT(DATA_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG);
	BIND_ENUM_CONSTANT(DATA_FORMAT_MAX);

	BIND_ENUM_CONSTANT(TEXTURE_TYPE_1D);
	BIND_ENUM_CONSTANT(TEXTURE_TYPE_2D);
	BIND_ENUM_CONSTANT(TEXTURE_TYPE_3D);
	BIND_ENUM_CONSTANT(TEXTURE_TYPE_CUBE);
	BIND_ENUM_CONSTANT(TEXTURE_TYPE_1D_ARRAY);
	BIND_ENUM_CONSTANT(TEXTURE_TYPE_2D_ARRAY);
	BIND_ENUM_CONSTANT(TEXTURE_TYPE_CUBE_ARRAY);
	BIND_ENUM_CONSTANT(TEXTURE_TYPE_MAX);

	BIND_ENUM_CONSTANT(TEXTURE_SAMPLES_1);
	BIND_ENUM_CONSTANT(TEXTURE_SAMPLES_2);
	BIND_ENUM_CONSTANT(TEXTURE_SAMPLES_4);
	BIND_ENUM_CONSTANT(TEXTURE_SAMPLES_8);
	BIND_ENUM_CONSTANT(TEXTURE_SAMPLES_16);
	BIND_ENUM_CONSTANT(TEXTURE_SAMPLES_32);
	BIND_ENUM_CONSTANT(TEXTURE_SAMPLES_64);
	BIND_ENUM_CONSTANT(TEXTURE_SAMPLES_MAX);

	BIND_ENUM_CONSTANT(TEXTURE_USAGE_SAMPLING_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_COLOR_ATTACHMENT_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_STORAGE_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_STORAGE_ATOMIC_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_CPU_READ_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_CAN_UPDATE_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_CAN_COPY_FROM_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_CAN_COPY_TO_BIT);
	BIND_ENUM_CONSTANT(TEXTURE_USAGE_INPUT_ATTACHMENT_BIT);

	BIND_ENUM_CONSTANT(TEXTURE_SWIZZLE_IDENTITY);
	BIND_ENUM_CONSTANT(TEXTURE_SWIZZLE_ZERO);
	BIND_ENUM_CONSTANT(TEXTURE_SWIZZLE_ONE);
	BIND_ENUM_CONSTANT(TEXTURE_SWIZZLE_R);
	BIND_ENUM_CONSTANT(TEXTURE_SWIZZLE_G);
	BIND_ENUM_CONSTANT(TEXTURE_SWIZZLE_B);
	BIND_ENUM_CONSTANT(TEXTURE_SWIZZLE_A);
	BIND_ENUM_CONSTANT(TEXTURE_SWIZZLE_MAX);

	BIND_ENUM_CONSTANT(TEXTURE_SLICE_2D);
	BIND_ENUM_CONSTANT(TEXTURE_SLICE_CUBEMAP);
	BIND_ENUM_CONSTANT(TEXTURE_SLICE_3D);

	BIND_ENUM_CONSTANT(SAMPLER_FILTER_NEAREST);
	BIND_ENUM_CONSTANT(SAMPLER_FILTER_LINEAR);
	BIND_ENUM_CONSTANT(SAMPLER_REPEAT_MODE_REPEAT);
	BIND_ENUM_CONSTANT(SAMPLER_REPEAT_MODE_MIRRORED_REPEAT);
	BIND_ENUM_CONSTANT(SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE);
	BIND_ENUM_CONSTANT(SAMPLER_REPEAT_MODE_CLAMP_TO_BORDER);
	BIND_ENUM_CONSTANT(SAMPLER_REPEAT_MODE_MIRROR_CLAMP_TO_EDGE);
	BIND_ENUM_CONSTANT(SAMPLER_REPEAT_MODE_MAX);

	BIND_ENUM_CONSTANT(SAMPLER_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK);
	BIND_ENUM_CONSTANT(SAMPLER_BORDER_COLOR_INT_TRANSPARENT_BLACK);
	BIND_ENUM_CONSTANT(SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_BLACK);
	BIND_ENUM_CONSTANT(SAMPLER_BORDER_COLOR_INT_OPAQUE_BLACK);
	BIND_ENUM_CONSTANT(SAMPLER_BORDER_COLOR_FLOAT_OPAQUE_WHITE);
	BIND_ENUM_CONSTANT(SAMPLER_BORDER_COLOR_INT_OPAQUE_WHITE);
	BIND_ENUM_CONSTANT(SAMPLER_BORDER_COLOR_MAX);

	BIND_ENUM_CONSTANT(VERTEX_FREQUENCY_VERTEX);
	BIND_ENUM_CONSTANT(VERTEX_FREQUENCY_INSTANCE);

	BIND_ENUM_CONSTANT(INDEX_BUFFER_FORMAT_UINT16);
	BIND_ENUM_CONSTANT(INDEX_BUFFER_FORMAT_UINT32);

	BIND_ENUM_CONSTANT(STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);

	BIND_ENUM_CONSTANT(UNIFORM_TYPE_SAMPLER); //for sampling only (sampler GLSL type)
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_SAMPLER_WITH_TEXTURE); // for sampling only); but includes a texture); (samplerXX GLSL type)); first a sampler then a texture
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_TEXTURE); //only texture); (textureXX GLSL type)
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_IMAGE); // storage image (imageXX GLSL type)); for compute mostly
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_TEXTURE_BUFFER); // buffer texture (or TBO); textureBuffer type)
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_SAMPLER_WITH_TEXTURE_BUFFER); // buffer texture with a sampler(or TBO); samplerBuffer type)
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_IMAGE_BUFFER); //texel buffer); (imageBuffer type)); for compute mostly
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_UNIFORM_BUFFER); //regular uniform buffer (or UBO).
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_STORAGE_BUFFER); //storage buffer ("buffer" qualifier) like UBO); but supports storage); for compute mostly
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_INPUT_ATTACHMENT); //used for sub-pass read/write); for mobile mostly
	BIND_ENUM_CONSTANT(UNIFORM_TYPE_MAX);

	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_POINTS);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_LINES);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_LINES_WITH_ADJACENCY);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_LINESTRIPS);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_LINESTRIPS_WITH_ADJACENCY);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_TRIANGLES);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_TRIANGLES_WITH_ADJACENCY);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_TRIANGLE_STRIPS);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_AJACENCY);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_TRIANGLE_STRIPS_WITH_RESTART_INDEX);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_TESSELATION_PATCH);
	BIND_ENUM_CONSTANT(RENDER_PRIMITIVE_MAX);

	BIND_ENUM_CONSTANT(POLYGON_CULL_DISABLED);
	BIND_ENUM_CONSTANT(POLYGON_CULL_FRONT);
	BIND_ENUM_CONSTANT(POLYGON_CULL_BACK);

	BIND_ENUM_CONSTANT(POLYGON_FRONT_FACE_CLOCKWISE);
	BIND_ENUM_CONSTANT(POLYGON_FRONT_FACE_COUNTER_CLOCKWISE);

	BIND_ENUM_CONSTANT(STENCIL_OP_KEEP);
	BIND_ENUM_CONSTANT(STENCIL_OP_ZERO);
	BIND_ENUM_CONSTANT(STENCIL_OP_REPLACE);
	BIND_ENUM_CONSTANT(STENCIL_OP_INCREMENT_AND_CLAMP);
	BIND_ENUM_CONSTANT(STENCIL_OP_DECREMENT_AND_CLAMP);
	BIND_ENUM_CONSTANT(STENCIL_OP_INVERT);
	BIND_ENUM_CONSTANT(STENCIL_OP_INCREMENT_AND_WRAP);
	BIND_ENUM_CONSTANT(STENCIL_OP_DECREMENT_AND_WRAP);
	BIND_ENUM_CONSTANT(STENCIL_OP_MAX); //not an actual operator); just the amount of operators :D

	BIND_ENUM_CONSTANT(COMPARE_OP_NEVER);
	BIND_ENUM_CONSTANT(COMPARE_OP_LESS);
	BIND_ENUM_CONSTANT(COMPARE_OP_EQUAL);
	BIND_ENUM_CONSTANT(COMPARE_OP_LESS_OR_EQUAL);
	BIND_ENUM_CONSTANT(COMPARE_OP_GREATER);
	BIND_ENUM_CONSTANT(COMPARE_OP_NOT_EQUAL);
	BIND_ENUM_CONSTANT(COMPARE_OP_GREATER_OR_EQUAL);
	BIND_ENUM_CONSTANT(COMPARE_OP_ALWAYS);
	BIND_ENUM_CONSTANT(COMPARE_OP_MAX);

	BIND_ENUM_CONSTANT(LOGIC_OP_CLEAR);
	BIND_ENUM_CONSTANT(LOGIC_OP_AND);
	BIND_ENUM_CONSTANT(LOGIC_OP_AND_REVERSE);
	BIND_ENUM_CONSTANT(LOGIC_OP_COPY);
	BIND_ENUM_CONSTANT(LOGIC_OP_AND_INVERTED);
	BIND_ENUM_CONSTANT(LOGIC_OP_NO_OP);
	BIND_ENUM_CONSTANT(LOGIC_OP_XOR);
	BIND_ENUM_CONSTANT(LOGIC_OP_OR);
	BIND_ENUM_CONSTANT(LOGIC_OP_NOR);
	BIND_ENUM_CONSTANT(LOGIC_OP_EQUIVALENT);
	BIND_ENUM_CONSTANT(LOGIC_OP_INVERT);
	BIND_ENUM_CONSTANT(LOGIC_OP_OR_REVERSE);
	BIND_ENUM_CONSTANT(LOGIC_OP_COPY_INVERTED);
	BIND_ENUM_CONSTANT(LOGIC_OP_OR_INVERTED);
	BIND_ENUM_CONSTANT(LOGIC_OP_NAND);
	BIND_ENUM_CONSTANT(LOGIC_OP_SET);
	BIND_ENUM_CONSTANT(LOGIC_OP_MAX); //not an actual operator); just the amount of operators :D

	BIND_ENUM_CONSTANT(BLEND_FACTOR_ZERO);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_SRC_COLOR);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE_MINUS_SRC_COLOR);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_DST_COLOR);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE_MINUS_DST_COLOR);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_SRC_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_DST_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE_MINUS_DST_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_CONSTANT_COLOR);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_CONSTANT_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_SRC_ALPHA_SATURATE);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_SRC1_COLOR);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE_MINUS_SRC1_COLOR);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_SRC1_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA);
	BIND_ENUM_CONSTANT(BLEND_FACTOR_MAX);

	BIND_ENUM_CONSTANT(BLEND_OP_ADD);
	BIND_ENUM_CONSTANT(BLEND_OP_SUBTRACT);
	BIND_ENUM_CONSTANT(BLEND_OP_REVERSE_SUBTRACT);
	BIND_ENUM_CONSTANT(BLEND_OP_MINIMUM);
	BIND_ENUM_CONSTANT(BLEND_OP_MAXIMUM);
	BIND_ENUM_CONSTANT(BLEND_OP_MAX);

	BIND_ENUM_CONSTANT(DYNAMIC_STATE_LINE_WIDTH);
	BIND_ENUM_CONSTANT(DYNAMIC_STATE_DEPTH_BIAS);
	BIND_ENUM_CONSTANT(DYNAMIC_STATE_BLEND_CONSTANTS);
	BIND_ENUM_CONSTANT(DYNAMIC_STATE_DEPTH_BOUNDS);
	BIND_ENUM_CONSTANT(DYNAMIC_STATE_STENCIL_COMPARE_MASK);
	BIND_ENUM_CONSTANT(DYNAMIC_STATE_STENCIL_WRITE_MASK);
	BIND_ENUM_CONSTANT(DYNAMIC_STATE_STENCIL_REFERENCE);

	BIND_ENUM_CONSTANT(INITIAL_ACTION_CLEAR); //start rendering and clear the framebuffer (supply params)
	BIND_ENUM_CONSTANT(INITIAL_ACTION_CLEAR_REGION); //start rendering and clear the framebuffer (supply params)
	BIND_ENUM_CONSTANT(INITIAL_ACTION_CLEAR_REGION_CONTINUE); //continue rendering and clear the framebuffer (supply params)
	BIND_ENUM_CONSTANT(INITIAL_ACTION_KEEP); //start rendering); but keep attached color texture contents (depth will be cleared)
	BIND_ENUM_CONSTANT(INITIAL_ACTION_DROP); //start rendering); ignore what is there); just write above it
	BIND_ENUM_CONSTANT(INITIAL_ACTION_CONTINUE); //continue rendering (framebuffer must have been left in "continue" state as final action previously)
	BIND_ENUM_CONSTANT(INITIAL_ACTION_MAX);

	BIND_ENUM_CONSTANT(FINAL_ACTION_READ); //will no longer render to it); allows attached textures to be read again); but depth buffer contents will be dropped (Can't be read from)
	BIND_ENUM_CONSTANT(FINAL_ACTION_DISCARD); // discard contents after rendering
	BIND_ENUM_CONSTANT(FINAL_ACTION_CONTINUE); //will continue rendering later); attached textures can't be read until re-bound with "finish"
	BIND_ENUM_CONSTANT(FINAL_ACTION_MAX);

	BIND_ENUM_CONSTANT(SHADER_STAGE_VERTEX);
	BIND_ENUM_CONSTANT(SHADER_STAGE_FRAGMENT);
	BIND_ENUM_CONSTANT(SHADER_STAGE_TESSELATION_CONTROL);
	BIND_ENUM_CONSTANT(SHADER_STAGE_TESSELATION_EVALUATION);
	BIND_ENUM_CONSTANT(SHADER_STAGE_COMPUTE);
	BIND_ENUM_CONSTANT(SHADER_STAGE_MAX);
	BIND_ENUM_CONSTANT(SHADER_STAGE_VERTEX_BIT);
	BIND_ENUM_CONSTANT(SHADER_STAGE_FRAGMENT_BIT);
	BIND_ENUM_CONSTANT(SHADER_STAGE_TESSELATION_CONTROL_BIT);
	BIND_ENUM_CONSTANT(SHADER_STAGE_TESSELATION_EVALUATION_BIT);
	BIND_ENUM_CONSTANT(SHADER_STAGE_COMPUTE_BIT);

	BIND_ENUM_CONSTANT(SHADER_LANGUAGE_GLSL);
	BIND_ENUM_CONSTANT(SHADER_LANGUAGE_HLSL);

	BIND_ENUM_CONSTANT(PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL);
	BIND_ENUM_CONSTANT(PIPELINE_SPECIALIZATION_CONSTANT_TYPE_INT);
	BIND_ENUM_CONSTANT(PIPELINE_SPECIALIZATION_CONSTANT_TYPE_FLOAT);

	BIND_ENUM_CONSTANT(LIMIT_MAX_BOUND_UNIFORM_SETS);
	BIND_ENUM_CONSTANT(LIMIT_MAX_FRAMEBUFFER_COLOR_ATTACHMENTS);
	BIND_ENUM_CONSTANT(LIMIT_MAX_TEXTURES_PER_UNIFORM_SET);
	BIND_ENUM_CONSTANT(LIMIT_MAX_SAMPLERS_PER_UNIFORM_SET);
	BIND_ENUM_CONSTANT(LIMIT_MAX_STORAGE_BUFFERS_PER_UNIFORM_SET);
	BIND_ENUM_CONSTANT(LIMIT_MAX_STORAGE_IMAGES_PER_UNIFORM_SET);
	BIND_ENUM_CONSTANT(LIMIT_MAX_UNIFORM_BUFFERS_PER_UNIFORM_SET);
	BIND_ENUM_CONSTANT(LIMIT_MAX_DRAW_INDEXED_INDEX);
	BIND_ENUM_CONSTANT(LIMIT_MAX_FRAMEBUFFER_HEIGHT);
	BIND_ENUM_CONSTANT(LIMIT_MAX_FRAMEBUFFER_WIDTH);
	BIND_ENUM_CONSTANT(LIMIT_MAX_TEXTURE_ARRAY_LAYERS);
	BIND_ENUM_CONSTANT(LIMIT_MAX_TEXTURE_SIZE_1D);
	BIND_ENUM_CONSTANT(LIMIT_MAX_TEXTURE_SIZE_2D);
	BIND_ENUM_CONSTANT(LIMIT_MAX_TEXTURE_SIZE_3D);
	BIND_ENUM_CONSTANT(LIMIT_MAX_TEXTURE_SIZE_CUBE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_SAMPLERS_PER_SHADER_STAGE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_STORAGE_BUFFERS_PER_SHADER_STAGE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_STORAGE_IMAGES_PER_SHADER_STAGE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_UNIFORM_BUFFERS_PER_SHADER_STAGE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_PUSH_CONSTANT_SIZE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_UNIFORM_BUFFER_SIZE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_VERTEX_INPUT_ATTRIBUTE_OFFSET);
	BIND_ENUM_CONSTANT(LIMIT_MAX_VERTEX_INPUT_ATTRIBUTES);
	BIND_ENUM_CONSTANT(LIMIT_MAX_VERTEX_INPUT_BINDINGS);
	BIND_ENUM_CONSTANT(LIMIT_MAX_VERTEX_INPUT_BINDING_STRIDE);
	BIND_ENUM_CONSTANT(LIMIT_MIN_UNIFORM_BUFFER_OFFSET_ALIGNMENT);
	BIND_ENUM_CONSTANT(LIMIT_MAX_COMPUTE_SHARED_MEMORY_SIZE);
	BIND_ENUM_CONSTANT(LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X);
	BIND_ENUM_CONSTANT(LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Y);
	BIND_ENUM_CONSTANT(LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_Z);
	BIND_ENUM_CONSTANT(LIMIT_MAX_COMPUTE_WORKGROUP_INVOCATIONS);
	BIND_ENUM_CONSTANT(LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_X);
	BIND_ENUM_CONSTANT(LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Y);
	BIND_ENUM_CONSTANT(LIMIT_MAX_COMPUTE_WORKGROUP_SIZE_Z);

	BIND_ENUM_CONSTANT(MEMORY_TEXTURES);
	BIND_ENUM_CONSTANT(MEMORY_BUFFERS);
	BIND_ENUM_CONSTANT(MEMORY_TOTAL);

	BIND_CONSTANT(INVALID_ID);
	BIND_CONSTANT(INVALID_FORMAT_ID);
}

RenderingDevice::RenderingDevice() {
	if (singleton == nullptr) { // there may be more rendering devices later
		singleton = this;
	}
}
