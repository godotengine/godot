/**************************************************************************/
/*  motion_blur_compositor_effect.cpp                                     */
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

#include "motion_blur_compositor_effect.h"

#include "servers/rendering/renderer_rd/storage_rd/render_data_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/render_scene_buffers_rd.h"
#include "servers/rendering/rendering_device_binds.h"

void MotionBlurCompositorEffect::_initialize_compute() {
	if (!DisplayServer::get_singleton()->window_can_draw()) {
		return;
	}

	RD::SamplerState sampler_state;
	sampler_state.min_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler_state.mag_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler_state.repeat_u = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;
	sampler_state.repeat_v = RD::SAMPLER_REPEAT_MODE_CLAMP_TO_EDGE;

	linear_sampler = RD::get_singleton()->sampler_create(sampler_state);
	String motion_blur_glsl = R"(#[compute]
#version 450

layout(set = 0, binding = 0) uniform sampler2D color_sampler;
layout(set = 0, binding = 1) uniform sampler2D vector_sampler;
layout(rgba16f, set = 0, binding = 2) uniform writeonly image2D output_image;

layout(push_constant, std430) uniform Params {
	vec2 samples_intensity;
	vec2 fade_padding;
} params;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
	ivec2 render_size = ivec2(textureSize(color_sampler, 0));
	ivec2 uvi = ivec2(gl_GlobalInvocationID.xy);
	if ((uvi.x >= render_size.x) || (uvi.y >= render_size.y)) {
		return;
	}
	vec2 uvn = (vec2(uvi) + 0.5) / render_size;

	int iteration_count = int(params.samples_intensity.x);
	vec2 velocity = textureLod(vector_sampler, uvn, 0.0).xy;
	vec2 sample_step = velocity * params.samples_intensity.y;
	sample_step /= max(1.0, params.samples_intensity.x - 1.0);

	float d = 1.0 - min(1.0, 2.0 * distance(uvn, vec2(0.5)));
	sample_step *= 1.0 - d * params.fade_padding.x;

	vec4 base = textureLod(color_sampler, uvn, 0.0);

	// There is no motion, early out.
	if (length(velocity) <= 0.0001) {
		imageStore(output_image, uvi, base);
		return;
	}

	float total_weight = 1.0;
	vec2 offset = vec2(0.0);
	vec4 col = base;
	for (int i = 1; i < iteration_count; i++) {
		offset += sample_step;
		vec2 uvo = uvn + offset;
		if (any(notEqual(uvo, clamp(uvo, vec2(0.0), vec2(1.0))))) {
			break;
		}

		vec2 step_velocity = textureLod(vector_sampler, uvo, 0.0).xy;
		// Attempt to prevent ghosting caused by surfaces with significantly different velocities.
		float sample_weight = clamp(dot(step_velocity, velocity) / dot(velocity, velocity), 0.0, 1.0);
		if (sample_weight <= 0.0) {
			continue;
		}
		total_weight += sample_weight;
		col += textureLod(color_sampler, uvo, 0.0) * sample_weight;
	}

	col /= total_weight;

	imageStore(output_image, uvi, col);
}
)";
	Ref<RDShaderFile> motion_blur_shader_file = memnew(RDShaderFile);
	motion_blur_shader_file->parse_versions_from_text(motion_blur_glsl);
	Ref<RDShaderSPIRV> motion_blur_bytecode = motion_blur_shader_file->get_spirv();
	motion_blur_shader = RD::get_singleton()->shader_create_from_spirv(motion_blur_bytecode->get_stages(), "Motion blur");
	motion_blur_pipeline = RD::get_singleton()->compute_pipeline_create(motion_blur_shader);

	String overlay_glsl = R"(#[compute]
#version 450

layout(set = 0, binding = 0) uniform sampler2D blur_sampler;
layout(rgba16f, set = 0, binding = 1) uniform image2D color_image;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
	ivec2 render_size = ivec2(textureSize(blur_sampler, 0));
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	if ((uv.x >= render_size.x) || (uv.y >= render_size.y)) {
		return;
	}
	imageStore(color_image, uv, textureLod(blur_sampler, (vec2(uv) + 0.5) / render_size, 0.0));
}
)";
	Ref<RDShaderFile> overlay_shader_file = memnew(RDShaderFile);
	overlay_shader_file->parse_versions_from_text(overlay_glsl);
	Ref<RDShaderSPIRV> overlay_bytecode = overlay_shader_file->get_spirv();
	overlay_shader = RD::get_singleton()->shader_create_from_spirv(overlay_bytecode->get_stages(), "Motion blur overlay");
	overlay_pipeline = RD::get_singleton()->compute_pipeline_create(overlay_shader);
}

RD::Uniform MotionBlurCompositorEffect::get_image_uniform(RID p_image, int p_binding) {
	RD::Uniform uniform;
	uniform.uniform_type = RD::UNIFORM_TYPE_IMAGE;
	uniform.binding = p_binding;
	uniform.append_id(p_image);
	return uniform;
}

RD::Uniform MotionBlurCompositorEffect::get_sampler_uniform(RID p_image, int p_binding) {
	RD::Uniform uniform;
	uniform.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	uniform.binding = p_binding;
	uniform.append_id(linear_sampler);
	uniform.append_id(p_image);
	return uniform;
}

void MotionBlurCompositorEffect::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_render_callback", "effect_callback_type", "render_data"), &MotionBlurCompositorEffect::_render_callback);

	ClassDB::bind_method(D_METHOD("get_motion_blur_samples"), &MotionBlurCompositorEffect::get_motion_blur_samples);
	ClassDB::bind_method(D_METHOD("set_motion_blur_samples", "samples"), &MotionBlurCompositorEffect::set_motion_blur_samples);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "motion_blur_samples"), "set_motion_blur_samples", "get_motion_blur_samples");

	ClassDB::bind_method(D_METHOD("get_motion_blur_intensity"), &MotionBlurCompositorEffect::get_motion_blur_intensity);
	ClassDB::bind_method(D_METHOD("set_motion_blur_intensity", "intensity"), &MotionBlurCompositorEffect::set_motion_blur_intensity);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "motion_blur_intensity", PROPERTY_HINT_RANGE, "0,0.5,or_greater"), "set_motion_blur_intensity", "get_motion_blur_intensity");

	ClassDB::bind_method(D_METHOD("get_motion_blur_center_fade"), &MotionBlurCompositorEffect::get_motion_blur_center_fade);
	ClassDB::bind_method(D_METHOD("set_motion_blur_center_fade", "center_fade"), &MotionBlurCompositorEffect::set_motion_blur_center_fade);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "motion_blur_center_fade", PROPERTY_HINT_RANGE, "0,1,or_greater"), "set_motion_blur_center_fade", "get_motion_blur_center_fade");
}

void MotionBlurCompositorEffect::_render_callback(int p_effect_callback_type, const RenderDataRD *p_render_data) {
	if (RD::get_singleton() && p_effect_callback_type != EFFECT_CALLBACK_TYPE_POST_TRANSPARENT) {
		return;
	}
	Ref<RenderSceneBuffersRD> render_scene_buffers = p_render_data->get_render_scene_buffers();
	RenderSceneData *render_scene_data = p_render_data->get_render_scene_data();
	if (render_scene_buffers.is_null() || !render_scene_data) {
		return;
	}
	Size2i render_size = render_scene_buffers->get_internal_size();
	if (render_size.x == 0 || render_size.y == 0) {
		return;
	}

	const StringName context = SNAME("MotionBlur");
	const StringName texture = SNAME("texture");
	if (render_scene_buffers->has_texture(context, texture)) {
		RD::TextureFormat tf = render_scene_buffers->get_texture_format(context, texture);
		if (static_cast<int32_t>(tf.width) != render_size.x || static_cast<int32_t>(tf.height) != render_size.y) {
			render_scene_buffers->clear_context(context);
		}
	}

	if (!render_scene_buffers->has_texture(context, texture)) {
		uint32_t usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT;
		render_scene_buffers->create_texture(context, texture, RD::DATA_FORMAT_R16G16B16A16_SFLOAT, usage_bits, RD::TEXTURE_SAMPLES_1, render_size, 1, 1, true);
	}

	RD::get_singleton()->draw_command_begin_label("Motion Blur");

	LocalVector<float> push_constant;
	push_constant.resize(4);
	push_constant[0] = motion_blur_samples;
	push_constant[1] = motion_blur_intensity;
	push_constant[2] = motion_blur_center_fade;
	push_constant[3] = 0.0f;

	int view_count = render_scene_buffers->get_view_count();
	for (int view = 0; view < view_count; view++) {
		RID color_image = render_scene_buffers->call("get_color_layer", view);
		RID velocity_image = render_scene_buffers->call("get_velocity_layer", view);
		RID texture_image = render_scene_buffers->get_texture_slice(context, texture, view, 0, 1, 1);
		RenderingDevice::get_singleton()->draw_command_begin_label("Compute blur " + itos(view), Color(1.0, 1.0, 1.0, 1.0));
		Vector<RenderingDevice::Uniform> uniforms;
		uniforms.push_back(get_sampler_uniform(color_image, 0));
		uniforms.push_back(get_sampler_uniform(velocity_image, 1));
		uniforms.push_back(get_image_uniform(texture_image, 2));

		RID tex_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, motion_blur_shader, 0);

		int x_groups = ((render_size.x - 1) / 8) + 1;
		int y_groups = ((render_size.y - 1) / 8) + 1;

		RenderingDevice::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, motion_blur_pipeline);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, tex_uniform_set, 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, push_constant.ptr(), push_constant.size() * sizeof(float));
		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
		RD::get_singleton()->compute_list_end();

		RD::get_singleton()->draw_command_end_label();

		RD::get_singleton()->draw_command_begin_label("Overlay result " + itos(view));

		uniforms.clear();
		uniforms.push_back(get_sampler_uniform(texture_image, 0));
		uniforms.push_back(get_image_uniform(color_image, 1));

		tex_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, overlay_shader, 0);

		compute_list = RD::get_singleton()->compute_list_begin();
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, overlay_pipeline);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, tex_uniform_set, 0);
		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
		RD::get_singleton()->compute_list_end();

		RD::get_singleton()->draw_command_end_label();
	}

	RD::get_singleton()->draw_command_end_label();
}

MotionBlurCompositorEffect::~MotionBlurCompositorEffect() {
	if (linear_sampler.is_valid()) {
		RD::get_singleton()->free(linear_sampler);
	}
	if (motion_blur_shader.is_valid()) {
		RD::get_singleton()->free(motion_blur_shader);
	}
	if (overlay_shader.is_valid()) {
		RD::get_singleton()->free(overlay_shader);
	}
}

MotionBlurCompositorEffect::MotionBlurCompositorEffect() {
	set_effect_callback_type(EFFECT_CALLBACK_TYPE_POST_TRANSPARENT);
	set_needs_motion_vectors(true);
	if (!DisplayServer::get_singleton()->window_can_draw()) {
		return;
	}
	_initialize_compute();
}

float MotionBlurCompositorEffect::get_motion_blur_center_fade() const {
	return motion_blur_center_fade;
}

void MotionBlurCompositorEffect::set_motion_blur_center_fade(float p_value) {
	motion_blur_center_fade = p_value;
}

void MotionBlurCompositorEffect::set_motion_blur_intensity(float p_intensity) { motion_blur_intensity = p_intensity; }

float MotionBlurCompositorEffect::get_motion_blur_intensity() const { return motion_blur_intensity; }

void MotionBlurCompositorEffect::set_motion_blur_samples(int p_samples) { motion_blur_samples = p_samples; }

int MotionBlurCompositorEffect::get_motion_blur_samples() const { return motion_blur_samples; }
