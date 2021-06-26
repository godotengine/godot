/*************************************************************************/
/*  tonemap_subpass_rd.cpp                                               */
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

#include "tonemap_subpass_rd.h"

#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"

RID TonemapSubpassRD::_get_uniform_set_from_texture(RD::UniformType p_type, RID p_texture, uint32_t p_set) {
	if (texture_to_uniform_set_cache.has(p_texture)) {
		RID uniform_set = texture_to_uniform_set_cache[p_texture];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.uniform_type = p_type;
	u.binding = 0;
	if (p_type == RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE) {
		u.ids.push_back(default_sampler);
	}
	u.ids.push_back(p_texture);
	uniforms.push_back(u);
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, shader.version_get_shader(shader_version, 0), p_set);

	texture_to_uniform_set_cache[p_texture] = uniform_set;

	return uniform_set;
}

void TonemapSubpassRD::tonemapper(RD::DrawListID p_draw_list, const Settings &p_settings) {
	memset(&push_constant, 0, sizeof(PushConstant));

	push_constant.use_bcs = p_settings.use_bcs;
	push_constant.bcs[0] = p_settings.brightness;
	push_constant.bcs[1] = p_settings.contrast;
	push_constant.bcs[2] = p_settings.saturation;

	int mode = p_settings.use_1d_color_correction ? TONEMAP_SP_MODE_1D_LUT : TONEMAP_SP_MODE_NORMAL;
	if (p_settings.view_count > 1) {
		// Use MULTIVIEW versions
		mode += 2;
	}

	push_constant.tonemapper = p_settings.tonemap_mode;
	push_constant.use_auto_exposure = p_settings.use_auto_exposure;
	push_constant.exposure = p_settings.exposure;
	push_constant.white = p_settings.white;
	push_constant.auto_exposure_grey = p_settings.auto_exposure_grey;

	push_constant.use_color_correction = p_settings.use_color_correction;

	push_constant.use_debanding = p_settings.use_debanding;

	RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipelines[mode].get_render_pipeline(RD::INVALID_ID, p_settings.format_id, false, RD::get_singleton()->get_current_subpass()));
	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, _get_uniform_set_from_texture(RD::UNIFORM_TYPE_INPUT_ATTACHMENT, p_settings.source_texture, 0), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, _get_uniform_set_from_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, p_settings.exposure_texture, 1), 1);
	RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, _get_uniform_set_from_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, p_settings.color_correction_texture, 2), 2);
	RD::get_singleton()->draw_list_bind_index_array(p_draw_list, index_array);

	RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
	RD::get_singleton()->draw_list_draw(p_draw_list, true);
}

TonemapSubpassRD::TonemapSubpassRD() {
	// Initialize tonemapper
	Vector<String> modes;
	modes.push_back("\n");
	modes.push_back("\n#define USE_1D_LUT\n");

	// multiview versions of our shaders
	modes.push_back("\n#define MULTIVIEW\n");
	modes.push_back("\n#define MULTIVIEW\n#define USE_1D_LUT\n");

	shader.initialize(modes);

	if (!RendererCompositorRD::singleton->is_xr_enabled()) {
		shader.set_variant_enabled(TONEMAP_SP_MODE_NORMAL_MULTIVIEW, false);
		shader.set_variant_enabled(TONEMAP_SP_MODE_1D_LUT_MULTIVIEW, false);
	}

	shader_version = shader.version_create();

	for (int i = 0; i < TONEMAP_SP_MODE_MAX; i++) {
		if (shader.is_variant_enabled(i)) {
			pipelines[i].setup(shader.version_get_shader(shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		} else {
			pipelines[i].clear();
		}
	}

	RD::SamplerState sampler;
	sampler.mag_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.max_lod = 0;

	default_sampler = RD::get_singleton()->sampler_create(sampler);
	RD::get_singleton()->set_resource_name(default_sampler, "Default Linear Tonemap Sampler");

	{ //create index array for copy shaders
		Vector<uint8_t> pv;
		pv.resize(6 * 4);
		{
			uint8_t *w = pv.ptrw();
			int *p32 = (int *)w;
			p32[0] = 0;
			p32[1] = 1;
			p32[2] = 2;
			p32[3] = 0;
			p32[4] = 2;
			p32[5] = 3;
		}
		index_buffer = RD::get_singleton()->index_buffer_create(6, RenderingDevice::INDEX_BUFFER_FORMAT_UINT32, pv);
		index_array = RD::get_singleton()->index_array_create(index_buffer, 0, 6);
	}
}

TonemapSubpassRD::~TonemapSubpassRD() {
	RD::get_singleton()->free(default_sampler);
	RD::get_singleton()->free(index_buffer); //array gets freed as dependency

	shader.version_free(shader_version);
}
