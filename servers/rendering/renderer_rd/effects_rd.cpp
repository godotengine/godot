/*************************************************************************/
/*  effects_rd.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "effects_rd.h"

#include "core/config/project_settings.h"
#include "core/math/math_defs.h"
#include "core/os/os.h"

#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "thirdparty/misc/cubemap_coeffs.h"

bool EffectsRD::get_prefer_raster_effects() {
	return prefer_raster_effects;
}

RID EffectsRD::_get_uniform_set_from_image(RID p_image) {
	if (image_to_uniform_set_cache.has(p_image)) {
		RID uniform_set = image_to_uniform_set_cache[p_image];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}
	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
	u.binding = 0;
	u.append_id(p_image);
	uniforms.push_back(u);
	//any thing with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, luminance_reduce.shader.version_get_shader(luminance_reduce.shader_version, 0), 1);

	image_to_uniform_set_cache[p_image] = uniform_set;

	return uniform_set;
}

RID EffectsRD::_get_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps) {
	if (texture_to_uniform_set_cache.has(p_texture)) {
		RID uniform_set = texture_to_uniform_set_cache[p_texture];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u.binding = 0;
	u.append_id(p_use_mipmaps ? default_mipmap_sampler : default_sampler);
	u.append_id(p_texture);
	uniforms.push_back(u);
	// anything with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, luminance_reduce_raster.shader.version_get_shader(luminance_reduce_raster.shader_version, 0), 0);

	texture_to_uniform_set_cache[p_texture] = uniform_set;

	return uniform_set;
}

RID EffectsRD::_get_compute_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps) {
	if (texture_to_compute_uniform_set_cache.has(p_texture)) {
		RID uniform_set = texture_to_compute_uniform_set_cache[p_texture];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.uniform_type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u.binding = 0;
	u.append_id(p_use_mipmaps ? default_mipmap_sampler : default_sampler);
	u.append_id(p_texture);
	uniforms.push_back(u);
	//any thing with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, luminance_reduce.shader.version_get_shader(luminance_reduce.shader_version, 0), 0);

	texture_to_compute_uniform_set_cache[p_texture] = uniform_set;

	return uniform_set;
}

void EffectsRD::luminance_reduction(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use compute version of luminance reduction with the mobile renderer.");

	luminance_reduce.push_constant.source_size[0] = p_source_size.x;
	luminance_reduce.push_constant.source_size[1] = p_source_size.y;
	luminance_reduce.push_constant.max_luminance = p_max_luminance;
	luminance_reduce.push_constant.min_luminance = p_min_luminance;
	luminance_reduce.push_constant.exposure_adjust = p_adjust;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	for (int i = 0; i < p_reduce.size(); i++) {
		if (i == 0) {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, luminance_reduce.pipelines[LUMINANCE_REDUCE_READ]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_texture), 0);
		} else {
			RD::get_singleton()->compute_list_add_barrier(compute_list); //needs barrier, wait until previous is done

			if (i == p_reduce.size() - 1 && !p_set) {
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, luminance_reduce.pipelines[LUMINANCE_REDUCE_WRITE]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_prev_luminance), 2);
			} else {
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, luminance_reduce.pipelines[LUMINANCE_REDUCE]);
			}

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_reduce[i - 1]), 0);
		}

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_reduce[i]), 1);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &luminance_reduce.push_constant, sizeof(LuminanceReducePushConstant));

		RD::get_singleton()->compute_list_dispatch_threads(compute_list, luminance_reduce.push_constant.source_size[0], luminance_reduce.push_constant.source_size[1], 1);

		luminance_reduce.push_constant.source_size[0] = MAX(luminance_reduce.push_constant.source_size[0] / 8, 1);
		luminance_reduce.push_constant.source_size[1] = MAX(luminance_reduce.push_constant.source_size[1] / 8, 1);
	}

	RD::get_singleton()->compute_list_end();
}

void EffectsRD::luminance_reduction_raster(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, Vector<RID> p_fb, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set) {
	ERR_FAIL_COND_MSG(!prefer_raster_effects, "Can't use raster version of luminance reduction with the clustered renderer.");
	ERR_FAIL_COND_MSG(p_reduce.size() != p_fb.size(), "Incorrect frame buffer account for luminance reduction.");

	luminance_reduce_raster.push_constant.max_luminance = p_max_luminance;
	luminance_reduce_raster.push_constant.min_luminance = p_min_luminance;
	luminance_reduce_raster.push_constant.exposure_adjust = p_adjust;

	for (int i = 0; i < p_reduce.size(); i++) {
		luminance_reduce_raster.push_constant.source_size[0] = i == 0 ? p_source_size.x : luminance_reduce_raster.push_constant.dest_size[0];
		luminance_reduce_raster.push_constant.source_size[1] = i == 0 ? p_source_size.y : luminance_reduce_raster.push_constant.dest_size[1];
		luminance_reduce_raster.push_constant.dest_size[0] = MAX(luminance_reduce_raster.push_constant.source_size[0] / 8, 1);
		luminance_reduce_raster.push_constant.dest_size[1] = MAX(luminance_reduce_raster.push_constant.source_size[1] / 8, 1);

		bool final = !p_set && (luminance_reduce_raster.push_constant.dest_size[0] == 1) && (luminance_reduce_raster.push_constant.dest_size[1] == 1);
		LuminanceReduceRasterMode mode = final ? LUMINANCE_REDUCE_FRAGMENT_FINAL : (i == 0 ? LUMINANCE_REDUCE_FRAGMENT_FIRST : LUMINANCE_REDUCE_FRAGMENT);

		RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_fb[i], RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
		RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, luminance_reduce_raster.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_fb[i])));
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(i == 0 ? p_source_texture : p_reduce[i - 1]), 0);
		if (final) {
			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_prev_luminance), 1);
		}
		RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

		RD::get_singleton()->draw_list_set_push_constant(draw_list, &luminance_reduce_raster.push_constant, sizeof(LuminanceReduceRasterPushConstant));

		RD::get_singleton()->draw_list_draw(draw_list, true);
		RD::get_singleton()->draw_list_end();
	}
}

void EffectsRD::roughness_limit(RID p_source_normal, RID p_roughness, const Size2i &p_size, float p_curve) {
	roughness_limiter.push_constant.screen_size[0] = p_size.x;
	roughness_limiter.push_constant.screen_size[1] = p_size.y;
	roughness_limiter.push_constant.curve = p_curve;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, roughness_limiter.pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_normal), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_roughness), 1);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &roughness_limiter.push_constant, sizeof(RoughnessLimiterPushConstant)); //not used but set anyway

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_size.x, p_size.y, 1);

	RD::get_singleton()->compute_list_end();
}

void EffectsRD::sort_buffer(RID p_uniform_set, int p_size) {
	Sort::PushConstant push_constant;
	push_constant.total_elements = p_size;

	bool done = true;

	int numThreadGroups = ((p_size - 1) >> 9) + 1;

	if (numThreadGroups > 1) {
		done = false;
	}

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sort.pipelines[SORT_MODE_BLOCK]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, p_uniform_set, 1);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(Sort::PushConstant));
	RD::get_singleton()->compute_list_dispatch(compute_list, numThreadGroups, 1, 1);

	int presorted = 512;

	while (!done) {
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		done = true;
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sort.pipelines[SORT_MODE_STEP]);

		numThreadGroups = 0;

		if (p_size > presorted) {
			if (p_size > presorted * 2) {
				done = false;
			}

			int pow2 = presorted;
			while (pow2 < p_size) {
				pow2 *= 2;
			}
			numThreadGroups = pow2 >> 9;
		}

		unsigned int nMergeSize = presorted * 2;

		for (unsigned int nMergeSubSize = nMergeSize >> 1; nMergeSubSize > 256; nMergeSubSize = nMergeSubSize >> 1) {
			push_constant.job_params[0] = nMergeSubSize;
			if (nMergeSubSize == nMergeSize >> 1) {
				push_constant.job_params[1] = (2 * nMergeSubSize - 1);
				push_constant.job_params[2] = -1;
			} else {
				push_constant.job_params[1] = nMergeSubSize;
				push_constant.job_params[2] = 1;
			}
			push_constant.job_params[3] = 0;

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(Sort::PushConstant));
			RD::get_singleton()->compute_list_dispatch(compute_list, numThreadGroups, 1, 1);
			RD::get_singleton()->compute_list_add_barrier(compute_list);
		}

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sort.pipelines[SORT_MODE_INNER]);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(Sort::PushConstant));
		RD::get_singleton()->compute_list_dispatch(compute_list, numThreadGroups, 1, 1);

		presorted *= 2;
	}

	RD::get_singleton()->compute_list_end();
}

EffectsRD::EffectsRD(bool p_prefer_raster_effects) {
	prefer_raster_effects = p_prefer_raster_effects;

	if (prefer_raster_effects) {
		Vector<String> luminance_reduce_modes;
		luminance_reduce_modes.push_back("\n#define FIRST_PASS\n"); // LUMINANCE_REDUCE_FRAGMENT_FIRST
		luminance_reduce_modes.push_back("\n"); // LUMINANCE_REDUCE_FRAGMENT
		luminance_reduce_modes.push_back("\n#define FINAL_PASS\n"); // LUMINANCE_REDUCE_FRAGMENT_FINAL

		luminance_reduce_raster.shader.initialize(luminance_reduce_modes);
		memset(&luminance_reduce_raster.push_constant, 0, sizeof(LuminanceReduceRasterPushConstant));
		luminance_reduce_raster.shader_version = luminance_reduce_raster.shader.version_create();

		for (int i = 0; i < LUMINANCE_REDUCE_FRAGMENT_MAX; i++) {
			luminance_reduce_raster.pipelines[i].setup(luminance_reduce_raster.shader.version_get_shader(luminance_reduce_raster.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}
	} else {
		// Initialize luminance_reduce
		Vector<String> luminance_reduce_modes;
		luminance_reduce_modes.push_back("\n#define READ_TEXTURE\n");
		luminance_reduce_modes.push_back("\n");
		luminance_reduce_modes.push_back("\n#define WRITE_LUMINANCE\n");

		luminance_reduce.shader.initialize(luminance_reduce_modes);

		luminance_reduce.shader_version = luminance_reduce.shader.version_create();

		for (int i = 0; i < LUMINANCE_REDUCE_MAX; i++) {
			luminance_reduce.pipelines[i] = RD::get_singleton()->compute_pipeline_create(luminance_reduce.shader.version_get_shader(luminance_reduce.shader_version, i));
		}

		for (int i = 0; i < LUMINANCE_REDUCE_FRAGMENT_MAX; i++) {
			luminance_reduce_raster.pipelines[i].clear();
		}
	}

	if (!prefer_raster_effects) {
		// Initialize roughness limiter
		Vector<String> shader_modes;
		shader_modes.push_back("");

		roughness_limiter.shader.initialize(shader_modes);

		roughness_limiter.shader_version = roughness_limiter.shader.version_create();

		roughness_limiter.pipeline = RD::get_singleton()->compute_pipeline_create(roughness_limiter.shader.version_get_shader(roughness_limiter.shader_version, 0));
	}

	{
		Vector<String> sort_modes;
		sort_modes.push_back("\n#define MODE_SORT_BLOCK\n");
		sort_modes.push_back("\n#define MODE_SORT_STEP\n");
		sort_modes.push_back("\n#define MODE_SORT_INNER\n");

		sort.shader.initialize(sort_modes);

		sort.shader_version = sort.shader.version_create();

		for (int i = 0; i < SORT_MODE_MAX; i++) {
			sort.pipelines[i] = RD::get_singleton()->compute_pipeline_create(sort.shader.version_get_shader(sort.shader_version, i));
		}
	}

	RD::SamplerState sampler;
	sampler.mag_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.max_lod = 0;

	default_sampler = RD::get_singleton()->sampler_create(sampler);
	RD::get_singleton()->set_resource_name(default_sampler, "Default Linear Sampler");

	sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.mip_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.max_lod = 1e20;

	default_mipmap_sampler = RD::get_singleton()->sampler_create(sampler);
	RD::get_singleton()->set_resource_name(default_mipmap_sampler, "Default MipMap Sampler");

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

EffectsRD::~EffectsRD() {
	RD::get_singleton()->free(default_sampler);
	RD::get_singleton()->free(default_mipmap_sampler);
	RD::get_singleton()->free(index_buffer); //array gets freed as dependency

	if (prefer_raster_effects) {
		luminance_reduce_raster.shader.version_free(luminance_reduce_raster.shader_version);
	} else {
		luminance_reduce.shader.version_free(luminance_reduce.shader_version);
	}
	if (!prefer_raster_effects) {
		roughness_limiter.shader.version_free(roughness_limiter.shader_version);
	}
	sort.shader.version_free(sort.shader_version);
}
