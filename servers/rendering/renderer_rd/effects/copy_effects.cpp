/*************************************************************************/
/*  copy_effects.cpp                                                     */
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

#include "copy_effects.h"
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/renderer_rd/storage_rd/material_storage.h"
#include "servers/rendering/renderer_rd/uniform_set_cache_rd.h"

using namespace RendererRD;

CopyEffects *CopyEffects::singleton = nullptr;

CopyEffects *CopyEffects::get_singleton() {
	return singleton;
}

CopyEffects::CopyEffects(bool p_prefer_raster_effects) {
	singleton = this;
	prefer_raster_effects = p_prefer_raster_effects;

	if (prefer_raster_effects) {
		// init blur shader (on compute use copy shader)

		Vector<String> blur_modes;
		blur_modes.push_back("\n#define MODE_MIPMAP\n"); // BLUR_MIPMAP
		blur_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n"); // BLUR_MODE_GAUSSIAN_BLUR
		blur_modes.push_back("\n#define MODE_GAUSSIAN_GLOW\n"); // BLUR_MODE_GAUSSIAN_GLOW
		blur_modes.push_back("\n#define MODE_GAUSSIAN_GLOW\n#define GLOW_USE_AUTO_EXPOSURE\n"); // BLUR_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE
		blur_modes.push_back("\n#define MODE_COPY\n"); // BLUR_MODE_COPY

		blur_raster.shader.initialize(blur_modes);
		memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));
		blur_raster.shader_version = blur_raster.shader.version_create();

		for (int i = 0; i < BLUR_MODE_MAX; i++) {
			blur_raster.pipelines[i].setup(blur_raster.shader.version_get_shader(blur_raster.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}

	} else {
		// not used in clustered
		for (int i = 0; i < BLUR_MODE_MAX; i++) {
			blur_raster.pipelines[i].clear();
		}

		Vector<String> copy_modes;
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n#define DST_IMAGE_8BIT\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n#define MODE_GLOW\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n#define MODE_GLOW\n#define GLOW_USE_AUTO_EXPOSURE\n");
		copy_modes.push_back("\n#define MODE_SIMPLE_COPY\n");
		copy_modes.push_back("\n#define MODE_SIMPLE_COPY\n#define DST_IMAGE_8BIT\n");
		copy_modes.push_back("\n#define MODE_SIMPLE_COPY_DEPTH\n");
		copy_modes.push_back("\n#define MODE_SET_COLOR\n");
		copy_modes.push_back("\n#define MODE_SET_COLOR\n#define DST_IMAGE_8BIT\n");
		copy_modes.push_back("\n#define MODE_MIPMAP\n");
		copy_modes.push_back("\n#define MODE_LINEARIZE_DEPTH_COPY\n");
		copy_modes.push_back("\n#define MODE_CUBEMAP_TO_PANORAMA\n");
		copy_modes.push_back("\n#define MODE_CUBEMAP_ARRAY_TO_PANORAMA\n");

		copy.shader.initialize(copy_modes);
		memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

		copy.shader_version = copy.shader.version_create();

		for (int i = 0; i < COPY_MODE_MAX; i++) {
			if (copy.shader.is_variant_enabled(i)) {
				copy.pipelines[i] = RD::get_singleton()->compute_pipeline_create(copy.shader.version_get_shader(copy.shader_version, i));
			}
		}
	}

	{
		Vector<String> copy_modes;
		copy_modes.push_back("\n");
		copy_modes.push_back("\n#define MODE_PANORAMA_TO_DP\n");
		copy_modes.push_back("\n#define MODE_TWO_SOURCES\n");
		copy_modes.push_back("\n#define MULTIVIEW\n");
		copy_modes.push_back("\n#define MULTIVIEW\n#define MODE_TWO_SOURCES\n");

		copy_to_fb.shader.initialize(copy_modes);

		if (!RendererCompositorRD::singleton->is_xr_enabled()) {
			copy_to_fb.shader.set_variant_enabled(COPY_TO_FB_MULTIVIEW, false);
			copy_to_fb.shader.set_variant_enabled(COPY_TO_FB_MULTIVIEW_WITH_DEPTH, false);
		}

		copy_to_fb.shader_version = copy_to_fb.shader.version_create();

		//use additive

		for (int i = 0; i < COPY_TO_FB_MAX; i++) {
			if (copy_to_fb.shader.is_variant_enabled(i)) {
				copy_to_fb.pipelines[i].setup(copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
			} else {
				copy_to_fb.pipelines[i].clear();
			}
		}
	}
}

CopyEffects::~CopyEffects() {
	if (prefer_raster_effects) {
		blur_raster.shader.version_free(blur_raster.shader_version);
	} else {
		copy.shader.version_free(copy.shader_version);
	}

	copy_to_fb.shader.version_free(copy_to_fb.shader_version);

	singleton = nullptr;
}

void CopyEffects::copy_to_rect(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y, bool p_force_luminance, bool p_all_source, bool p_8_bit_dst, bool p_alpha_to_one) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the copy_to_rect shader with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));
	if (p_flip_y) {
		copy.push_constant.flags |= COPY_FLAG_FLIP_Y;
	}

	if (p_force_luminance) {
		copy.push_constant.flags |= COPY_FLAG_FORCE_LUMINANCE;
	}

	if (p_all_source) {
		copy.push_constant.flags |= COPY_FLAG_ALL_SOURCE;
	}

	if (p_alpha_to_one) {
		copy.push_constant.flags |= COPY_FLAG_ALPHA_TO_ONE;
	}

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_rect.size.width;
	copy.push_constant.section[3] = p_rect.size.height;
	copy.push_constant.target[0] = p_rect.position.x;
	copy.push_constant.target[1] = p_rect.position.y;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = p_8_bit_dst ? COPY_MODE_SIMPLY_COPY_8BIT : COPY_MODE_SIMPLY_COPY;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_rect.size.width, p_rect.size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::copy_cubemap_to_panorama(RID p_source_cube, RID p_dest_panorama, const Size2i &p_panorama_size, float p_lod, bool p_is_array) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the copy_cubemap_to_panorama shader with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_panorama_size.width;
	copy.push_constant.section[3] = p_panorama_size.height;
	copy.push_constant.target[0] = 0;
	copy.push_constant.target[1] = 0;
	copy.push_constant.camera_z_far = p_lod;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_cube(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_cube }));
	RD::Uniform u_dest_panorama(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_panorama);

	CopyMode mode = p_is_array ? COPY_MODE_CUBE_ARRAY_TO_PANORAMA : COPY_MODE_CUBE_TO_PANORAMA;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_cube), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_panorama), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_panorama_size.width, p_panorama_size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::copy_depth_to_rect(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the copy_depth_to_rect shader with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));
	if (p_flip_y) {
		copy.push_constant.flags |= COPY_FLAG_FLIP_Y;
	}

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_rect.size.width;
	copy.push_constant.section[3] = p_rect.size.height;
	copy.push_constant.target[0] = p_rect.position.x;
	copy.push_constant.target[1] = p_rect.position.y;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = COPY_MODE_SIMPLY_COPY_DEPTH;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_rect.size.width, p_rect.size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::copy_depth_to_rect_and_linearize(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y, float p_z_near, float p_z_far) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the copy_depth_to_rect_and_linearize shader with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));
	if (p_flip_y) {
		copy.push_constant.flags |= COPY_FLAG_FLIP_Y;
	}

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_rect.size.width;
	copy.push_constant.section[3] = p_rect.size.height;
	copy.push_constant.target[0] = p_rect.position.x;
	copy.push_constant.target[1] = p_rect.position.y;
	copy.push_constant.camera_z_far = p_z_far;
	copy.push_constant.camera_z_near = p_z_near;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = COPY_MODE_LINEARIZE_DEPTH;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_rect.size.width, p_rect.size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::copy_to_atlas_fb(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_uv_rect, RD::DrawListID p_draw_list, bool p_flip_y, bool p_panorama) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the copy_to_atlas_fb shader with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy_to_fb.push_constant, 0, sizeof(CopyToFbPushConstant));

	copy_to_fb.push_constant.use_section = true;
	copy_to_fb.push_constant.section[0] = p_uv_rect.position.x;
	copy_to_fb.push_constant.section[1] = p_uv_rect.position.y;
	copy_to_fb.push_constant.section[2] = p_uv_rect.size.x;
	copy_to_fb.push_constant.section[3] = p_uv_rect.size.y;

	if (p_flip_y) {
		copy_to_fb.push_constant.flip_y = true;
	}

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	CopyToFBMode mode = p_panorama ? COPY_TO_FB_COPY_PANORAMA_TO_DP : COPY_TO_FB_COPY;
	RID shader = copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = p_draw_list;
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy_to_fb.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
}

void CopyEffects::copy_to_fb_rect(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2i &p_rect, bool p_flip_y, bool p_force_luminance, bool p_alpha_to_zero, bool p_srgb, RID p_secondary, bool p_multiview) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the copy_to_fb_rect shader with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy_to_fb.push_constant, 0, sizeof(CopyToFbPushConstant));

	if (p_flip_y) {
		copy_to_fb.push_constant.flip_y = true;
	}
	if (p_force_luminance) {
		copy_to_fb.push_constant.force_luminance = true;
	}
	if (p_alpha_to_zero) {
		copy_to_fb.push_constant.alpha_to_zero = true;
	}
	if (p_srgb) {
		copy_to_fb.push_constant.srgb = true;
	}

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	CopyToFBMode mode;
	if (p_multiview) {
		mode = p_secondary.is_valid() ? COPY_TO_FB_MULTIVIEW_WITH_DEPTH : COPY_TO_FB_MULTIVIEW;
	} else {
		mode = p_secondary.is_valid() ? COPY_TO_FB_COPY2 : COPY_TO_FB_COPY;
	}

	RID shader = copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, Vector<Color>(), 1.0, 0, p_rect);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy_to_fb.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	if (p_secondary.is_valid()) {
		// TODO may need to do this differently when reading from depth buffer for multiview
		RD::Uniform u_secondary(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_secondary }));
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 1, u_secondary), 1);
	}
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::copy_raster(RID p_source_texture, RID p_dest_framebuffer) {
	ERR_FAIL_COND_MSG(!prefer_raster_effects, "Can't use the raster version of the copy with the clustered renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_texture }));

	RID shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, BLUR_MODE_COPY);
	ERR_FAIL_COND(shader.is_null());

	// Just copy it back (we use our blur raster shader here)..
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[BLUR_MODE_COPY].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::gaussian_blur(RID p_source_rd_texture, RID p_texture, const Rect2i &p_region, bool p_8bit_dst) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the gaussian blur with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = p_region.position.x;
	copy.push_constant.section[1] = p_region.position.y;
	copy.push_constant.section[2] = p_region.size.width;
	copy.push_constant.section[3] = p_region.size.height;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_texture);

	CopyMode mode = p_8bit_dst ? COPY_MODE_GAUSSIAN_COPY_8BIT : COPY_MODE_GAUSSIAN_COPY;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	//HORIZONTAL
	RD::DrawListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_texture), 3);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_region.size.width, p_region.size.height, 1);

	RD::get_singleton()->compute_list_end();
}

void CopyEffects::gaussian_glow(RID p_source_rd_texture, RID p_back_texture, const Size2i &p_size, float p_strength, bool p_high_quality, bool p_first_pass, float p_luminance_cap, float p_exposure, float p_bloom, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, RID p_auto_exposure, float p_auto_exposure_grey) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the gaussian glow with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	CopyMode copy_mode = p_first_pass && p_auto_exposure.is_valid() ? COPY_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE : COPY_MODE_GAUSSIAN_GLOW;
	uint32_t base_flags = 0;

	copy.push_constant.section[2] = p_size.x;
	copy.push_constant.section[3] = p_size.y;

	copy.push_constant.glow_strength = p_strength;
	copy.push_constant.glow_bloom = p_bloom;
	copy.push_constant.glow_hdr_threshold = p_hdr_bleed_threshold;
	copy.push_constant.glow_hdr_scale = p_hdr_bleed_scale;
	copy.push_constant.glow_exposure = p_exposure;
	copy.push_constant.glow_white = 0; //actually unused
	copy.push_constant.glow_luminance_cap = p_luminance_cap;

	copy.push_constant.glow_auto_exposure_grey = p_auto_exposure_grey; //unused also

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_back_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_back_texture);

	RID shader = copy.shader.version_get_shader(copy.shader_version, copy_mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[copy_mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_back_texture), 3);
	if (p_auto_exposure.is_valid() && p_first_pass) {
		RD::Uniform u_auto_exposure(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_auto_exposure }));
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 1, u_auto_exposure), 1);
	}

	copy.push_constant.flags = base_flags | (p_first_pass ? COPY_FLAG_GLOW_FIRST_PASS : 0) | (p_high_quality ? COPY_FLAG_HIGH_QUALITY_GLOW : 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_size.width, p_size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::gaussian_glow_raster(RID p_source_rd_texture, float p_luminance_multiplier, RID p_framebuffer_half, RID p_rd_texture_half, RID p_dest_framebuffer, const Size2i &p_size, float p_strength, bool p_high_quality, bool p_first_pass, float p_luminance_cap, float p_exposure, float p_bloom, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, RID p_auto_exposure, float p_auto_exposure_grey) {
	ERR_FAIL_COND_MSG(!prefer_raster_effects, "Can't use the raster version of the gaussian glow with the clustered renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));

	BlurRasterMode blur_mode = p_first_pass && p_auto_exposure.is_valid() ? BLUR_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE : BLUR_MODE_GAUSSIAN_GLOW;
	uint32_t base_flags = 0;

	blur_raster.push_constant.pixel_size[0] = 1.0 / float(p_size.x);
	blur_raster.push_constant.pixel_size[1] = 1.0 / float(p_size.y);

	blur_raster.push_constant.glow_strength = p_strength;
	blur_raster.push_constant.glow_bloom = p_bloom;
	blur_raster.push_constant.glow_hdr_threshold = p_hdr_bleed_threshold;
	blur_raster.push_constant.glow_hdr_scale = p_hdr_bleed_scale;
	blur_raster.push_constant.glow_exposure = p_exposure;
	blur_raster.push_constant.glow_white = 0; //actually unused
	blur_raster.push_constant.glow_luminance_cap = p_luminance_cap;

	blur_raster.push_constant.glow_auto_exposure_grey = p_auto_exposure_grey; //unused also

	blur_raster.push_constant.luminance_multiplier = p_luminance_multiplier;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_rd_texture_half(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_rd_texture_half }));

	RID shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, blur_mode);
	ERR_FAIL_COND(shader.is_null());

	//HORIZONTAL
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer_half, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[blur_mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer_half)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	if (p_auto_exposure.is_valid() && p_first_pass) {
		RD::Uniform u_auto_exposure(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_auto_exposure }));
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 1, u_auto_exposure), 1);
	}
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());

	blur_raster.push_constant.flags = base_flags | BLUR_FLAG_HORIZONTAL | (p_first_pass ? BLUR_FLAG_GLOW_FIRST_PASS : 0);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();

	blur_mode = BLUR_MODE_GAUSSIAN_GLOW;

	shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, blur_mode);
	ERR_FAIL_COND(shader.is_null());

	//VERTICAL
	draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[blur_mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_rd_texture_half), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());

	blur_raster.push_constant.flags = base_flags;
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::make_mipmap(RID p_source_rd_texture, RID p_dest_texture, const Size2i &p_size) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the make_mipmap shader with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_size.width;
	copy.push_constant.section[3] = p_size.height;

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = COPY_MODE_MIPMAP;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_size.width, p_size.height, 1);
	RD::get_singleton()->compute_list_end();
}

void CopyEffects::make_mipmap_raster(RID p_source_rd_texture, RID p_dest_framebuffer, const Size2i &p_size) {
	ERR_FAIL_COND_MSG(!prefer_raster_effects, "Can't use the raster version of mipmap with the clustered renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);
	MaterialStorage *material_storage = MaterialStorage::get_singleton();
	ERR_FAIL_NULL(material_storage);

	memset(&blur_raster.push_constant, 0, sizeof(BlurRasterPushConstant));

	BlurRasterMode mode = BLUR_MIPMAP;

	blur_raster.push_constant.pixel_size[0] = 1.0 / float(p_size.x);
	blur_raster.push_constant.pixel_size[1] = 1.0 / float(p_size.y);

	// setup our uniforms
	RID default_sampler = material_storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED);

	RD::Uniform u_source_rd_texture(RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE, 0, Vector<RID>({ default_sampler, p_source_rd_texture }));

	RID shader = blur_raster.shader.version_get_shader(blur_raster.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur_raster.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, uniform_set_cache->get_cache(shader, 0, u_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, material_storage->get_quad_index_array());
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur_raster.push_constant, sizeof(BlurRasterPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void CopyEffects::set_color(RID p_dest_texture, const Color &p_color, const Rect2i &p_region, bool p_8bit_dst) {
	ERR_FAIL_COND_MSG(prefer_raster_effects, "Can't use the compute version of the set_color shader with the mobile renderer.");

	UniformSetCacheRD *uniform_set_cache = UniformSetCacheRD::get_singleton();
	ERR_FAIL_NULL(uniform_set_cache);

	memset(&copy.push_constant, 0, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_region.size.width;
	copy.push_constant.section[3] = p_region.size.height;
	copy.push_constant.target[0] = p_region.position.x;
	copy.push_constant.target[1] = p_region.position.y;
	copy.push_constant.set_color[0] = p_color.r;
	copy.push_constant.set_color[1] = p_color.g;
	copy.push_constant.set_color[2] = p_color.b;
	copy.push_constant.set_color[3] = p_color.a;

	// setup our uniforms
	RD::Uniform u_dest_texture(RD::UNIFORM_TYPE_IMAGE, 0, p_dest_texture);

	CopyMode mode = p_8bit_dst ? COPY_MODE_SET_COLOR_8BIT : COPY_MODE_SET_COLOR;
	RID shader = copy.shader.version_get_shader(copy.shader_version, mode);
	ERR_FAIL_COND(shader.is_null());

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, uniform_set_cache->get_cache(shader, 3, u_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_region.size.width, p_region.size.height, 1);
	RD::get_singleton()->compute_list_end();
}
