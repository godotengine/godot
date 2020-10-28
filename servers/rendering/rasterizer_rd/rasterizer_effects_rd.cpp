/*************************************************************************/
/*  rasterizer_effects_rd.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "rasterizer_effects_rd.h"

#include "core/os/os.h"
#include "core/project_settings.h"

#include "thirdparty/misc/cubemap_coeffs.h"

static _FORCE_INLINE_ void store_transform_3x3(const Basis &p_basis, float *p_array) {
	p_array[0] = p_basis.elements[0][0];
	p_array[1] = p_basis.elements[1][0];
	p_array[2] = p_basis.elements[2][0];
	p_array[3] = 0;
	p_array[4] = p_basis.elements[0][1];
	p_array[5] = p_basis.elements[1][1];
	p_array[6] = p_basis.elements[2][1];
	p_array[7] = 0;
	p_array[8] = p_basis.elements[0][2];
	p_array[9] = p_basis.elements[1][2];
	p_array[10] = p_basis.elements[2][2];
	p_array[11] = 0;
}

static _FORCE_INLINE_ void store_camera(const CameraMatrix &p_mtx, float *p_array) {
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			p_array[i * 4 + j] = p_mtx.matrix[i][j];
		}
	}
}

RID RasterizerEffectsRD::_get_uniform_set_from_image(RID p_image) {
	if (image_to_uniform_set_cache.has(p_image)) {
		RID uniform_set = image_to_uniform_set_cache[p_image];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}
	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.type = RD::UNIFORM_TYPE_IMAGE;
	u.binding = 0;
	u.ids.push_back(p_image);
	uniforms.push_back(u);
	//any thing with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, luminance_reduce.shader.version_get_shader(luminance_reduce.shader_version, 0), 1);

	image_to_uniform_set_cache[p_image] = uniform_set;

	return uniform_set;
}

RID RasterizerEffectsRD::_get_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps) {
	if (texture_to_uniform_set_cache.has(p_texture)) {
		RID uniform_set = texture_to_uniform_set_cache[p_texture];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u.binding = 0;
	u.ids.push_back(p_use_mipmaps ? default_mipmap_sampler : default_sampler);
	u.ids.push_back(p_texture);
	uniforms.push_back(u);
	//any thing with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, tonemap.shader.version_get_shader(tonemap.shader_version, 0), 0);

	texture_to_uniform_set_cache[p_texture] = uniform_set;

	return uniform_set;
}

RID RasterizerEffectsRD::_get_compute_uniform_set_from_texture(RID p_texture, bool p_use_mipmaps) {
	if (texture_to_compute_uniform_set_cache.has(p_texture)) {
		RID uniform_set = texture_to_compute_uniform_set_cache[p_texture];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}

	Vector<RD::Uniform> uniforms;
	RD::Uniform u;
	u.type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
	u.binding = 0;
	u.ids.push_back(p_use_mipmaps ? default_mipmap_sampler : default_sampler);
	u.ids.push_back(p_texture);
	uniforms.push_back(u);
	//any thing with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, luminance_reduce.shader.version_get_shader(luminance_reduce.shader_version, 0), 0);

	texture_to_compute_uniform_set_cache[p_texture] = uniform_set;

	return uniform_set;
}

RID RasterizerEffectsRD::_get_compute_uniform_set_from_texture_pair(RID p_texture1, RID p_texture2, bool p_use_mipmaps) {
	TexturePair tp;
	tp.texture1 = p_texture1;
	tp.texture2 = p_texture2;

	if (texture_pair_to_compute_uniform_set_cache.has(tp)) {
		RID uniform_set = texture_pair_to_compute_uniform_set_cache[tp];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}

	Vector<RD::Uniform> uniforms;
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
		u.binding = 0;
		u.ids.push_back(p_use_mipmaps ? default_mipmap_sampler : default_sampler);
		u.ids.push_back(p_texture1);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_SAMPLER_WITH_TEXTURE;
		u.binding = 1;
		u.ids.push_back(p_use_mipmaps ? default_mipmap_sampler : default_sampler);
		u.ids.push_back(p_texture2);
		uniforms.push_back(u);
	}
	//any thing with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ssr_scale.shader.version_get_shader(ssr_scale.shader_version, 0), 1);

	texture_pair_to_compute_uniform_set_cache[tp] = uniform_set;

	return uniform_set;
}

RID RasterizerEffectsRD::_get_compute_uniform_set_from_image_pair(RID p_texture1, RID p_texture2) {
	TexturePair tp;
	tp.texture1 = p_texture1;
	tp.texture2 = p_texture2;

	if (image_pair_to_compute_uniform_set_cache.has(tp)) {
		RID uniform_set = image_pair_to_compute_uniform_set_cache[tp];
		if (RD::get_singleton()->uniform_set_is_valid(uniform_set)) {
			return uniform_set;
		}
	}

	Vector<RD::Uniform> uniforms;
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_IMAGE;
		u.binding = 0;
		u.ids.push_back(p_texture1);
		uniforms.push_back(u);
	}
	{
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_IMAGE;
		u.binding = 1;
		u.ids.push_back(p_texture2);
		uniforms.push_back(u);
	}
	//any thing with the same configuration (one texture in binding 0 for set 0), is good
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, ssr_scale.shader.version_get_shader(ssr_scale.shader_version, 0), 3);

	image_pair_to_compute_uniform_set_cache[tp] = uniform_set;

	return uniform_set;
}

void RasterizerEffectsRD::copy_to_atlas_fb(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_uv_rect, RD::DrawListID p_draw_list, bool p_flip_y, bool p_panorama) {
	zeromem(&copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));

	copy_to_fb.push_constant.use_section = true;
	copy_to_fb.push_constant.section[0] = p_uv_rect.position.x;
	copy_to_fb.push_constant.section[1] = p_uv_rect.position.y;
	copy_to_fb.push_constant.section[2] = p_uv_rect.size.x;
	copy_to_fb.push_constant.section[3] = p_uv_rect.size.y;

	if (p_flip_y) {
		copy_to_fb.push_constant.flip_y = true;
	}

	RD::DrawListID draw_list = p_draw_list;
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy_to_fb.pipelines[p_panorama ? COPY_TO_FB_COPY_PANORAMA_TO_DP : COPY_TO_FB_COPY].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
}

void RasterizerEffectsRD::copy_to_fb_rect(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2i &p_rect, bool p_flip_y, bool p_force_luminance, bool p_alpha_to_zero, bool p_srgb, RID p_secondary) {
	zeromem(&copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));

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

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, Vector<Color>(), 1.0, 0, p_rect);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy_to_fb.pipelines[p_secondary.is_valid() ? COPY_TO_FB_COPY2 : COPY_TO_FB_COPY].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	if (p_secondary.is_valid()) {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_secondary), 1);
	}
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &copy_to_fb.push_constant, sizeof(CopyToFbPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::copy_to_rect(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y, bool p_force_luminance, bool p_all_source, bool p_8_bit_dst, bool p_alpha_to_one) {
	zeromem(&copy.push_constant, sizeof(CopyPushConstant));
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

	int32_t x_groups = (p_rect.size.width - 1) / 8 + 1;
	int32_t y_groups = (p_rect.size.height - 1) / 8 + 1;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[p_8_bit_dst ? COPY_MODE_SIMPLY_COPY_8BIT : COPY_MODE_SIMPLY_COPY]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::copy_cubemap_to_panorama(RID p_source_cube, RID p_dest_panorama, const Size2i &p_panorama_size, float p_lod, bool p_is_array) {
	zeromem(&copy.push_constant, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_panorama_size.width;
	copy.push_constant.section[3] = p_panorama_size.height;
	copy.push_constant.target[0] = 0;
	copy.push_constant.target[1] = 0;
	copy.push_constant.camera_z_far = p_lod;

	int32_t x_groups = (p_panorama_size.width - 1) / 8 + 1;
	int32_t y_groups = (p_panorama_size.height - 1) / 8 + 1;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[p_is_array ? COPY_MODE_CUBE_ARRAY_TO_PANORAMA : COPY_MODE_CUBE_TO_PANORAMA]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_cube), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_panorama), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::copy_depth_to_rect_and_linearize(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y, float p_z_near, float p_z_far) {
	zeromem(&copy.push_constant, sizeof(CopyPushConstant));
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

	int32_t x_groups = (p_rect.size.width - 1) / 8 + 1;
	int32_t y_groups = (p_rect.size.height - 1) / 8 + 1;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[COPY_MODE_LINEARIZE_DEPTH]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::copy_depth_to_rect(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, bool p_flip_y) {
	zeromem(&copy.push_constant, sizeof(CopyPushConstant));
	if (p_flip_y) {
		copy.push_constant.flags |= COPY_FLAG_FLIP_Y;
	}

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_rect.size.width;
	copy.push_constant.section[3] = p_rect.size.height;
	copy.push_constant.target[0] = p_rect.position.x;
	copy.push_constant.target[1] = p_rect.position.y;

	int32_t x_groups = (p_rect.size.width - 1) / 8 + 1;
	int32_t y_groups = (p_rect.size.height - 1) / 8 + 1;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[COPY_MODE_SIMPLY_COPY_DEPTH]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::set_color(RID p_dest_texture, const Color &p_color, const Rect2i &p_region, bool p_8bit_dst) {
	zeromem(&copy.push_constant, sizeof(CopyPushConstant));

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

	int32_t x_groups = (p_region.size.width - 1) / 8 + 1;
	int32_t y_groups = (p_region.size.height - 1) / 8 + 1;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[p_8bit_dst ? COPY_MODE_SET_COLOR_8BIT : COPY_MODE_SET_COLOR]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::gaussian_blur(RID p_source_rd_texture, RID p_texture, RID p_back_texture, const Rect2i &p_region, bool p_8bit_dst) {
	zeromem(&copy.push_constant, sizeof(CopyPushConstant));

	uint32_t base_flags = 0;
	copy.push_constant.section[0] = p_region.position.x;
	copy.push_constant.section[1] = p_region.position.y;
	copy.push_constant.section[2] = p_region.size.width;
	copy.push_constant.section[3] = p_region.size.height;

	int32_t x_groups = (p_region.size.width - 1) / 8 + 1;
	int32_t y_groups = (p_region.size.height - 1) / 8 + 1;
	//HORIZONTAL
	RD::DrawListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[p_8bit_dst ? COPY_MODE_GAUSSIAN_COPY_8BIT : COPY_MODE_GAUSSIAN_COPY]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_back_texture), 3);

	copy.push_constant.flags = base_flags | COPY_FLAG_HORIZONTAL;
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

	RD::get_singleton()->compute_list_add_barrier(compute_list);

	//VERTICAL
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_back_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_texture), 3);

	copy.push_constant.flags = base_flags;
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::gaussian_glow(RID p_source_rd_texture, RID p_back_texture, const Size2i &p_size, float p_strength, bool p_high_quality, bool p_first_pass, float p_luminance_cap, float p_exposure, float p_bloom, float p_hdr_bleed_treshold, float p_hdr_bleed_scale, RID p_auto_exposure, float p_auto_exposure_grey) {
	zeromem(&copy.push_constant, sizeof(CopyPushConstant));

	CopyMode copy_mode = p_first_pass && p_auto_exposure.is_valid() ? COPY_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE : COPY_MODE_GAUSSIAN_GLOW;
	uint32_t base_flags = 0;

	int32_t x_groups = (p_size.width + 7) / 8;
	int32_t y_groups = (p_size.height + 7) / 8;

	copy.push_constant.section[2] = p_size.x;
	copy.push_constant.section[3] = p_size.y;

	copy.push_constant.glow_strength = p_strength;
	copy.push_constant.glow_bloom = p_bloom;
	copy.push_constant.glow_hdr_threshold = p_hdr_bleed_treshold;
	copy.push_constant.glow_hdr_scale = p_hdr_bleed_scale;
	copy.push_constant.glow_exposure = p_exposure;
	copy.push_constant.glow_white = 0; //actually unused
	copy.push_constant.glow_luminance_cap = p_luminance_cap;

	copy.push_constant.glow_auto_exposure_grey = p_auto_exposure_grey; //unused also

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[copy_mode]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_back_texture), 3);
	if (p_auto_exposure.is_valid() && p_first_pass) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_auto_exposure), 1);
	}

	copy.push_constant.flags = base_flags | (p_first_pass ? COPY_FLAG_GLOW_FIRST_PASS : 0) | (p_high_quality ? COPY_FLAG_HIGH_QUALITY_GLOW : 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::screen_space_reflection(RID p_diffuse, RID p_normal_roughness, RenderingServer::EnvironmentSSRRoughnessQuality p_roughness_quality, RID p_blur_radius, RID p_blur_radius2, RID p_metallic, const Color &p_metallic_mask, RID p_depth, RID p_scale_depth, RID p_scale_normal, RID p_output, RID p_output_blur, const Size2i &p_screen_size, int p_max_steps, float p_fade_in, float p_fade_out, float p_tolerance, const CameraMatrix &p_camera) {
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	int32_t x_groups = (p_screen_size.width - 1) / 8 + 1;
	int32_t y_groups = (p_screen_size.height - 1) / 8 + 1;

	{ //scale color and depth to half
		ssr_scale.push_constant.camera_z_far = p_camera.get_z_far();
		ssr_scale.push_constant.camera_z_near = p_camera.get_z_near();
		ssr_scale.push_constant.orthogonal = p_camera.is_orthogonal();
		ssr_scale.push_constant.filter = false; //enabling causes arctifacts
		ssr_scale.push_constant.screen_size[0] = p_screen_size.x;
		ssr_scale.push_constant.screen_size[1] = p_screen_size.y;

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssr_scale.pipeline);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_diffuse), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture_pair(p_depth, p_normal_roughness), 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_output_blur), 2);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_scale_depth, p_scale_normal), 3);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssr_scale.push_constant, sizeof(ScreenSpaceReflectionScalePushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}

	{
		ssr.push_constant.camera_z_far = p_camera.get_z_far();
		ssr.push_constant.camera_z_near = p_camera.get_z_near();
		ssr.push_constant.orthogonal = p_camera.is_orthogonal();
		ssr.push_constant.screen_size[0] = p_screen_size.x;
		ssr.push_constant.screen_size[1] = p_screen_size.y;
		ssr.push_constant.curve_fade_in = p_fade_in;
		ssr.push_constant.distance_fade = p_fade_out;
		ssr.push_constant.num_steps = p_max_steps;
		ssr.push_constant.depth_tolerance = p_tolerance;
		ssr.push_constant.use_half_res = true;
		ssr.push_constant.proj_info[0] = -2.0f / (p_screen_size.width * p_camera.matrix[0][0]);
		ssr.push_constant.proj_info[1] = -2.0f / (p_screen_size.height * p_camera.matrix[1][1]);
		ssr.push_constant.proj_info[2] = (1.0f - p_camera.matrix[0][2]) / p_camera.matrix[0][0];
		ssr.push_constant.proj_info[3] = (1.0f + p_camera.matrix[1][2]) / p_camera.matrix[1][1];
		ssr.push_constant.metallic_mask[0] = CLAMP(p_metallic_mask.r * 255.0, 0, 255);
		ssr.push_constant.metallic_mask[1] = CLAMP(p_metallic_mask.g * 255.0, 0, 255);
		ssr.push_constant.metallic_mask[2] = CLAMP(p_metallic_mask.b * 255.0, 0, 255);
		ssr.push_constant.metallic_mask[3] = CLAMP(p_metallic_mask.a * 255.0, 0, 255);
		store_camera(p_camera, ssr.push_constant.projection);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssr.pipelines[(p_roughness_quality != RS::ENV_SSR_ROUGNESS_QUALITY_DISABLED) ? SCREEN_SPACE_REFLECTION_ROUGH : SCREEN_SPACE_REFLECTION_NORMAL]);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssr.push_constant, sizeof(ScreenSpaceReflectionPushConstant));

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_output_blur, p_scale_depth), 0);

		if (p_roughness_quality != RS::ENV_SSR_ROUGNESS_QUALITY_DISABLED) {
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_output, p_blur_radius), 1);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture_pair(p_metallic, p_normal_roughness), 3);
		} else {
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_output), 1);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_metallic), 3);
		}
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_scale_normal), 2);

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	}

	if (p_roughness_quality != RS::ENV_SSR_ROUGNESS_QUALITY_DISABLED) {
		//blur

		RD::get_singleton()->compute_list_add_barrier(compute_list);

		ssr_filter.push_constant.orthogonal = p_camera.is_orthogonal();
		ssr_filter.push_constant.edge_tolerance = Math::sin(Math::deg2rad(15.0));
		ssr_filter.push_constant.proj_info[0] = -2.0f / (p_screen_size.width * p_camera.matrix[0][0]);
		ssr_filter.push_constant.proj_info[1] = -2.0f / (p_screen_size.height * p_camera.matrix[1][1]);
		ssr_filter.push_constant.proj_info[2] = (1.0f - p_camera.matrix[0][2]) / p_camera.matrix[0][0];
		ssr_filter.push_constant.proj_info[3] = (1.0f + p_camera.matrix[1][2]) / p_camera.matrix[1][1];
		ssr_filter.push_constant.vertical = 0;
		if (p_roughness_quality == RS::ENV_SSR_ROUGNESS_QUALITY_LOW) {
			ssr_filter.push_constant.steps = p_max_steps / 3;
			ssr_filter.push_constant.increment = 3;
		} else if (p_roughness_quality == RS::ENV_SSR_ROUGNESS_QUALITY_MEDIUM) {
			ssr_filter.push_constant.steps = p_max_steps / 2;
			ssr_filter.push_constant.increment = 2;
		} else {
			ssr_filter.push_constant.steps = p_max_steps;
			ssr_filter.push_constant.increment = 1;
		}

		ssr_filter.push_constant.screen_size[0] = p_screen_size.width;
		ssr_filter.push_constant.screen_size[1] = p_screen_size.height;

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssr_filter.pipelines[SCREEN_SPACE_REFLECTION_FILTER_HORIZONTAL]);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_output, p_blur_radius), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_scale_normal), 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_output_blur, p_blur_radius2), 2);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_scale_depth), 3);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssr_filter.push_constant, sizeof(ScreenSpaceReflectionFilterPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

		RD::get_singleton()->compute_list_add_barrier(compute_list);

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssr_filter.pipelines[SCREEN_SPACE_REFLECTION_FILTER_VERTICAL]);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_output_blur, p_blur_radius2), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_scale_normal), 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_output), 2);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_scale_depth), 3);

		ssr_filter.push_constant.vertical = 1;

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssr_filter.push_constant, sizeof(ScreenSpaceReflectionFilterPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	}

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::sub_surface_scattering(RID p_diffuse, RID p_diffuse2, RID p_depth, const CameraMatrix &p_camera, const Size2i &p_screen_size, float p_scale, float p_depth_scale, RenderingServer::SubSurfaceScatteringQuality p_quality) {
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	int32_t x_groups = (p_screen_size.width - 1) / 8 + 1;
	int32_t y_groups = (p_screen_size.height - 1) / 8 + 1;

	Plane p = p_camera.xform4(Plane(1, 0, -1, 1));
	p.normal /= p.d;
	float unit_size = p.normal.x;

	{ //scale color and depth to half
		sss.push_constant.camera_z_far = p_camera.get_z_far();
		sss.push_constant.camera_z_near = p_camera.get_z_near();
		sss.push_constant.orthogonal = p_camera.is_orthogonal();
		sss.push_constant.unit_size = unit_size;
		sss.push_constant.screen_size[0] = p_screen_size.x;
		sss.push_constant.screen_size[1] = p_screen_size.y;
		sss.push_constant.vertical = false;
		sss.push_constant.scale = p_scale;
		sss.push_constant.depth_scale = p_depth_scale;

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, sss.pipelines[p_quality - 1]);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_diffuse), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_diffuse2), 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth), 2);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &sss.push_constant, sizeof(SubSurfaceScatteringPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

		RD::get_singleton()->compute_list_add_barrier(compute_list);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_diffuse2), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_diffuse), 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth), 2);

		sss.push_constant.vertical = true;
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &sss.push_constant, sizeof(SubSurfaceScatteringPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

		RD::get_singleton()->compute_list_end();
	}
}

void RasterizerEffectsRD::merge_specular(RID p_dest_framebuffer, RID p_specular, RID p_base, RID p_reflection) {
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, Vector<Color>());

	if (p_reflection.is_valid()) {
		if (p_base.is_valid()) {
			RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, specular_merge.pipelines[SPECULAR_MERGE_SSR].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_base), 2);
		} else {
			RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, specular_merge.pipelines[SPECULAR_MERGE_ADDITIVE_SSR].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
		}

		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_specular), 0);
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_reflection), 1);

	} else {
		if (p_base.is_valid()) {
			RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, specular_merge.pipelines[SPECULAR_MERGE_ADD].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
			RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_base), 2);
		} else {
			RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, specular_merge.pipelines[SPECULAR_MERGE_ADDITIVE_ADD].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
		}

		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_specular), 0);
	}

	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::make_mipmap(RID p_source_rd_texture, RID p_dest_texture, const Size2i &p_size) {
	zeromem(&copy.push_constant, sizeof(CopyPushConstant));

	copy.push_constant.section[0] = 0;
	copy.push_constant.section[1] = 0;
	copy.push_constant.section[2] = p_size.width;
	copy.push_constant.section[3] = p_size.height;

	int32_t x_groups = (p_size.width - 1) / 8 + 1;
	int32_t y_groups = (p_size.height - 1) / 8 + 1;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, copy.pipelines[COPY_MODE_MIPMAP]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_texture), 3);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &copy.push_constant, sizeof(CopyPushConstant));
	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::copy_cubemap_to_dp(RID p_source_rd_texture, RID p_dest_texture, const Rect2i &p_rect, float p_z_near, float p_z_far, float p_bias, bool p_dp_flip) {
	CopyToDPPushConstant push_constant;
	push_constant.screen_size[0] = p_rect.size.x;
	push_constant.screen_size[1] = p_rect.size.y;
	push_constant.dest_offset[0] = p_rect.position.x;
	push_constant.dest_offset[1] = p_rect.position.y;
	push_constant.bias = p_bias;
	push_constant.z_far = p_z_far;
	push_constant.z_near = p_z_near;
	push_constant.z_flip = p_dp_flip;

	int32_t x_groups = (p_rect.size.width - 1) / 8 + 1;
	int32_t y_groups = (p_rect.size.height - 1) / 8 + 1;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, cube_to_dp.pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_texture), 1);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(CopyToDPPushConstant));
	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::tonemapper(RID p_source_color, RID p_dst_framebuffer, const TonemapSettings &p_settings) {
	zeromem(&tonemap.push_constant, sizeof(TonemapPushConstant));

	tonemap.push_constant.use_bcs = p_settings.use_bcs;
	tonemap.push_constant.bcs[0] = p_settings.brightness;
	tonemap.push_constant.bcs[1] = p_settings.contrast;
	tonemap.push_constant.bcs[2] = p_settings.saturation;

	tonemap.push_constant.use_glow = p_settings.use_glow;
	tonemap.push_constant.glow_intensity = p_settings.glow_intensity;
	tonemap.push_constant.glow_levels[0] = p_settings.glow_levels[0]; // clean this up to just pass by pointer or something
	tonemap.push_constant.glow_levels[1] = p_settings.glow_levels[1];
	tonemap.push_constant.glow_levels[2] = p_settings.glow_levels[2];
	tonemap.push_constant.glow_levels[3] = p_settings.glow_levels[3];
	tonemap.push_constant.glow_levels[4] = p_settings.glow_levels[4];
	tonemap.push_constant.glow_levels[5] = p_settings.glow_levels[5];
	tonemap.push_constant.glow_levels[6] = p_settings.glow_levels[6];
	tonemap.push_constant.glow_texture_size[0] = p_settings.glow_texture_size.x;
	tonemap.push_constant.glow_texture_size[1] = p_settings.glow_texture_size.y;
	tonemap.push_constant.glow_mode = p_settings.glow_mode;

	TonemapMode mode = p_settings.glow_use_bicubic_upscale ? TONEMAP_MODE_BICUBIC_GLOW_FILTER : TONEMAP_MODE_NORMAL;

	tonemap.push_constant.tonemapper = p_settings.tonemap_mode;
	tonemap.push_constant.use_auto_exposure = p_settings.use_auto_exposure;
	tonemap.push_constant.exposure = p_settings.exposure;
	tonemap.push_constant.white = p_settings.white;
	tonemap.push_constant.auto_exposure_grey = p_settings.auto_exposure_grey;

	tonemap.push_constant.use_color_correction = p_settings.use_color_correction;

	tonemap.push_constant.use_fxaa = p_settings.use_fxaa;
	tonemap.push_constant.use_debanding = p_settings.use_debanding;
	tonemap.push_constant.pixel_size[0] = 1.0 / p_settings.texture_size.x;
	tonemap.push_constant.pixel_size[1] = 1.0 / p_settings.texture_size.y;

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer, RD::INITIAL_ACTION_DROP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_DROP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, tonemap.pipelines[mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dst_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_color), 0);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_settings.exposure_texture), 1);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_settings.glow_texture, true), 2);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_settings.color_correction_texture), 3);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &tonemap.push_constant, sizeof(TonemapPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::luminance_reduction(RID p_source_texture, const Size2i p_source_size, const Vector<RID> p_reduce, RID p_prev_luminance, float p_min_luminance, float p_max_luminance, float p_adjust, bool p_set) {
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

		int32_t x_groups = (luminance_reduce.push_constant.source_size[0] - 1) / 8 + 1;
		int32_t y_groups = (luminance_reduce.push_constant.source_size[1] - 1) / 8 + 1;

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

		luminance_reduce.push_constant.source_size[0] = MAX(luminance_reduce.push_constant.source_size[0] / 8, 1);
		luminance_reduce.push_constant.source_size[1] = MAX(luminance_reduce.push_constant.source_size[1] / 8, 1);
	}

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::bokeh_dof(RID p_base_texture, RID p_depth_texture, const Size2i &p_base_texture_size, RID p_secondary_texture, RID p_halfsize_texture1, RID p_halfsize_texture2, bool p_dof_far, float p_dof_far_begin, float p_dof_far_size, bool p_dof_near, float p_dof_near_begin, float p_dof_near_size, float p_bokeh_size, RenderingServer::DOFBokehShape p_bokeh_shape, RS::DOFBlurQuality p_quality, bool p_use_jitter, float p_cam_znear, float p_cam_zfar, bool p_cam_orthogonal) {
	bokeh.push_constant.blur_far_active = p_dof_far;
	bokeh.push_constant.blur_far_begin = p_dof_far_begin;
	bokeh.push_constant.blur_far_end = p_dof_far_begin + p_dof_far_size;

	bokeh.push_constant.blur_near_active = p_dof_near;
	bokeh.push_constant.blur_near_begin = p_dof_near_begin;
	bokeh.push_constant.blur_near_end = MAX(0, p_dof_near_begin - p_dof_near_size);
	bokeh.push_constant.use_jitter = p_use_jitter;
	bokeh.push_constant.jitter_seed = Math::randf() * 1000.0;

	bokeh.push_constant.z_near = p_cam_znear;
	bokeh.push_constant.z_far = p_cam_zfar;
	bokeh.push_constant.orthogonal = p_cam_orthogonal;
	bokeh.push_constant.blur_size = p_bokeh_size;

	bokeh.push_constant.second_pass = false;
	bokeh.push_constant.half_size = false;

	bokeh.push_constant.blur_scale = 0.5;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	/* FIRST PASS */
	// The alpha channel of the source color texture is filled with the expected circle size
	// If used for DOF far, the size is positive, if used for near, its negative.

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, bokeh.pipelines[BOKEH_GEN_BLUR_SIZE]);

	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_base_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_texture), 1);

	int32_t x_groups = (p_base_texture_size.x - 1) / 8 + 1;
	int32_t y_groups = (p_base_texture_size.y - 1) / 8 + 1;
	bokeh.push_constant.size[0] = p_base_texture_size.x;
	bokeh.push_constant.size[1] = p_base_texture_size.y;

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	if (p_bokeh_shape == RS::DOF_BOKEH_BOX || p_bokeh_shape == RS::DOF_BOKEH_HEXAGON) {
		//second pass
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, bokeh.pipelines[p_bokeh_shape == RS::DOF_BOKEH_BOX ? BOKEH_GEN_BOKEH_BOX : BOKEH_GEN_BOKEH_HEXAGONAL]);

		static const int quality_samples[4] = { 6, 12, 12, 24 };

		bokeh.push_constant.steps = quality_samples[p_quality];

		if (p_quality == RS::DOF_BLUR_QUALITY_VERY_LOW || p_quality == RS::DOF_BLUR_QUALITY_LOW) {
			//box and hexagon are more or less the same, and they can work in either half (very low and low quality) or full (medium and high quality_ sizes)

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_halfsize_texture1), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_base_texture), 1);

			x_groups = ((p_base_texture_size.x >> 1) - 1) / 8 + 1;
			y_groups = ((p_base_texture_size.y >> 1) - 1) / 8 + 1;
			bokeh.push_constant.size[0] = p_base_texture_size.x >> 1;
			bokeh.push_constant.size[1] = p_base_texture_size.y >> 1;
			bokeh.push_constant.half_size = true;
			bokeh.push_constant.blur_size *= 0.5;

		} else {
			//medium and high quality use full size
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_secondary_texture), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_base_texture), 1);
		}

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		//third pass
		bokeh.push_constant.second_pass = true;

		if (p_quality == RS::DOF_BLUR_QUALITY_VERY_LOW || p_quality == RS::DOF_BLUR_QUALITY_LOW) {
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_halfsize_texture2), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_halfsize_texture1), 1);
		} else {
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_base_texture), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_secondary_texture), 1);
		}

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		if (p_quality == RS::DOF_BLUR_QUALITY_VERY_LOW || p_quality == RS::DOF_BLUR_QUALITY_LOW) {
			//forth pass, upscale for low quality

			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, bokeh.pipelines[BOKEH_COMPOSITE]);

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_base_texture), 0);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_halfsize_texture2), 1);

			x_groups = (p_base_texture_size.x - 1) / 8 + 1;
			y_groups = (p_base_texture_size.y - 1) / 8 + 1;
			bokeh.push_constant.size[0] = p_base_texture_size.x;
			bokeh.push_constant.size[1] = p_base_texture_size.y;
			bokeh.push_constant.half_size = false;
			bokeh.push_constant.second_pass = false;

			RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

			RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
		}
	} else {
		//circle

		//second pass
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, bokeh.pipelines[BOKEH_GEN_BOKEH_CIRCULAR]);

		static const float quality_scale[4] = { 8.0, 4.0, 1.0, 0.5 };

		bokeh.push_constant.steps = 0;
		bokeh.push_constant.blur_scale = quality_scale[p_quality];

		//circle always runs in half size, otherwise too expensive

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_halfsize_texture1), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_base_texture), 1);

		x_groups = ((p_base_texture_size.x >> 1) - 1) / 8 + 1;
		y_groups = ((p_base_texture_size.y >> 1) - 1) / 8 + 1;
		bokeh.push_constant.size[0] = p_base_texture_size.x >> 1;
		bokeh.push_constant.size[1] = p_base_texture_size.y >> 1;
		bokeh.push_constant.half_size = true;

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		//circle is just one pass, then upscale

		// upscale

		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, bokeh.pipelines[BOKEH_COMPOSITE]);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_base_texture), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_halfsize_texture1), 1);

		x_groups = (p_base_texture_size.x - 1) / 8 + 1;
		y_groups = (p_base_texture_size.y - 1) / 8 + 1;
		bokeh.push_constant.size[0] = p_base_texture_size.x;
		bokeh.push_constant.size[1] = p_base_texture_size.y;
		bokeh.push_constant.half_size = false;
		bokeh.push_constant.second_pass = false;

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	}

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::generate_ssao(RID p_depth_buffer, RID p_normal_buffer, const Size2i &p_depth_buffer_size, RID p_depth_mipmaps_texture, const Vector<RID> &depth_mipmaps, RID p_ao1, bool p_half_size, RID p_ao2, RID p_upscale_buffer, float p_intensity, float p_radius, float p_bias, const CameraMatrix &p_projection, RS::EnvironmentSSAOQuality p_quality, RS::EnvironmentSSAOBlur p_blur, float p_edge_sharpness) {
	//minify first
	ssao.minify_push_constant.orthogonal = p_projection.is_orthogonal();
	ssao.minify_push_constant.z_near = p_projection.get_z_near();
	ssao.minify_push_constant.z_far = p_projection.get_z_far();
	ssao.minify_push_constant.pixel_size[0] = 1.0 / p_depth_buffer_size.x;
	ssao.minify_push_constant.pixel_size[1] = 1.0 / p_depth_buffer_size.y;
	ssao.minify_push_constant.source_size[0] = p_depth_buffer_size.x;
	ssao.minify_push_constant.source_size[1] = p_depth_buffer_size.y;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	/* FIRST PASS */
	// Minify the depth buffer.

	for (int i = 0; i < depth_mipmaps.size(); i++) {
		if (i == 0) {
			RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_MINIFY_FIRST]);
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_buffer), 0);
		} else {
			if (i == 1) {
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_MINIFY_MIPMAP]);
			}

			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(depth_mipmaps[i - 1]), 0);
		}
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(depth_mipmaps[i]), 1);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.minify_push_constant, sizeof(SSAOMinifyPushConstant));
		// shrink after set
		ssao.minify_push_constant.source_size[0] = MAX(1, ssao.minify_push_constant.source_size[0] >> 1);
		ssao.minify_push_constant.source_size[1] = MAX(1, ssao.minify_push_constant.source_size[1] >> 1);

		int x_groups = (ssao.minify_push_constant.source_size[0] - 1) / 8 + 1;
		int y_groups = (ssao.minify_push_constant.source_size[1] - 1) / 8 + 1;

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}

	/* SECOND PASS */
	// Gather samples

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[(SSAO_GATHER_LOW + p_quality) + (p_half_size ? 4 : 0)]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_mipmaps_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_ao1), 1);
	if (!p_half_size) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_buffer), 2);
	}
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_normal_buffer), 3);

	ssao.gather_push_constant.screen_size[0] = p_depth_buffer_size.x;
	ssao.gather_push_constant.screen_size[1] = p_depth_buffer_size.y;
	if (p_half_size) {
		ssao.gather_push_constant.screen_size[0] >>= 1;
		ssao.gather_push_constant.screen_size[1] >>= 1;
	}
	ssao.gather_push_constant.z_far = p_projection.get_z_far();
	ssao.gather_push_constant.z_near = p_projection.get_z_near();
	ssao.gather_push_constant.orthogonal = p_projection.is_orthogonal();

	ssao.gather_push_constant.proj_info[0] = -2.0f / (ssao.gather_push_constant.screen_size[0] * p_projection.matrix[0][0]);
	ssao.gather_push_constant.proj_info[1] = -2.0f / (ssao.gather_push_constant.screen_size[1] * p_projection.matrix[1][1]);
	ssao.gather_push_constant.proj_info[2] = (1.0f - p_projection.matrix[0][2]) / p_projection.matrix[0][0];
	ssao.gather_push_constant.proj_info[3] = (1.0f + p_projection.matrix[1][2]) / p_projection.matrix[1][1];
	//ssao.gather_push_constant.proj_info[2] = (1.0f - p_projection.matrix[0][2]) / p_projection.matrix[0][0];
	//ssao.gather_push_constant.proj_info[3] = -(1.0f + p_projection.matrix[1][2]) / p_projection.matrix[1][1];

	ssao.gather_push_constant.radius = p_radius;

	ssao.gather_push_constant.proj_scale = float(p_projection.get_pixels_per_meter(ssao.gather_push_constant.screen_size[0]));
	ssao.gather_push_constant.bias = p_bias;
	ssao.gather_push_constant.intensity_div_r6 = p_intensity / pow(p_radius, 6.0f);

	ssao.gather_push_constant.pixel_size[0] = 1.0 / p_depth_buffer_size.x;
	ssao.gather_push_constant.pixel_size[1] = 1.0 / p_depth_buffer_size.y;

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.gather_push_constant, sizeof(SSAOGatherPushConstant));

	int x_groups = (ssao.gather_push_constant.screen_size[0] - 1) / 8 + 1;
	int y_groups = (ssao.gather_push_constant.screen_size[1] - 1) / 8 + 1;

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	/* THIRD PASS */
	// Blur horizontal

	ssao.blur_push_constant.edge_sharpness = p_edge_sharpness;
	ssao.blur_push_constant.filter_scale = p_blur;
	ssao.blur_push_constant.screen_size[0] = ssao.gather_push_constant.screen_size[0];
	ssao.blur_push_constant.screen_size[1] = ssao.gather_push_constant.screen_size[1];
	ssao.blur_push_constant.z_far = p_projection.get_z_far();
	ssao.blur_push_constant.z_near = p_projection.get_z_near();
	ssao.blur_push_constant.orthogonal = p_projection.is_orthogonal();
	ssao.blur_push_constant.axis[0] = 1;
	ssao.blur_push_constant.axis[1] = 0;

	if (p_blur != RS::ENV_SSAO_BLUR_DISABLED) {
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[p_half_size ? SSAO_BLUR_PASS_HALF : SSAO_BLUR_PASS]);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_ao1), 0);
		if (p_half_size) {
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_mipmaps_texture), 1);
		} else {
			RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_buffer), 1);
		}
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_ao2), 3);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.blur_push_constant, sizeof(SSAOBlurPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
		RD::get_singleton()->compute_list_add_barrier(compute_list);

		/* THIRD PASS */
		// Blur vertical

		ssao.blur_push_constant.axis[0] = 0;
		ssao.blur_push_constant.axis[1] = 1;

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_ao2), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_ao1), 3);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.blur_push_constant, sizeof(SSAOBlurPushConstant));

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	}
	if (p_half_size) { //must upscale

		/* FOURTH PASS */
		// upscale if half size
		//back to full size
		ssao.blur_push_constant.screen_size[0] = p_depth_buffer_size.x;
		ssao.blur_push_constant.screen_size[1] = p_depth_buffer_size.y;

		RD::get_singleton()->compute_list_add_barrier(compute_list);
		RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, ssao.pipelines[SSAO_BLUR_UPSCALE]);

		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_ao1), 0);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_upscale_buffer), 3);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_buffer), 1);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_mipmaps_texture), 2);

		RD::get_singleton()->compute_list_set_push_constant(compute_list, &ssao.blur_push_constant, sizeof(SSAOBlurPushConstant)); //not used but set anyway

		x_groups = (p_depth_buffer_size.x - 1) / 8 + 1;
		y_groups = (p_depth_buffer_size.y - 1) / 8 + 1;

		RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	}

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::roughness_limit(RID p_source_normal, RID p_roughness, const Size2i &p_size, float p_curve) {
	roughness_limiter.push_constant.screen_size[0] = p_size.x;
	roughness_limiter.push_constant.screen_size[1] = p_size.y;
	roughness_limiter.push_constant.curve = p_curve;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, roughness_limiter.pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_normal), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_roughness), 1);

	int x_groups = (p_size.x - 1) / 8 + 1;
	int y_groups = (p_size.y - 1) / 8 + 1;

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &roughness_limiter.push_constant, sizeof(RoughnessLimiterPushConstant)); //not used but set anyway

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::cubemap_roughness(RID p_source_rd_texture, RID p_dest_framebuffer, uint32_t p_face_id, uint32_t p_sample_count, float p_roughness, float p_size) {
	zeromem(&roughness.push_constant, sizeof(CubemapRoughnessPushConstant));

	roughness.push_constant.face_id = p_face_id > 9 ? 0 : p_face_id;
	roughness.push_constant.roughness = p_roughness;
	roughness.push_constant.sample_count = p_sample_count;
	roughness.push_constant.use_direct_write = p_roughness == 0.0;
	roughness.push_constant.face_size = p_size;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, roughness.pipeline);

	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_framebuffer), 1);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &roughness.push_constant, sizeof(CubemapRoughnessPushConstant));

	int x_groups = (p_size - 1) / 8 + 1;
	int y_groups = (p_size - 1) / 8 + 1;

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, p_face_id > 9 ? 6 : 1);

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::cubemap_downsample(RID p_source_cubemap, RID p_dest_cubemap, const Size2i &p_size) {
	cubemap_downsampler.push_constant.face_size = p_size.x;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, cubemap_downsampler.pipeline);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_cubemap), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_cubemap), 1);

	int x_groups = (p_size.x - 1) / 8 + 1;
	int y_groups = (p_size.y - 1) / 8 + 1;

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &cubemap_downsampler.push_constant, sizeof(CubemapDownsamplerPushConstant));

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 6); // one z_group for each face

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::cubemap_filter(RID p_source_cubemap, Vector<RID> p_dest_cubemap, bool p_use_array) {
	Vector<RD::Uniform> uniforms;
	for (int i = 0; i < p_dest_cubemap.size(); i++) {
		RD::Uniform u;
		u.type = RD::UNIFORM_TYPE_IMAGE;
		u.binding = i;
		u.ids.push_back(p_dest_cubemap[i]);
		uniforms.push_back(u);
	}
	if (RD::get_singleton()->uniform_set_is_valid(filter.image_uniform_set)) {
		RD::get_singleton()->free(filter.image_uniform_set);
	}
	filter.image_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, filter.shader.version_get_shader(filter.shader_version, 0), 2);

	int pipeline = p_use_array ? FILTER_MODE_HIGH_QUALITY_ARRAY : FILTER_MODE_HIGH_QUALITY;
	pipeline = filter.use_high_quality ? pipeline : pipeline + 1;
	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, filter.pipelines[pipeline]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_cubemap, true), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, filter.uniform_set, 1);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, filter.image_uniform_set, 2);

	int x_groups = p_use_array ? 1792 : 342; // (128 * 128 * 7) / 64 : (128*128 + 64*64 + 32*32 + 16*16 + 8*8 + 4*4 + 2*2) / 64

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, 6, 1); // one y_group for each face

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::render_sky(RD::DrawListID p_list, float p_time, RID p_fb, RID p_samplers, RID p_fog, RenderPipelineVertexFormatCacheRD *p_pipeline, RID p_uniform_set, RID p_texture_set, const CameraMatrix &p_camera, const Basis &p_orientation, float p_multiplier, const Vector3 &p_position) {
	SkyPushConstant sky_push_constant;

	zeromem(&sky_push_constant, sizeof(SkyPushConstant));

	sky_push_constant.proj[0] = p_camera.matrix[2][0];
	sky_push_constant.proj[1] = p_camera.matrix[0][0];
	sky_push_constant.proj[2] = p_camera.matrix[2][1];
	sky_push_constant.proj[3] = p_camera.matrix[1][1];
	sky_push_constant.position[0] = p_position.x;
	sky_push_constant.position[1] = p_position.y;
	sky_push_constant.position[2] = p_position.z;
	sky_push_constant.multiplier = p_multiplier;
	sky_push_constant.time = p_time;
	store_transform_3x3(p_orientation, sky_push_constant.orientation);

	RenderingDevice::FramebufferFormatID fb_format = RD::get_singleton()->framebuffer_get_format(p_fb);

	RD::DrawListID draw_list = p_list;

	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, p_pipeline->get_render_pipeline(RD::INVALID_ID, fb_format));

	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_samplers, 0);
	if (p_uniform_set.is_valid()) { //material may not have uniform set
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_uniform_set, 1);
	}
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_texture_set, 2);
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, p_fog, 3);

	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &sky_push_constant, sizeof(SkyPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
}

void RasterizerEffectsRD::resolve_gi(RID p_source_depth, RID p_source_normal_roughness, RID p_source_giprobe, RID p_dest_depth, RID p_dest_normal_roughness, RID p_dest_giprobe, Vector2i p_screen_size, int p_samples) {
	ResolvePushConstant push_constant;
	push_constant.screen_size[0] = p_screen_size.x;
	push_constant.screen_size[1] = p_screen_size.y;
	push_constant.samples = p_samples;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, resolve.pipelines[p_source_giprobe.is_valid() ? RESOLVE_MODE_GI_GIPROBE : RESOLVE_MODE_GI]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture_pair(p_source_depth, p_source_normal_roughness), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_dest_depth, p_dest_normal_roughness), 1);
	if (p_source_giprobe.is_valid()) {
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_source_giprobe), 2);
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_dest_giprobe), 3);
	}

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(ResolvePushConstant));

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_screen_size.x, p_screen_size.y, 1, 8, 8, 1);

	RD::get_singleton()->compute_list_end();
}

void RasterizerEffectsRD::reduce_shadow(RID p_source_shadow, RID p_dest_shadow, const Size2i &p_source_size, const Rect2i &p_source_rect, int p_shrink_limit, RD::ComputeListID compute_list) {
	uint32_t push_constant[8] = { (uint32_t)p_source_size.x, (uint32_t)p_source_size.y, (uint32_t)p_source_rect.position.x, (uint32_t)p_source_rect.position.y, (uint32_t)p_shrink_limit, 0, 0, 0 };

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, shadow_reduce.pipelines[SHADOW_REDUCE_REDUCE]);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_source_shadow, p_dest_shadow), 0);
	RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(uint32_t) * 8);

	RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_source_rect.size.width, p_source_rect.size.height, 1, 8, 8, 1);
}
void RasterizerEffectsRD::filter_shadow(RID p_shadow, RID p_backing_shadow, const Size2i &p_source_size, const Rect2i &p_source_rect, RenderingServer::EnvVolumetricFogShadowFilter p_filter, RD::ComputeListID compute_list, bool p_vertical, bool p_horizontal) {
	uint32_t push_constant[8] = { (uint32_t)p_source_size.x, (uint32_t)p_source_size.y, (uint32_t)p_source_rect.position.x, (uint32_t)p_source_rect.position.y, 0, 0, 0, 0 };

	switch (p_filter) {
		case RS::ENV_VOLUMETRIC_FOG_SHADOW_FILTER_DISABLED:
		case RS::ENV_VOLUMETRIC_FOG_SHADOW_FILTER_LOW: {
			push_constant[5] = 0;
		} break;
		case RS::ENV_VOLUMETRIC_FOG_SHADOW_FILTER_MEDIUM: {
			push_constant[5] = 9;
		} break;
		case RS::ENV_VOLUMETRIC_FOG_SHADOW_FILTER_HIGH: {
			push_constant[5] = 18;
		} break;
	}

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, shadow_reduce.pipelines[SHADOW_REDUCE_FILTER]);
	if (p_vertical) {
		push_constant[6] = 1;
		push_constant[7] = 0;
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_shadow, p_backing_shadow), 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(uint32_t) * 8);
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_source_rect.size.width, p_source_rect.size.height, 1, 8, 8, 1);
	}
	if (p_vertical && p_horizontal) {
		RD::get_singleton()->compute_list_add_barrier(compute_list);
	}
	if (p_horizontal) {
		push_constant[6] = 0;
		push_constant[7] = 1;
		RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_image_pair(p_backing_shadow, p_shadow), 0);
		RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(uint32_t) * 8);
		RD::get_singleton()->compute_list_dispatch_threads(compute_list, p_source_rect.size.width, p_source_rect.size.height, 1, 8, 8, 1);
	}
}

void RasterizerEffectsRD::sort_buffer(RID p_uniform_set, int p_size) {
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

RasterizerEffectsRD::RasterizerEffectsRD() {
	{ // Initialize copy
		Vector<String> copy_modes;
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n#define DST_IMAGE_8BIT\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_GLOW\n");
		copy_modes.push_back("\n#define MODE_GAUSSIAN_GLOW\n#define GLOW_USE_AUTO_EXPOSURE\n");
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
		zeromem(&copy.push_constant, sizeof(CopyPushConstant));
		copy.shader_version = copy.shader.version_create();

		for (int i = 0; i < COPY_MODE_MAX; i++) {
			copy.pipelines[i] = RD::get_singleton()->compute_pipeline_create(copy.shader.version_get_shader(copy.shader_version, i));
		}
	}
	{
		Vector<String> copy_modes;
		copy_modes.push_back("\n");
		copy_modes.push_back("\n#define MODE_PANORAMA_TO_DP\n");
		copy_modes.push_back("\n#define MODE_TWO_SOURCES\n");

		copy_to_fb.shader.initialize(copy_modes);

		copy_to_fb.shader_version = copy_to_fb.shader.version_create();

		//use additive

		for (int i = 0; i < COPY_TO_FB_MAX; i++) {
			copy_to_fb.pipelines[i].setup(copy_to_fb.shader.version_get_shader(copy_to_fb.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	{
		// Initialize roughness
		Vector<String> cubemap_roughness_modes;
		cubemap_roughness_modes.push_back("");
		roughness.shader.initialize(cubemap_roughness_modes);

		roughness.shader_version = roughness.shader.version_create();

		roughness.pipeline = RD::get_singleton()->compute_pipeline_create(roughness.shader.version_get_shader(roughness.shader_version, 0));
	}

	{
		// Initialize tonemapper
		Vector<String> tonemap_modes;
		tonemap_modes.push_back("\n");
		tonemap_modes.push_back("\n#define USE_GLOW_FILTER_BICUBIC\n");

		tonemap.shader.initialize(tonemap_modes);

		tonemap.shader_version = tonemap.shader.version_create();

		for (int i = 0; i < TONEMAP_MODE_MAX; i++) {
			tonemap.pipelines[i].setup(tonemap.shader.version_get_shader(tonemap.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	{
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
	}

	{
		// Initialize copier
		Vector<String> copy_modes;
		copy_modes.push_back("\n");

		cube_to_dp.shader.initialize(copy_modes);

		cube_to_dp.shader_version = cube_to_dp.shader.version_create();

		cube_to_dp.pipeline = RD::get_singleton()->compute_pipeline_create(cube_to_dp.shader.version_get_shader(cube_to_dp.shader_version, 0));
	}

	{
		// Initialize bokeh
		Vector<String> bokeh_modes;
		bokeh_modes.push_back("\n#define MODE_GEN_BLUR_SIZE\n");
		bokeh_modes.push_back("\n#define MODE_BOKEH_BOX\n");
		bokeh_modes.push_back("\n#define MODE_BOKEH_HEXAGONAL\n");
		bokeh_modes.push_back("\n#define MODE_BOKEH_CIRCULAR\n");
		bokeh_modes.push_back("\n#define MODE_COMPOSITE_BOKEH\n");

		bokeh.shader.initialize(bokeh_modes);

		bokeh.shader_version = bokeh.shader.version_create();

		for (int i = 0; i < BOKEH_MAX; i++) {
			bokeh.pipelines[i] = RD::get_singleton()->compute_pipeline_create(bokeh.shader.version_get_shader(bokeh.shader_version, i));
		}
	}

	{
		// Initialize ssao
		uint32_t pipeline = 0;
		{
			Vector<String> ssao_modes;
			ssao_modes.push_back("\n#define MINIFY_START\n");
			ssao_modes.push_back("\n");

			ssao.minify_shader.initialize(ssao_modes);

			ssao.minify_shader_version = ssao.minify_shader.version_create();

			for (int i = 0; i <= SSAO_MINIFY_MIPMAP; i++) {
				ssao.pipelines[pipeline] = RD::get_singleton()->compute_pipeline_create(ssao.minify_shader.version_get_shader(ssao.minify_shader_version, i));
				pipeline++;
			}
		}
		{
			Vector<String> ssao_modes;
			ssao_modes.push_back("\n#define SSAO_QUALITY_LOW\n");
			ssao_modes.push_back("\n");
			ssao_modes.push_back("\n#define SSAO_QUALITY_HIGH\n");
			ssao_modes.push_back("\n#define SSAO_QUALITY_ULTRA\n");
			ssao_modes.push_back("\n#define SSAO_QUALITY_LOW\n#define USE_HALF_SIZE\n");
			ssao_modes.push_back("\n#define USE_HALF_SIZE\n");
			ssao_modes.push_back("\n#define SSAO_QUALITY_HIGH\n#define USE_HALF_SIZE\n");
			ssao_modes.push_back("\n#define SSAO_QUALITY_ULTRA\n#define USE_HALF_SIZE\n");

			ssao.gather_shader.initialize(ssao_modes);

			ssao.gather_shader_version = ssao.gather_shader.version_create();

			for (int i = SSAO_GATHER_LOW; i <= SSAO_GATHER_ULTRA_HALF; i++) {
				ssao.pipelines[pipeline] = RD::get_singleton()->compute_pipeline_create(ssao.gather_shader.version_get_shader(ssao.gather_shader_version, i - SSAO_GATHER_LOW));
				pipeline++;
			}
		}
		{
			Vector<String> ssao_modes;
			ssao_modes.push_back("\n#define MODE_FULL_SIZE\n");
			ssao_modes.push_back("\n");
			ssao_modes.push_back("\n#define MODE_UPSCALE\n");

			ssao.blur_shader.initialize(ssao_modes);

			ssao.blur_shader_version = ssao.blur_shader.version_create();

			for (int i = SSAO_BLUR_PASS; i <= SSAO_BLUR_UPSCALE; i++) {
				ssao.pipelines[pipeline] = RD::get_singleton()->compute_pipeline_create(ssao.blur_shader.version_get_shader(ssao.blur_shader_version, i - SSAO_BLUR_PASS));

				pipeline++;
			}
		}

		ERR_FAIL_COND(pipeline != SSAO_MAX);
	}

	{
		// Initialize roughness limiter
		Vector<String> shader_modes;
		shader_modes.push_back("");

		roughness_limiter.shader.initialize(shader_modes);

		roughness_limiter.shader_version = roughness_limiter.shader.version_create();

		roughness_limiter.pipeline = RD::get_singleton()->compute_pipeline_create(roughness_limiter.shader.version_get_shader(roughness_limiter.shader_version, 0));
	}

	{
		//Initialize cubemap downsampler
		Vector<String> cubemap_downsampler_modes;
		cubemap_downsampler_modes.push_back("");
		cubemap_downsampler.shader.initialize(cubemap_downsampler_modes);

		cubemap_downsampler.shader_version = cubemap_downsampler.shader.version_create();

		cubemap_downsampler.pipeline = RD::get_singleton()->compute_pipeline_create(cubemap_downsampler.shader.version_get_shader(cubemap_downsampler.shader_version, 0));
	}

	{
		// Initialize cubemap filter
		filter.use_high_quality = GLOBAL_GET("rendering/quality/reflections/fast_filter_high_quality");

		Vector<String> cubemap_filter_modes;
		cubemap_filter_modes.push_back("\n#define USE_HIGH_QUALITY\n");
		cubemap_filter_modes.push_back("\n#define USE_LOW_QUALITY\n");
		cubemap_filter_modes.push_back("\n#define USE_HIGH_QUALITY\n#define USE_TEXTURE_ARRAY\n");
		cubemap_filter_modes.push_back("\n#define USE_LOW_QUALITY\n#define USE_TEXTURE_ARRAY\n");
		filter.shader.initialize(cubemap_filter_modes);
		filter.shader_version = filter.shader.version_create();

		for (int i = 0; i < FILTER_MODE_MAX; i++) {
			filter.pipelines[i] = RD::get_singleton()->compute_pipeline_create(filter.shader.version_get_shader(filter.shader_version, i));
		}

		if (filter.use_high_quality) {
			filter.coefficient_buffer = RD::get_singleton()->storage_buffer_create(sizeof(high_quality_coeffs));
			RD::get_singleton()->buffer_update(filter.coefficient_buffer, 0, sizeof(high_quality_coeffs), &high_quality_coeffs[0], false);
		} else {
			filter.coefficient_buffer = RD::get_singleton()->storage_buffer_create(sizeof(low_quality_coeffs));
			RD::get_singleton()->buffer_update(filter.coefficient_buffer, 0, sizeof(low_quality_coeffs), &low_quality_coeffs[0], false);
		}

		Vector<RD::Uniform> uniforms;
		{
			RD::Uniform u;
			u.type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
			u.binding = 0;
			u.ids.push_back(filter.coefficient_buffer);
			uniforms.push_back(u);
		}
		filter.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, filter.shader.version_get_shader(filter.shader_version, filter.use_high_quality ? 0 : 1), 1);
	}

	{
		Vector<String> specular_modes;
		specular_modes.push_back("\n#define MODE_MERGE\n");
		specular_modes.push_back("\n#define MODE_MERGE\n#define MODE_SSR\n");
		specular_modes.push_back("\n");
		specular_modes.push_back("\n#define MODE_SSR\n");

		specular_merge.shader.initialize(specular_modes);

		specular_merge.shader_version = specular_merge.shader.version_create();

		//use additive

		RD::PipelineColorBlendState::Attachment ba;
		ba.enable_blend = true;
		ba.src_color_blend_factor = RD::BLEND_FACTOR_ONE;
		ba.dst_color_blend_factor = RD::BLEND_FACTOR_ONE;
		ba.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
		ba.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
		ba.color_blend_op = RD::BLEND_OP_ADD;
		ba.alpha_blend_op = RD::BLEND_OP_ADD;

		RD::PipelineColorBlendState blend_additive;
		blend_additive.attachments.push_back(ba);

		for (int i = 0; i < SPECULAR_MERGE_MAX; i++) {
			RD::PipelineColorBlendState blend_state;
			if (i == SPECULAR_MERGE_ADDITIVE_ADD || i == SPECULAR_MERGE_ADDITIVE_SSR) {
				blend_state = blend_additive;
			} else {
				blend_state = RD::PipelineColorBlendState::create_disabled();
			}
			specular_merge.pipelines[i].setup(specular_merge.shader.version_get_shader(specular_merge.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), blend_state, 0);
		}
	}

	{
		Vector<String> ssr_modes;
		ssr_modes.push_back("\n");
		ssr_modes.push_back("\n#define MODE_ROUGH\n");

		ssr.shader.initialize(ssr_modes);

		ssr.shader_version = ssr.shader.version_create();

		for (int i = 0; i < SCREEN_SPACE_REFLECTION_MAX; i++) {
			ssr.pipelines[i] = RD::get_singleton()->compute_pipeline_create(ssr.shader.version_get_shader(ssr.shader_version, i));
		}
	}

	{
		Vector<String> ssr_filter_modes;
		ssr_filter_modes.push_back("\n");
		ssr_filter_modes.push_back("\n#define VERTICAL_PASS\n");

		ssr_filter.shader.initialize(ssr_filter_modes);

		ssr_filter.shader_version = ssr_filter.shader.version_create();

		for (int i = 0; i < SCREEN_SPACE_REFLECTION_FILTER_MAX; i++) {
			ssr_filter.pipelines[i] = RD::get_singleton()->compute_pipeline_create(ssr_filter.shader.version_get_shader(ssr_filter.shader_version, i));
		}
	}

	{
		Vector<String> ssr_scale_modes;
		ssr_scale_modes.push_back("\n");

		ssr_scale.shader.initialize(ssr_scale_modes);

		ssr_scale.shader_version = ssr_scale.shader.version_create();

		ssr_scale.pipeline = RD::get_singleton()->compute_pipeline_create(ssr_scale.shader.version_get_shader(ssr_scale.shader_version, 0));
	}

	{
		Vector<String> sss_modes;
		sss_modes.push_back("\n#define USE_11_SAMPLES\n");
		sss_modes.push_back("\n#define USE_17_SAMPLES\n");
		sss_modes.push_back("\n#define USE_25_SAMPLES\n");

		sss.shader.initialize(sss_modes);

		sss.shader_version = sss.shader.version_create();

		for (int i = 0; i < sss_modes.size(); i++) {
			sss.pipelines[i] = RD::get_singleton()->compute_pipeline_create(sss.shader.version_get_shader(sss.shader_version, i));
		}
	}

	{
		Vector<String> resolve_modes;
		resolve_modes.push_back("\n#define MODE_RESOLVE_GI\n");
		resolve_modes.push_back("\n#define MODE_RESOLVE_GI\n#define GIPROBE_RESOLVE\n");

		resolve.shader.initialize(resolve_modes);

		resolve.shader_version = resolve.shader.version_create();

		for (int i = 0; i < RESOLVE_MODE_MAX; i++) {
			resolve.pipelines[i] = RD::get_singleton()->compute_pipeline_create(resolve.shader.version_get_shader(resolve.shader_version, i));
		}
	}

	{
		Vector<String> shadow_reduce_modes;
		shadow_reduce_modes.push_back("\n#define MODE_REDUCE\n");
		shadow_reduce_modes.push_back("\n#define MODE_FILTER\n");

		shadow_reduce.shader.initialize(shadow_reduce_modes);

		shadow_reduce.shader_version = shadow_reduce.shader.version_create();

		for (int i = 0; i < SHADOW_REDUCE_MAX; i++) {
			shadow_reduce.pipelines[i] = RD::get_singleton()->compute_pipeline_create(shadow_reduce.shader.version_get_shader(shadow_reduce.shader_version, i));
		}
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

	sampler.min_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.mip_filter = RD::SAMPLER_FILTER_LINEAR;
	sampler.max_lod = 1e20;

	default_mipmap_sampler = RD::get_singleton()->sampler_create(sampler);

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

RasterizerEffectsRD::~RasterizerEffectsRD() {
	if (RD::get_singleton()->uniform_set_is_valid(filter.image_uniform_set)) {
		RD::get_singleton()->free(filter.image_uniform_set);
	}

	if (RD::get_singleton()->uniform_set_is_valid(filter.uniform_set)) {
		RD::get_singleton()->free(filter.uniform_set);
	}

	RD::get_singleton()->free(default_sampler);
	RD::get_singleton()->free(default_mipmap_sampler);
	RD::get_singleton()->free(index_buffer); //array gets freed as dependency
	RD::get_singleton()->free(filter.coefficient_buffer);

	bokeh.shader.version_free(bokeh.shader_version);
	copy.shader.version_free(copy.shader_version);
	copy_to_fb.shader.version_free(copy_to_fb.shader_version);
	cube_to_dp.shader.version_free(cube_to_dp.shader_version);
	cubemap_downsampler.shader.version_free(cubemap_downsampler.shader_version);
	filter.shader.version_free(filter.shader_version);
	luminance_reduce.shader.version_free(luminance_reduce.shader_version);
	resolve.shader.version_free(resolve.shader_version);
	roughness.shader.version_free(roughness.shader_version);
	roughness_limiter.shader.version_free(roughness_limiter.shader_version);
	sort.shader.version_free(sort.shader_version);
	specular_merge.shader.version_free(specular_merge.shader_version);
	ssao.blur_shader.version_free(ssao.blur_shader_version);
	ssao.gather_shader.version_free(ssao.gather_shader_version);
	ssao.minify_shader.version_free(ssao.minify_shader_version);
	ssr.shader.version_free(ssr.shader_version);
	ssr_filter.shader.version_free(ssr_filter.shader_version);
	ssr_scale.shader.version_free(ssr_scale.shader_version);
	sss.shader.version_free(sss.shader_version);
	tonemap.shader.version_free(tonemap.shader_version);
	shadow_reduce.shader.version_free(shadow_reduce.shader_version);
}
