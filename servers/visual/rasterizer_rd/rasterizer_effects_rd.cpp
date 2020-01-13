/*************************************************************************/
/*  rasterizer_effects_rd.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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
	RID uniform_set = RD::get_singleton()->uniform_set_create(uniforms, blur.shader.version_get_shader(blur.shader_version, 0), 0);

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

void RasterizerEffectsRD::copy_to_rect(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_rect, bool p_flip_y) {

	zeromem(&blur.push_constant, sizeof(BlurPushConstant));
	if (p_flip_y) {
		blur.push_constant.flags |= BLUR_FLAG_FLIP_Y;
	}

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, Vector<Color>(), 1.0, 0, p_rect);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[BLUR_MODE_SIMPLY_COPY].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::region_copy(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_region) {

	zeromem(&blur.push_constant, sizeof(BlurPushConstant));

	if (p_region != Rect2()) {
		blur.push_constant.flags = BLUR_FLAG_USE_BLUR_SECTION;
		blur.push_constant.section[0] = p_region.position.x;
		blur.push_constant.section[1] = p_region.position.y;
		blur.push_constant.section[2] = p_region.size.width;
		blur.push_constant.section[3] = p_region.size.height;
	}

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[BLUR_MODE_SIMPLY_COPY].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::gaussian_blur(RID p_source_rd_texture, RID p_framebuffer_half, RID p_rd_texture_half, RID p_dest_framebuffer, const Vector2 &p_pixel_size, const Rect2 &p_region) {

	zeromem(&blur.push_constant, sizeof(BlurPushConstant));

	uint32_t base_flags = 0;
	if (p_region != Rect2()) {
		base_flags = BLUR_FLAG_USE_BLUR_SECTION;
		blur.push_constant.section[0] = p_region.position.x;
		blur.push_constant.section[1] = p_region.position.y;
		blur.push_constant.section[2] = p_region.size.width;
		blur.push_constant.section[3] = p_region.size.height;
	}

	blur.push_constant.pixel_size[0] = p_pixel_size.x;
	blur.push_constant.pixel_size[1] = p_pixel_size.y;

	//HORIZONTAL
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer_half, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[BLUR_MODE_GAUSSIAN_BLUR].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer_half)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	blur.push_constant.flags = base_flags | BLUR_FLAG_HORIZONTAL;
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();

	//VERTICAL
	draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[BLUR_MODE_GAUSSIAN_BLUR].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_rd_texture_half), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	blur.push_constant.flags = base_flags;
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::gaussian_glow(RID p_source_rd_texture, RID p_framebuffer_half, RID p_rd_texture_half, RID p_dest_framebuffer, const Vector2 &p_pixel_size, float p_strength, bool p_first_pass, float p_luminance_cap, float p_exposure, float p_bloom, float p_hdr_bleed_treshold, float p_hdr_bleed_scale, RID p_auto_exposure, float p_auto_exposure_grey) {

	zeromem(&blur.push_constant, sizeof(BlurPushConstant));

	BlurMode blur_mode = p_first_pass && p_auto_exposure.is_valid() ? BLUR_MODE_GAUSSIAN_GLOW_AUTO_EXPOSURE : BLUR_MODE_GAUSSIAN_GLOW;
	uint32_t base_flags = 0;

	blur.push_constant.pixel_size[0] = p_pixel_size.x;
	blur.push_constant.pixel_size[1] = p_pixel_size.y;

	blur.push_constant.glow_strength = p_strength;
	blur.push_constant.glow_bloom = p_bloom;
	blur.push_constant.glow_hdr_threshold = p_hdr_bleed_treshold;
	blur.push_constant.glow_hdr_scale = p_hdr_bleed_scale;
	blur.push_constant.glow_exposure = p_exposure;
	blur.push_constant.glow_white = 0; //actually unused
	blur.push_constant.glow_luminance_cap = p_luminance_cap;
	blur.push_constant.glow_auto_exposure_grey = p_auto_exposure_grey; //unused also

	//HORIZONTAL
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_framebuffer_half, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[blur_mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_framebuffer_half)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	if (p_auto_exposure.is_valid() && p_first_pass) {
		RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_auto_exposure), 1);
	}
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	blur.push_constant.flags = base_flags | BLUR_FLAG_HORIZONTAL | (p_first_pass ? BLUR_FLAG_GLOW_FIRST_PASS : 0);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();

	blur_mode = BLUR_MODE_GAUSSIAN_GLOW;

	//VERTICAL
	draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[blur_mode].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_rd_texture_half), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	blur.push_constant.flags = base_flags;
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::cubemap_roughness(RID p_source_rd_texture, bool p_source_is_panorama, RID p_dest_framebuffer, uint32_t p_face_id, uint32_t p_sample_count, float p_roughness) {

	zeromem(&roughness.push_constant, sizeof(CubemapRoughnessPushConstant));

	roughness.push_constant.face_id = p_face_id;
	roughness.push_constant.roughness = p_roughness;
	roughness.push_constant.sample_count = p_sample_count;
	roughness.push_constant.use_direct_write = p_roughness == 0.0;

	//RUN
	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, roughness.pipelines[p_source_is_panorama ? CUBEMAP_ROUGHNESS_SOURCE_PANORAMA : CUBEMAP_ROUGHNESS_SOURCE_CUBEMAP].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));

	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &roughness.push_constant, sizeof(CubemapRoughnessPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::render_panorama(RD::DrawListID p_list, RenderingDevice::FramebufferFormatID p_fb_format, RID p_panorama, const CameraMatrix &p_camera, const Basis &p_orientation, float p_alpha, float p_multipler) {

	zeromem(&sky.push_constant, sizeof(SkyPushConstant));

	sky.push_constant.proj[0] = p_camera.matrix[2][0];
	sky.push_constant.proj[1] = p_camera.matrix[0][0];
	sky.push_constant.proj[2] = p_camera.matrix[2][1];
	sky.push_constant.proj[3] = p_camera.matrix[1][1];
	sky.push_constant.alpha = p_alpha;
	sky.push_constant.depth = 1.0;
	sky.push_constant.multiplier = p_multipler;
	store_transform_3x3(p_orientation, sky.push_constant.orientation);

	RD::DrawListID draw_list = p_list;

	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, sky.pipeline.get_render_pipeline(RD::INVALID_ID, p_fb_format));

	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_panorama), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);

	RD::get_singleton()->draw_list_set_push_constant(draw_list, &sky.push_constant, sizeof(SkyPushConstant));

	RD::get_singleton()->draw_list_draw(draw_list, true);
}

void RasterizerEffectsRD::make_mipmap(RID p_source_rd_texture, RID p_dest_framebuffer, const Vector2 &p_pixel_size) {

	zeromem(&blur.push_constant, sizeof(BlurPushConstant));

	blur.push_constant.pixel_size[0] = p_pixel_size.x;
	blur.push_constant.pixel_size[1] = p_pixel_size.y;

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, blur.pipelines[BLUR_MODE_MIPMAP].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &blur.push_constant, sizeof(BlurPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::copy_cubemap_to_dp(RID p_source_rd_texture, RID p_dest_framebuffer, const Rect2 &p_rect, float p_z_near, float p_z_far, float p_bias, bool p_dp_flip) {

	CopyToDPPushConstant push_constant;
	push_constant.bias = p_bias;
	push_constant.z_far = p_z_far;
	push_constant.z_near = p_z_near;
	push_constant.z_flip = p_dp_flip;

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dest_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD, Vector<Color>(), 1.0, 0, p_rect);
	RD::get_singleton()->draw_list_bind_render_pipeline(draw_list, copy.pipelines[COPY_MODE_CUBE_TO_DP].get_render_pipeline(RD::INVALID_ID, RD::get_singleton()->framebuffer_get_format(p_dest_framebuffer)));
	RD::get_singleton()->draw_list_bind_uniform_set(draw_list, _get_uniform_set_from_texture(p_source_rd_texture), 0);
	RD::get_singleton()->draw_list_bind_index_array(draw_list, index_array);
	RD::get_singleton()->draw_list_set_push_constant(draw_list, &push_constant, sizeof(CopyToDPPushConstant));
	RD::get_singleton()->draw_list_draw(draw_list, true);
	RD::get_singleton()->draw_list_end();
}

void RasterizerEffectsRD::tonemapper(RID p_source_color, RID p_dst_framebuffer, const TonemapSettings &p_settings) {

	zeromem(&tonemap.push_constant, sizeof(TonemapPushConstant));

	tonemap.push_constant.use_bcs = p_settings.use_bcs;
	tonemap.push_constant.bcs[0] = p_settings.brightness;
	tonemap.push_constant.bcs[1] = p_settings.contrast;
	tonemap.push_constant.bcs[2] = p_settings.saturation;

	tonemap.push_constant.use_glow = p_settings.use_glow;
	tonemap.push_constant.glow_intensity = p_settings.glow_intensity;
	tonemap.push_constant.glow_level_flags = p_settings.glow_level_flags;
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

	RD::DrawListID draw_list = RD::get_singleton()->draw_list_begin(p_dst_framebuffer, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_READ, RD::INITIAL_ACTION_KEEP, RD::FINAL_ACTION_DISCARD);
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

void RasterizerEffectsRD::bokeh_dof(RID p_base_texture, RID p_depth_texture, const Size2i &p_base_texture_size, RID p_bokeh_texture, bool p_dof_far, float p_dof_far_begin, float p_dof_far_size, bool p_dof_near, float p_dof_near_begin, float p_dof_near_size, float p_bokeh_size, VS::DOFBlurQuality p_quality, float p_cam_znear, float p_cam_zfar, bool p_cam_orthogonal) {

	bokeh.push_constant.blur_far_active = p_dof_far;
	bokeh.push_constant.blur_far_begin = p_dof_far_begin;
	bokeh.push_constant.blur_far_end = p_dof_far_begin + p_dof_far_size;

	bokeh.push_constant.blur_near_active = p_dof_near;
	bokeh.push_constant.blur_near_begin = p_dof_near_begin;
	bokeh.push_constant.blur_near_end = MAX(0, p_dof_near_begin - p_dof_near_size);

	bokeh.push_constant.size[0] = p_base_texture_size.x;
	bokeh.push_constant.size[1] = p_base_texture_size.y;

	bokeh.push_constant.z_near = p_cam_znear;
	bokeh.push_constant.z_far = p_cam_zfar;
	bokeh.push_constant.orthogonal = p_cam_orthogonal;
	bokeh.push_constant.blur_size = p_bokeh_size;
	bokeh.push_constant.blur_scale = 0.5;

	RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, bokeh.pipelines[BOKEH_GEN_BLUR_SIZE]);

	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_base_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_texture), 1);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

	int32_t x_groups = (p_base_texture_size.x - 1) / 8 + 1;
	int32_t y_groups = (p_base_texture_size.y - 1) / 8 + 1;

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, bokeh.pipelines[BOKEH_GEN_BOKEH]);

	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_bokeh_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_depth_texture), 1);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_base_texture), 2);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

	x_groups = ((p_base_texture_size.x >> 1) - 1) / 8 + 1;
	y_groups = ((p_base_texture_size.y >> 1) - 1) / 8 + 1;

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, bokeh.pipelines[BOKEH_COMPOSITE]);

	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_uniform_set_from_image(p_base_texture), 0);
	RD::get_singleton()->compute_list_bind_uniform_set(compute_list, _get_compute_uniform_set_from_texture(p_bokeh_texture), 1);

	RD::get_singleton()->compute_list_set_push_constant(compute_list, &bokeh.push_constant, sizeof(BokehPushConstant));

	x_groups = (p_base_texture_size.x - 1) / 8 + 1;
	y_groups = (p_base_texture_size.y - 1) / 8 + 1;

	RD::get_singleton()->compute_list_dispatch(compute_list, x_groups, y_groups, 1);
	RD::get_singleton()->compute_list_add_barrier(compute_list);

	RD::get_singleton()->compute_list_end();
}

RasterizerEffectsRD::RasterizerEffectsRD() {

	{
		// Initialize blur
		Vector<String> blur_modes;
		blur_modes.push_back("\n#define MODE_GAUSSIAN_BLUR\n");
		blur_modes.push_back("\n#define MODE_GAUSSIAN_GLOW\n");
		blur_modes.push_back("\n#define MODE_GAUSSIAN_GLOW\n#define GLOW_USE_AUTO_EXPOSURE\n");
		blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_LOW\n");
		blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_MEDIUM\n");
		blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_HIGH\n");
		blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_LOW\n#define DOF_NEAR_BLUR_MERGE\n");
		blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_MEDIUM\n#define DOF_NEAR_BLUR_MERGE\n");
		blur_modes.push_back("\n#define MODE_DOF_NEAR_BLUR\n#define DOF_QUALITY_HIGH\n#define DOF_NEAR_BLUR_MERGE\n");
		blur_modes.push_back("\n#define MODE_DOF_FAR_BLUR\n#define DOF_QUALITY_LOW\n");
		blur_modes.push_back("\n#define MODE_DOF_FAR_BLUR\n#define DOF_QUALITY_MEDIUM\n");
		blur_modes.push_back("\n#define MODE_DOF_FAR_BLUR\n#define DOF_QUALITY_HIGH\n");
		blur_modes.push_back("\n#define MODE_SSAO_MERGE\n");
		blur_modes.push_back("\n#define MODE_SIMPLE_COPY\n");
		blur_modes.push_back("\n#define MODE_MIPMAP\n");

		blur.shader.initialize(blur_modes);
		zeromem(&blur.push_constant, sizeof(BlurPushConstant));
		blur.shader_version = blur.shader.version_create();

		for (int i = 0; i < BLUR_MODE_MAX; i++) {
			blur.pipelines[i].setup(blur.shader.version_get_shader(blur.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	{
		// Initialize roughness
		Vector<String> cubemap_roughness_modes;
		cubemap_roughness_modes.push_back("\n#define MODE_SOURCE_PANORAMA\n");
		cubemap_roughness_modes.push_back("\n#define MODE_SOURCE_CUBEMAP\n");
		roughness.shader.initialize(cubemap_roughness_modes);

		roughness.shader_version = roughness.shader.version_create();

		for (int i = 0; i < CUBEMAP_ROUGHNESS_SOURCE_MAX; i++) {
			roughness.pipelines[i].setup(roughness.shader.version_get_shader(roughness.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	{
		// Initialize sky
		Vector<String> sky_modes;
		sky_modes.push_back("");
		sky.shader.initialize(sky_modes);

		sky.shader_version = sky.shader.version_create();

		RD::PipelineDepthStencilState depth_stencil_state;

		depth_stencil_state.enable_depth_test = true;
		depth_stencil_state.depth_compare_operator = RD::COMPARE_OP_LESS_OR_EQUAL;

		sky.pipeline.setup(sky.shader.version_get_shader(sky.shader_version, 0), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), depth_stencil_state, RD::PipelineColorBlendState::create_disabled(), 0);
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
		copy_modes.push_back("\n#define MODE_CUBE_TO_DP\n");

		copy.shader.initialize(copy_modes);

		copy.shader_version = copy.shader.version_create();

		for (int i = 0; i < COPY_MODE_MAX; i++) {
			copy.pipelines[i].setup(copy.shader.version_get_shader(copy.shader_version, i), RD::RENDER_PRIMITIVE_TRIANGLES, RD::PipelineRasterizationState(), RD::PipelineMultisampleState(), RD::PipelineDepthStencilState(), RD::PipelineColorBlendState::create_disabled(), 0);
		}
	}

	{
		// Initialize bokeh
		Vector<String> bokeh_modes;
		bokeh_modes.push_back("\n#define MODE_GEN_BLUR_SIZE\n");
		bokeh_modes.push_back("\n#define MODE_GEN_BOKEH\n");
		bokeh_modes.push_back("\n#define MODE_COMPOSITE_BOKEH\n");

		bokeh.shader.initialize(bokeh_modes);

		bokeh.shader_version = bokeh.shader.version_create();

		for (int i = 0; i < BOKEH_MAX; i++) {
			bokeh.pipelines[i] = RD::get_singleton()->compute_pipeline_create(bokeh.shader.version_get_shader(bokeh.shader_version, i));
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
		PoolVector<uint8_t> pv;
		pv.resize(6 * 4);
		{
			PoolVector<uint8_t>::Write w = pv.write();
			int *p32 = (int *)w.ptr();
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
	RD::get_singleton()->free(default_sampler);
	blur.shader.version_free(blur.shader_version);
	RD::get_singleton()->free(index_buffer); //array gets freed as dependency
}
