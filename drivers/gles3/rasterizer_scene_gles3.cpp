/*************************************************************************/
/*  rasterizer_scene_gles3.cpp                                           */
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

#include "rasterizer_scene_gles3.h"
#include "core/config/project_settings.h"
#include "servers/rendering/rendering_server_default.h"

#ifdef GLES3_ENABLED

uint64_t RasterizerSceneGLES3::auto_exposure_counter = 2;

RasterizerSceneGLES3 *RasterizerSceneGLES3::singleton = nullptr;

RasterizerSceneGLES3 *RasterizerSceneGLES3::get_singleton() {
	return singleton;
}

RasterizerSceneGLES3::GeometryInstance *RasterizerSceneGLES3::geometry_instance_create(RID p_base) {
	return nullptr;
}

void RasterizerSceneGLES3::geometry_instance_set_skeleton(GeometryInstance *p_geometry_instance, RID p_skeleton) {
}

void RasterizerSceneGLES3::geometry_instance_set_material_override(GeometryInstance *p_geometry_instance, RID p_override) {
}

void RasterizerSceneGLES3::geometry_instance_set_material_overlay(GeometryInstance *p_geometry_instance, RID p_overlay) {
}

void RasterizerSceneGLES3::geometry_instance_set_surface_materials(GeometryInstance *p_geometry_instance, const Vector<RID> &p_material) {
}

void RasterizerSceneGLES3::geometry_instance_set_mesh_instance(GeometryInstance *p_geometry_instance, RID p_mesh_instance) {
}

void RasterizerSceneGLES3::geometry_instance_set_transform(GeometryInstance *p_geometry_instance, const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabbb) {
}

void RasterizerSceneGLES3::geometry_instance_set_layer_mask(GeometryInstance *p_geometry_instance, uint32_t p_layer_mask) {
}

void RasterizerSceneGLES3::geometry_instance_set_lod_bias(GeometryInstance *p_geometry_instance, float p_lod_bias) {
}

void RasterizerSceneGLES3::geometry_instance_set_transparency(GeometryInstance *p_geometry_instance, float p_transparency) {
}

void RasterizerSceneGLES3::geometry_instance_set_fade_range(GeometryInstance *p_geometry_instance, bool p_enable_near, float p_near_begin, float p_near_end, bool p_enable_far, float p_far_begin, float p_far_end) {
}

void RasterizerSceneGLES3::geometry_instance_set_parent_fade_alpha(GeometryInstance *p_geometry_instance, float p_alpha) {
}

void RasterizerSceneGLES3::geometry_instance_set_use_baked_light(GeometryInstance *p_geometry_instance, bool p_enable) {
}

void RasterizerSceneGLES3::geometry_instance_set_use_dynamic_gi(GeometryInstance *p_geometry_instance, bool p_enable) {
}

void RasterizerSceneGLES3::geometry_instance_set_use_lightmap(GeometryInstance *p_geometry_instance, RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) {
}

void RasterizerSceneGLES3::geometry_instance_set_lightmap_capture(GeometryInstance *p_geometry_instance, const Color *p_sh9) {
}

void RasterizerSceneGLES3::geometry_instance_set_instance_shader_parameters_offset(GeometryInstance *p_geometry_instance, int32_t p_offset) {
}

void RasterizerSceneGLES3::geometry_instance_set_cast_double_sided_shadows(GeometryInstance *p_geometry_instance, bool p_enable) {
}

uint32_t RasterizerSceneGLES3::geometry_instance_get_pair_mask() {
	return 0;
}

void RasterizerSceneGLES3::geometry_instance_pair_light_instances(GeometryInstance *p_geometry_instance, const RID *p_light_instances, uint32_t p_light_instance_count) {
}

void RasterizerSceneGLES3::geometry_instance_pair_reflection_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) {
}

void RasterizerSceneGLES3::geometry_instance_pair_decal_instances(GeometryInstance *p_geometry_instance, const RID *p_decal_instances, uint32_t p_decal_instance_count) {
}

void RasterizerSceneGLES3::geometry_instance_pair_voxel_gi_instances(GeometryInstance *p_geometry_instance, const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) {
}

void RasterizerSceneGLES3::geometry_instance_set_softshadow_projector_pairing(GeometryInstance *p_geometry_instance, bool p_softshadow, bool p_projector) {
}

void RasterizerSceneGLES3::geometry_instance_free(GeometryInstance *p_geometry_instance) {
}

/* SHADOW ATLAS API */

RID RasterizerSceneGLES3::shadow_atlas_create() {
	return RID();
}

void RasterizerSceneGLES3::shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits) {
}

void RasterizerSceneGLES3::shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) {
}

bool RasterizerSceneGLES3::shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) {
	return false;
}

void RasterizerSceneGLES3::directional_shadow_atlas_set_size(int p_size, bool p_16_bits) {
}

int RasterizerSceneGLES3::get_directional_light_shadow_size(RID p_light_intance) {
	return 0;
}

void RasterizerSceneGLES3::set_directional_shadow_count(int p_count) {
}

/* SKY API */

void RasterizerSceneGLES3::Sky::free() {
	if (radiance != 0) {
		glDeleteTextures(1, &radiance);
		radiance = 0;
		glDeleteFramebuffers(1, &radiance_framebuffer);
		radiance_framebuffer = 0;
	}
}

RID RasterizerSceneGLES3::sky_allocate() {
	return sky_owner.allocate_rid();
}

void RasterizerSceneGLES3::sky_initialize(RID p_rid) {
	sky_owner.initialize_rid(p_rid);
}

void RasterizerSceneGLES3::sky_set_radiance_size(RID p_sky, int p_radiance_size) {
	Sky *sky = sky_owner.get_or_null(p_sky);
	ERR_FAIL_COND(!sky);
	ERR_FAIL_COND_MSG(p_radiance_size < 32 || p_radiance_size > 2048, "Sky radiance size must be between 32 and 2048");

	if (sky->radiance_size == p_radiance_size) {
		return; // No need to update
	}

	sky->radiance_size = p_radiance_size;

	sky->free();
}

void RasterizerSceneGLES3::sky_set_mode(RID p_sky, RS::SkyMode p_mode) {
	Sky *sky = sky_owner.get_or_null(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->mode == p_mode) {
		return;
	}

	sky->mode = p_mode;

	if (sky->mode == RS::SKY_MODE_REALTIME) {
		WARN_PRINT_ONCE("The OpenGL renderer does not support the Real Time Sky Update Mode yet. Please use High Quality Mode instead");
	}
}

void RasterizerSceneGLES3::sky_set_material(RID p_sky, RID p_material) {
	Sky *sky = sky_owner.get_or_null(p_sky);
	ERR_FAIL_COND(!sky);

	if (sky->material == p_material) {
		return;
	}

	sky->material = p_material;
}

void RasterizerSceneGLES3::_invalidate_sky(Sky *p_sky) {
	if (!p_sky->dirty) {
		p_sky->dirty = true;
		p_sky->dirty_list = dirty_sky_list;
		dirty_sky_list = p_sky;
	}
}

void RasterizerSceneGLES3::_update_dirty_skys() {
	Sky *sky = dirty_sky_list;

	while (sky) {
		if (sky->radiance == 0) {
			//int mipmaps = Image::get_image_required_mipmaps(sky->radiance_size, sky->radiance_size, Image::FORMAT_RGBAH) + 1;

			//uint32_t w = sky->radiance_size, h = sky->radiance_size;
			//int layers = sky_globals.roughness_layers;
			glGenFramebuffers(1, &sky->radiance_framebuffer);

			glGenTextures(1, &sky->radiance);
		}

		sky->reflection_dirty = true;
		sky->processing_layer = 0;

		Sky *next = sky->dirty_list;
		sky->dirty_list = nullptr;
		sky->dirty = false;
		sky = next;
	}

	dirty_sky_list = nullptr;
}

void RasterizerSceneGLES3::_draw_sky(Sky *p_sky, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_custom_fov, float p_energy, const Basis &p_sky_orientation) {
	ERR_FAIL_COND(!p_sky);

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDepthFunc(GL_LEQUAL);
	glColorMask(1, 1, 1, 1);

	//state.sky_shader.version_bind_shader(sky_globals.default_shader, SkyShaderGLES3::MODE_BACKGROUND);
	//glBindBufferBase(GL_UNIFORM_BUFFER, 0, state.canvas_instance_data_buffers[state.current_buffer]); // Canvas data updated here
	//glBindBufferBase(GL_UNIFORM_BUFFER, 1, state.canvas_instance_data_buffers[state.current_buffer]); // Global data
	//glBindBufferBase(GL_UNIFORM_BUFFER, 2, state.canvas_instance_data_buffers[state.current_buffer]); // Directional light data
	//glBindBufferBase(GL_UNIFORM_BUFFER, 3, state.canvas_instance_data_buffers[state.current_buffer]); // Material uniforms

	// Camera
	CameraMatrix camera;

	if (p_custom_fov) {
		float near_plane = p_projection.get_z_near();
		float far_plane = p_projection.get_z_far();
		float aspect = p_projection.get_aspect();

		camera.set_perspective(p_custom_fov, aspect, near_plane, far_plane);

	} else {
		camera = p_projection;
	}

	glDrawArrays(GL_TRIANGLES, 0, 3);
}

Ref<Image> RasterizerSceneGLES3::sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) {
	return Ref<Image>();
}

/* ENVIRONMENT API */

RID RasterizerSceneGLES3::environment_allocate() {
	return environment_owner.allocate_rid();
}

void RasterizerSceneGLES3::environment_initialize(RID p_rid) {
	environment_owner.initialize_rid(p_rid);
}

void RasterizerSceneGLES3::environment_set_background(RID p_env, RS::EnvironmentBG p_bg) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->background = p_bg;
}

void RasterizerSceneGLES3::environment_set_sky(RID p_env, RID p_sky) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky = p_sky;
}

void RasterizerSceneGLES3::environment_set_sky_custom_fov(RID p_env, float p_scale) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky_custom_fov = p_scale;
}

void RasterizerSceneGLES3::environment_set_sky_orientation(RID p_env, const Basis &p_orientation) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->sky_orientation = p_orientation;
}

void RasterizerSceneGLES3::environment_set_bg_color(RID p_env, const Color &p_color) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->bg_color = p_color;
}

void RasterizerSceneGLES3::environment_set_bg_energy(RID p_env, float p_energy) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->bg_energy = p_energy;
}

void RasterizerSceneGLES3::environment_set_canvas_max_layer(RID p_env, int p_max_layer) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->canvas_max_layer = p_max_layer;
}

void RasterizerSceneGLES3::environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RS::EnvironmentReflectionSource p_reflection_source) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->ambient_light = p_color;
	env->ambient_source = p_ambient;
	env->ambient_light_energy = p_energy;
	env->ambient_sky_contribution = p_sky_contribution;
	env->reflection_source = p_reflection_source;
}

void RasterizerSceneGLES3::environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, RID p_glow_map) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	ERR_FAIL_COND_MSG(p_levels.size() != 7, "Size of array of glow levels must be 7");
	env->glow_enabled = p_enable;
	env->glow_levels = p_levels;
	env->glow_intensity = p_intensity;
	env->glow_strength = p_strength;
	env->glow_mix = p_mix;
	env->glow_bloom = p_bloom_threshold;
	env->glow_blend_mode = p_blend_mode;
	env->glow_hdr_bleed_threshold = p_hdr_bleed_threshold;
	env->glow_hdr_bleed_scale = p_hdr_bleed_scale;
	env->glow_hdr_luminance_cap = p_hdr_luminance_cap;
	env->glow_map_strength = p_glow_map_strength;
	env->glow_map = p_glow_map;
}

void RasterizerSceneGLES3::environment_glow_set_use_bicubic_upscale(bool p_enable) {
	glow_bicubic_upscale = p_enable;
}

void RasterizerSceneGLES3::environment_glow_set_use_high_quality(bool p_enable) {
	glow_high_quality = p_enable;
}

void RasterizerSceneGLES3::environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->ssr_enabled = p_enable;
	env->ssr_max_steps = p_max_steps;
	env->ssr_fade_in = p_fade_int;
	env->ssr_fade_out = p_fade_out;
	env->ssr_depth_tolerance = p_depth_tolerance;
}

void RasterizerSceneGLES3::environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) {
}

void RasterizerSceneGLES3::environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
}

void RasterizerSceneGLES3::environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) {
}

void RasterizerSceneGLES3::environment_set_ssil(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_sharpness, float p_normal_rejection) {
}
void RasterizerSceneGLES3::environment_set_ssil_quality(RS::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) {
}

void RasterizerSceneGLES3::environment_set_sdfgi(RID p_env, bool p_enable, int p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) {
}

void RasterizerSceneGLES3::environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) {
}

void RasterizerSceneGLES3::environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) {
}

void RasterizerSceneGLES3::environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) {
}

void RasterizerSceneGLES3::environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->exposure = p_exposure;
	env->tone_mapper = p_tone_mapper;
	if (!env->auto_exposure && p_auto_exposure) {
		env->auto_exposure_version = ++auto_exposure_counter;
	}
	env->auto_exposure = p_auto_exposure;
	env->white = p_white;
	env->min_luminance = p_min_luminance;
	env->max_luminance = p_max_luminance;
	env->auto_exp_speed = p_auto_exp_speed;
	env->auto_exp_scale = p_auto_exp_scale;
}

void RasterizerSceneGLES3::environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->adjustments_enabled = p_enable;
	env->adjustments_brightness = p_brightness;
	env->adjustments_contrast = p_contrast;
	env->adjustments_saturation = p_saturation;
	env->use_1d_color_correction = p_use_1d_color_correction;
	env->color_correction = p_color_correction;
}

void RasterizerSceneGLES3::environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND(!env);
	env->fog_enabled = p_enable;
	env->fog_light_color = p_light_color;
	env->fog_light_energy = p_light_energy;
	env->fog_sun_scatter = p_sun_scatter;
	env->fog_density = p_density;
	env->fog_height = p_height;
	env->fog_height_density = p_height_density;
	env->fog_aerial_perspective = p_aerial_perspective;
}

void RasterizerSceneGLES3::environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject) {
}

void RasterizerSceneGLES3::environment_set_volumetric_fog_volume_size(int p_size, int p_depth) {
}

void RasterizerSceneGLES3::environment_set_volumetric_fog_filter_active(bool p_enable) {
}

Ref<Image> RasterizerSceneGLES3::environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, Ref<Image>());
	return Ref<Image>();
}

bool RasterizerSceneGLES3::is_environment(RID p_env) const {
	return environment_owner.owns(p_env);
}

RS::EnvironmentBG RasterizerSceneGLES3::environment_get_background(RID p_env) const {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, RS::ENV_BG_MAX);
	return env->background;
}

int RasterizerSceneGLES3::environment_get_canvas_max_layer(RID p_env) const {
	Environment *env = environment_owner.get_or_null(p_env);
	ERR_FAIL_COND_V(!env, 0);
	return env->canvas_max_layer;
}

RID RasterizerSceneGLES3::camera_effects_allocate() {
	return RID();
}

void RasterizerSceneGLES3::camera_effects_initialize(RID p_rid) {
}

void RasterizerSceneGLES3::camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) {
}

void RasterizerSceneGLES3::camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) {
}

void RasterizerSceneGLES3::camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) {
}

void RasterizerSceneGLES3::camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) {
}

void RasterizerSceneGLES3::shadows_quality_set(RS::ShadowQuality p_quality) {
}

void RasterizerSceneGLES3::directional_shadow_quality_set(RS::ShadowQuality p_quality) {
}

RID RasterizerSceneGLES3::light_instance_create(RID p_light) {
	return RID();
}

void RasterizerSceneGLES3::light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) {
}

void RasterizerSceneGLES3::light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) {
}

void RasterizerSceneGLES3::light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale, float p_range_begin, const Vector2 &p_uv_scale) {
}

void RasterizerSceneGLES3::light_instance_mark_visible(RID p_light_instance) {
}

RID RasterizerSceneGLES3::fog_volume_instance_create(RID p_fog_volume) {
	return RID();
}

void RasterizerSceneGLES3::fog_volume_instance_set_transform(RID p_fog_volume_instance, const Transform3D &p_transform) {
}

void RasterizerSceneGLES3::fog_volume_instance_set_active(RID p_fog_volume_instance, bool p_active) {
}

RID RasterizerSceneGLES3::fog_volume_instance_get_volume(RID p_fog_volume_instance) const {
	return RID();
}

Vector3 RasterizerSceneGLES3::fog_volume_instance_get_position(RID p_fog_volume_instance) const {
	return Vector3();
}

RID RasterizerSceneGLES3::reflection_atlas_create() {
	return RID();
}

int RasterizerSceneGLES3::reflection_atlas_get_size(RID p_ref_atlas) const {
	return 0;
}

void RasterizerSceneGLES3::reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) {
}

RID RasterizerSceneGLES3::reflection_probe_instance_create(RID p_probe) {
	return RID();
}

void RasterizerSceneGLES3::reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) {
}

void RasterizerSceneGLES3::reflection_probe_release_atlas_index(RID p_instance) {
}

bool RasterizerSceneGLES3::reflection_probe_instance_needs_redraw(RID p_instance) {
	return false;
}

bool RasterizerSceneGLES3::reflection_probe_instance_has_reflection(RID p_instance) {
	return false;
}

bool RasterizerSceneGLES3::reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) {
	return false;
}

bool RasterizerSceneGLES3::reflection_probe_instance_postprocess_step(RID p_instance) {
	return true;
}

RID RasterizerSceneGLES3::decal_instance_create(RID p_decal) {
	return RID();
}

void RasterizerSceneGLES3::decal_instance_set_transform(RID p_decal, const Transform3D &p_transform) {
}

RID RasterizerSceneGLES3::lightmap_instance_create(RID p_lightmap) {
	return RID();
}

void RasterizerSceneGLES3::lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) {
}

RID RasterizerSceneGLES3::voxel_gi_instance_create(RID p_voxel_gi) {
	return RID();
}

void RasterizerSceneGLES3::voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform) {
}

bool RasterizerSceneGLES3::voxel_gi_needs_update(RID p_probe) const {
	return false;
}

void RasterizerSceneGLES3::voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RendererSceneRender::GeometryInstance *> &p_dynamic_objects) {
}

void RasterizerSceneGLES3::voxel_gi_set_quality(RS::VoxelGIQuality) {
}

void RasterizerSceneGLES3::render_scene(RID p_render_buffers, const CameraData *p_camera_data, const PagedArray<GeometryInstance *> &p_instances, const PagedArray<RID> &p_lights, const PagedArray<RID> &p_reflection_probes, const PagedArray<RID> &p_voxel_gi_instances, const PagedArray<RID> &p_decals, const PagedArray<RID> &p_lightmaps, const PagedArray<RID> &p_fog_volumes, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_occluder_debug_tex, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_mesh_lod_threshold, const RenderShadowData *p_render_shadows, int p_render_shadow_count, const RenderSDFGIData *p_render_sdfgi_regions, int p_render_sdfgi_region_count, const RenderSDFGIUpdateData *p_sdfgi_update_data, RendererScene::RenderInfo *r_render_info) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	RENDER_TIMESTAMP("Setup 3D Scene");
	// assign render data
	// Use the format from rendererRD
	RenderDataGLES3 render_data;
	{
		render_data.render_buffers = p_render_buffers;

		// Our first camera is used by default
		render_data.cam_transform = p_camera_data->main_transform;
		render_data.cam_projection = p_camera_data->main_projection;
		render_data.view_projection[0] = p_camera_data->main_projection;
		render_data.cam_ortogonal = p_camera_data->is_orthogonal;

		render_data.view_count = p_camera_data->view_count;
		for (uint32_t v = 0; v < p_camera_data->view_count; v++) {
			render_data.view_projection[v] = p_camera_data->view_projection[v];
		}

		render_data.z_near = p_camera_data->main_projection.get_z_near();
		render_data.z_far = p_camera_data->main_projection.get_z_far();

		render_data.instances = &p_instances;
		render_data.lights = &p_lights;
		render_data.reflection_probes = &p_reflection_probes;
		//render_data.voxel_gi_instances = &p_voxel_gi_instances;
		//render_data.decals = &p_decals;
		//render_data.lightmaps = &p_lightmaps;
		//render_data.fog_volumes = &p_fog_volumes;
		render_data.environment = p_environment;
		render_data.camera_effects = p_camera_effects;
		render_data.shadow_atlas = p_shadow_atlas;
		render_data.reflection_atlas = p_reflection_atlas;
		render_data.reflection_probe = p_reflection_probe;
		render_data.reflection_probe_pass = p_reflection_probe_pass;

		// this should be the same for all cameras..
		render_data.lod_distance_multiplier = p_camera_data->main_projection.get_lod_multiplier();
		render_data.lod_camera_plane = Plane(-p_camera_data->main_transform.basis.get_column(Vector3::AXIS_Z), p_camera_data->main_transform.get_origin());

		if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_DISABLE_LOD) {
			render_data.screen_mesh_lod_threshold = 0.0;
		} else {
			render_data.screen_mesh_lod_threshold = p_screen_mesh_lod_threshold;
		}
		render_data.render_info = r_render_info;
	}

	PagedArray<RID> empty;

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_UNSHADED) {
		render_data.lights = &empty;
		render_data.reflection_probes = &empty;
	}

	RenderBuffers *rb = nullptr;
	//RasterizerStorageGLES3::RenderTarget *rt = nullptr;
	if (p_render_buffers.is_valid()) {
		rb = render_buffers_owner.get_or_null(p_render_buffers);
		ERR_FAIL_COND(!rb);
		//rt = texture_storage->render_target_owner.get_or_null(rb->render_target);
		//ERR_FAIL_COND(!rt);
	}

	Color clear_color;
	if (p_render_buffers.is_valid()) {
		clear_color = texture_storage->render_target_get_clear_request_color(rb->render_target);
	} else {
		clear_color = storage->get_default_clear_color();
	}

	Environment *env = environment_owner.get_or_null(p_environment);

	bool fb_cleared = false;

	glDepthFunc(GL_LEQUAL);

	/* Depth Prepass */

	glBindFramebuffer(GL_FRAMEBUFFER, rb->framebuffer);

	if (!fb_cleared) {
		glClearDepth(1.0f);
		glClear(GL_DEPTH_BUFFER_BIT);
	}

	bool draw_sky = false;
	bool keep_color = false;

	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW) {
		clear_color = Color(0, 0, 0, 1);
	}
	if (get_debug_draw_mode() == RS::VIEWPORT_DEBUG_DRAW_OVERDRAW) {
		clear_color = Color(0, 0, 0, 1); //in overdraw mode, BG should always be black
	} else if (is_environment(p_environment)) {
		RS::EnvironmentBG bg_mode = environment_get_background(p_environment);
		float bg_energy = env->bg_energy; //environment_get_bg_energy(p_environment);
		switch (bg_mode) {
			case RS::ENV_BG_CLEAR_COLOR: {
				clear_color.r *= bg_energy;
				clear_color.g *= bg_energy;
				clear_color.b *= bg_energy;
			} break;
			case RS::ENV_BG_COLOR: {
				clear_color = env->bg_color; //environment_get_bg_color(p_environment);
				clear_color.r *= bg_energy;
				clear_color.g *= bg_energy;
				clear_color.b *= bg_energy;
			} break;
			case RS::ENV_BG_SKY: {
				draw_sky = true;
			} break;
			case RS::ENV_BG_CANVAS: {
				keep_color = true;
			} break;
			case RS::ENV_BG_KEEP: {
				keep_color = true;
			} break;
			case RS::ENV_BG_CAMERA_FEED: {
			} break;
			default: {
			}
		}
	}

	if (!keep_color) {
		glClearBufferfv(GL_COLOR, 0, clear_color.components);
	}

	if (draw_sky) {
		//_draw_sky(sky, render_data.cam_projection, render_data.cam_transform, env->sky_custom_fov, env->bg_energy, env->sky_orientation);
	}

	if (p_render_buffers.is_valid()) {
		/*
		RENDER_TIMESTAMP("Tonemap");
		_render_buffers_post_process_and_tonemap(&render_data);
		*/

		_render_buffers_debug_draw(p_render_buffers, p_shadow_atlas, p_occluder_debug_tex);
	}
	texture_storage->render_target_disable_clear_request(rb->render_target);
}

void RasterizerSceneGLES3::render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) {
}

void RasterizerSceneGLES3::render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<GeometryInstance *> &p_instances) {
}

void RasterizerSceneGLES3::set_time(double p_time, double p_step) {
	time = p_time;
	time_step = p_step;
}

void RasterizerSceneGLES3::set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) {
	debug_draw = p_debug_draw;
}

RID RasterizerSceneGLES3::render_buffers_create() {
	RenderBuffers rb;
	return render_buffers_owner.make_rid(rb);
}

/* BACK FBO */
/* For MSAA */
/*
#ifndef JAVASCRIPT_ENABLED
	if (rt->msaa >= RS::VIEWPORT_MSAA_2X && rt->msaa <= RS::VIEWPORT_MSAA_8X) {
		rt->multisample_active = true;

		static const int msaa_value[] = { 0, 2, 4, 8, 16 };
		int msaa = msaa_value[rt->msaa];

		int max_samples = 0;
		glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
		if (msaa > max_samples) {
			WARN_PRINT("MSAA must be <= GL_MAX_SAMPLES, falling-back to GL_MAX_SAMPLES = " + itos(max_samples));
			msaa = max_samples;
		}

		//regular fbo
		glGenFramebuffers(1, &rt->multisample_fbo);
		bind_framebuffer(rt->multisample_fbo);

		glGenRenderbuffers(1, &rt->multisample_depth);
		glBindRenderbuffer(GL_RENDERBUFFER, rt->multisample_depth);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, config.depth_buffer_internalformat, rt->size.x, rt->size.y);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->multisample_depth);

		glGenRenderbuffers(1, &rt->multisample_color);
		glBindRenderbuffer(GL_RENDERBUFFER, rt->multisample_color);
		glRenderbufferStorageMultisample(GL_RENDERBUFFER, msaa, color_internal_format, rt->size.x, rt->size.y);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rt->multisample_color);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			// Delete allocated resources and default to no MSAA
			WARN_PRINT_ONCE("Cannot allocate back framebuffer for MSAA");
			printf("err status: %x\n", status);
			rt->multisample_active = false;

			glDeleteFramebuffers(1, &rt->multisample_fbo);
			rt->multisample_fbo = 0;

			glDeleteRenderbuffers(1, &rt->multisample_depth);
			rt->multisample_depth = 0;

			glDeleteRenderbuffers(1, &rt->multisample_color);
			rt->multisample_color = 0;
		}

		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		bind_framebuffer(0);

	} else
#endif // JAVASCRIPT_ENABLED
	{
		rt->multisample_active = false;
	}
	*/

// copy texscreen buffers
//	if (!(rt->flags[RendererStorage::RENDER_TARGET_NO_SAMPLING])) {
/*
if (false) {
glGenTextures(1, &rt->copy_screen_effect.color);
glBindTexture(GL_TEXTURE_2D, rt->copy_screen_effect.color);

if (rt->flags[RendererStorage::RENDER_TARGET_TRANSPARENT]) {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rt->size.x, rt->size.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
} else {
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, rt->size.x, rt->size.y, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
}

glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

glGenFramebuffers(1, &rt->copy_screen_effect.fbo);
bind_framebuffer(rt->copy_screen_effect.fbo);
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->copy_screen_effect.color, 0);

glClearColor(0, 0, 0, 0);
glClear(GL_COLOR_BUFFER_BIT);

GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
if (status != GL_FRAMEBUFFER_COMPLETE) {
	_clear_render_target(rt);
	ERR_FAIL_COND(status != GL_FRAMEBUFFER_COMPLETE);
}
}
*/

void RasterizerSceneGLES3::render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_internal_width, int p_internal_height, int p_width, int p_height, float p_fsr_sharpness, float p_fsr_mipmap_bias, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding, uint32_t p_view_count) {
	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();

	RenderBuffers *rb = render_buffers_owner.get_or_null(p_render_buffers);
	ERR_FAIL_COND(!rb);

	//rb->internal_width = p_internal_width; // ignore for now
	//rb->internal_height = p_internal_height;
	rb->width = p_width;
	rb->height = p_height;
	//rb->fsr_sharpness = p_fsr_sharpness;
	rb->render_target = p_render_target;
	//rb->msaa = p_msaa;
	//rb->screen_space_aa = p_screen_space_aa;
	//rb->use_debanding = p_use_debanding;
	//rb->view_count = p_view_count;

	_free_render_buffer_data(rb);

	GLES3::RenderTarget *rt = texture_storage->get_render_target(p_render_target);

	// framebuffer
	glGenFramebuffers(1, &rb->framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, rb->framebuffer);

	glBindTexture(GL_TEXTURE_2D, rt->color);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->color, 0);

	glGenTextures(1, &rb->depth_texture);
	glBindTexture(GL_TEXTURE_2D, rb->depth_texture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, rt->size.x, rt->size.y, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, nullptr);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rb->depth_texture, 0);

	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, texture_storage->system_fbo);

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		_free_render_buffer_data(rb);
		WARN_PRINT("Could not create 3D renderbuffer, status: " + texture_storage->get_framebuffer_error(status));
		return;
	}
}

void RasterizerSceneGLES3::_free_render_buffer_data(RenderBuffers *rb) {
	if (rb->depth_texture) {
		glDeleteTextures(1, &rb->depth_texture);
		rb->depth_texture = 0;
	}
	if (rb->framebuffer) {
		glDeleteFramebuffers(1, &rb->framebuffer);
		rb->framebuffer = 0;
	}
}

//clear render buffers
/*


		if (rt->copy_screen_effect.color) {
		glDeleteFramebuffers(1, &rt->copy_screen_effect.fbo);
		rt->copy_screen_effect.fbo = 0;

		glDeleteTextures(1, &rt->copy_screen_effect.color);
		rt->copy_screen_effect.color = 0;
	}

	if (rt->multisample_active) {
		glDeleteFramebuffers(1, &rt->multisample_fbo);
		rt->multisample_fbo = 0;

		glDeleteRenderbuffers(1, &rt->multisample_depth);
		rt->multisample_depth = 0;

		glDeleteRenderbuffers(1, &rt->multisample_color);

		rt->multisample_color = 0;
	}
*/

void RasterizerSceneGLES3::_render_buffers_debug_draw(RID p_render_buffers, RID p_shadow_atlas, RID p_occlusion_buffer) {
}

void RasterizerSceneGLES3::gi_set_use_half_resolution(bool p_enable) {
}

void RasterizerSceneGLES3::screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_curve) {
}

bool RasterizerSceneGLES3::screen_space_roughness_limiter_is_active() const {
	return false;
}

void RasterizerSceneGLES3::sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) {
}

void RasterizerSceneGLES3::sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) {
}

TypedArray<Image> RasterizerSceneGLES3::bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) {
	return TypedArray<Image>();
}

bool RasterizerSceneGLES3::free(RID p_rid) {
	if (environment_owner.owns(p_rid)) {
		environment_owner.free(p_rid);
	} else if (sky_owner.owns(p_rid)) {
		Sky *sky = sky_owner.get_or_null(p_rid);
		ERR_FAIL_COND_V(!sky, false);
		sky->free();
		sky_owner.free(p_rid);
	} else if (render_buffers_owner.owns(p_rid)) {
		RenderBuffers *rb = render_buffers_owner.get_or_null(p_rid);
		ERR_FAIL_COND_V(!rb, false);
		_free_render_buffer_data(rb);
		render_buffers_owner.free(p_rid);

	} else {
		return false;
	}
	return true;
}

void RasterizerSceneGLES3::update() {
	_update_dirty_skys();
}

void RasterizerSceneGLES3::sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) {
}

void RasterizerSceneGLES3::decals_set_filter(RS::DecalFilter p_filter) {
}

void RasterizerSceneGLES3::light_projectors_set_filter(RS::LightProjectorFilter p_filter) {
}

RasterizerSceneGLES3::RasterizerSceneGLES3(RasterizerStorageGLES3 *p_storage) {
	GLES3::MaterialStorage *material_storage = GLES3::MaterialStorage::get_singleton();
	storage = p_storage;

	{
		// Initialize Sky stuff
		sky_globals.roughness_layers = GLOBAL_GET("rendering/reflections/sky_reflections/roughness_layers");
		sky_globals.ggx_samples = GLOBAL_GET("rendering/reflections/sky_reflections/ggx_samples");

		String global_defines;
		global_defines += "#define MAX_GLOBAL_VARIABLES 256\n"; // TODO: this is arbitrary for now
		global_defines += "\n#define MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS " + itos(sky_globals.max_directional_lights) + "\n";
		state.sky_shader.initialize(global_defines);
		sky_globals.shader_default_version = state.sky_shader.version_create();
		state.sky_shader.version_bind_shader(sky_globals.shader_default_version, SkyShaderGLES3::MODE_BACKGROUND);
	}

	{
		sky_globals.default_shader = material_storage->shader_allocate();

		material_storage->shader_initialize(sky_globals.default_shader);

		material_storage->shader_set_code(sky_globals.default_shader, R"(
// Default sky shader.

shader_type sky;

void sky() {
	COLOR = vec3(0.0);
}
)");
		sky_globals.default_material = material_storage->material_allocate();
		material_storage->material_initialize(sky_globals.default_material);

		material_storage->material_set_shader(sky_globals.default_material, sky_globals.default_shader);
	}
}

RasterizerSceneGLES3::~RasterizerSceneGLES3() {
	state.sky_shader.version_free(sky_globals.shader_default_version);
	storage->free(sky_globals.default_material);
	storage->free(sky_globals.default_shader);
}

#endif // GLES3_ENABLED
