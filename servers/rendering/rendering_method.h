/**************************************************************************/
/*  rendering_method.h                                                    */
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

#ifndef RENDERING_METHOD_H
#define RENDERING_METHOD_H

#include "servers/rendering/storage/render_scene_buffers.h"
#include "servers/rendering_server.h"
#include "servers/xr/xr_interface.h"

class RenderingMethod {
public:
	virtual RID camera_allocate() = 0;
	virtual void camera_initialize(RID p_rid) = 0;

	virtual void camera_set_perspective(RID p_camera, float p_fovy_degrees, float p_z_near, float p_z_far) = 0;
	virtual void camera_set_orthogonal(RID p_camera, float p_size, float p_z_near, float p_z_far) = 0;
	virtual void camera_set_frustum(RID p_camera, float p_size, Vector2 p_offset, float p_z_near, float p_z_far) = 0;
	virtual void camera_set_transform(RID p_camera, const Transform3D &p_transform) = 0;
	virtual void camera_set_cull_mask(RID p_camera, uint32_t p_layers) = 0;
	virtual void camera_set_environment(RID p_camera, RID p_env) = 0;
	virtual void camera_set_camera_attributes(RID p_camera, RID p_attributes) = 0;
	virtual void camera_set_use_vertical_aspect(RID p_camera, bool p_enable) = 0;
	virtual bool is_camera(RID p_camera) const = 0;

	virtual RID occluder_allocate() = 0;
	virtual void occluder_initialize(RID p_occluder) = 0;
	virtual void occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices) = 0;

	virtual RID scenario_allocate() = 0;
	virtual void scenario_initialize(RID p_rid) = 0;

	virtual void scenario_set_environment(RID p_scenario, RID p_environment) = 0;
	virtual void scenario_set_camera_attributes(RID p_scenario, RID p_attributes) = 0;
	virtual void scenario_set_fallback_environment(RID p_scenario, RID p_environment) = 0;
	virtual void scenario_set_reflection_atlas_size(RID p_scenario, int p_reflection_size, int p_reflection_count) = 0;
	virtual bool is_scenario(RID p_scenario) const = 0;
	virtual RID scenario_get_environment(RID p_scenario) = 0;
	virtual void scenario_add_viewport_visibility_mask(RID p_scenario, RID p_viewport) = 0;
	virtual void scenario_remove_viewport_visibility_mask(RID p_scenario, RID p_viewport) = 0;

	virtual RID instance_allocate() = 0;
	virtual void instance_initialize(RID p_rid) = 0;

	virtual void instance_set_base(RID p_instance, RID p_base) = 0;
	virtual void instance_set_scenario(RID p_instance, RID p_scenario) = 0;
	virtual void instance_set_layer_mask(RID p_instance, uint32_t p_mask) = 0;
	virtual void instance_set_pivot_data(RID p_instance, float p_sorting_offset, bool p_use_aabb_center) = 0;
	virtual void instance_set_transform(RID p_instance, const Transform3D &p_transform) = 0;
	virtual void instance_attach_object_instance_id(RID p_instance, ObjectID p_id) = 0;
	virtual void instance_set_blend_shape_weight(RID p_instance, int p_shape, float p_weight) = 0;
	virtual void instance_set_surface_override_material(RID p_instance, int p_surface, RID p_material) = 0;
	virtual void instance_set_visible(RID p_instance, bool p_visible) = 0;
	virtual void instance_geometry_set_transparency(RID p_instance, float p_transparency) = 0;

	virtual void instance_set_custom_aabb(RID p_instance, AABB p_aabb) = 0;

	virtual void instance_attach_skeleton(RID p_instance, RID p_skeleton) = 0;

	virtual void instance_set_extra_visibility_margin(RID p_instance, real_t p_margin) = 0;
	virtual void instance_set_visibility_parent(RID p_instance, RID p_parent_instance) = 0;

	virtual void instance_set_ignore_culling(RID p_instance, bool p_enabled) = 0;

	// don't use these in a game!
	virtual Vector<ObjectID> instances_cull_aabb(const AABB &p_aabb, RID p_scenario = RID()) const = 0;
	virtual Vector<ObjectID> instances_cull_ray(const Vector3 &p_from, const Vector3 &p_to, RID p_scenario = RID()) const = 0;
	virtual Vector<ObjectID> instances_cull_convex(const Vector<Plane> &p_convex, RID p_scenario = RID()) const = 0;

	virtual void instance_geometry_set_flag(RID p_instance, RS::InstanceFlags p_flags, bool p_enabled) = 0;
	virtual void instance_geometry_set_cast_shadows_setting(RID p_instance, RS::ShadowCastingSetting p_shadow_casting_setting) = 0;
	virtual void instance_geometry_set_material_override(RID p_instance, RID p_material) = 0;
	virtual void instance_geometry_set_material_overlay(RID p_instance, RID p_material) = 0;

	virtual void instance_geometry_set_visibility_range(RID p_instance, float p_min, float p_max, float p_min_margin, float p_max_margin, RS::VisibilityRangeFadeMode p_fade_mode) = 0;
	virtual void instance_geometry_set_lightmap(RID p_instance, RID p_lightmap, const Rect2 &p_lightmap_uv_scale, int p_slice_index) = 0;
	virtual void instance_geometry_set_lod_bias(RID p_instance, float p_lod_bias) = 0;
	virtual void instance_geometry_set_shader_parameter(RID p_instance, const StringName &p_parameter, const Variant &p_value) = 0;
	virtual void instance_geometry_get_shader_parameter_list(RID p_instance, List<PropertyInfo> *p_parameters) const = 0;
	virtual Variant instance_geometry_get_shader_parameter(RID p_instance, const StringName &p_parameter) const = 0;
	virtual Variant instance_geometry_get_shader_parameter_default_value(RID p_instance, const StringName &p_parameter) const = 0;

	/* SKY API */

	virtual RID sky_allocate() = 0;
	virtual void sky_initialize(RID p_rid) = 0;

	virtual void sky_set_radiance_size(RID p_sky, int p_radiance_size) = 0;
	virtual void sky_set_mode(RID p_sky, RS::SkyMode p_samples) = 0;
	virtual void sky_set_material(RID p_sky, RID p_material) = 0;
	virtual Ref<Image> sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) = 0;

	/* ENVIRONMENT API */

	virtual RID environment_allocate() = 0;
	virtual void environment_initialize(RID p_rid) = 0;

	// Background
	virtual void environment_set_background(RID p_env, RS::EnvironmentBG p_bg) = 0;
	virtual void environment_set_sky(RID p_env, RID p_sky) = 0;
	virtual void environment_set_sky_custom_fov(RID p_env, float p_scale) = 0;
	virtual void environment_set_sky_orientation(RID p_env, const Basis &p_orientation) = 0;
	virtual void environment_set_bg_color(RID p_env, const Color &p_color) = 0;
	virtual void environment_set_bg_energy(RID p_env, float p_multiplier, float p_exposure_value) = 0;
	virtual void environment_set_canvas_max_layer(RID p_env, int p_max_layer) = 0;
	virtual void environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient = RS::ENV_AMBIENT_SOURCE_BG, float p_energy = 1.0, float p_sky_contribution = 0.0, RS::EnvironmentReflectionSource p_reflection_source = RS::ENV_REFLECTION_SOURCE_BG) = 0;

	virtual RS::EnvironmentBG environment_get_background(RID p_Env) const = 0;
	virtual RID environment_get_sky(RID p_env) const = 0;
	virtual float environment_get_sky_custom_fov(RID p_env) const = 0;
	virtual Basis environment_get_sky_orientation(RID p_env) const = 0;
	virtual Color environment_get_bg_color(RID p_env) const = 0;
	virtual float environment_get_bg_energy_multiplier(RID p_env) const = 0;
	virtual float environment_get_bg_intensity(RID p_env) const = 0;
	virtual int environment_get_canvas_max_layer(RID p_env) const = 0;
	virtual RS::EnvironmentAmbientSource environment_get_ambient_source(RID p_env) const = 0;
	virtual Color environment_get_ambient_light(RID p_env) const = 0;
	virtual float environment_get_ambient_light_energy(RID p_env) const = 0;
	virtual float environment_get_ambient_sky_contribution(RID p_env) const = 0;
	virtual RS::EnvironmentReflectionSource environment_get_reflection_source(RID p_env) const = 0;

	// Tonemap
	virtual void environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white) = 0;

	virtual RS::EnvironmentToneMapper environment_get_tone_mapper(RID p_env) const = 0;
	virtual float environment_get_exposure(RID p_env) const = 0;
	virtual float environment_get_white(RID p_env) const = 0;

	// Fog
	virtual void environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective, float p_sky_affect) = 0;

	virtual bool environment_get_fog_enabled(RID p_env) const = 0;
	virtual Color environment_get_fog_light_color(RID p_env) const = 0;
	virtual float environment_get_fog_light_energy(RID p_env) const = 0;
	virtual float environment_get_fog_sun_scatter(RID p_env) const = 0;
	virtual float environment_get_fog_density(RID p_env) const = 0;
	virtual float environment_get_fog_height(RID p_env) const = 0;
	virtual float environment_get_fog_height_density(RID p_env) const = 0;
	virtual float environment_get_fog_aerial_perspective(RID p_env) const = 0;
	virtual float environment_get_fog_sky_affect(RID p_env) const = 0;

	// Volumetric Fog
	virtual void environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject, float p_sky_affect) = 0;

	virtual bool environment_get_volumetric_fog_enabled(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_density(RID p_env) const = 0;
	virtual Color environment_get_volumetric_fog_scattering(RID p_env) const = 0;
	virtual Color environment_get_volumetric_fog_emission(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_emission_energy(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_anisotropy(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_length(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_detail_spread(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_gi_inject(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_sky_affect(RID p_env) const = 0;
	virtual bool environment_get_volumetric_fog_temporal_reprojection(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_temporal_reprojection_amount(RID p_env) const = 0;
	virtual float environment_get_volumetric_fog_ambient_inject(RID p_env) const = 0;

	virtual void environment_set_volumetric_fog_volume_size(int p_size, int p_depth) = 0;
	virtual void environment_set_volumetric_fog_filter_active(bool p_enable) = 0;

	// Glow

	virtual void environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, RID p_glow_map) = 0;

	virtual bool environment_get_glow_enabled(RID p_env) const = 0;
	virtual Vector<float> environment_get_glow_levels(RID p_env) const = 0;
	virtual float environment_get_glow_intensity(RID p_env) const = 0;
	virtual float environment_get_glow_strength(RID p_env) const = 0;
	virtual float environment_get_glow_bloom(RID p_env) const = 0;
	virtual float environment_get_glow_mix(RID p_env) const = 0;
	virtual RS::EnvironmentGlowBlendMode environment_get_glow_blend_mode(RID p_env) const = 0;
	virtual float environment_get_glow_hdr_bleed_threshold(RID p_env) const = 0;
	virtual float environment_get_glow_hdr_luminance_cap(RID p_env) const = 0;
	virtual float environment_get_glow_hdr_bleed_scale(RID p_env) const = 0;
	virtual float environment_get_glow_map_strength(RID p_env) const = 0;
	virtual RID environment_get_glow_map(RID p_env) const = 0;

	virtual void environment_glow_set_use_bicubic_upscale(bool p_enable) = 0;

	// SSR

	virtual void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) = 0;

	virtual bool environment_get_ssr_enabled(RID p_env) const = 0;
	virtual int environment_get_ssr_max_steps(RID p_env) const = 0;
	virtual float environment_get_ssr_fade_in(RID p_env) const = 0;
	virtual float environment_get_ssr_fade_out(RID p_env) const = 0;
	virtual float environment_get_ssr_depth_tolerance(RID p_env) const = 0;

	virtual void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) = 0;

	// SSAO
	virtual void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) = 0;

	virtual bool environment_get_ssao_enabled(RID p_env) const = 0;
	virtual float environment_get_ssao_radius(RID p_env) const = 0;
	virtual float environment_get_ssao_intensity(RID p_env) const = 0;
	virtual float environment_get_ssao_power(RID p_env) const = 0;
	virtual float environment_get_ssao_detail(RID p_env) const = 0;
	virtual float environment_get_ssao_horizon(RID p_env) const = 0;
	virtual float environment_get_ssao_sharpness(RID p_env) const = 0;
	virtual float environment_get_ssao_direct_light_affect(RID p_env) const = 0;
	virtual float environment_get_ssao_ao_channel_affect(RID p_env) const = 0;

	virtual void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) = 0;

	// SSIL

	virtual void environment_set_ssil(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_sharpness, float p_normal_rejection) = 0;

	virtual bool environment_get_ssil_enabled(RID p_env) const = 0;
	virtual float environment_get_ssil_radius(RID p_env) const = 0;
	virtual float environment_get_ssil_intensity(RID p_env) const = 0;
	virtual float environment_get_ssil_sharpness(RID p_env) const = 0;
	virtual float environment_get_ssil_normal_rejection(RID p_env) const = 0;

	virtual void environment_set_ssil_quality(RS::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) = 0;

	// SDFGI
	virtual void environment_set_sdfgi(RID p_env, bool p_enable, int p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) = 0;

	virtual bool environment_get_sdfgi_enabled(RID p_env) const = 0;
	virtual int environment_get_sdfgi_cascades(RID p_env) const = 0;
	virtual float environment_get_sdfgi_min_cell_size(RID p_env) const = 0;
	virtual bool environment_get_sdfgi_use_occlusion(RID p_env) const = 0;
	virtual float environment_get_sdfgi_bounce_feedback(RID p_env) const = 0;
	virtual bool environment_get_sdfgi_read_sky_light(RID p_env) const = 0;
	virtual float environment_get_sdfgi_energy(RID p_env) const = 0;
	virtual float environment_get_sdfgi_normal_bias(RID p_env) const = 0;
	virtual float environment_get_sdfgi_probe_bias(RID p_env) const = 0;
	virtual RS::EnvironmentSDFGIYScale environment_get_sdfgi_y_scale(RID p_env) const = 0;

	virtual void environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) = 0;
	virtual void environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) = 0;
	virtual void environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) = 0;

	virtual void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) = 0;

	virtual bool environment_get_adjustments_enabled(RID p_env) const = 0;
	virtual float environment_get_adjustments_brightness(RID p_env) const = 0;
	virtual float environment_get_adjustments_contrast(RID p_env) const = 0;
	virtual float environment_get_adjustments_saturation(RID p_env) const = 0;
	virtual bool environment_get_use_1d_color_correction(RID p_env) const = 0;
	virtual RID environment_get_color_correction(RID p_env) const = 0;

	virtual Ref<Image> environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) = 0;

	virtual bool is_environment(RID p_environment) const = 0;

	virtual void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit) = 0;
	virtual void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) = 0;
	virtual void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) = 0;

	virtual void positional_soft_shadow_filter_set_quality(RS::ShadowQuality p_quality) = 0;
	virtual void directional_soft_shadow_filter_set_quality(RS::ShadowQuality p_quality) = 0;

	/* Render Buffers */

	virtual Ref<RenderSceneBuffers> render_buffers_create() = 0;

	virtual void gi_set_use_half_resolution(bool p_enable) = 0;

	virtual void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) = 0;

	virtual TypedArray<Image> bake_render_uv2(RID p_base, const TypedArray<RID> &p_material_overrides, const Size2i &p_image_size) = 0;
	virtual void voxel_gi_set_quality(RS::VoxelGIQuality) = 0;

	virtual void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) = 0;

	virtual void render_empty_scene(const Ref<RenderSceneBuffers> &p_render_buffers, RID p_scenario, RID p_shadow_atlas) = 0;

	struct RenderInfo {
		int info[RS::VIEWPORT_RENDER_INFO_TYPE_MAX][RS::VIEWPORT_RENDER_INFO_MAX] = {};
	};

	virtual void render_camera(const Ref<RenderSceneBuffers> &p_render_buffers, RID p_camera, RID p_scenario, RID p_viewport, Size2 p_viewport_size, uint32_t p_jitter_phase_count, float p_mesh_lod_threshold, RID p_shadow_atlas, Ref<XRInterface> &p_xr_interface, RenderInfo *r_render_info = nullptr) = 0;

	virtual void update() = 0;
	virtual void render_probes() = 0;
	virtual void update_visibility_notifiers() = 0;

	virtual void decals_set_filter(RS::DecalFilter p_filter) = 0;
	virtual void light_projectors_set_filter(RS::LightProjectorFilter p_filter) = 0;

	virtual bool free(RID p_rid) = 0;

	RenderingMethod();
	virtual ~RenderingMethod();
};

#endif // RENDERING_METHOD_H
