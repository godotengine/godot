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
#ifdef GLES3_ENABLED

// TODO: 3D support not implemented yet.

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

/* SDFGI UPDATE */

void RasterizerSceneGLES3::sdfgi_update(RID p_render_buffers, RID p_environment, const Vector3 &p_world_position) {
}

int RasterizerSceneGLES3::sdfgi_get_pending_region_count(RID p_render_buffers) const {
	return 0;
}

AABB RasterizerSceneGLES3::sdfgi_get_pending_region_bounds(RID p_render_buffers, int p_region) const {
	return AABB();
}

uint32_t RasterizerSceneGLES3::sdfgi_get_pending_region_cascade(RID p_render_buffers, int p_region) const {
	return 0;
}

/* SKY API */

RID RasterizerSceneGLES3::sky_allocate() {
	return RID();
}

void RasterizerSceneGLES3::sky_initialize(RID p_rid) {
}

void RasterizerSceneGLES3::sky_set_radiance_size(RID p_sky, int p_radiance_size) {
}

void RasterizerSceneGLES3::sky_set_mode(RID p_sky, RS::SkyMode p_samples) {
}

void RasterizerSceneGLES3::sky_set_material(RID p_sky, RID p_material) {
}

Ref<Image> RasterizerSceneGLES3::sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) {
	return Ref<Image>();
}

/* ENVIRONMENT API */

RID RasterizerSceneGLES3::environment_allocate() {
	return RID();
}

void RasterizerSceneGLES3::environment_initialize(RID p_rid) {
}

void RasterizerSceneGLES3::environment_set_background(RID p_env, RS::EnvironmentBG p_bg) {
}

void RasterizerSceneGLES3::environment_set_sky(RID p_env, RID p_sky) {
}

void RasterizerSceneGLES3::environment_set_sky_custom_fov(RID p_env, float p_scale) {
}

void RasterizerSceneGLES3::environment_set_sky_orientation(RID p_env, const Basis &p_orientation) {
}

void RasterizerSceneGLES3::environment_set_bg_color(RID p_env, const Color &p_color) {
}

void RasterizerSceneGLES3::environment_set_bg_energy(RID p_env, float p_energy) {
}

void RasterizerSceneGLES3::environment_set_canvas_max_layer(RID p_env, int p_max_layer) {
}

void RasterizerSceneGLES3::environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient, float p_energy, float p_sky_contribution, RS::EnvironmentReflectionSource p_reflection_source) {
}

void RasterizerSceneGLES3::environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, RID p_glow_map) {
}

void RasterizerSceneGLES3::environment_glow_set_use_bicubic_upscale(bool p_enable) {
}

void RasterizerSceneGLES3::environment_glow_set_use_high_quality(bool p_enable) {
}

void RasterizerSceneGLES3::environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) {
}

void RasterizerSceneGLES3::environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) {
}

void RasterizerSceneGLES3::environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) {
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
}

void RasterizerSceneGLES3::environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) {
}

void RasterizerSceneGLES3::environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective) {
}

void RasterizerSceneGLES3::environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject) {
}

void RasterizerSceneGLES3::environment_set_volumetric_fog_volume_size(int p_size, int p_depth) {
}

void RasterizerSceneGLES3::environment_set_volumetric_fog_filter_active(bool p_enable) {
}

Ref<Image> RasterizerSceneGLES3::environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) {
	return Ref<Image>();
}

bool RasterizerSceneGLES3::is_environment(RID p_env) const {
	return false;
}

RS::EnvironmentBG RasterizerSceneGLES3::environment_get_background(RID p_env) const {
	return RS::ENV_BG_KEEP;
}

int RasterizerSceneGLES3::environment_get_canvas_max_layer(RID p_env) const {
	return 0;
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
}

void RasterizerSceneGLES3::render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) {
}

void RasterizerSceneGLES3::render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<GeometryInstance *> &p_instances) {
}

void RasterizerSceneGLES3::set_scene_pass(uint64_t p_pass) {
}

void RasterizerSceneGLES3::set_time(double p_time, double p_step) {
}

void RasterizerSceneGLES3::set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) {
}

RID RasterizerSceneGLES3::render_buffers_create() {
	return RID();
}

void RasterizerSceneGLES3::render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_internal_width, int p_internal_height, int p_width, int p_height, float p_fsr_sharpness, float p_fsr_mipmap_bias, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding, uint32_t p_view_count) {
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
	return false;
}

void RasterizerSceneGLES3::update() {
}

void RasterizerSceneGLES3::sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) {
}

void RasterizerSceneGLES3::decals_set_filter(RS::DecalFilter p_filter) {
}

void RasterizerSceneGLES3::light_projectors_set_filter(RS::LightProjectorFilter p_filter) {
}

RasterizerSceneGLES3::RasterizerSceneGLES3() {
}

#endif // GLES3_ENABLED
