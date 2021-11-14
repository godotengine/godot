/*************************************************************************/
/*  renderer_scene_render.h                                              */
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

#ifndef RENDERINGSERVERSCENERENDER_H
#define RENDERINGSERVERSCENERENDER_H

#include "core/math/camera_matrix.h"
#include "core/templates/paged_array.h"
#include "servers/rendering/renderer_scene.h"
#include "servers/rendering/renderer_storage.h"

class RendererSceneRender {
public:
	enum {
		MAX_DIRECTIONAL_LIGHTS = 8,
		MAX_DIRECTIONAL_LIGHT_CASCADES = 4,
		MAX_RENDER_VIEWS = 2
	};

	struct GeometryInstance {
		virtual ~GeometryInstance() {}
	};

	virtual GeometryInstance *geometry_instance_create(RID p_base) = 0;
	virtual void geometry_instance_set_skeleton(GeometryInstance *p_geometry_instance, RID p_skeleton) = 0;
	virtual void geometry_instance_set_material_override(GeometryInstance *p_geometry_instance, RID p_override) = 0;
	virtual void geometry_instance_set_surface_materials(GeometryInstance *p_geometry_instance, const Vector<RID> &p_material) = 0;
	virtual void geometry_instance_set_mesh_instance(GeometryInstance *p_geometry_instance, RID p_mesh_instance) = 0;
	virtual void geometry_instance_set_transform(GeometryInstance *p_geometry_instance, const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabbb) = 0;
	virtual void geometry_instance_set_layer_mask(GeometryInstance *p_geometry_instance, uint32_t p_layer_mask) = 0;
	virtual void geometry_instance_set_lod_bias(GeometryInstance *p_geometry_instance, float p_lod_bias) = 0;
	virtual void geometry_instance_set_transparency(GeometryInstance *p_geometry_instance, float p_transparency) = 0;
	virtual void geometry_instance_set_fade_range(GeometryInstance *p_geometry_instance, bool p_enable_near, float p_near_begin, float p_near_end, bool p_enable_far, float p_far_begin, float p_far_end) = 0;
	virtual void geometry_instance_set_parent_fade_alpha(GeometryInstance *p_geometry_instance, float p_alpha) = 0;
	virtual void geometry_instance_set_use_baked_light(GeometryInstance *p_geometry_instance, bool p_enable) = 0;
	virtual void geometry_instance_set_use_dynamic_gi(GeometryInstance *p_geometry_instance, bool p_enable) = 0;
	virtual void geometry_instance_set_use_lightmap(GeometryInstance *p_geometry_instance, RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) = 0;
	virtual void geometry_instance_set_lightmap_capture(GeometryInstance *p_geometry_instance, const Color *p_sh9) = 0;
	virtual void geometry_instance_set_instance_shader_parameters_offset(GeometryInstance *p_geometry_instance, int32_t p_offset) = 0;
	virtual void geometry_instance_set_cast_double_sided_shadows(GeometryInstance *p_geometry_instance, bool p_enable) = 0;

	virtual uint32_t geometry_instance_get_pair_mask() = 0;
	virtual void geometry_instance_pair_light_instances(GeometryInstance *p_geometry_instance, const RID *p_light_instances, uint32_t p_light_instance_count) = 0;
	virtual void geometry_instance_pair_reflection_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) = 0;
	virtual void geometry_instance_pair_decal_instances(GeometryInstance *p_geometry_instance, const RID *p_decal_instances, uint32_t p_decal_instance_count) = 0;
	virtual void geometry_instance_pair_voxel_gi_instances(GeometryInstance *p_geometry_instance, const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) = 0;

	virtual void geometry_instance_set_softshadow_projector_pairing(GeometryInstance *p_geometry_instance, bool p_softshadow, bool p_projector) = 0;

	virtual void geometry_instance_free(GeometryInstance *p_geometry_instance) = 0;

	/* SHADOW ATLAS API */

	virtual RID shadow_atlas_create() = 0;
	virtual void shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits = false) = 0;
	virtual void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) = 0;
	virtual bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) = 0;

	virtual void directional_shadow_atlas_set_size(int p_size, bool p_16_bits = false) = 0;
	virtual int get_directional_light_shadow_size(RID p_light_intance) = 0;
	virtual void set_directional_shadow_count(int p_count) = 0;

	/* SDFGI UPDATE */

	virtual void sdfgi_update(RID p_render_buffers, RID p_environment, const Vector3 &p_world_position) = 0;
	virtual int sdfgi_get_pending_region_count(RID p_render_buffers) const = 0;
	virtual AABB sdfgi_get_pending_region_bounds(RID p_render_buffers, int p_region) const = 0;
	virtual uint32_t sdfgi_get_pending_region_cascade(RID p_render_buffers, int p_region) const = 0;

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

	virtual void environment_set_background(RID p_env, RS::EnvironmentBG p_bg) = 0;
	virtual void environment_set_sky(RID p_env, RID p_sky) = 0;
	virtual void environment_set_sky_custom_fov(RID p_env, float p_scale) = 0;
	virtual void environment_set_sky_orientation(RID p_env, const Basis &p_orientation) = 0;
	virtual void environment_set_bg_color(RID p_env, const Color &p_color) = 0;
	virtual void environment_set_bg_energy(RID p_env, float p_energy) = 0;
	virtual void environment_set_canvas_max_layer(RID p_env, int p_max_layer) = 0;
	virtual void environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient = RS::ENV_AMBIENT_SOURCE_BG, float p_energy = 1.0, float p_sky_contribution = 0.0, RS::EnvironmentReflectionSource p_reflection_source = RS::ENV_REFLECTION_SOURCE_BG) = 0;
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	virtual void environment_set_camera_feed_id(RID p_env, int p_camera_feed_id) = 0;
#endif

	virtual void environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap) = 0;
	virtual void environment_glow_set_use_bicubic_upscale(bool p_enable) = 0;
	virtual void environment_glow_set_use_high_quality(bool p_enable) = 0;

	virtual void environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject) = 0;
	virtual void environment_set_volumetric_fog_volume_size(int p_size, int p_depth) = 0;
	virtual void environment_set_volumetric_fog_filter_active(bool p_enable) = 0;

	virtual void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) = 0;
	virtual void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) = 0;

	virtual void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) = 0;

	virtual void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) = 0;

	virtual void environment_set_sdfgi(RID p_env, bool p_enable, RS::EnvironmentSDFGICascades p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) = 0;

	virtual void environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) = 0;
	virtual void environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) = 0;
	virtual void environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) = 0;

	virtual void environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) = 0;

	virtual void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) = 0;

	virtual void environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective) = 0;

	virtual Ref<Image> environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) = 0;

	virtual bool is_environment(RID p_env) const = 0;
	virtual RS::EnvironmentBG environment_get_background(RID p_env) const = 0;
	virtual int environment_get_canvas_max_layer(RID p_env) const = 0;

	virtual RID camera_effects_allocate() = 0;
	virtual void camera_effects_initialize(RID p_rid) = 0;

	virtual void camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) = 0;
	virtual void camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) = 0;

	virtual void camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) = 0;
	virtual void camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) = 0;

	virtual void shadows_quality_set(RS::ShadowQuality p_quality) = 0;
	virtual void directional_shadow_quality_set(RS::ShadowQuality p_quality) = 0;

	virtual RID light_instance_create(RID p_light) = 0;
	virtual void light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) = 0;
	virtual void light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) = 0;
	virtual void light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale = 1.0, float p_range_begin = 0, const Vector2 &p_uv_scale = Vector2()) = 0;
	virtual void light_instance_mark_visible(RID p_light_instance) = 0;
	virtual bool light_instances_can_render_shadow_cube() const {
		return true;
	}

	virtual RID fog_volume_instance_create(RID p_fog_volume) = 0;
	virtual void fog_volume_instance_set_transform(RID p_fog_volume_instance, const Transform3D &p_transform) = 0;
	virtual void fog_volume_instance_set_active(RID p_fog_volume_instance, bool p_active) = 0;
	virtual RID fog_volume_instance_get_volume(RID p_fog_volume_instance) const = 0;
	virtual Vector3 fog_volume_instance_get_position(RID p_fog_volume_instance) const = 0;

	virtual RID reflection_atlas_create() = 0;
	virtual void reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) = 0;
	virtual int reflection_atlas_get_size(RID p_ref_atlas) const = 0;

	virtual RID reflection_probe_instance_create(RID p_probe) = 0;
	virtual void reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) = 0;
	virtual void reflection_probe_release_atlas_index(RID p_instance) = 0;
	virtual bool reflection_probe_instance_needs_redraw(RID p_instance) = 0;
	virtual bool reflection_probe_instance_has_reflection(RID p_instance) = 0;
	virtual bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) = 0;
	virtual bool reflection_probe_instance_postprocess_step(RID p_instance) = 0;

	virtual RID decal_instance_create(RID p_decal) = 0;
	virtual void decal_instance_set_transform(RID p_decal, const Transform3D &p_transform) = 0;

	virtual RID lightmap_instance_create(RID p_lightmap) = 0;
	virtual void lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) = 0;

	virtual RID voxel_gi_instance_create(RID p_voxel_gi) = 0;
	virtual void voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform) = 0;
	virtual bool voxel_gi_needs_update(RID p_probe) const = 0;
	virtual void voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<GeometryInstance *> &p_dynamic_objects) = 0;

	virtual void voxel_gi_set_quality(RS::VoxelGIQuality) = 0;

	struct RenderShadowData {
		RID light;
		int pass = 0;
		PagedArray<GeometryInstance *> instances;
	};

	struct RenderSDFGIData {
		int region = 0;
		PagedArray<GeometryInstance *> instances;
	};

	struct RenderSDFGIUpdateData {
		bool update_static = false;
		uint32_t static_cascade_count;
		uint32_t *static_cascade_indices;
		PagedArray<RID> *static_positional_lights;

		const Vector<RID> *directional_lights;
		const RID *positional_light_instances;
		uint32_t positional_light_count;
	};

	struct CameraData {
		// flags
		uint32_t view_count;
		bool is_ortogonal;
		bool vaspect;

		// Main/center projection
		Transform3D main_transform;
		CameraMatrix main_projection;

		Transform3D view_offset[RendererSceneRender::MAX_RENDER_VIEWS];
		CameraMatrix view_projection[RendererSceneRender::MAX_RENDER_VIEWS];

		void set_camera(const Transform3D p_transform, const CameraMatrix p_projection, bool p_is_ortogonal, bool p_vaspect);
		void set_multiview_camera(uint32_t p_view_count, const Transform3D *p_transforms, const CameraMatrix *p_projections, bool p_is_ortogonal, bool p_vaspect);
	};

	virtual void render_scene(RID p_render_buffers, const CameraData *p_camera_data, const PagedArray<GeometryInstance *> &p_instances, const PagedArray<RID> &p_lights, const PagedArray<RID> &p_reflection_probes, const PagedArray<RID> &p_voxel_gi_instances, const PagedArray<RID> &p_decals, const PagedArray<RID> &p_lightmaps, const PagedArray<RID> &p_fog_volumes, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_occluder_debug_tex, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_lod_threshold, const RenderShadowData *p_render_shadows, int p_render_shadow_count, const RenderSDFGIData *p_render_sdfgi_regions, int p_render_sdfgi_region_count, const RenderSDFGIUpdateData *p_sdfgi_update_data = nullptr, RendererScene::RenderInfo *r_render_info = nullptr) = 0;

	virtual void render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) = 0;
	virtual void render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<GeometryInstance *> &p_instances) = 0;

	virtual void set_scene_pass(uint64_t p_pass) = 0;
	virtual void set_time(double p_time, double p_step) = 0;
	virtual void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) = 0;

	virtual RID render_buffers_create() = 0;
	virtual void render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding, uint32_t p_view_count) = 0;
	virtual void gi_set_use_half_resolution(bool p_enable) = 0;

	virtual void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit) = 0;
	virtual bool screen_space_roughness_limiter_is_active() const = 0;

	virtual void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) = 0;
	virtual void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) = 0;

	virtual TypedArray<Image> bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) = 0;

	virtual bool free(RID p_rid) = 0;

	virtual void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) = 0;

	virtual void decals_set_filter(RS::DecalFilter p_filter) = 0;
	virtual void light_projectors_set_filter(RS::LightProjectorFilter p_filter) = 0;

	virtual void update() = 0;
	virtual ~RendererSceneRender() {}
};

#endif // RENDERINGSERVERSCENERENDER_H
