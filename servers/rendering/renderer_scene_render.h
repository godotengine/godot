/*************************************************************************/
/*  renderer_scene_render.h                                              */
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

#ifndef RENDERINGSERVERSCENERENDER_H
#define RENDERINGSERVERSCENERENDER_H

#include "core/math/camera_matrix.h"
#include "servers/rendering/renderer_storage.h"

class RendererSceneRender {
public:
	/* SHADOW ATLAS API */

	virtual RID shadow_atlas_create() = 0;
	virtual void shadow_atlas_set_size(RID p_atlas, int p_size) = 0;
	virtual void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) = 0;
	virtual bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) = 0;

	virtual void directional_shadow_atlas_set_size(int p_size) = 0;
	virtual int get_directional_light_shadow_size(RID p_light_intance) = 0;
	virtual void set_directional_shadow_count(int p_count) = 0;

	/* SDFGI UPDATE */

	struct InstanceBase;

	virtual void sdfgi_update(RID p_render_buffers, RID p_environment, const Vector3 &p_world_position) = 0;
	virtual int sdfgi_get_pending_region_count(RID p_render_buffers) const = 0;
	virtual AABB sdfgi_get_pending_region_bounds(RID p_render_buffers, int p_region) const = 0;
	virtual uint32_t sdfgi_get_pending_region_cascade(RID p_render_buffers, int p_region) const = 0;
	virtual void sdfgi_update_probes(RID p_render_buffers, RID p_environment, const RID *p_directional_light_instances, uint32_t p_directional_light_count, const RID *p_positional_light_instances, uint32_t p_positional_light_count) = 0;

	/* SKY API */

	virtual RID sky_create() = 0;
	virtual void sky_set_radiance_size(RID p_sky, int p_radiance_size) = 0;
	virtual void sky_set_mode(RID p_sky, RS::SkyMode p_samples) = 0;
	virtual void sky_set_material(RID p_sky, RID p_material) = 0;
	virtual Ref<Image> sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) = 0;

	/* ENVIRONMENT API */

	virtual RID environment_create() = 0;

	virtual void environment_set_background(RID p_env, RS::EnvironmentBG p_bg) = 0;
	virtual void environment_set_sky(RID p_env, RID p_sky) = 0;
	virtual void environment_set_sky_custom_fov(RID p_env, float p_scale) = 0;
	virtual void environment_set_sky_orientation(RID p_env, const Basis &p_orientation) = 0;
	virtual void environment_set_bg_color(RID p_env, const Color &p_color) = 0;
	virtual void environment_set_bg_energy(RID p_env, float p_energy) = 0;
	virtual void environment_set_canvas_max_layer(RID p_env, int p_max_layer) = 0;
	virtual void environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient = RS::ENV_AMBIENT_SOURCE_BG, float p_energy = 1.0, float p_sky_contribution = 0.0, RS::EnvironmentReflectionSource p_reflection_source = RS::ENV_REFLECTION_SOURCE_BG, const Color &p_ao_color = Color()) = 0;
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	virtual void environment_set_camera_feed_id(RID p_env, int p_camera_feed_id) = 0;
#endif

	virtual void environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap) = 0;
	virtual void environment_glow_set_use_bicubic_upscale(bool p_enable) = 0;
	virtual void environment_glow_set_use_high_quality(bool p_enable) = 0;

	virtual void environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_light, float p_light_energy, float p_length, float p_detail_spread, float p_gi_inject, RS::EnvVolumetricFogShadowFilter p_shadow_filter) = 0;

	virtual void environment_set_volumetric_fog_volume_size(int p_size, int p_depth) = 0;
	virtual void environment_set_volumetric_fog_filter_active(bool p_enable) = 0;
	virtual void environment_set_volumetric_fog_directional_shadow_shrink_size(int p_shrink_size) = 0;
	virtual void environment_set_volumetric_fog_positional_shadow_shrink_size(int p_shrink_size) = 0;

	virtual void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) = 0;
	virtual void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) = 0;

	virtual void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_bias, float p_light_affect, float p_ao_channel_affect, RS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness) = 0;

	virtual void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size) = 0;

	virtual void environment_set_sdfgi(RID p_env, bool p_enable, RS::EnvironmentSDFGICascades p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, bool p_use_multibounce, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) = 0;

	virtual void environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) = 0;
	virtual void environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) = 0;

	virtual void environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) = 0;

	virtual void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) = 0;

	virtual void environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective) = 0;

	virtual Ref<Image> environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) = 0;

	virtual bool is_environment(RID p_env) const = 0;
	virtual RS::EnvironmentBG environment_get_background(RID p_env) const = 0;
	virtual int environment_get_canvas_max_layer(RID p_env) const = 0;

	virtual RID camera_effects_create() = 0;

	virtual void camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) = 0;
	virtual void camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) = 0;

	virtual void camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) = 0;
	virtual void camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) = 0;

	virtual void shadows_quality_set(RS::ShadowQuality p_quality) = 0;
	virtual void directional_shadow_quality_set(RS::ShadowQuality p_quality) = 0;

	struct InstanceBase : public RendererStorage::InstanceBaseDependency {
		RS::InstanceType base_type;
		RID base;

		RID skeleton;
		RID material_override;

		RID instance_data;

		Transform transform;

		int depth_layer;
		uint32_t layer_mask;

		//RID sampled_light;

		Vector<RID> materials;
		Vector<RID> light_instances;
		Vector<RID> reflection_probe_instances;
		Vector<RID> gi_probe_instances;

		Vector<float> blend_values;

		RS::ShadowCastingSetting cast_shadows;

		//fit in 32 bits
		bool mirror : 8;
		bool receive_shadows : 8;
		bool visible : 8;
		bool baked_light : 2; //this flag is only to know if it actually did use baked light
		bool dynamic_gi : 2; //this flag is only to know if it actually did use baked light
		bool redraw_if_visible : 4;

		float depth; //used for sorting

		InstanceBase *lightmap;
		Rect2 lightmap_uv_scale;
		int lightmap_slice_index;
		uint32_t lightmap_cull_index;
		Vector<Color> lightmap_sh; //spherical harmonic

		AABB aabb;
		AABB transformed_aabb;

		struct InstanceShaderParameter {
			int32_t index = -1;
			Variant value;
			Variant default_value;
			PropertyInfo info;
		};

		Map<StringName, InstanceShaderParameter> instance_shader_parameters;
		bool instance_allocated_shader_parameters = false;
		int32_t instance_allocated_shader_parameters_offset = -1;

		InstanceBase() {
			base_type = RS::INSTANCE_NONE;
			cast_shadows = RS::SHADOW_CASTING_SETTING_ON;
			receive_shadows = true;
			visible = true;
			depth_layer = 0;
			layer_mask = 1;
			instance_version = 0;
			baked_light = false;
			dynamic_gi = false;
			redraw_if_visible = false;
			lightmap_slice_index = 0;
			lightmap = nullptr;
			lightmap_cull_index = 0;
		}

		virtual ~InstanceBase() {
		}
	};

	virtual RID light_instance_create(RID p_light) = 0;
	virtual void light_instance_set_transform(RID p_light_instance, const Transform &p_transform) = 0;
	virtual void light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) = 0;
	virtual void light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale = 1.0, float p_range_begin = 0, const Vector2 &p_uv_scale = Vector2()) = 0;
	virtual void light_instance_mark_visible(RID p_light_instance) = 0;
	virtual bool light_instances_can_render_shadow_cube() const {
		return true;
	}

	virtual RID reflection_atlas_create() = 0;
	virtual void reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) = 0;

	virtual RID reflection_probe_instance_create(RID p_probe) = 0;
	virtual void reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform) = 0;
	virtual void reflection_probe_release_atlas_index(RID p_instance) = 0;
	virtual bool reflection_probe_instance_needs_redraw(RID p_instance) = 0;
	virtual bool reflection_probe_instance_has_reflection(RID p_instance) = 0;
	virtual bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) = 0;
	virtual bool reflection_probe_instance_postprocess_step(RID p_instance) = 0;

	virtual RID decal_instance_create(RID p_decal) = 0;
	virtual void decal_instance_set_transform(RID p_decal, const Transform &p_transform) = 0;

	virtual RID gi_probe_instance_create(RID p_gi_probe) = 0;
	virtual void gi_probe_instance_set_transform_to_data(RID p_probe, const Transform &p_xform) = 0;
	virtual bool gi_probe_needs_update(RID p_probe) const = 0;
	virtual void gi_probe_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, int p_dynamic_object_count, InstanceBase **p_dynamic_objects) = 0;

	virtual void gi_probe_set_quality(RS::GIProbeQuality) = 0;

	virtual void render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, RID *p_decal_cull_result, int p_decal_cull_count, InstanceBase **p_lightmap_cull_result, int p_lightmap_cull_count, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) = 0;

	virtual void render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count) = 0;
	virtual void render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region) = 0;
	virtual void render_sdfgi(RID p_render_buffers, int p_region, InstanceBase **p_cull_result, int p_cull_count) = 0;
	virtual void render_sdfgi_static_lights(RID p_render_buffers, uint32_t p_cascade_count, const uint32_t *p_cascade_indices, const RID **p_positional_light_cull_result, const uint32_t *p_positional_light_cull_count) = 0;
	virtual void render_particle_collider_heightfield(RID p_collider, const Transform &p_transform, InstanceBase **p_cull_result, int p_cull_count) = 0;

	virtual void set_scene_pass(uint64_t p_pass) = 0;
	virtual void set_time(double p_time, double p_step) = 0;
	virtual void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) = 0;

	virtual RID render_buffers_create() = 0;
	virtual void render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding) = 0;

	virtual void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_limit) = 0;
	virtual bool screen_space_roughness_limiter_is_active() const = 0;

	virtual void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) = 0;
	virtual void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) = 0;

	virtual TypedArray<Image> bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) = 0;

	virtual bool free(RID p_rid) = 0;

	virtual void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) = 0;

	virtual void update() = 0;
	virtual ~RendererSceneRender() {}
};

#endif // RENDERINGSERVERSCENERENDER_H
