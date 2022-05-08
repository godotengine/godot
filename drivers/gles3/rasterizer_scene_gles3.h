/*************************************************************************/
/*  rasterizer_scene_gles3.h                                             */
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

#ifndef RASTERIZER_SCENE_OPENGL_H
#define RASTERIZER_SCENE_OPENGL_H

#ifdef GLES3_ENABLED

#include "core/math/camera_matrix.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "rasterizer_storage_gles3.h"
#include "scene/resources/mesh.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_scene_render.h"
#include "servers/rendering_server.h"
#include "shader_gles3.h"
#include "shaders/sky.glsl.gen.h"

// Copied from renderer_scene_render_rd
struct RenderDataGLES3 {
	RID render_buffers = RID();

	Transform3D cam_transform = Transform3D();
	CameraMatrix cam_projection = CameraMatrix();
	bool cam_ortogonal = false;

	// For stereo rendering
	uint32_t view_count = 1;
	CameraMatrix view_projection[RendererSceneRender::MAX_RENDER_VIEWS];

	float z_near = 0.0;
	float z_far = 0.0;

	const PagedArray<RendererSceneRender::GeometryInstance *> *instances = nullptr;
	const PagedArray<RID> *lights = nullptr;
	const PagedArray<RID> *reflection_probes = nullptr;
	//const PagedArray<RID> *voxel_gi_instances = nullptr;
	//const PagedArray<RID> *decals = nullptr;
	//const PagedArray<RID> *lightmaps = nullptr;
	//const PagedArray<RID> *fog_volumes = nullptr;
	RID environment = RID();
	RID camera_effects = RID();
	RID shadow_atlas = RID();
	RID reflection_atlas = RID();
	RID reflection_probe = RID();
	int reflection_probe_pass = 0;

	float lod_distance_multiplier = 0.0;
	Plane lod_camera_plane = Plane();
	float screen_mesh_lod_threshold = 0.0;

	uint32_t directional_light_count = 0;

	RendererScene::RenderInfo *render_info = nullptr;
};

class RasterizerStorageGLES3;
class RasterizerCanvasGLES3;

class RasterizerSceneGLES3 : public RendererSceneRender {
private:
	static RasterizerSceneGLES3 *singleton;
	RS::ViewportDebugDraw debug_draw = RS::VIEWPORT_DEBUG_DRAW_DISABLED;
	uint64_t scene_pass = 0;

	/* Sky */
	struct SkyGlobals {
		RID shader_current_version;
		RID shader_default_version;
		RID default_material;
		RID default_shader;
		uint32_t max_directional_lights = 4;
		uint32_t roughness_layers = 8;
		uint32_t ggx_samples = 128;
	} sky_globals;

protected:
	double time;
	double time_step = 0;

	struct RenderBuffers {
		int internal_width = 0;
		int internal_height = 0;
		int width = 0;
		int height = 0;
		//float fsr_sharpness = 0.2f;
		RS::ViewportMSAA msaa = RS::VIEWPORT_MSAA_DISABLED;
		//RS::ViewportScreenSpaceAA screen_space_aa = RS::VIEWPORT_SCREEN_SPACE_AA_DISABLED;
		//bool use_debanding = false;
		//uint32_t view_count = 1;

		RID render_target;
		GLuint internal_texture = 0; // Used for rendering when post effects are enabled
		GLuint depth_texture = 0; // Main depth texture
		GLuint framebuffer = 0; // Main framebuffer, contains internal_texture and depth_texture or render_target->color and depth_texture

		//built-in textures used for ping pong image processing and blurring
		struct Blur {
			RID texture;

			struct Mipmap {
				RID texture;
				int width;
				int height;
				GLuint fbo;
			};

			Vector<Mipmap> mipmaps;
		};

		Blur blur[2]; //the second one starts from the first mipmap

		/*
		GLuint fbo = 0;
		GLuint color = 0;
		GLuint depth = 0;

		GLuint multisample_fbo = 0;
		GLuint multisample_color = 0;
		GLuint multisample_depth = 0;
		bool multisample_active = false;

		struct Effect {
			GLuint fbo = 0;
			int width = 0;
			int height = 0;

			GLuint color = 0;

			Effect() {
			}
		};

		Effect copy_screen_effect;

		struct MipMaps {
			struct Size {
				GLuint fbo;
				GLuint color;
				int width;
				int height;
			};

			Vector<Size> sizes;
			GLuint color = 0;
			int levels = 0;

			MipMaps() {
			}
		};

		MipMaps mip_maps[2];

		*/
	};

	bool screen_space_roughness_limiter = false;
	float screen_space_roughness_limiter_amount = 0.25;
	float screen_space_roughness_limiter_limit = 0.18;

	mutable RID_Owner<RenderBuffers, true> render_buffers_owner;

	void _free_render_buffer_data(RenderBuffers *rb);
	void _allocate_blur_textures(RenderBuffers *rb);
	void _allocate_depth_backbuffer_textures(RenderBuffers *rb);

	void _render_buffers_debug_draw(RID p_render_buffers, RID p_shadow_atlas, RID p_occlusion_buffer);

	/* Environment */

	struct Environment {
		// BG
		RS::EnvironmentBG background = RS::ENV_BG_CLEAR_COLOR;
		RID sky;
		float sky_custom_fov = 0.0;
		Basis sky_orientation;
		Color bg_color;
		float bg_energy = 1.0;
		int canvas_max_layer = 0;
		RS::EnvironmentAmbientSource ambient_source = RS::ENV_AMBIENT_SOURCE_BG;
		Color ambient_light;
		float ambient_light_energy = 1.0;
		float ambient_sky_contribution = 1.0;
		RS::EnvironmentReflectionSource reflection_source = RS::ENV_REFLECTION_SOURCE_BG;
		Color ao_color;

		/// Tonemap

		RS::EnvironmentToneMapper tone_mapper;
		float exposure = 1.0;
		float white = 1.0;
		bool auto_exposure = false;
		float min_luminance = 0.2;
		float max_luminance = 8.0;
		float auto_exp_speed = 0.2;
		float auto_exp_scale = 0.5;
		uint64_t auto_exposure_version = 0;

		// Fog
		bool fog_enabled = false;
		Color fog_light_color = Color(0.5, 0.6, 0.7);
		float fog_light_energy = 1.0;
		float fog_sun_scatter = 0.0;
		float fog_density = 0.001;
		float fog_height = 0.0;
		float fog_height_density = 0.0; //can be negative to invert effect
		float fog_aerial_perspective = 0.0;

		/// Glow
		bool glow_enabled = false;
		Vector<float> glow_levels;
		float glow_intensity = 0.8;
		float glow_strength = 1.0;
		float glow_bloom = 0.0;
		float glow_mix = 0.01;
		RS::EnvironmentGlowBlendMode glow_blend_mode = RS::ENV_GLOW_BLEND_MODE_SOFTLIGHT;
		float glow_hdr_bleed_threshold = 1.0;
		float glow_hdr_luminance_cap = 12.0;
		float glow_hdr_bleed_scale = 2.0;
		float glow_map_strength = 1.0;
		RID glow_map = RID();

		/// SSAO
		bool ssao_enabled = false;
		float ssao_radius = 1.0;
		float ssao_intensity = 2.0;
		float ssao_power = 1.5;
		float ssao_detail = 0.5;
		float ssao_horizon = 0.06;
		float ssao_sharpness = 0.98;
		float ssao_direct_light_affect = 0.0;
		float ssao_ao_channel_affect = 0.0;

		/// SSR
		bool ssr_enabled = false;
		int ssr_max_steps = 64;
		float ssr_fade_in = 0.15;
		float ssr_fade_out = 2.0;
		float ssr_depth_tolerance = 0.2;

		/// Adjustments
		bool adjustments_enabled = false;
		float adjustments_brightness = 1.0f;
		float adjustments_contrast = 1.0f;
		float adjustments_saturation = 1.0f;
		bool use_1d_color_correction = false;
		RID color_correction = RID();
	};

	RS::EnvironmentSSAOQuality ssao_quality = RS::ENV_SSAO_QUALITY_MEDIUM;
	bool ssao_half_size = false;
	bool ssao_using_half_size = false;
	float ssao_adaptive_target = 0.5;
	int ssao_blur_passes = 2;
	float ssao_fadeout_from = 50.0;
	float ssao_fadeout_to = 300.0;

	bool glow_bicubic_upscale = false;
	bool glow_high_quality = false;
	RS::EnvironmentSSRRoughnessQuality ssr_roughness_quality = RS::ENV_SSR_ROUGHNESS_QUALITY_LOW;

	static uint64_t auto_exposure_counter;

	mutable RID_Owner<Environment, true> environment_owner;

	/* Sky */

	struct Sky {
		// Screen Buffers
		GLuint half_res_pass = 0;
		GLuint half_res_framebuffer = 0;
		GLuint quarter_res_pass = 0;
		GLuint quarter_res_framebuffer = 0;
		Size2i screen_size = Size2i(0, 0);

		// Radiance Cubemap
		GLuint radiance = 0;
		GLuint radiance_framebuffer = 0;

		RID material;
		RID uniform_buffer;

		int radiance_size = 256;

		RS::SkyMode mode = RS::SKY_MODE_AUTOMATIC;

		//ReflectionData reflection;
		bool reflection_dirty = false;
		bool dirty = false;
		int processing_layer = 0;
		Sky *dirty_list = nullptr;

		//State to track when radiance cubemap needs updating
		//SkyMaterialData *prev_material;
		Vector3 prev_position = Vector3(0.0, 0.0, 0.0);
		float prev_time = 0.0f;

		void free();
		bool set_radiance_size(int p_radiance_size);
		bool set_mode(RS::SkyMode p_mode);
		bool set_material(RID p_material);
		Ref<Image> bake_panorama(float p_energy, int p_roughness_layers, const Size2i &p_size);
	};

	Sky *dirty_sky_list = nullptr;
	mutable RID_Owner<Sky, true> sky_owner;

	void _invalidate_sky(Sky *p_sky);
	void _update_dirty_skys();
	void _draw_sky(Sky *p_sky, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_custom_fov, float p_energy, const Basis &p_sky_orientation);

public:
	RasterizerStorageGLES3 *storage;
	RasterizerCanvasGLES3 *canvas;

	// References to shaders are needed in public space so they can be accessed in RasterizerStorageGLES3
	struct State {
		SkyShaderGLES3 sky_shader;
	} state;

	GeometryInstance *geometry_instance_create(RID p_base) override;
	void geometry_instance_set_skeleton(GeometryInstance *p_geometry_instance, RID p_skeleton) override;
	void geometry_instance_set_material_override(GeometryInstance *p_geometry_instance, RID p_override) override;
	void geometry_instance_set_material_overlay(GeometryInstance *p_geometry_instance, RID p_overlay) override;
	void geometry_instance_set_surface_materials(GeometryInstance *p_geometry_instance, const Vector<RID> &p_material) override;
	void geometry_instance_set_mesh_instance(GeometryInstance *p_geometry_instance, RID p_mesh_instance) override;
	void geometry_instance_set_transform(GeometryInstance *p_geometry_instance, const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabbb) override;
	void geometry_instance_set_layer_mask(GeometryInstance *p_geometry_instance, uint32_t p_layer_mask) override;
	void geometry_instance_set_lod_bias(GeometryInstance *p_geometry_instance, float p_lod_bias) override;
	void geometry_instance_set_transparency(GeometryInstance *p_geometry_instance, float p_transparency) override;
	void geometry_instance_set_fade_range(GeometryInstance *p_geometry_instance, bool p_enable_near, float p_near_begin, float p_near_end, bool p_enable_far, float p_far_begin, float p_far_end) override;
	void geometry_instance_set_parent_fade_alpha(GeometryInstance *p_geometry_instance, float p_alpha) override;
	void geometry_instance_set_use_baked_light(GeometryInstance *p_geometry_instance, bool p_enable) override;
	void geometry_instance_set_use_dynamic_gi(GeometryInstance *p_geometry_instance, bool p_enable) override;
	void geometry_instance_set_use_lightmap(GeometryInstance *p_geometry_instance, RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) override;
	void geometry_instance_set_lightmap_capture(GeometryInstance *p_geometry_instance, const Color *p_sh9) override;
	void geometry_instance_set_instance_shader_parameters_offset(GeometryInstance *p_geometry_instance, int32_t p_offset) override;
	void geometry_instance_set_cast_double_sided_shadows(GeometryInstance *p_geometry_instance, bool p_enable) override;

	uint32_t geometry_instance_get_pair_mask() override;
	void geometry_instance_pair_light_instances(GeometryInstance *p_geometry_instance, const RID *p_light_instances, uint32_t p_light_instance_count) override;
	void geometry_instance_pair_reflection_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) override;
	void geometry_instance_pair_decal_instances(GeometryInstance *p_geometry_instance, const RID *p_decal_instances, uint32_t p_decal_instance_count) override;
	void geometry_instance_pair_voxel_gi_instances(GeometryInstance *p_geometry_instance, const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) override;
	void geometry_instance_set_softshadow_projector_pairing(GeometryInstance *p_geometry_instance, bool p_softshadow, bool p_projector) override;

	void geometry_instance_free(GeometryInstance *p_geometry_instance) override;

	/* SHADOW ATLAS API */

	RID shadow_atlas_create() override;
	void shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits = true) override;
	void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) override;
	bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) override;

	void directional_shadow_atlas_set_size(int p_size, bool p_16_bits = true) override;
	int get_directional_light_shadow_size(RID p_light_intance) override;
	void set_directional_shadow_count(int p_count) override;

	/* SDFGI UPDATE */

	void sdfgi_update(RID p_render_buffers, RID p_environment, const Vector3 &p_world_position) override {}
	int sdfgi_get_pending_region_count(RID p_render_buffers) const override { return 0; }
	AABB sdfgi_get_pending_region_bounds(RID p_render_buffers, int p_region) const override { return AABB(); }
	uint32_t sdfgi_get_pending_region_cascade(RID p_render_buffers, int p_region) const override { return 0; }

	/* SKY API */

	RID sky_allocate() override;
	void sky_initialize(RID p_rid) override;
	void sky_set_radiance_size(RID p_sky, int p_radiance_size) override;
	void sky_set_mode(RID p_sky, RS::SkyMode p_mode) override;
	void sky_set_material(RID p_sky, RID p_material) override;
	Ref<Image> sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) override;

	/* ENVIRONMENT API */

	RID environment_allocate() override;
	void environment_initialize(RID p_rid) override;
	void environment_set_background(RID p_env, RS::EnvironmentBG p_bg) override;
	void environment_set_sky(RID p_env, RID p_sky) override;
	void environment_set_sky_custom_fov(RID p_env, float p_scale) override;
	void environment_set_sky_orientation(RID p_env, const Basis &p_orientation) override;
	void environment_set_bg_color(RID p_env, const Color &p_color) override;
	void environment_set_bg_energy(RID p_env, float p_energy) override;
	void environment_set_canvas_max_layer(RID p_env, int p_max_layer) override;
	void environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient = RS::ENV_AMBIENT_SOURCE_BG, float p_energy = 1.0, float p_sky_contribution = 0.0, RS::EnvironmentReflectionSource p_reflection_source = RS::ENV_REFLECTION_SOURCE_BG) override;

	void environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap, float p_glow_map_strength, RID p_glow_map) override;
	void environment_glow_set_use_bicubic_upscale(bool p_enable) override;
	void environment_glow_set_use_high_quality(bool p_enable) override;

	void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) override;
	void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) override;
	void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) override;
	void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override;
	void environment_set_ssil(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_sharpness, float p_normal_rejection) override;
	void environment_set_ssil_quality(RS::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override;

	void environment_set_sdfgi(RID p_env, bool p_enable, int p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) override;

	void environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) override;
	void environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) override;
	void environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) override;

	void environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) override;

	void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) override;

	void environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective) override;
	void environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_albedo, const Color &p_emission, float p_emission_energy, float p_anisotropy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount, float p_ambient_inject) override;
	void environment_set_volumetric_fog_volume_size(int p_size, int p_depth) override;
	void environment_set_volumetric_fog_filter_active(bool p_enable) override;

	Ref<Image> environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) override;

	bool is_environment(RID p_env) const override;
	RS::EnvironmentBG environment_get_background(RID p_env) const override;
	int environment_get_canvas_max_layer(RID p_env) const override;

	RID camera_effects_allocate() override;
	void camera_effects_initialize(RID p_rid) override;
	void camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) override;
	void camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) override;

	void camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) override;
	void camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) override;

	void shadows_quality_set(RS::ShadowQuality p_quality) override;
	void directional_shadow_quality_set(RS::ShadowQuality p_quality) override;

	RID light_instance_create(RID p_light) override;
	void light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) override;
	void light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) override;
	void light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale = 1.0, float p_range_begin = 0, const Vector2 &p_uv_scale = Vector2()) override;
	void light_instance_mark_visible(RID p_light_instance) override;

	RID fog_volume_instance_create(RID p_fog_volume) override;
	void fog_volume_instance_set_transform(RID p_fog_volume_instance, const Transform3D &p_transform) override;
	void fog_volume_instance_set_active(RID p_fog_volume_instance, bool p_active) override;
	RID fog_volume_instance_get_volume(RID p_fog_volume_instance) const override;
	Vector3 fog_volume_instance_get_position(RID p_fog_volume_instance) const override;

	RID reflection_atlas_create() override;
	int reflection_atlas_get_size(RID p_ref_atlas) const override;
	void reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) override;

	RID reflection_probe_instance_create(RID p_probe) override;
	void reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) override;
	void reflection_probe_release_atlas_index(RID p_instance) override;
	bool reflection_probe_instance_needs_redraw(RID p_instance) override;
	bool reflection_probe_instance_has_reflection(RID p_instance) override;
	bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) override;
	bool reflection_probe_instance_postprocess_step(RID p_instance) override;

	RID decal_instance_create(RID p_decal) override;
	void decal_instance_set_transform(RID p_decal, const Transform3D &p_transform) override;

	RID lightmap_instance_create(RID p_lightmap) override;
	void lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) override;

	RID voxel_gi_instance_create(RID p_voxel_gi) override;
	void voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform) override;
	bool voxel_gi_needs_update(RID p_probe) const override;
	void voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RendererSceneRender::GeometryInstance *> &p_dynamic_objects) override;

	void voxel_gi_set_quality(RS::VoxelGIQuality) override;

	void render_scene(RID p_render_buffers, const CameraData *p_camera_data, const PagedArray<GeometryInstance *> &p_instances, const PagedArray<RID> &p_lights, const PagedArray<RID> &p_reflection_probes, const PagedArray<RID> &p_voxel_gi_instances, const PagedArray<RID> &p_decals, const PagedArray<RID> &p_lightmaps, const PagedArray<RID> &p_fog_volumes, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_occluder_debug_tex, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_mesh_lod_threshold, const RenderShadowData *p_render_shadows, int p_render_shadow_count, const RenderSDFGIData *p_render_sdfgi_regions, int p_render_sdfgi_region_count, const RenderSDFGIUpdateData *p_sdfgi_update_data = nullptr, RendererScene::RenderInfo *r_render_info = nullptr) override;
	void render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_orthogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) override;
	void render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<GeometryInstance *> &p_instances) override;

	void set_scene_pass(uint64_t p_pass) override {
		scene_pass = p_pass;
	}

	_FORCE_INLINE_ uint64_t get_scene_pass() {
		return scene_pass;
	}

	void set_time(double p_time, double p_step) override;
	void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) override;
	_FORCE_INLINE_ RS::ViewportDebugDraw get_debug_draw_mode() const {
		return debug_draw;
	}

	RID render_buffers_create() override;
	void render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_internal_width, int p_internal_height, int p_width, int p_height, float p_fsr_sharpness, float p_fsr_mipmap_bias, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding, uint32_t p_view_count) override;
	void gi_set_use_half_resolution(bool p_enable) override;

	void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_curve) override;
	bool screen_space_roughness_limiter_is_active() const override;

	void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) override;
	void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) override;

	TypedArray<Image> bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) override;

	bool free(RID p_rid) override;
	void update() override;
	void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) override;

	void decals_set_filter(RS::DecalFilter p_filter) override;
	void light_projectors_set_filter(RS::LightProjectorFilter p_filter) override;

	static RasterizerSceneGLES3 *get_singleton();
	RasterizerSceneGLES3(RasterizerStorageGLES3 *p_storage);
	~RasterizerSceneGLES3();
};

#endif // GLES3_ENABLED

#endif // RASTERIZER_SCENE_OPENGL_H
