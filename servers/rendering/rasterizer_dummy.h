/*************************************************************************/
/*  rasterizer_dummy.h                                                   */
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

#ifndef RASTERIZER_DUMMY_H
#define RASTERIZER_DUMMY_H

#include "core/math/camera_matrix.h"
#include "core/templates/rid_owner.h"
#include "core/templates/self_list.h"
#include "scene/resources/mesh.h"
#include "servers/rendering/renderer_compositor.h"
#include "servers/rendering/renderer_scene_render.h"
#include "servers/rendering_server.h"

class RasterizerSceneDummy : public RendererSceneRender {
public:
	GeometryInstance *geometry_instance_create(RID p_base) override { return nullptr; }
	void geometry_instance_set_skeleton(GeometryInstance *p_geometry_instance, RID p_skeleton) override {}
	void geometry_instance_set_material_override(GeometryInstance *p_geometry_instance, RID p_override) override {}
	void geometry_instance_set_surface_materials(GeometryInstance *p_geometry_instance, const Vector<RID> &p_material) override {}
	void geometry_instance_set_mesh_instance(GeometryInstance *p_geometry_instance, RID p_mesh_instance) override {}
	void geometry_instance_set_transform(GeometryInstance *p_geometry_instance, const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabbb) override {}
	void geometry_instance_set_layer_mask(GeometryInstance *p_geometry_instance, uint32_t p_layer_mask) override {}
	void geometry_instance_set_lod_bias(GeometryInstance *p_geometry_instance, float p_lod_bias) override {}
	void geometry_instance_set_use_baked_light(GeometryInstance *p_geometry_instance, bool p_enable) override {}
	void geometry_instance_set_use_dynamic_gi(GeometryInstance *p_geometry_instance, bool p_enable) override {}
	void geometry_instance_set_use_lightmap(GeometryInstance *p_geometry_instance, RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) override {}
	void geometry_instance_set_lightmap_capture(GeometryInstance *p_geometry_instance, const Color *p_sh9) override {}
	void geometry_instance_set_instance_shader_parameters_offset(GeometryInstance *p_geometry_instance, int32_t p_offset) override {}
	void geometry_instance_set_cast_double_sided_shadows(GeometryInstance *p_geometry_instance, bool p_enable) override {}
	void geometry_instance_set_fade_range(GeometryInstance *p_geometry_instance, bool p_enable_near, float p_near_begin, float p_near_end, bool p_enable_far, float p_far_begin, float p_far_end) override {}
	void geometry_instance_set_parent_fade_alpha(GeometryInstance *p_geometry_instance, float p_alpha) override {}
	void geometry_instance_set_transparency(GeometryInstance *p_geometry_instance, float p_transparency) override {}

	uint32_t geometry_instance_get_pair_mask() override { return 0; }
	void geometry_instance_pair_light_instances(GeometryInstance *p_geometry_instance, const RID *p_light_instances, uint32_t p_light_instance_count) override {}
	void geometry_instance_pair_reflection_probe_instances(GeometryInstance *p_geometry_instance, const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) override {}
	void geometry_instance_pair_decal_instances(GeometryInstance *p_geometry_instance, const RID *p_decal_instances, uint32_t p_decal_instance_count) override {}
	void geometry_instance_pair_voxel_gi_instances(GeometryInstance *p_geometry_instance, const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) override {}
	void geometry_instance_set_softshadow_projector_pairing(GeometryInstance *p_geometry_instance, bool p_softshadow, bool p_projector) override {}

	void geometry_instance_free(GeometryInstance *p_geometry_instance) override {}

	/* SHADOW ATLAS API */

	RID shadow_atlas_create() override { return RID(); }
	void shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits = false) override {}
	void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) override {}
	bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) override { return false; }

	void directional_shadow_atlas_set_size(int p_size, bool p_16_bits = false) override {}
	int get_directional_light_shadow_size(RID p_light_intance) override { return 0; }
	void set_directional_shadow_count(int p_count) override {}

	/* SDFGI UPDATE */

	void sdfgi_update(RID p_render_buffers, RID p_environment, const Vector3 &p_world_position) override {}
	int sdfgi_get_pending_region_count(RID p_render_buffers) const override { return 0; }
	AABB sdfgi_get_pending_region_bounds(RID p_render_buffers, int p_region) const override { return AABB(); }
	uint32_t sdfgi_get_pending_region_cascade(RID p_render_buffers, int p_region) const override { return 0; }

	/* SKY API */

	RID sky_allocate() override { return RID(); }
	void sky_initialize(RID p_rid) override {}
	void sky_set_radiance_size(RID p_sky, int p_radiance_size) override {}
	void sky_set_mode(RID p_sky, RS::SkyMode p_samples) override {}
	void sky_set_material(RID p_sky, RID p_material) override {}
	Ref<Image> sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) override { return Ref<Image>(); }

	/* ENVIRONMENT API */

	RID environment_allocate() override { return RID(); }
	void environment_initialize(RID p_rid) override {}
	void environment_set_background(RID p_env, RS::EnvironmentBG p_bg) override {}
	void environment_set_sky(RID p_env, RID p_sky) override {}
	void environment_set_sky_custom_fov(RID p_env, float p_scale) override {}
	void environment_set_sky_orientation(RID p_env, const Basis &p_orientation) override {}
	void environment_set_bg_color(RID p_env, const Color &p_color) override {}
	void environment_set_bg_energy(RID p_env, float p_energy) override {}
	void environment_set_canvas_max_layer(RID p_env, int p_max_layer) override {}
	void environment_set_ambient_light(RID p_env, const Color &p_color, RS::EnvironmentAmbientSource p_ambient = RS::ENV_AMBIENT_SOURCE_BG, float p_energy = 1.0, float p_sky_contribution = 0.0, RS::EnvironmentReflectionSource p_reflection_source = RS::ENV_REFLECTION_SOURCE_BG) override {}

	void environment_set_glow(RID p_env, bool p_enable, Vector<float> p_levels, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap) override {}
	void environment_glow_set_use_bicubic_upscale(bool p_enable) override {}
	void environment_glow_set_use_high_quality(bool p_enable) override {}

	void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) override {}
	void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) override {}
	void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_power, float p_detail, float p_horizon, float p_sharpness, float p_light_affect, float p_ao_channel_affect) override {}
	void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override {}

	void environment_set_sdfgi(RID p_env, bool p_enable, RS::EnvironmentSDFGICascades p_cascades, float p_min_cell_size, RS::EnvironmentSDFGIYScale p_y_scale, bool p_use_occlusion, float p_bounce_feedback, bool p_read_sky, float p_energy, float p_normal_bias, float p_probe_bias) override {}

	void environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) override {}
	void environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) override {}
	void environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) override {}

	void environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) override {}

	void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, bool p_use_1d_color_correction, RID p_color_correction) override {}

	void environment_set_fog(RID p_env, bool p_enable, const Color &p_light_color, float p_light_energy, float p_sun_scatter, float p_density, float p_height, float p_height_density, float p_aerial_perspective) override {}
	void environment_set_volumetric_fog(RID p_env, bool p_enable, float p_density, const Color &p_light, float p_light_energy, float p_length, float p_detail_spread, float p_gi_inject, bool p_temporal_reprojection, float p_temporal_reprojection_amount) override {}
	void environment_set_volumetric_fog_volume_size(int p_size, int p_depth) override {}
	void environment_set_volumetric_fog_filter_active(bool p_enable) override {}

	Ref<Image> environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) override { return Ref<Image>(); }

	bool is_environment(RID p_env) const override { return false; }
	RS::EnvironmentBG environment_get_background(RID p_env) const override { return RS::ENV_BG_KEEP; }
	int environment_get_canvas_max_layer(RID p_env) const override { return 0; }

	RID camera_effects_allocate() override { return RID(); }
	void camera_effects_initialize(RID p_rid) override {}
	void camera_effects_set_dof_blur_quality(RS::DOFBlurQuality p_quality, bool p_use_jitter) override {}
	void camera_effects_set_dof_blur_bokeh_shape(RS::DOFBokehShape p_shape) override {}

	void camera_effects_set_dof_blur(RID p_camera_effects, bool p_far_enable, float p_far_distance, float p_far_transition, bool p_near_enable, float p_near_distance, float p_near_transition, float p_amount) override {}
	void camera_effects_set_custom_exposure(RID p_camera_effects, bool p_enable, float p_exposure) override {}

	void shadows_quality_set(RS::ShadowQuality p_quality) override {}
	void directional_shadow_quality_set(RS::ShadowQuality p_quality) override {}

	RID light_instance_create(RID p_light) override { return RID(); }
	void light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) override {}
	void light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) override {}
	void light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale = 1.0, float p_range_begin = 0, const Vector2 &p_uv_scale = Vector2()) override {}
	void light_instance_mark_visible(RID p_light_instance) override {}

	RID reflection_atlas_create() override { return RID(); }
	int reflection_atlas_get_size(RID p_ref_atlas) const override { return 0; }
	void reflection_atlas_set_size(RID p_ref_atlas, int p_reflection_size, int p_reflection_count) override {}

	RID reflection_probe_instance_create(RID p_probe) override { return RID(); }
	void reflection_probe_instance_set_transform(RID p_instance, const Transform3D &p_transform) override {}
	void reflection_probe_release_atlas_index(RID p_instance) override {}
	bool reflection_probe_instance_needs_redraw(RID p_instance) override { return false; }
	bool reflection_probe_instance_has_reflection(RID p_instance) override { return false; }
	bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) override { return false; }
	bool reflection_probe_instance_postprocess_step(RID p_instance) override { return true; }

	RID decal_instance_create(RID p_decal) override { return RID(); }
	void decal_instance_set_transform(RID p_decal, const Transform3D &p_transform) override {}

	RID lightmap_instance_create(RID p_lightmap) override { return RID(); }
	void lightmap_instance_set_transform(RID p_lightmap, const Transform3D &p_transform) override {}

	RID voxel_gi_instance_create(RID p_voxel_gi) override { return RID(); }
	void voxel_gi_instance_set_transform_to_data(RID p_probe, const Transform3D &p_xform) override {}
	bool voxel_gi_needs_update(RID p_probe) const override { return false; }
	void voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RendererSceneRender::GeometryInstance *> &p_dynamic_objects) override {}

	void voxel_gi_set_quality(RS::VoxelGIQuality) override {}

	void render_scene(RID p_render_buffers, const CameraData *p_camera_data, const PagedArray<GeometryInstance *> &p_instances, const PagedArray<RID> &p_lights, const PagedArray<RID> &p_reflection_probes, const PagedArray<RID> &p_voxel_gi_instances, const PagedArray<RID> &p_decals, const PagedArray<RID> &p_lightmaps, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_occluder_debug_tex, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_lod_threshold, const RenderShadowData *p_render_shadows, int p_render_shadow_count, const RenderSDFGIData *p_render_sdfgi_regions, int p_render_sdfgi_region_count, const RenderSDFGIUpdateData *p_sdfgi_update_data = nullptr, RendererScene::RenderInfo *r_info = nullptr) override {}
	void render_material(const Transform3D &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, const PagedArray<GeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) override {}
	void render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<GeometryInstance *> &p_instances) override {}

	void set_scene_pass(uint64_t p_pass) override {}
	void set_time(double p_time, double p_step) override {}
	void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) override {}

	RID render_buffers_create() override { return RID(); }
	void render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa, bool p_use_debanding, uint32_t p_view_count) override {}
	void gi_set_use_half_resolution(bool p_enable) override {}

	void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_curve) override {}
	bool screen_space_roughness_limiter_is_active() const override { return false; }

	void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) override {}
	void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) override {}

	TypedArray<Image> bake_render_uv2(RID p_base, const Vector<RID> &p_material_overrides, const Size2i &p_image_size) override { return TypedArray<Image>(); }

	bool free(RID p_rid) override { return false; }
	void update() override {}
	void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) override {}

	virtual void decals_set_filter(RS::DecalFilter p_filter) override {}
	virtual void light_projectors_set_filter(RS::LightProjectorFilter p_filter) override {}

	RasterizerSceneDummy() {}
	~RasterizerSceneDummy() {}
};

class RasterizerStorageDummy : public RendererStorage {
public:
	bool can_create_resources_async() const override { return false; }

	/* TEXTURE API */
	struct DummyTexture {
		Ref<Image> image;
	};
	mutable RID_PtrOwner<DummyTexture> texture_owner;

	RID texture_allocate() override {
		DummyTexture *texture = memnew(DummyTexture);
		ERR_FAIL_COND_V(!texture, RID());
		return texture_owner.make_rid(texture);
	}
	void texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) override {
		DummyTexture *t = texture_owner.get_or_null(p_texture);
		ERR_FAIL_COND(!t);
		t->image = p_image->duplicate();
	}

	void texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) override {}
	void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) override {}
	void texture_3d_initialize(RID p_texture, Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) override {}
	void texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) override {}
	void texture_proxy_initialize(RID p_texture, RID p_base) override {}
	void texture_proxy_update(RID p_proxy, RID p_base) override {}

	void texture_2d_placeholder_initialize(RID p_texture) override {}
	void texture_2d_layered_placeholder_initialize(RID p_texture, RenderingServer::TextureLayeredType p_layered_type) override {}
	void texture_3d_placeholder_initialize(RID p_texture) override {}

	Ref<Image> texture_2d_get(RID p_texture) const override {
		DummyTexture *t = texture_owner.get_or_null(p_texture);
		ERR_FAIL_COND_V(!t, Ref<Image>());
		return t->image;
	}

	Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const override { return Ref<Image>(); }
	Vector<Ref<Image>> texture_3d_get(RID p_texture) const override { return Vector<Ref<Image>>(); }

	void texture_replace(RID p_texture, RID p_by_texture) override {}
	void texture_set_size_override(RID p_texture, int p_width, int p_height) override {}

	void texture_set_path(RID p_texture, const String &p_path) override {}
	String texture_get_path(RID p_texture) const override { return String(); }

	void texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) override {}
	void texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) override {}
	void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) override {}

	void texture_debug_usage(List<RS::TextureInfo> *r_info) override {}
	void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) override {}
	Size2 texture_size_with_proxy(RID p_proxy) override { return Size2(); }

	void texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) override {}
	void texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) override {}

	/* CANVAS TEXTURE API */

	RID canvas_texture_allocate() override { return RID(); }
	void canvas_texture_initialize(RID p_rid) override {}
	void canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) override {}
	void canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_base_color, float p_shininess) override {}

	void canvas_texture_set_texture_filter(RID p_item, RS::CanvasItemTextureFilter p_filter) override {}
	void canvas_texture_set_texture_repeat(RID p_item, RS::CanvasItemTextureRepeat p_repeat) override {}

	/* SHADER API */

	RID shader_allocate() override { return RID(); }
	void shader_initialize(RID p_rid) override {}
	void shader_set_code(RID p_shader, const String &p_code) override {}
	String shader_get_code(RID p_shader) const override { return ""; }
	void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const override {}

	void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) override {}
	RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const override { return RID(); }
	Variant shader_get_param_default(RID p_material, const StringName &p_param) const override { return Variant(); }

	RS::ShaderNativeSourceCode shader_get_native_source_code(RID p_shader) const override { return RS::ShaderNativeSourceCode(); };

	/* COMMON MATERIAL API */

	RID material_allocate() override { return RID(); }
	void material_initialize(RID p_rid) override {}
	void material_set_render_priority(RID p_material, int priority) override {}
	void material_set_shader(RID p_shader_material, RID p_shader) override {}

	void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) override {}
	Variant material_get_param(RID p_material, const StringName &p_param) const override { return Variant(); }

	void material_set_next_pass(RID p_material, RID p_next_material) override {}

	bool material_is_animated(RID p_material) override { return false; }
	bool material_casts_shadows(RID p_material) override { return false; }
	void material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) override {}
	void material_update_dependency(RID p_material, DependencyTracker *p_instance) override {}

	/* MESH API */

	RID mesh_allocate() override { return RID(); }
	void mesh_initialize(RID p_rid) override {}
	void mesh_set_blend_shape_count(RID p_mesh, int p_blend_shape_count) override {}
	bool mesh_needs_instance(RID p_mesh, bool p_has_skeleton) override { return false; }
	RID mesh_instance_create(RID p_base) override { return RID(); }
	void mesh_instance_set_skeleton(RID p_mesh_instance, RID p_skeleton) override {}
	void mesh_instance_set_blend_shape_weight(RID p_mesh_instance, int p_shape, float p_weight) override {}
	void mesh_instance_check_for_update(RID p_mesh_instance) override {}
	void update_mesh_instances() override {}
	void reflection_probe_set_lod_threshold(RID p_probe, float p_ratio) override {}
	float reflection_probe_get_lod_threshold(RID p_probe) const override { return 0.0; }

	void mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) override {}

	int mesh_get_blend_shape_count(RID p_mesh) const override { return 0; }

	void mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) override {}
	RS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const override { return RS::BLEND_SHAPE_MODE_NORMALIZED; }

	void mesh_surface_update_vertex_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) override {}
	void mesh_surface_update_attribute_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) override {}
	void mesh_surface_update_skin_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) override {}

	void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) override {}
	RID mesh_surface_get_material(RID p_mesh, int p_surface) const override { return RID(); }

	RS::SurfaceData mesh_get_surface(RID p_mesh, int p_surface) const override { return RS::SurfaceData(); }
	int mesh_get_surface_count(RID p_mesh) const override { return 0; }

	void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) override {}
	AABB mesh_get_custom_aabb(RID p_mesh) const override { return AABB(); }

	AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID()) override { return AABB(); }
	void mesh_set_shadow_mesh(RID p_mesh, RID p_shadow_mesh) override {}
	void mesh_clear(RID p_mesh) override {}

	/* MULTIMESH API */

	RID multimesh_allocate() override { return RID(); }
	void multimesh_initialize(RID p_rid) override {}
	void multimesh_allocate_data(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors = false, bool p_use_custom_data = false) override {}
	int multimesh_get_instance_count(RID p_multimesh) const override { return 0; }

	void multimesh_set_mesh(RID p_multimesh, RID p_mesh) override {}
	void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform3D &p_transform) override {}
	void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) override {}
	void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) override {}
	void multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) override {}

	RID multimesh_get_mesh(RID p_multimesh) const override { return RID(); }
	AABB multimesh_get_aabb(RID p_multimesh) const override { return AABB(); }

	Transform3D multimesh_instance_get_transform(RID p_multimesh, int p_index) const override { return Transform3D(); }
	Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const override { return Transform2D(); }
	Color multimesh_instance_get_color(RID p_multimesh, int p_index) const override { return Color(); }
	Color multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const override { return Color(); }
	void multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) override {}
	Vector<float> multimesh_get_buffer(RID p_multimesh) const override { return Vector<float>(); }

	void multimesh_set_visible_instances(RID p_multimesh, int p_visible) override {}
	int multimesh_get_visible_instances(RID p_multimesh) const override { return 0; }

	/* SKELETON API */

	RID skeleton_allocate() override { return RID(); }
	void skeleton_initialize(RID p_rid) override {}
	void skeleton_allocate_data(RID p_skeleton, int p_bones, bool p_2d_skeleton = false) override {}
	void skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) override {}
	int skeleton_get_bone_count(RID p_skeleton) const override { return 0; }
	void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform3D &p_transform) override {}
	Transform3D skeleton_bone_get_transform(RID p_skeleton, int p_bone) const override { return Transform3D(); }
	void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) override {}
	Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const override { return Transform2D(); }

	/* Light API */

	RID directional_light_allocate() override { return RID(); }
	void directional_light_initialize(RID p_rid) override {}
	RID omni_light_allocate() override { return RID(); }
	void omni_light_initialize(RID p_rid) override {}
	RID spot_light_allocate() override { return RID(); }
	void spot_light_initialize(RID p_rid) override {}
	RID reflection_probe_allocate() override { return RID(); }
	void reflection_probe_initialize(RID p_rid) override {}

	void light_set_color(RID p_light, const Color &p_color) override {}
	void light_set_param(RID p_light, RS::LightParam p_param, float p_value) override {}
	void light_set_shadow(RID p_light, bool p_enabled) override {}
	void light_set_shadow_color(RID p_light, const Color &p_color) override {}
	void light_set_projector(RID p_light, RID p_texture) override {}
	void light_set_negative(RID p_light, bool p_enable) override {}
	void light_set_cull_mask(RID p_light, uint32_t p_mask) override {}
	void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) override {}
	void light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) override {}
	void light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) override {}

	void light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) override {}

	void light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) override {}
	void light_directional_set_blend_splits(RID p_light, bool p_enable) override {}
	bool light_directional_get_blend_splits(RID p_light) const override { return false; }
	void light_directional_set_sky_only(RID p_light, bool p_sky_only) override {}
	bool light_directional_is_sky_only(RID p_light) const override { return false; }

	RS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) override { return RS::LIGHT_DIRECTIONAL_SHADOW_ORTHOGONAL; }
	RS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) override { return RS::LIGHT_OMNI_SHADOW_DUAL_PARABOLOID; }

	bool light_has_shadow(RID p_light) const override { return false; }
	bool light_has_projector(RID p_light) const override { return false; }

	RS::LightType light_get_type(RID p_light) const override { return RS::LIGHT_OMNI; }
	AABB light_get_aabb(RID p_light) const override { return AABB(); }
	float light_get_param(RID p_light, RS::LightParam p_param) override { return 0.0; }
	Color light_get_color(RID p_light) override { return Color(); }
	RS::LightBakeMode light_get_bake_mode(RID p_light) override { return RS::LIGHT_BAKE_DISABLED; }
	uint32_t light_get_max_sdfgi_cascade(RID p_light) override { return 0; }
	uint64_t light_get_version(RID p_light) const override { return 0; }

	/* PROBE API */

	void reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) override {}
	void reflection_probe_set_intensity(RID p_probe, float p_intensity) override {}
	void reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) override {}
	void reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) override {}
	void reflection_probe_set_ambient_energy(RID p_probe, float p_energy) override {}
	void reflection_probe_set_max_distance(RID p_probe, float p_distance) override {}
	void reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) override {}
	void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) override {}
	void reflection_probe_set_as_interior(RID p_probe, bool p_enable) override {}
	void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) override {}
	void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) override {}
	void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) override {}
	void reflection_probe_set_resolution(RID p_probe, int p_resolution) override {}

	AABB reflection_probe_get_aabb(RID p_probe) const override { return AABB(); }
	RS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const override { return RenderingServer::REFLECTION_PROBE_UPDATE_ONCE; }
	uint32_t reflection_probe_get_cull_mask(RID p_probe) const override { return 0; }
	Vector3 reflection_probe_get_extents(RID p_probe) const override { return Vector3(); }
	Vector3 reflection_probe_get_origin_offset(RID p_probe) const override { return Vector3(); }
	float reflection_probe_get_origin_max_distance(RID p_probe) const override { return 0.0; }
	bool reflection_probe_renders_shadows(RID p_probe) const override { return false; }

	void base_update_dependency(RID p_base, DependencyTracker *p_instance) override {}
	void skeleton_update_dependency(RID p_base, DependencyTracker *p_instance) override {}

	/* DECAL API */

	RID decal_allocate() override { return RID(); }
	void decal_initialize(RID p_rid) override {}
	void decal_set_extents(RID p_decal, const Vector3 &p_extents) override {}
	void decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) override {}
	void decal_set_emission_energy(RID p_decal, float p_energy) override {}
	void decal_set_albedo_mix(RID p_decal, float p_mix) override {}
	void decal_set_modulate(RID p_decal, const Color &p_modulate) override {}
	void decal_set_cull_mask(RID p_decal, uint32_t p_layers) override {}
	void decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) override {}
	void decal_set_fade(RID p_decal, float p_above, float p_below) override {}
	void decal_set_normal_fade(RID p_decal, float p_fade) override {}

	AABB decal_get_aabb(RID p_decal) const override { return AABB(); }

	/* VOXEL GI API */

	RID voxel_gi_allocate() override { return RID(); }
	void voxel_gi_initialize(RID p_rid) override {}
	void voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) override {}

	AABB voxel_gi_get_bounds(RID p_voxel_gi) const override { return AABB(); }
	Vector3i voxel_gi_get_octree_size(RID p_voxel_gi) const override { return Vector3i(); }
	Vector<uint8_t> voxel_gi_get_octree_cells(RID p_voxel_gi) const override { return Vector<uint8_t>(); }
	Vector<uint8_t> voxel_gi_get_data_cells(RID p_voxel_gi) const override { return Vector<uint8_t>(); }
	Vector<uint8_t> voxel_gi_get_distance_field(RID p_voxel_gi) const override { return Vector<uint8_t>(); }

	Vector<int> voxel_gi_get_level_counts(RID p_voxel_gi) const override { return Vector<int>(); }
	Transform3D voxel_gi_get_to_cell_xform(RID p_voxel_gi) const override { return Transform3D(); }

	void voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_dynamic_range(RID p_voxel_gi) const override { return 0; }

	void voxel_gi_set_propagation(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_propagation(RID p_voxel_gi) const override { return 0; }

	void voxel_gi_set_energy(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_energy(RID p_voxel_gi) const override { return 0.0; }

	void voxel_gi_set_bias(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_bias(RID p_voxel_gi) const override { return 0.0; }

	void voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) override {}
	float voxel_gi_get_normal_bias(RID p_voxel_gi) const override { return 0.0; }

	void voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) override {}
	bool voxel_gi_is_interior(RID p_voxel_gi) const override { return false; }

	void voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) override {}
	bool voxel_gi_is_using_two_bounces(RID p_voxel_gi) const override { return false; }

	void voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) override {}
	float voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const override { return 0; }

	uint32_t voxel_gi_get_version(RID p_voxel_gi) override { return 0; }

	/* LIGHTMAP CAPTURE */
	RID lightmap_allocate() override { return RID(); }
	void lightmap_initialize(RID p_rid) override {}
	void lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) override {}
	void lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) override {}
	void lightmap_set_probe_interior(RID p_lightmap, bool p_interior) override {}
	void lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) override {}
	PackedVector3Array lightmap_get_probe_capture_points(RID p_lightmap) const override { return PackedVector3Array(); }
	PackedColorArray lightmap_get_probe_capture_sh(RID p_lightmap) const override { return PackedColorArray(); }
	PackedInt32Array lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const override { return PackedInt32Array(); }
	PackedInt32Array lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const override { return PackedInt32Array(); }
	AABB lightmap_get_aabb(RID p_lightmap) const override { return AABB(); }
	void lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) override {}
	bool lightmap_is_interior(RID p_lightmap) const override { return false; }
	void lightmap_set_probe_capture_update_speed(float p_speed) override {}
	float lightmap_get_probe_capture_update_speed() const override { return 0; }

	/* OCCLUDER */

	void occluder_set_mesh(RID p_occluder, const PackedVector3Array &p_vertices, const PackedInt32Array &p_indices) {}

	/* PARTICLES */

	RID particles_allocate() override { return RID(); }
	void particles_initialize(RID p_rid) override {}
	void particles_set_mode(RID p_particles, RS::ParticlesMode p_mode) override {}
	void particles_emit(RID p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) override {}
	void particles_set_emitting(RID p_particles, bool p_emitting) override {}
	void particles_set_amount(RID p_particles, int p_amount) override {}
	void particles_set_lifetime(RID p_particles, double p_lifetime) override {}
	void particles_set_one_shot(RID p_particles, bool p_one_shot) override {}
	void particles_set_pre_process_time(RID p_particles, double p_time) override {}
	void particles_set_explosiveness_ratio(RID p_particles, real_t p_ratio) override {}
	void particles_set_randomness_ratio(RID p_particles, real_t p_ratio) override {}
	void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) override {}
	void particles_set_speed_scale(RID p_particles, double p_scale) override {}
	void particles_set_use_local_coordinates(RID p_particles, bool p_enable) override {}
	void particles_set_process_material(RID p_particles, RID p_material) override {}
	void particles_set_fixed_fps(RID p_particles, int p_fps) override {}
	void particles_set_interpolate(RID p_particles, bool p_enable) override {}
	void particles_set_fractional_delta(RID p_particles, bool p_enable) override {}
	void particles_set_subemitter(RID p_particles, RID p_subemitter_particles) override {}
	void particles_set_view_axis(RID p_particles, const Vector3 &p_axis, const Vector3 &p_up_axis) override {}
	void particles_set_collision_base_size(RID p_particles, real_t p_size) override {}

	void particles_set_transform_align(RID p_particles, RS::ParticlesTransformAlign p_transform_align) override {}

	void particles_set_trails(RID p_particles, bool p_enable, double p_length) override {}
	void particles_set_trail_bind_poses(RID p_particles, const Vector<Transform3D> &p_bind_poses) override {}

	void particles_restart(RID p_particles) override {}

	void particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) override {}

	void particles_set_draw_passes(RID p_particles, int p_count) override {}
	void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) override {}

	void particles_request_process(RID p_particles) override {}
	AABB particles_get_current_aabb(RID p_particles) override { return AABB(); }
	AABB particles_get_aabb(RID p_particles) const override { return AABB(); }

	void particles_set_emission_transform(RID p_particles, const Transform3D &p_transform) override {}

	bool particles_get_emitting(RID p_particles) override { return false; }
	int particles_get_draw_passes(RID p_particles) const override { return 0; }
	RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const override { return RID(); }

	void particles_add_collision(RID p_particles, RID p_instance) override {}
	void particles_remove_collision(RID p_particles, RID p_instance) override {}

	void particles_set_canvas_sdf_collision(RID p_particles, bool p_enable, const Transform2D &p_xform, const Rect2 &p_to_screen, RID p_texture) override {}

	void update_particles() override {}

	/* PARTICLES COLLISION */

	RID particles_collision_allocate() override { return RID(); }
	void particles_collision_initialize(RID p_rid) override {}
	void particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) override {}
	void particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) override {}
	void particles_collision_set_sphere_radius(RID p_particles_collision, real_t p_radius) override {}
	void particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) override {}
	void particles_collision_set_attractor_strength(RID p_particles_collision, real_t p_strength) override {}
	void particles_collision_set_attractor_directionality(RID p_particles_collision, real_t p_directionality) override {}
	void particles_collision_set_attractor_attenuation(RID p_particles_collision, real_t p_curve) override {}
	void particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) override {}
	void particles_collision_height_field_update(RID p_particles_collision) override {}
	void particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) override {}
	AABB particles_collision_get_aabb(RID p_particles_collision) const override { return AABB(); }
	bool particles_collision_is_heightfield(RID p_particles_collision) const override { return false; }
	RID particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const override { return RID(); }

	RID particles_collision_instance_create(RID p_collision) override { return RID(); }
	void particles_collision_instance_set_transform(RID p_collision_instance, const Transform3D &p_transform) override {}
	void particles_collision_instance_set_active(RID p_collision_instance, bool p_active) override {}

	/* VISIBILITY NOTIFIER */
	virtual RID visibility_notifier_allocate() override { return RID(); }
	virtual void visibility_notifier_initialize(RID p_notifier) override {}
	virtual void visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) override {}
	virtual void visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) override {}

	virtual AABB visibility_notifier_get_aabb(RID p_notifier) const override { return AABB(); }
	virtual void visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) override {}

	/* GLOBAL VARIABLES */

	void global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value) override {}
	void global_variable_remove(const StringName &p_name) override {}
	Vector<StringName> global_variable_get_list() const override { return Vector<StringName>(); }

	void global_variable_set(const StringName &p_name, const Variant &p_value) override {}
	void global_variable_set_override(const StringName &p_name, const Variant &p_value) override {}
	Variant global_variable_get(const StringName &p_name) const override { return Variant(); }
	RS::GlobalVariableType global_variable_get_type(const StringName &p_name) const override { return RS::GLOBAL_VAR_TYPE_MAX; }

	void global_variables_load_settings(bool p_load_textures = true) override {}
	void global_variables_clear() override {}

	int32_t global_variables_instance_allocate(RID p_instance) override { return 0; }
	void global_variables_instance_free(RID p_instance) override {}
	void global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value) override {}

	bool particles_is_inactive(RID p_particles) const override { return false; }

	/* RENDER TARGET */

	RID render_target_create() override { return RID(); }
	void render_target_set_position(RID p_render_target, int p_x, int p_y) override {}
	void render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) override {}
	RID render_target_get_texture(RID p_render_target) override { return RID(); }
	void render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) override {}
	void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) override {}
	bool render_target_was_used(RID p_render_target) override { return false; }
	void render_target_set_as_unused(RID p_render_target) override {}

	void render_target_request_clear(RID p_render_target, const Color &p_clear_color) override {}
	bool render_target_is_clear_requested(RID p_render_target) override { return false; }
	Color render_target_get_clear_request_color(RID p_render_target) override { return Color(); }
	void render_target_disable_clear_request(RID p_render_target) override {}
	void render_target_do_clear_request(RID p_render_target) override {}

	void render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) override {}
	Rect2i render_target_get_sdf_rect(RID p_render_target) const override { return Rect2i(); }
	void render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) override {}

	RS::InstanceType get_base_type(RID p_rid) const override { return RS::INSTANCE_NONE; }
	bool free(RID p_rid) override {
		if (texture_owner.owns(p_rid)) {
			// delete the texture
			DummyTexture *texture = texture_owner.get_or_null(p_rid);
			texture_owner.free(p_rid);
			memdelete(texture);
			return true;
		}
		return false;
	}

	virtual void update_memory_info() override {}
	virtual uint64_t get_rendering_info(RS::RenderingInfo p_info) override { return 0; }

	bool has_os_feature(const String &p_feature) const override { return false; }

	void update_dirty_resources() override {}

	void set_debug_generate_wireframes(bool p_generate) override {}

	String get_video_adapter_name() const override { return String(); }
	String get_video_adapter_vendor() const override { return String(); }

	static RendererStorage *base_singleton;

	void capture_timestamps_begin() override {}
	void capture_timestamp(const String &p_name) override {}
	uint32_t get_captured_timestamps_count() const override { return 0; }
	uint64_t get_captured_timestamps_frame() const override { return 0; }
	uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const override { return 0; }
	uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const override { return 0; }
	String get_captured_timestamp_name(uint32_t p_index) const override { return String(); }

	RasterizerStorageDummy() {}
	~RasterizerStorageDummy() {}
};

class RasterizerCanvasDummy : public RendererCanvasRender {
public:
	PolygonID request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), const Vector<int> &p_bones = Vector<int>(), const Vector<float> &p_weights = Vector<float>()) override { return 0; }
	void free_polygon(PolygonID p_polygon) override {}

	void canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, Light *p_directional_list, const Transform2D &p_canvas_transform, RS::CanvasItemTextureFilter p_default_filter, RS::CanvasItemTextureRepeat p_default_repeat, bool p_snap_2d_vertices_to_pixel, bool &r_sdf_used) override {}
	void canvas_debug_viewport_shadows(Light *p_lights_with_shadow) override {}

	RID light_create() override { return RID(); }
	void light_set_texture(RID p_rid, RID p_texture) override {}
	void light_set_use_shadow(RID p_rid, bool p_enable) override {}
	void light_update_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) override {}
	void light_update_directional_shadow(RID p_rid, int p_shadow_index, const Transform2D &p_light_xform, int p_light_mask, float p_cull_distance, const Rect2 &p_clip_rect, LightOccluderInstance *p_occluders) override {}

	void render_sdf(RID p_render_target, LightOccluderInstance *p_occluders) override {}
	RID occluder_polygon_create() override { return RID(); }
	void occluder_polygon_set_shape(RID p_occluder, const Vector<Vector2> &p_points, bool p_closed) override {}
	void occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) override {}
	void set_shadow_texture_size(int p_size) override {}

	bool free(RID p_rid) override { return true; }
	void update() override {}

	RasterizerCanvasDummy() {}
	~RasterizerCanvasDummy() {}
};

class RasterizerDummy : public RendererCompositor {
private:
	uint64_t frame = 1;
	double delta = 0;

protected:
	RasterizerCanvasDummy canvas;
	RasterizerStorageDummy storage;
	RasterizerSceneDummy scene;

public:
	RendererStorage *get_storage() override { return &storage; }
	RendererCanvasRender *get_canvas() override { return &canvas; }
	RendererSceneRender *get_scene() override { return &scene; }

	void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter = true) override {}

	void initialize() override {}
	void begin_frame(double frame_step) override {
		frame++;
		delta = frame_step;
	}

	void prepare_for_blitting_render_targets() override {}
	void blit_render_targets_to_screen(int p_screen, const BlitToScreen *p_render_targets, int p_amount) override {}

	void end_frame(bool p_swap_buffers) override {
		if (p_swap_buffers) {
			DisplayServer::get_singleton()->swap_buffers();
		}
	}

	void finalize() override {}

	static RendererCompositor *_create_current() {
		return memnew(RasterizerDummy);
	}

	static void make_current() {
		_create_func = _create_current;
	}

	bool is_low_end() const override { return true; }
	uint64_t get_frame_number() const override { return frame; }
	double get_frame_delta_time() const override { return delta; }

	RasterizerDummy() {}
	~RasterizerDummy() {}
};

#endif // RASTERIZER_DUMMY_H
