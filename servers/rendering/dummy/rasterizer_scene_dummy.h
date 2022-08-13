/*************************************************************************/
/*  rasterizer_scene_dummy.h                                             */
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

#ifndef RASTERIZER_SCENE_DUMMY_H
#define RASTERIZER_SCENE_DUMMY_H

#include "core/templates/paged_allocator.h"
#include "servers/rendering/renderer_scene_render.h"
#include "storage/utilities.h"

class RasterizerSceneDummy : public RendererSceneRender {
public:
	class GeometryInstanceDummy : public RenderGeometryInstance {
	public:
		GeometryInstanceDummy() {}

		virtual void _mark_dirty() override {}

		virtual void set_skeleton(RID p_skeleton) override {}
		virtual void set_material_override(RID p_override) override {}
		virtual void set_material_overlay(RID p_overlay) override {}
		virtual void set_surface_materials(const Vector<RID> &p_materials) override {}
		virtual void set_mesh_instance(RID p_mesh_instance) override {}
		virtual void set_transform(const Transform3D &p_transform, const AABB &p_aabb, const AABB &p_transformed_aabb) override {}
		virtual void set_lod_bias(float p_lod_bias) override {}
		virtual void set_layer_mask(uint32_t p_layer_mask) override {}
		virtual void set_fade_range(bool p_enable_near, float p_near_begin, float p_near_end, bool p_enable_far, float p_far_begin, float p_far_end) override {}
		virtual void set_parent_fade_alpha(float p_alpha) override {}
		virtual void set_transparency(float p_transparency) override {}
		virtual void set_use_baked_light(bool p_enable) override {}
		virtual void set_use_dynamic_gi(bool p_enable) override {}
		virtual void set_use_lightmap(RID p_lightmap_instance, const Rect2 &p_lightmap_uv_scale, int p_lightmap_slice_index) override {}
		virtual void set_lightmap_scale(float p_lightmap_scale) override {}
		virtual void set_lightmap_capture(const Color *p_sh9) override {}
		virtual void set_instance_shader_uniforms_offset(int32_t p_offset) override {}
		virtual void set_cast_double_sided_shadows(bool p_enable) override {}

		virtual Transform3D get_transform() override { return Transform3D(); }
		virtual AABB get_aabb() override { return AABB(); }

		virtual void pair_light_instances(const RID *p_light_instances, uint32_t p_light_instance_count) override {}
		virtual void pair_reflection_probe_instances(const RID *p_reflection_probe_instances, uint32_t p_reflection_probe_instance_count) override {}
		virtual void pair_decal_instances(const RID *p_decal_instances, uint32_t p_decal_instance_count) override {}
		virtual void pair_voxel_gi_instances(const RID *p_voxel_gi_instances, uint32_t p_voxel_gi_instance_count) override {}

		virtual void set_softshadow_projector_pairing(bool p_softshadow, bool p_projector) override {}
	};

	PagedAllocator<GeometryInstanceDummy> geometry_instance_alloc;

public:
	RenderGeometryInstance *geometry_instance_create(RID p_base) override {
		RS::InstanceType type = RendererDummy::Utilities::get_singleton()->get_base_type(p_base);
		ERR_FAIL_COND_V(!((1 << type) & RS::INSTANCE_GEOMETRY_MASK), nullptr);

		GeometryInstanceDummy *ginstance = geometry_instance_alloc.alloc();

		return ginstance;
	}

	void geometry_instance_free(RenderGeometryInstance *p_geometry_instance) override {
		GeometryInstanceDummy *ginstance = static_cast<GeometryInstanceDummy *>(p_geometry_instance);
		ERR_FAIL_COND(!ginstance);

		geometry_instance_alloc.free(ginstance);
	}

	uint32_t geometry_instance_get_pair_mask() override { return 0; }

	/* SHADOW ATLAS API */

	RID shadow_atlas_create() override { return RID(); }
	void shadow_atlas_set_size(RID p_atlas, int p_size, bool p_16_bits = true) override {}
	void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) override {}
	bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) override { return false; }

	void directional_shadow_atlas_set_size(int p_size, bool p_16_bits = true) override {}
	int get_directional_light_shadow_size(RID p_light_intance) override { return 0; }
	void set_directional_shadow_count(int p_count) override {}

	/* SDFGI UPDATE */

	void sdfgi_update(const Ref<RenderSceneBuffers> &p_render_buffers, RID p_environment, const Vector3 &p_world_position) override {}
	int sdfgi_get_pending_region_count(const Ref<RenderSceneBuffers> &p_render_buffers) const override { return 0; }
	AABB sdfgi_get_pending_region_bounds(const Ref<RenderSceneBuffers> &p_render_buffers, int p_region) const override { return AABB(); }
	uint32_t sdfgi_get_pending_region_cascade(const Ref<RenderSceneBuffers> &p_render_buffers, int p_region) const override { return 0; }

	/* SKY API */

	RID sky_allocate() override { return RID(); }
	void sky_initialize(RID p_rid) override {}
	void sky_set_radiance_size(RID p_sky, int p_radiance_size) override {}
	void sky_set_mode(RID p_sky, RS::SkyMode p_samples) override {}
	void sky_set_material(RID p_sky, RID p_material) override {}
	Ref<Image> sky_bake_panorama(RID p_sky, float p_energy, bool p_bake_irradiance, const Size2i &p_size) override { return Ref<Image>(); }

	/* ENVIRONMENT API */

	void environment_glow_set_use_bicubic_upscale(bool p_enable) override {}
	void environment_glow_set_use_high_quality(bool p_enable) override {}

	void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) override {}

	void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override {}

	void environment_set_ssil_quality(RS::EnvironmentSSILQuality p_quality, bool p_half_size, float p_adaptive_target, int p_blur_passes, float p_fadeout_from, float p_fadeout_to) override {}

	void environment_set_sdfgi_ray_count(RS::EnvironmentSDFGIRayCount p_ray_count) override {}
	void environment_set_sdfgi_frames_to_converge(RS::EnvironmentSDFGIFramesToConverge p_frames) override {}
	void environment_set_sdfgi_frames_to_update_light(RS::EnvironmentSDFGIFramesToUpdateLight p_update) override {}

	void environment_set_volumetric_fog_volume_size(int p_size, int p_depth) override {}
	void environment_set_volumetric_fog_filter_active(bool p_enable) override {}

	Ref<Image> environment_bake_panorama(RID p_env, bool p_bake_irradiance, const Size2i &p_size) override { return Ref<Image>(); }

	void positional_soft_shadow_filter_set_quality(RS::ShadowQuality p_quality) override {}
	void directional_soft_shadow_filter_set_quality(RS::ShadowQuality p_quality) override {}

	RID light_instance_create(RID p_light) override { return RID(); }
	void light_instance_set_transform(RID p_light_instance, const Transform3D &p_transform) override {}
	void light_instance_set_aabb(RID p_light_instance, const AABB &p_aabb) override {}
	void light_instance_set_shadow_transform(RID p_light_instance, const Projection &p_projection, const Transform3D &p_transform, float p_far, float p_split, int p_pass, float p_shadow_texel_size, float p_bias_scale = 1.0, float p_range_begin = 0, const Vector2 &p_uv_scale = Vector2()) override {}
	void light_instance_mark_visible(RID p_light_instance) override {}

	RID fog_volume_instance_create(RID p_fog_volume) override { return RID(); }
	void fog_volume_instance_set_transform(RID p_fog_volume_instance, const Transform3D &p_transform) override {}
	void fog_volume_instance_set_active(RID p_fog_volume_instance, bool p_active) override {}
	RID fog_volume_instance_get_volume(RID p_fog_volume_instance) const override { return RID(); }
	Vector3 fog_volume_instance_get_position(RID p_fog_volume_instance) const override { return Vector3(); }

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
	void voxel_gi_update(RID p_probe, bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RenderGeometryInstance *> &p_dynamic_objects) override {}

	void voxel_gi_set_quality(RS::VoxelGIQuality) override {}

	void render_scene(const Ref<RenderSceneBuffers> &p_render_buffers, const CameraData *p_camera_data, const CameraData *p_prev_camera_data, const PagedArray<RenderGeometryInstance *> &p_instances, const PagedArray<RID> &p_lights, const PagedArray<RID> &p_reflection_probes, const PagedArray<RID> &p_voxel_gi_instances, const PagedArray<RID> &p_decals, const PagedArray<RID> &p_lightmaps, const PagedArray<RID> &p_fog_volumes, RID p_environment, RID p_camera_attributes, RID p_shadow_atlas, RID p_occluder_debug_tex, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass, float p_screen_mesh_lod_threshold, const RenderShadowData *p_render_shadows, int p_render_shadow_count, const RenderSDFGIData *p_render_sdfgi_regions, int p_render_sdfgi_region_count, const RenderSDFGIUpdateData *p_sdfgi_update_data = nullptr, RendererScene::RenderInfo *r_info = nullptr) override {}
	void render_material(const Transform3D &p_cam_transform, const Projection &p_cam_projection, bool p_cam_orthogonal, const PagedArray<RenderGeometryInstance *> &p_instances, RID p_framebuffer, const Rect2i &p_region) override {}
	void render_particle_collider_heightfield(RID p_collider, const Transform3D &p_transform, const PagedArray<RenderGeometryInstance *> &p_instances) override {}

	void set_scene_pass(uint64_t p_pass) override {}
	void set_time(double p_time, double p_step) override {}
	void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) override {}

	Ref<RenderSceneBuffers> render_buffers_create() override { return Ref<RenderSceneBuffers>(); }
	void gi_set_use_half_resolution(bool p_enable) override {}

	void screen_space_roughness_limiter_set_active(bool p_enable, float p_amount, float p_curve) override {}
	bool screen_space_roughness_limiter_is_active() const override { return false; }

	void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) override {}
	void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) override {}

	TypedArray<Image> bake_render_uv2(RID p_base, const TypedArray<RID> &p_material_overrides, const Size2i &p_image_size) override { return TypedArray<Image>(); }

	bool free(RID p_rid) override {
		if (is_environment(p_rid)) {
			environment_free(p_rid);
			return true;
		} else if (RSG::camera_attributes->owns_camera_attributes(p_rid)) {
			RSG::camera_attributes->camera_attributes_free(p_rid);
			return true;
		} else {
			return false;
		}
	}
	void update() override {}
	void sdfgi_set_debug_probe_select(const Vector3 &p_position, const Vector3 &p_dir) override {}

	virtual void decals_set_filter(RS::DecalFilter p_filter) override {}
	virtual void light_projectors_set_filter(RS::LightProjectorFilter p_filter) override {}

	RasterizerSceneDummy() {}
	~RasterizerSceneDummy() {}
};

#endif // RASTERIZER_SCENE_DUMMY_H
