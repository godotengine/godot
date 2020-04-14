/*************************************************************************/
/*  rasterizer.h                                                         */
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

#ifndef RASTERIZER_H
#define RASTERIZER_H

#include "core/math/camera_matrix.h"
#include "servers/rendering_server.h"

#include "core/pair.h"
#include "core/self_list.h"

class RasterizerScene {

public:
	/* SHADOW ATLAS API */

	virtual RID shadow_atlas_create() = 0;
	virtual void shadow_atlas_set_size(RID p_atlas, int p_size) = 0;
	virtual void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) = 0;
	virtual bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) = 0;

	virtual void directional_shadow_atlas_set_size(int p_size) = 0;
	virtual int get_directional_light_shadow_size(RID p_light_intance) = 0;
	virtual void set_directional_shadow_count(int p_count) = 0;

	/* SKY API */

	virtual RID sky_create() = 0;
	virtual void sky_set_radiance_size(RID p_sky, int p_radiance_size) = 0;
	virtual void sky_set_mode(RID p_sky, RS::SkyMode p_samples) = 0;
	virtual void sky_set_material(RID p_sky, RID p_material) = 0;

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

	virtual void environment_set_glow(RID p_env, bool p_enable, int p_level_flags, float p_intensity, float p_strength, float p_mix, float p_bloom_threshold, RS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, float p_hdr_luminance_cap) = 0;
	virtual void environment_glow_set_use_bicubic_upscale(bool p_enable) = 0;
	virtual void environment_set_fog(RID p_env, bool p_enable, float p_begin, float p_end, RID p_gradient_texture) = 0;

	virtual void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance) = 0;
	virtual void environment_set_ssr_roughness_quality(RS::EnvironmentSSRRoughnessQuality p_quality) = 0;

	virtual void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_bias, float p_light_affect, float p_ao_channel_affect, RS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness) = 0;

	virtual void environment_set_ssao_quality(RS::EnvironmentSSAOQuality p_quality, bool p_half_size) = 0;

	virtual void environment_set_tonemap(RID p_env, RS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) = 0;

	virtual void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, RID p_ramp) = 0;

	virtual void environment_set_fog(RID p_env, bool p_enable, const Color &p_color, const Color &p_sun_color, float p_sun_amount) = 0;
	virtual void environment_set_fog_depth(RID p_env, bool p_enable, float p_depth_begin, float p_depth_end, float p_depth_curve, bool p_transmit, float p_transmit_curve) = 0;
	virtual void environment_set_fog_height(RID p_env, bool p_enable, float p_min_height, float p_max_height, float p_height_curve) = 0;

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

	struct InstanceBase;

	struct InstanceDependency {
		void instance_notify_changed(bool p_aabb, bool p_dependencies);
		void instance_notify_deleted(RID p_deleted);

		~InstanceDependency();

	private:
		friend struct InstanceBase;
		Map<InstanceBase *, uint32_t> instances;
	};

	struct InstanceBase {
		RS::InstanceType base_type;
		RID base;

		RID skeleton;
		RID material_override;

		RID instance_data;

		Transform transform;

		int depth_layer;
		uint32_t layer_mask;
		uint32_t instance_version;

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

		SelfList<InstanceBase> dependency_item;

		InstanceBase *lightmap_capture;
		RID lightmap;
		Vector<Color> lightmap_capture_data; //in a array (12 values) to avoid wasting space if unused. Alpha is unused, but needed to send to shader

		AABB aabb;
		AABB transformed_aabb;

		virtual void dependency_deleted(RID p_dependency) = 0;
		virtual void dependency_changed(bool p_aabb, bool p_dependencies) = 0;

		Set<InstanceDependency *> dependencies;

		void instance_increase_version() {
			instance_version++;
		}

		void update_dependency(InstanceDependency *p_dependency) {
			dependencies.insert(p_dependency);
			p_dependency->instances[this] = instance_version;
		}

		void clean_up_dependencies() {
			List<Pair<InstanceDependency *, Map<InstanceBase *, uint32_t>::Element *>> to_clean_up;
			for (Set<InstanceDependency *>::Element *E = dependencies.front(); E; E = E->next()) {
				InstanceDependency *dep = E->get();
				Map<InstanceBase *, uint32_t>::Element *F = dep->instances.find(this);
				ERR_CONTINUE(!F);
				if (F->get() != instance_version) {
					Pair<InstanceDependency *, Map<InstanceBase *, uint32_t>::Element *> p;
					p.first = dep;
					p.second = F;
					to_clean_up.push_back(p);
				}
			}

			while (to_clean_up.size()) {
				to_clean_up.front()->get().first->instances.erase(to_clean_up.front()->get().second);
				to_clean_up.pop_front();
			}
		}

		void clear_dependencies() {
			for (Set<InstanceDependency *>::Element *E = dependencies.front(); E; E = E->next()) {
				InstanceDependency *dep = E->get();
				dep->instances.erase(this);
			}
			dependencies.clear();
		}

		InstanceBase() :
				dependency_item(this) {

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
			lightmap_capture = nullptr;
		}

		virtual ~InstanceBase() {
			clear_dependencies();
		}
	};

	virtual RID light_instance_create(RID p_light) = 0;
	virtual void light_instance_set_transform(RID p_light_instance, const Transform &p_transform) = 0;
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

	virtual void render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID *p_gi_probe_cull_result, int p_gi_probe_cull_count, RID *p_decal_cull_result, int p_decal_cull_count, RID p_environment, RID p_camera_effects, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) = 0;

	virtual void render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count) = 0;
	virtual void render_material(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID p_framebuffer, const Rect2i &p_region) = 0;

	virtual void set_scene_pass(uint64_t p_pass) = 0;
	virtual void set_time(double p_time, double p_step) = 0;
	virtual void set_debug_draw_mode(RS::ViewportDebugDraw p_debug_draw) = 0;

	virtual RID render_buffers_create() = 0;
	virtual void render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, RS::ViewportMSAA p_msaa, RS::ViewportScreenSpaceAA p_screen_space_aa) = 0;

	virtual void screen_space_roughness_limiter_set_active(bool p_enable, float p_curve) = 0;
	virtual bool screen_space_roughness_limiter_is_active() const = 0;

	virtual void sub_surface_scattering_set_quality(RS::SubSurfaceScatteringQuality p_quality) = 0;
	virtual void sub_surface_scattering_set_scale(float p_scale, float p_depth_scale) = 0;

	virtual bool free(RID p_rid) = 0;

	virtual void update() = 0;
	virtual ~RasterizerScene() {}
};

class RasterizerStorage {

	Color default_clear_color;

public:
	/* TEXTURE API */

	virtual RID texture_2d_create(const Ref<Image> &p_image) = 0;
	virtual RID texture_2d_layered_create(const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) = 0;
	virtual RID texture_3d_create(const Vector<Ref<Image>> &p_slices) = 0; //all slices, then all the mipmaps, must be coherent
	virtual RID texture_proxy_create(RID p_base) = 0; //all slices, then all the mipmaps, must be coherent

	virtual void texture_2d_update_immediate(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) = 0; //mostly used for video and streaming
	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) = 0;
	virtual void texture_3d_update(RID p_texture, const Ref<Image> &p_image, int p_depth, int p_mipmap) = 0;
	virtual void texture_proxy_update(RID p_proxy, RID p_base) = 0;

	//these two APIs can be used together or in combination with the others.
	virtual RID texture_2d_placeholder_create() = 0;
	virtual RID texture_2d_layered_placeholder_create() = 0;
	virtual RID texture_3d_placeholder_create() = 0;

	virtual Ref<Image> texture_2d_get(RID p_texture) const = 0;
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const = 0;
	virtual Ref<Image> texture_3d_slice_get(RID p_texture, int p_depth, int p_mipmap) const = 0;

	virtual void texture_replace(RID p_texture, RID p_by_texture) = 0;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) = 0;
// FIXME: Disabled during Vulkan refactoring, should be ported.
#if 0
	virtual void texture_bind(RID p_texture, uint32_t p_texture_no) = 0;
#endif

	virtual void texture_set_path(RID p_texture, const String &p_path) = 0;
	virtual String texture_get_path(RID p_texture) const = 0;

	virtual void texture_set_detect_3d_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) = 0;
	virtual void texture_set_detect_normal_callback(RID p_texture, RS::TextureDetectCallback p_callback, void *p_userdata) = 0;
	virtual void texture_set_detect_roughness_callback(RID p_texture, RS::TextureDetectRoughnessCallback p_callback, void *p_userdata) = 0;

	virtual void texture_debug_usage(List<RS::TextureInfo> *r_info) = 0;

	virtual void texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) = 0;

	virtual Size2 texture_size_with_proxy(RID p_proxy) = 0;

	virtual void texture_add_to_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) = 0;
	virtual void texture_remove_from_decal_atlas(RID p_texture, bool p_panorama_to_dp = false) = 0;

	/* SHADER API */

	virtual RID shader_create() = 0;

	virtual void shader_set_code(RID p_shader, const String &p_code) = 0;
	virtual String shader_get_code(RID p_shader) const = 0;
	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const = 0;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) = 0;
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const = 0;
	virtual Variant shader_get_param_default(RID p_material, const StringName &p_param) const = 0;

	/* COMMON MATERIAL API */

	virtual RID material_create() = 0;

	virtual void material_set_render_priority(RID p_material, int priority) = 0;
	virtual void material_set_shader(RID p_shader_material, RID p_shader) = 0;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) = 0;
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const = 0;

	virtual void material_set_next_pass(RID p_material, RID p_next_material) = 0;

	virtual bool material_is_animated(RID p_material) = 0;
	virtual bool material_casts_shadows(RID p_material) = 0;

	virtual void material_update_dependency(RID p_material, RasterizerScene::InstanceBase *p_instance) = 0;

	/* MESH API */

	virtual RID mesh_create() = 0;

	/// Returns stride
	virtual void mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) = 0;

	virtual int mesh_get_blend_shape_count(RID p_mesh) const = 0;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) = 0;
	virtual RS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const = 0;

	virtual void mesh_surface_update_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) = 0;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) = 0;
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const = 0;

	virtual RS::SurfaceData mesh_get_surface(RID p_mesh, int p_surface) const = 0;

	virtual int mesh_get_surface_count(RID p_mesh) const = 0;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) = 0;
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const = 0;

	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID()) = 0;

	virtual void mesh_clear(RID p_mesh) = 0;

	/* MULTIMESH API */

	virtual RID multimesh_create() = 0;

	virtual void multimesh_allocate(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors = false, bool p_use_custom_data = false) = 0;

	virtual int multimesh_get_instance_count(RID p_multimesh) const = 0;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh) = 0;
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) = 0;
	virtual void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) = 0;
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) = 0;
	virtual void multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) = 0;

	virtual RID multimesh_get_mesh(RID p_multimesh) const = 0;

	virtual Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const = 0;
	virtual Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const = 0;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const = 0;
	virtual Color multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const = 0;

	virtual void multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) = 0;
	virtual Vector<float> multimesh_get_buffer(RID p_multimesh) const = 0;

	virtual void multimesh_set_visible_instances(RID p_multimesh, int p_visible) = 0;
	virtual int multimesh_get_visible_instances(RID p_multimesh) const = 0;

	virtual AABB multimesh_get_aabb(RID p_multimesh) const = 0;

	/* IMMEDIATE API */

	virtual RID immediate_create() = 0;
	virtual void immediate_begin(RID p_immediate, RS::PrimitiveType p_rimitive, RID p_texture = RID()) = 0;
	virtual void immediate_vertex(RID p_immediate, const Vector3 &p_vertex) = 0;
	virtual void immediate_normal(RID p_immediate, const Vector3 &p_normal) = 0;
	virtual void immediate_tangent(RID p_immediate, const Plane &p_tangent) = 0;
	virtual void immediate_color(RID p_immediate, const Color &p_color) = 0;
	virtual void immediate_uv(RID p_immediate, const Vector2 &tex_uv) = 0;
	virtual void immediate_uv2(RID p_immediate, const Vector2 &tex_uv) = 0;
	virtual void immediate_end(RID p_immediate) = 0;
	virtual void immediate_clear(RID p_immediate) = 0;
	virtual void immediate_set_material(RID p_immediate, RID p_material) = 0;
	virtual RID immediate_get_material(RID p_immediate) const = 0;
	virtual AABB immediate_get_aabb(RID p_immediate) const = 0;

	/* SKELETON API */

	virtual RID skeleton_create() = 0;
	virtual void skeleton_allocate(RID p_skeleton, int p_bones, bool p_2d_skeleton = false) = 0;
	virtual int skeleton_get_bone_count(RID p_skeleton) const = 0;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform) = 0;
	virtual Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone) const = 0;
	virtual void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) = 0;
	virtual Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const = 0;
	virtual void skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) = 0;

	/* Light API */

	virtual RID light_create(RS::LightType p_type) = 0;

	RID directional_light_create() { return light_create(RS::LIGHT_DIRECTIONAL); }
	RID omni_light_create() { return light_create(RS::LIGHT_OMNI); }
	RID spot_light_create() { return light_create(RS::LIGHT_SPOT); }

	virtual void light_set_color(RID p_light, const Color &p_color) = 0;
	virtual void light_set_param(RID p_light, RS::LightParam p_param, float p_value) = 0;
	virtual void light_set_shadow(RID p_light, bool p_enabled) = 0;
	virtual void light_set_shadow_color(RID p_light, const Color &p_color) = 0;
	virtual void light_set_projector(RID p_light, RID p_texture) = 0;
	virtual void light_set_negative(RID p_light, bool p_enable) = 0;
	virtual void light_set_cull_mask(RID p_light, uint32_t p_mask) = 0;
	virtual void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) = 0;
	virtual void light_set_use_gi(RID p_light, bool p_enable) = 0;

	virtual void light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) = 0;

	virtual void light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) = 0;
	virtual void light_directional_set_blend_splits(RID p_light, bool p_enable) = 0;
	virtual bool light_directional_get_blend_splits(RID p_light) const = 0;
	virtual void light_directional_set_shadow_depth_range_mode(RID p_light, RS::LightDirectionalShadowDepthRangeMode p_range_mode) = 0;
	virtual RS::LightDirectionalShadowDepthRangeMode light_directional_get_shadow_depth_range_mode(RID p_light) const = 0;

	virtual RS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) = 0;
	virtual RS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) = 0;

	virtual bool light_has_shadow(RID p_light) const = 0;

	virtual RS::LightType light_get_type(RID p_light) const = 0;
	virtual AABB light_get_aabb(RID p_light) const = 0;
	virtual float light_get_param(RID p_light, RS::LightParam p_param) = 0;
	virtual Color light_get_color(RID p_light) = 0;
	virtual bool light_get_use_gi(RID p_light) = 0;
	virtual uint64_t light_get_version(RID p_light) const = 0;

	/* PROBE API */

	virtual RID reflection_probe_create() = 0;

	virtual void reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) = 0;
	virtual void reflection_probe_set_resolution(RID p_probe, int p_resolution) = 0;
	virtual void reflection_probe_set_intensity(RID p_probe, float p_intensity) = 0;
	virtual void reflection_probe_set_interior_ambient(RID p_probe, const Color &p_ambient) = 0;
	virtual void reflection_probe_set_interior_ambient_energy(RID p_probe, float p_energy) = 0;
	virtual void reflection_probe_set_interior_ambient_probe_contribution(RID p_probe, float p_contrib) = 0;
	virtual void reflection_probe_set_max_distance(RID p_probe, float p_distance) = 0;
	virtual void reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) = 0;
	virtual void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) = 0;
	virtual void reflection_probe_set_as_interior(RID p_probe, bool p_enable) = 0;
	virtual void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) = 0;
	virtual void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) = 0;
	virtual void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) = 0;

	virtual AABB reflection_probe_get_aabb(RID p_probe) const = 0;
	virtual RS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const = 0;
	virtual uint32_t reflection_probe_get_cull_mask(RID p_probe) const = 0;
	virtual Vector3 reflection_probe_get_extents(RID p_probe) const = 0;
	virtual Vector3 reflection_probe_get_origin_offset(RID p_probe) const = 0;
	virtual float reflection_probe_get_origin_max_distance(RID p_probe) const = 0;
	virtual bool reflection_probe_renders_shadows(RID p_probe) const = 0;

	virtual void base_update_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) = 0;
	virtual void skeleton_update_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) = 0;

	/* DECAL API */

	virtual RID decal_create() = 0;
	virtual void decal_set_extents(RID p_decal, const Vector3 &p_extents) = 0;
	virtual void decal_set_texture(RID p_decal, RS::DecalTexture p_type, RID p_texture) = 0;
	virtual void decal_set_emission_energy(RID p_decal, float p_energy) = 0;
	virtual void decal_set_albedo_mix(RID p_decal, float p_mix) = 0;
	virtual void decal_set_modulate(RID p_decal, const Color &p_modulate) = 0;
	virtual void decal_set_cull_mask(RID p_decal, uint32_t p_layers) = 0;
	virtual void decal_set_distance_fade(RID p_decal, bool p_enabled, float p_begin, float p_length) = 0;
	virtual void decal_set_fade(RID p_decal, float p_above, float p_below) = 0;
	virtual void decal_set_normal_fade(RID p_decal, float p_fade) = 0;

	virtual AABB decal_get_aabb(RID p_decal) const = 0;

	/* GI PROBE API */

	virtual RID gi_probe_create() = 0;

	virtual void gi_probe_allocate(RID p_gi_probe, const Transform &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) = 0;

	virtual AABB gi_probe_get_bounds(RID p_gi_probe) const = 0;
	virtual Vector3i gi_probe_get_octree_size(RID p_gi_probe) const = 0;
	virtual Vector<uint8_t> gi_probe_get_octree_cells(RID p_gi_probe) const = 0;
	virtual Vector<uint8_t> gi_probe_get_data_cells(RID p_gi_probe) const = 0;
	virtual Vector<uint8_t> gi_probe_get_distance_field(RID p_gi_probe) const = 0;

	virtual Vector<int> gi_probe_get_level_counts(RID p_gi_probe) const = 0;
	virtual Transform gi_probe_get_to_cell_xform(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_dynamic_range(RID p_gi_probe, float p_range) = 0;
	virtual float gi_probe_get_dynamic_range(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_propagation(RID p_gi_probe, float p_range) = 0;
	virtual float gi_probe_get_propagation(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_energy(RID p_gi_probe, float p_energy) = 0;
	virtual float gi_probe_get_energy(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_ao(RID p_gi_probe, float p_ao) = 0;
	virtual float gi_probe_get_ao(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_ao_size(RID p_gi_probe, float p_strength) = 0;
	virtual float gi_probe_get_ao_size(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_bias(RID p_gi_probe, float p_bias) = 0;
	virtual float gi_probe_get_bias(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_normal_bias(RID p_gi_probe, float p_range) = 0;
	virtual float gi_probe_get_normal_bias(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_interior(RID p_gi_probe, bool p_enable) = 0;
	virtual bool gi_probe_is_interior(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_use_two_bounces(RID p_gi_probe, bool p_enable) = 0;
	virtual bool gi_probe_is_using_two_bounces(RID p_gi_probe) const = 0;

	virtual void gi_probe_set_anisotropy_strength(RID p_gi_probe, float p_strength) = 0;
	virtual float gi_probe_get_anisotropy_strength(RID p_gi_probe) const = 0;

	virtual uint32_t gi_probe_get_version(RID p_probe) = 0;

	/* LIGHTMAP CAPTURE */

	struct LightmapCaptureOctree {

		enum {
			CHILD_EMPTY = 0xFFFFFFFF
		};

		uint16_t light[6][3]; //anisotropic light
		float alpha;
		uint32_t children[8];
	};

	virtual RID lightmap_capture_create() = 0;
	virtual void lightmap_capture_set_bounds(RID p_capture, const AABB &p_bounds) = 0;
	virtual AABB lightmap_capture_get_bounds(RID p_capture) const = 0;
	virtual void lightmap_capture_set_octree(RID p_capture, const Vector<uint8_t> &p_octree) = 0;
	virtual Vector<uint8_t> lightmap_capture_get_octree(RID p_capture) const = 0;
	virtual void lightmap_capture_set_octree_cell_transform(RID p_capture, const Transform &p_xform) = 0;
	virtual Transform lightmap_capture_get_octree_cell_transform(RID p_capture) const = 0;
	virtual void lightmap_capture_set_octree_cell_subdiv(RID p_capture, int p_subdiv) = 0;
	virtual int lightmap_capture_get_octree_cell_subdiv(RID p_capture) const = 0;
	virtual void lightmap_capture_set_energy(RID p_capture, float p_energy) = 0;
	virtual float lightmap_capture_get_energy(RID p_capture) const = 0;
	virtual const Vector<LightmapCaptureOctree> *lightmap_capture_get_octree_ptr(RID p_capture) const = 0;

	/* PARTICLES */

	virtual RID particles_create() = 0;

	virtual void particles_set_emitting(RID p_particles, bool p_emitting) = 0;
	virtual bool particles_get_emitting(RID p_particles) = 0;

	virtual void particles_set_amount(RID p_particles, int p_amount) = 0;
	virtual void particles_set_lifetime(RID p_particles, float p_lifetime) = 0;
	virtual void particles_set_one_shot(RID p_particles, bool p_one_shot) = 0;
	virtual void particles_set_pre_process_time(RID p_particles, float p_time) = 0;
	virtual void particles_set_explosiveness_ratio(RID p_particles, float p_ratio) = 0;
	virtual void particles_set_randomness_ratio(RID p_particles, float p_ratio) = 0;
	virtual void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) = 0;
	virtual void particles_set_speed_scale(RID p_particles, float p_scale) = 0;
	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable) = 0;
	virtual void particles_set_process_material(RID p_particles, RID p_material) = 0;
	virtual void particles_set_fixed_fps(RID p_particles, int p_fps) = 0;
	virtual void particles_set_fractional_delta(RID p_particles, bool p_enable) = 0;
	virtual void particles_restart(RID p_particles) = 0;

	virtual bool particles_is_inactive(RID p_particles) const = 0;

	virtual void particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) = 0;

	virtual void particles_set_draw_passes(RID p_particles, int p_count) = 0;
	virtual void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) = 0;

	virtual void particles_request_process(RID p_particles) = 0;
	virtual AABB particles_get_current_aabb(RID p_particles) = 0;
	virtual AABB particles_get_aabb(RID p_particles) const = 0;

	virtual void particles_set_emission_transform(RID p_particles, const Transform &p_transform) = 0;

	virtual int particles_get_draw_passes(RID p_particles) const = 0;
	virtual RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const = 0;

	/* RENDER TARGET */

	enum RenderTargetFlags {
		RENDER_TARGET_TRANSPARENT,
		RENDER_TARGET_DIRECT_TO_SCREEN,
		RENDER_TARGET_FLAG_MAX
	};

	virtual RID render_target_create() = 0;
	virtual void render_target_set_position(RID p_render_target, int p_x, int p_y) = 0;
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height) = 0;
	virtual RID render_target_get_texture(RID p_render_target) = 0;
	virtual void render_target_set_external_texture(RID p_render_target, unsigned int p_texture_id) = 0;
	virtual void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) = 0;
	virtual bool render_target_was_used(RID p_render_target) = 0;
	virtual void render_target_set_as_unused(RID p_render_target) = 0;

	virtual void render_target_request_clear(RID p_render_target, const Color &p_clear_color) = 0;
	virtual bool render_target_is_clear_requested(RID p_render_target) = 0;
	virtual Color render_target_get_clear_request_color(RID p_render_target) = 0;
	virtual void render_target_disable_clear_request(RID p_render_target) = 0;
	virtual void render_target_do_clear_request(RID p_render_target) = 0;

	virtual RS::InstanceType get_base_type(RID p_rid) const = 0;
	virtual bool free(RID p_rid) = 0;

	virtual bool has_os_feature(const String &p_feature) const = 0;

	virtual void update_dirty_resources() = 0;

	virtual void set_debug_generate_wireframes(bool p_generate) = 0;

	virtual void render_info_begin_capture() = 0;
	virtual void render_info_end_capture() = 0;
	virtual int get_captured_render_info(RS::RenderInfo p_info) = 0;

	virtual int get_render_info(RS::RenderInfo p_info) = 0;
	virtual String get_video_adapter_name() const = 0;
	virtual String get_video_adapter_vendor() const = 0;

	static RasterizerStorage *base_singleton;

	void set_default_clear_color(const Color &p_color) {
		default_clear_color = p_color;
	}

	Color get_default_clear_color() const {
		return default_clear_color;
	}
#define TIMESTAMP_BEGIN()                                                                 \
	{                                                                                     \
		if (RSG::storage->capturing_timestamps) RSG::storage->capture_timestamps_begin(); \
	}

#define RENDER_TIMESTAMP(m_text)                                                         \
	{                                                                                    \
		if (RSG::storage->capturing_timestamps) RSG::storage->capture_timestamp(m_text); \
	}

	bool capturing_timestamps = false;

	virtual void capture_timestamps_begin() = 0;
	virtual void capture_timestamp(const String &p_name) = 0;
	virtual uint32_t get_captured_timestamps_count() const = 0;
	virtual uint64_t get_captured_timestamps_frame() const = 0;
	virtual uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const = 0;
	virtual uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const = 0;
	virtual String get_captured_timestamp_name(uint32_t p_index) const = 0;

	RasterizerStorage();
	virtual ~RasterizerStorage() {}
};

class RasterizerCanvas {
public:
	static RasterizerCanvas *singleton;

	enum CanvasRectFlags {

		CANVAS_RECT_REGION = 1,
		CANVAS_RECT_TILE = 2,
		CANVAS_RECT_FLIP_H = 4,
		CANVAS_RECT_FLIP_V = 8,
		CANVAS_RECT_TRANSPOSE = 16,
		CANVAS_RECT_CLIP_UV = 32
	};

	struct Light {

		bool enabled;
		Color color;
		Transform2D xform;
		float height;
		float energy;
		float scale;
		int z_min;
		int z_max;
		int layer_min;
		int layer_max;
		int item_mask;
		int item_shadow_mask;
		RS::CanvasLightMode mode;
		RID texture;
		Vector2 texture_offset;
		RID canvas;
		bool use_shadow;
		int shadow_buffer_size;
		RS::CanvasLightShadowFilter shadow_filter;
		Color shadow_color;
		float shadow_smooth;

		//void *texture_cache; // implementation dependent
		Rect2 rect_cache;
		Transform2D xform_cache;
		float radius_cache; //used for shadow far plane
		//CameraMatrix shadow_matrix_cache;

		Transform2D light_shader_xform;
		//Vector2 light_shader_pos;

		Light *shadows_next_ptr;
		Light *filter_next_ptr;
		Light *next_ptr;
		Light *mask_next_ptr;

		RID light_internal;
		uint64_t version;

		int32_t render_index_cache;

		Light() {
			version = 0;
			enabled = true;
			color = Color(1, 1, 1);
			shadow_color = Color(0, 0, 0, 0);
			height = 0;
			z_min = -1024;
			z_max = 1024;
			layer_min = 0;
			layer_max = 0;
			item_mask = 1;
			scale = 1.0;
			energy = 1.0;
			item_shadow_mask = -1;
			mode = RS::CANVAS_LIGHT_MODE_ADD;
			//			texture_cache = nullptr;
			next_ptr = nullptr;
			mask_next_ptr = nullptr;
			filter_next_ptr = nullptr;
			use_shadow = false;
			shadow_buffer_size = 2048;
			shadow_filter = RS::CANVAS_LIGHT_FILTER_NONE;
			shadow_smooth = 0.0;
			render_index_cache = -1;
		}
	};

	typedef uint64_t TextureBindingID;

	virtual TextureBindingID request_texture_binding(RID p_texture, RID p_normalmap, RID p_specular, RS::CanvasItemTextureFilter p_filter, RS::CanvasItemTextureRepeat p_repeat, RID p_multimesh) = 0;
	virtual void free_texture_binding(TextureBindingID p_binding) = 0;

	//easier wrap to avoid mistakes

	struct Item;

	struct TextureBinding {

		TextureBindingID binding_id;

		_FORCE_INLINE_ void create(RS::CanvasItemTextureFilter p_item_filter, RS::CanvasItemTextureRepeat p_item_repeat, RID p_texture, RID p_normalmap, RID p_specular, RS::CanvasItemTextureFilter p_filter, RS::CanvasItemTextureRepeat p_repeat, RID p_multimesh) {
			if (p_filter == RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT) {
				p_filter = p_item_filter;
			}
			if (p_repeat == RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) {
				p_repeat = p_item_repeat;
			}
			if (p_texture != RID() || p_normalmap != RID() || p_specular != RID() || p_filter != RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT || p_repeat != RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT || p_multimesh.is_valid()) {
				ERR_FAIL_COND(binding_id != 0);
				binding_id = singleton->request_texture_binding(p_texture, p_normalmap, p_specular, p_filter, p_repeat, p_multimesh);
			}
		}

		_FORCE_INLINE_ TextureBinding() { binding_id = 0; }
		_FORCE_INLINE_ ~TextureBinding() {
			if (binding_id) singleton->free_texture_binding(binding_id);
		}
	};

	typedef uint64_t PolygonID;
	virtual PolygonID request_polygon(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), const Vector<int> &p_bones = Vector<int>(), const Vector<float> &p_weights = Vector<float>()) = 0;
	virtual void free_polygon(PolygonID p_polygon) = 0;

	//also easier to wrap to avoid mistakes
	struct Polygon {

		PolygonID polygon_id;
		Rect2 rect_cache;

		_FORCE_INLINE_ void create(const Vector<int> &p_indices, const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs = Vector<Point2>(), const Vector<int> &p_bones = Vector<int>(), const Vector<float> &p_weights = Vector<float>()) {
			ERR_FAIL_COND(polygon_id != 0);
			{
				uint32_t pc = p_points.size();
				const Vector2 *v2 = p_points.ptr();
				rect_cache.position = *v2;
				for (uint32_t i = 1; i < pc; i++) {
					rect_cache.expand_to(v2[i]);
				}
			}
			polygon_id = singleton->request_polygon(p_indices, p_points, p_colors, p_uvs, p_bones, p_weights);
		}

		_FORCE_INLINE_ Polygon() { polygon_id = 0; }
		_FORCE_INLINE_ ~Polygon() {
			if (polygon_id) singleton->free_polygon(polygon_id);
		}
	};

	//item

	struct Item {

		//commands are allocated in blocks of 4k to improve performance
		//and cache coherence.
		//blocks always grow but never shrink.

		struct CommandBlock {
			enum {
				MAX_SIZE = 4096
			};
			uint32_t usage;
			uint8_t *memory;
		};

		struct Command {

			enum Type {

				TYPE_RECT,
				TYPE_NINEPATCH,
				TYPE_POLYGON,
				TYPE_PRIMITIVE,
				TYPE_MESH,
				TYPE_MULTIMESH,
				TYPE_PARTICLES,
				TYPE_TRANSFORM,
				TYPE_CLIP_IGNORE,
			};

			Command *next;
			Type type;
			virtual ~Command() {}
		};

		struct CommandRect : public Command {

			Rect2 rect;
			Color modulate;
			Rect2 source;
			uint8_t flags;
			Color specular_shininess;

			TextureBinding texture_binding;

			CommandRect() {
				flags = 0;
				type = TYPE_RECT;
			}
		};

		struct CommandNinePatch : public Command {

			Rect2 rect;
			Rect2 source;
			float margin[4];
			bool draw_center;
			Color color;
			RS::NinePatchAxisMode axis_x;
			RS::NinePatchAxisMode axis_y;
			Color specular_shininess;
			TextureBinding texture_binding;
			CommandNinePatch() {
				draw_center = true;
				type = TYPE_NINEPATCH;
			}
		};

		struct CommandPolygon : public Command {

			RS::PrimitiveType primitive;
			Polygon polygon;
			Color specular_shininess;
			TextureBinding texture_binding;
			CommandPolygon() {
				type = TYPE_POLYGON;
			}
		};

		struct CommandPrimitive : public Command {

			uint32_t point_count;
			Vector2 points[4];
			Vector2 uvs[4];
			Color colors[4];
			Color specular_shininess;
			TextureBinding texture_binding;
			CommandPrimitive() {
				type = TYPE_PRIMITIVE;
			}
		};

		struct CommandMesh : public Command {

			RID mesh;
			Transform2D transform;
			Color modulate;
			Color specular_shininess;
			TextureBinding texture_binding;
			CommandMesh() { type = TYPE_MESH; }
		};

		struct CommandMultiMesh : public Command {

			RID multimesh;
			Color specular_shininess;
			TextureBinding texture_binding;
			CommandMultiMesh() { type = TYPE_MULTIMESH; }
		};

		struct CommandParticles : public Command {

			RID particles;
			Color specular_shininess;
			TextureBinding texture_binding;
			CommandParticles() { type = TYPE_PARTICLES; }
		};

		struct CommandTransform : public Command {

			Transform2D xform;
			CommandTransform() { type = TYPE_TRANSFORM; }
		};

		struct CommandClipIgnore : public Command {

			bool ignore;
			CommandClipIgnore() {
				type = TYPE_CLIP_IGNORE;
				ignore = false;
			}
		};

		struct ViewportRender {
			RenderingServer *owner;
			void *udata;
			Rect2 rect;
		};

		Transform2D xform;
		bool clip;
		bool visible;
		bool behind;
		bool update_when_visible;
		//RS::MaterialBlendMode blend_mode;
		int light_mask;
		int z_final;

		mutable bool custom_rect;
		mutable bool rect_dirty;
		mutable Rect2 rect;
		RID material;
		RID skeleton;

		Item *next;

		struct CopyBackBuffer {
			Rect2 rect;
			Rect2 screen_rect;
			bool full;
		};
		CopyBackBuffer *copy_back_buffer;

		Color final_modulate;
		Transform2D final_transform;
		Rect2 final_clip_rect;
		Item *final_clip_owner;
		Item *material_owner;
		ViewportRender *vp_render;
		bool distance_field;
		bool light_masked;

		Rect2 global_rect_cache;

		const Rect2 &get_rect() const {
			if (custom_rect || (!rect_dirty && !update_when_visible))
				return rect;

			//must update rect

			if (commands == nullptr) {

				rect = Rect2();
				rect_dirty = false;
				return rect;
			}

			Transform2D xf;
			bool found_xform = false;
			bool first = true;

			const Item::Command *c = commands;

			while (c) {

				Rect2 r;

				switch (c->type) {
					case Item::Command::TYPE_RECT: {

						const Item::CommandRect *crect = static_cast<const Item::CommandRect *>(c);
						r = crect->rect;

					} break;
					case Item::Command::TYPE_NINEPATCH: {

						const Item::CommandNinePatch *style = static_cast<const Item::CommandNinePatch *>(c);
						r = style->rect;
					} break;

					case Item::Command::TYPE_POLYGON: {

						const Item::CommandPolygon *polygon = static_cast<const Item::CommandPolygon *>(c);
						r = polygon->polygon.rect_cache;
					} break;
					case Item::Command::TYPE_PRIMITIVE: {

						const Item::CommandPrimitive *primitive = static_cast<const Item::CommandPrimitive *>(c);
						for (uint32_t j = 0; j < primitive->point_count; j++) {
							if (j == 0) {
								r.position = primitive->points[0];
							} else {
								r.expand_to(primitive->points[j]);
							}
						}
					} break;
					case Item::Command::TYPE_MESH: {

						const Item::CommandMesh *mesh = static_cast<const Item::CommandMesh *>(c);
						AABB aabb = RasterizerStorage::base_singleton->mesh_get_aabb(mesh->mesh, RID());

						r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);

					} break;
					case Item::Command::TYPE_MULTIMESH: {

						const Item::CommandMultiMesh *multimesh = static_cast<const Item::CommandMultiMesh *>(c);
						AABB aabb = RasterizerStorage::base_singleton->multimesh_get_aabb(multimesh->multimesh);

						r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);

					} break;
					case Item::Command::TYPE_PARTICLES: {

						const Item::CommandParticles *particles_cmd = static_cast<const Item::CommandParticles *>(c);
						if (particles_cmd->particles.is_valid()) {
							AABB aabb = RasterizerStorage::base_singleton->particles_get_aabb(particles_cmd->particles);
							r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);
						}

					} break;
					case Item::Command::TYPE_TRANSFORM: {

						const Item::CommandTransform *transform = static_cast<const Item::CommandTransform *>(c);
						xf = transform->xform;
						found_xform = true;
						[[fallthrough]];
					}
					default: {
						c = c->next;
						continue;
					}
				}

				if (found_xform) {
					r = xf.xform(r);
					found_xform = false;
				}

				if (first) {
					rect = r;
					first = false;
				} else {
					rect = rect.merge(r);
				}
				c = c->next;
			}

			rect_dirty = false;
			return rect;
		}

		Command *commands;
		Command *last_command;
		Vector<CommandBlock> blocks;
		uint32_t current_block;

		template <class T>
		T *alloc_command() {
			T *command;
			if (commands == nullptr) {
				// As the most common use case of canvas items is to
				// use only one command, the first is done with it's
				// own allocation. The rest of them use blocks.
				command = memnew(T);
				command->next = nullptr;
				commands = command;
				last_command = command;
			} else {
				//Subsequent commands go into a block.

				while (true) {
					if (unlikely(current_block == (uint32_t)blocks.size())) {
						// If we need more blocks, we allocate them
						// (they won't be freed until this CanvasItem is
						// deleted, though).
						CommandBlock cb;
						cb.memory = (uint8_t *)memalloc(CommandBlock::MAX_SIZE);
						cb.usage = 0;
						blocks.push_back(cb);
					}

					CommandBlock *c = &blocks.write[current_block];
					size_t space_left = CommandBlock::MAX_SIZE - c->usage;
					if (space_left < sizeof(T)) {
						current_block++;
						continue;
					}

					//allocate block and add to the linked list
					void *memory = c->memory + c->usage;
					command = memnew_placement(memory, T);
					command->next = nullptr;
					last_command->next = command;
					last_command = command;
					c->usage += sizeof(T);
					break;
				}
			}

			rect_dirty = true;
			return command;
		}

		struct CustomData {

			virtual ~CustomData() {}
		};

		mutable CustomData *custom_data; //implementation dependent

		void clear() {
			Command *c = commands;
			while (c) {
				Command *n = c->next;
				if (c == commands) {
					memdelete(commands);
					commands = nullptr;
				} else {
					c->~Command();
				}
				c = n;
			}
			{
				uint32_t cbc = MIN((current_block + 1), (uint32_t)blocks.size());
				CommandBlock *blockptr = blocks.ptrw();
				for (uint32_t i = 0; i < cbc; i++) {
					blockptr[i].usage = 0;
				}
			}

			last_command = nullptr;
			commands = nullptr;
			current_block = 0;
			clip = false;
			rect_dirty = true;
			final_clip_owner = nullptr;
			material_owner = nullptr;
			light_masked = false;
		}
		Item() {
			commands = nullptr;
			last_command = nullptr;
			current_block = 0;
			light_mask = 1;
			vp_render = nullptr;
			next = nullptr;
			final_clip_owner = nullptr;
			clip = false;
			final_modulate = Color(1, 1, 1, 1);
			visible = true;
			rect_dirty = true;
			custom_rect = false;
			behind = false;
			material_owner = nullptr;
			copy_back_buffer = nullptr;
			distance_field = false;
			light_masked = false;
			update_when_visible = false;
			z_final = 0;
			custom_data = nullptr;
		}
		virtual ~Item() {
			clear();
			for (int i = 0; i < blocks.size(); i++) {
				memfree(blocks[i].memory);
			}
			if (copy_back_buffer) memdelete(copy_back_buffer);
			if (custom_data) {
				memdelete(custom_data);
			}
		}
	};

	virtual void canvas_render_items(RID p_to_render_target, Item *p_item_list, const Color &p_modulate, Light *p_light_list, const Transform2D &p_canvas_transform) = 0;
	virtual void canvas_debug_viewport_shadows(Light *p_lights_with_shadow) = 0;

	struct LightOccluderInstance {

		bool enabled;
		RID canvas;
		RID polygon;
		RID occluder;
		Rect2 aabb_cache;
		Transform2D xform;
		Transform2D xform_cache;
		int light_mask;
		RS::CanvasOccluderPolygonCullMode cull_cache;

		LightOccluderInstance *next;

		LightOccluderInstance() {
			enabled = true;
			next = nullptr;
			light_mask = 1;
			cull_cache = RS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;
		}
	};

	virtual RID light_create() = 0;
	virtual void light_set_texture(RID p_rid, RID p_texture) = 0;
	virtual void light_set_use_shadow(RID p_rid, bool p_enable, int p_resolution) = 0;
	virtual void light_update_shadow(RID p_rid, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders) = 0;

	virtual RID occluder_polygon_create() = 0;
	virtual void occluder_polygon_set_shape_as_lines(RID p_occluder, const Vector<Vector2> &p_lines) = 0;
	virtual void occluder_polygon_set_cull_mode(RID p_occluder, RS::CanvasOccluderPolygonCullMode p_mode) = 0;

	virtual void draw_window_margins(int *p_margins, RID *p_margin_textures) = 0;

	virtual bool free(RID p_rid) = 0;
	virtual void update() = 0;

	RasterizerCanvas() { singleton = this; }
	virtual ~RasterizerCanvas() {}
};

class Rasterizer {
protected:
	static Rasterizer *(*_create_func)();

public:
	static Rasterizer *create();

	virtual RasterizerStorage *get_storage() = 0;
	virtual RasterizerCanvas *get_canvas() = 0;
	virtual RasterizerScene *get_scene() = 0;

	virtual void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter = true) = 0;

	virtual void initialize() = 0;
	virtual void begin_frame(double frame_step) = 0;

	struct BlitToScreen {
		RID render_target;
		Rect2i rect;
		//lens distorted parameters for VR should go here
	};

	virtual void prepare_for_blitting_render_targets() = 0;
	virtual void blit_render_targets_to_screen(DisplayServer::WindowID p_screen, const BlitToScreen *p_render_targets, int p_amount) = 0;

	virtual void end_frame(bool p_swap_buffers) = 0;
	virtual void finalize() = 0;

	virtual bool is_low_end() const = 0;

	virtual ~Rasterizer() {}
};

#endif // RASTERIZER_H
