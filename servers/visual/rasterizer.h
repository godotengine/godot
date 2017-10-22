/*************************************************************************/
/*  rasterizer.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "camera_matrix.h"
#include "servers/visual_server.h"

#include "self_list.h"

class RasterizerScene {
public:
	/* SHADOW ATLAS API */

	virtual RID shadow_atlas_create() = 0;
	virtual void shadow_atlas_set_size(RID p_atlas, int p_size) = 0;
	virtual void shadow_atlas_set_quadrant_subdivision(RID p_atlas, int p_quadrant, int p_subdivision) = 0;
	virtual bool shadow_atlas_update_light(RID p_atlas, RID p_light_intance, float p_coverage, uint64_t p_light_version) = 0;

	virtual int get_directional_light_shadow_size(RID p_light_intance) = 0;
	virtual void set_directional_shadow_count(int p_count) = 0;

	/* ENVIRONMENT API */

	virtual RID environment_create() = 0;

	virtual void environment_set_background(RID p_env, VS::EnvironmentBG p_bg) = 0;
	virtual void environment_set_sky(RID p_env, RID p_sky) = 0;
	virtual void environment_set_sky_custom_fov(RID p_env, float p_scale) = 0;
	virtual void environment_set_bg_color(RID p_env, const Color &p_color) = 0;
	virtual void environment_set_bg_energy(RID p_env, float p_energy) = 0;
	virtual void environment_set_canvas_max_layer(RID p_env, int p_max_layer) = 0;
	virtual void environment_set_ambient_light(RID p_env, const Color &p_color, float p_energy = 1.0, float p_sky_contribution = 0.0) = 0;

	virtual void environment_set_dof_blur_near(RID p_env, bool p_enable, float p_distance, float p_transition, float p_far_amount, VS::EnvironmentDOFBlurQuality p_quality) = 0;
	virtual void environment_set_dof_blur_far(RID p_env, bool p_enable, float p_distance, float p_transition, float p_far_amount, VS::EnvironmentDOFBlurQuality p_quality) = 0;
	virtual void environment_set_glow(RID p_env, bool p_enable, int p_level_flags, float p_intensity, float p_strength, float p_bloom_threshold, VS::EnvironmentGlowBlendMode p_blend_mode, float p_hdr_bleed_threshold, float p_hdr_bleed_scale, bool p_bicubic_upscale) = 0;
	virtual void environment_set_fog(RID p_env, bool p_enable, float p_begin, float p_end, RID p_gradient_texture) = 0;

	virtual void environment_set_ssr(RID p_env, bool p_enable, int p_max_steps, float p_fade_int, float p_fade_out, float p_depth_tolerance, bool p_roughness) = 0;
	virtual void environment_set_ssao(RID p_env, bool p_enable, float p_radius, float p_intensity, float p_radius2, float p_intensity2, float p_bias, float p_light_affect, const Color &p_color, VS::EnvironmentSSAOQuality p_quality, VS::EnvironmentSSAOBlur p_blur, float p_bilateral_sharpness) = 0;

	virtual void environment_set_tonemap(RID p_env, VS::EnvironmentToneMapper p_tone_mapper, float p_exposure, float p_white, bool p_auto_exposure, float p_min_luminance, float p_max_luminance, float p_auto_exp_speed, float p_auto_exp_scale) = 0;

	virtual void environment_set_adjustment(RID p_env, bool p_enable, float p_brightness, float p_contrast, float p_saturation, RID p_ramp) = 0;

	virtual void environment_set_fog(RID p_env, bool p_enable, const Color &p_color, const Color &p_sun_color, float p_sun_amount) = 0;
	virtual void environment_set_fog_depth(RID p_env, bool p_enable, float p_depth_begin, float p_depth_curve, bool p_transmit, float p_transmit_curve) = 0;
	virtual void environment_set_fog_height(RID p_env, bool p_enable, float p_min_height, float p_max_height, float p_height_curve) = 0;

	virtual bool is_environment(RID p_env) = 0;
	virtual VS::EnvironmentBG environment_get_background(RID p_env) = 0;
	virtual int environment_get_canvas_max_layer(RID p_env) = 0;

	struct InstanceBase : RID_Data {

		VS::InstanceType base_type;
		RID base;

		RID skeleton;
		RID material_override;

		Transform transform;

		int depth_layer;
		uint32_t layer_mask;

		//RID sampled_light;

		Vector<RID> materials;
		Vector<RID> light_instances;
		Vector<RID> reflection_probe_instances;
		Vector<RID> gi_probe_instances;

		Vector<float> blend_values;

		VS::ShadowCastingSetting cast_shadows;

		bool mirror : 8;
		bool receive_shadows : 8;
		bool visible : 8;
		bool baked_light : 8; //this flag is only to know if it actually did use baked light

		float depth; //used for sorting

		SelfList<InstanceBase> dependency_item;

		virtual void base_removed() = 0;
		virtual void base_changed() = 0;
		virtual void base_material_changed() = 0;

		InstanceBase()
			: dependency_item(this) {

			base_type = VS::INSTANCE_NONE;
			cast_shadows = VS::SHADOW_CASTING_SETTING_ON;
			receive_shadows = true;
			visible = true;
			depth_layer = 0;
			layer_mask = 1;
			baked_light = false;
		}
	};

	virtual RID light_instance_create(RID p_light) = 0;
	virtual void light_instance_set_transform(RID p_light_instance, const Transform &p_transform) = 0;
	virtual void light_instance_set_shadow_transform(RID p_light_instance, const CameraMatrix &p_projection, const Transform &p_transform, float p_far, float p_split, int p_pass, float p_bias_scale = 1.0) = 0;
	virtual void light_instance_mark_visible(RID p_light_instance) = 0;

	virtual RID reflection_atlas_create() = 0;
	virtual void reflection_atlas_set_size(RID p_ref_atlas, int p_size) = 0;
	virtual void reflection_atlas_set_subdivision(RID p_ref_atlas, int p_subdiv) = 0;

	virtual RID reflection_probe_instance_create(RID p_probe) = 0;
	virtual void reflection_probe_instance_set_transform(RID p_instance, const Transform &p_transform) = 0;
	virtual void reflection_probe_release_atlas_index(RID p_instance) = 0;
	virtual bool reflection_probe_instance_needs_redraw(RID p_instance) = 0;
	virtual bool reflection_probe_instance_has_reflection(RID p_instance) = 0;
	virtual bool reflection_probe_instance_begin_render(RID p_instance, RID p_reflection_atlas) = 0;
	virtual bool reflection_probe_instance_postprocess_step(RID p_instance) = 0;

	virtual RID gi_probe_instance_create() = 0;
	virtual void gi_probe_instance_set_light_data(RID p_probe, RID p_base, RID p_data) = 0;
	virtual void gi_probe_instance_set_transform_to_data(RID p_probe, const Transform &p_xform) = 0;
	virtual void gi_probe_instance_set_bounds(RID p_probe, const Vector3 &p_bounds) = 0;

	virtual void render_scene(const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID p_environment, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) = 0;
	virtual void render_shadow(RID p_light, RID p_shadow_atlas, int p_pass, InstanceBase **p_cull_result, int p_cull_count) = 0;

	virtual void set_scene_pass(uint64_t p_pass) = 0;
	virtual void set_debug_draw_mode(VS::ViewportDebugDraw p_debug_draw) = 0;

	virtual bool free(RID p_rid) = 0;

	virtual ~RasterizerScene() {}
};

class RasterizerStorage {
public:
	/* TEXTURE API */

	virtual RID texture_create() = 0;
	virtual void texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags = VS::TEXTURE_FLAGS_DEFAULT) = 0;
	virtual void texture_set_data(RID p_texture, const Ref<Image> &p_image, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT) = 0;
	virtual Ref<Image> texture_get_data(RID p_texture, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT) const = 0;
	virtual void texture_set_flags(RID p_texture, uint32_t p_flags) = 0;
	virtual uint32_t texture_get_flags(RID p_texture) const = 0;
	virtual Image::Format texture_get_format(RID p_texture) const = 0;
	virtual uint32_t texture_get_texid(RID p_texture) const = 0;
	virtual uint32_t texture_get_width(RID p_texture) const = 0;
	virtual uint32_t texture_get_height(RID p_texture) const = 0;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) = 0;

	virtual void texture_set_path(RID p_texture, const String &p_path) = 0;
	virtual String texture_get_path(RID p_texture) const = 0;

	virtual void texture_set_shrink_all_x2_on_set_data(bool p_enable) = 0;

	virtual void texture_debug_usage(List<VS::TextureInfo> *r_info) = 0;

	virtual RID texture_create_radiance_cubemap(RID p_source, int p_resolution = -1) const = 0;

	virtual void texture_set_detect_3d_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata) = 0;
	virtual void texture_set_detect_srgb_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata) = 0;
	virtual void texture_set_detect_normal_callback(RID p_texture, VisualServer::TextureDetectCallback p_callback, void *p_userdata) = 0;

	virtual void textures_keep_original(bool p_enable) = 0;

	/* SKY API */

	virtual RID sky_create() = 0;
	virtual void sky_set_texture(RID p_sky, RID p_cube_map, int p_radiance_size) = 0;

	/* SHADER API */

	virtual RID shader_create() = 0;

	virtual void shader_set_code(RID p_shader, const String &p_code) = 0;
	virtual String shader_get_code(RID p_shader) const = 0;
	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const = 0;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) = 0;
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const = 0;

	/* COMMON MATERIAL API */

	virtual RID material_create() = 0;

	virtual void material_set_render_priority(RID p_material, int priority) = 0;
	virtual void material_set_shader(RID p_shader_material, RID p_shader) = 0;
	virtual RID material_get_shader(RID p_shader_material) const = 0;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) = 0;
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const = 0;

	virtual void material_set_line_width(RID p_material, float p_width) = 0;

	virtual void material_set_next_pass(RID p_material, RID p_next_material) = 0;

	virtual bool material_is_animated(RID p_material) = 0;
	virtual bool material_casts_shadows(RID p_material) = 0;

	virtual void material_add_instance_owner(RID p_material, RasterizerScene::InstanceBase *p_instance) = 0;
	virtual void material_remove_instance_owner(RID p_material, RasterizerScene::InstanceBase *p_instance) = 0;

	/* MESH API */

	virtual RID mesh_create() = 0;

	virtual void mesh_add_surface(RID p_mesh, uint32_t p_format, VS::PrimitiveType p_primitive, const PoolVector<uint8_t> &p_array, int p_vertex_count, const PoolVector<uint8_t> &p_index_array, int p_index_count, const Rect3 &p_aabb, const Vector<PoolVector<uint8_t> > &p_blend_shapes = Vector<PoolVector<uint8_t> >(), const Vector<Rect3> &p_bone_aabbs = Vector<Rect3>()) = 0;

	virtual void mesh_set_blend_shape_count(RID p_mesh, int p_amount) = 0;
	virtual int mesh_get_blend_shape_count(RID p_mesh) const = 0;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, VS::BlendShapeMode p_mode) = 0;
	virtual VS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const = 0;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) = 0;
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const = 0;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const = 0;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const = 0;

	virtual PoolVector<uint8_t> mesh_surface_get_array(RID p_mesh, int p_surface) const = 0;
	virtual PoolVector<uint8_t> mesh_surface_get_index_array(RID p_mesh, int p_surface) const = 0;

	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const = 0;
	virtual VS::PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const = 0;

	virtual Rect3 mesh_surface_get_aabb(RID p_mesh, int p_surface) const = 0;
	virtual Vector<PoolVector<uint8_t> > mesh_surface_get_blend_shapes(RID p_mesh, int p_surface) const = 0;
	virtual Vector<Rect3> mesh_surface_get_skeleton_aabb(RID p_mesh, int p_surface) const = 0;

	virtual void mesh_remove_surface(RID p_mesh, int p_index) = 0;
	virtual int mesh_get_surface_count(RID p_mesh) const = 0;

	virtual void mesh_set_custom_aabb(RID p_mesh, const Rect3 &p_aabb) = 0;
	virtual Rect3 mesh_get_custom_aabb(RID p_mesh) const = 0;

	virtual Rect3 mesh_get_aabb(RID p_mesh, RID p_skeleton) const = 0;
	virtual void mesh_clear(RID p_mesh) = 0;

	/* MULTIMESH API */

	virtual RID multimesh_create() = 0;

	virtual void multimesh_allocate(RID p_multimesh, int p_instances, VS::MultimeshTransformFormat p_transform_format, VS::MultimeshColorFormat p_color_format) = 0;
	virtual int multimesh_get_instance_count(RID p_multimesh) const = 0;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh) = 0;
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) = 0;
	virtual void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) = 0;
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) = 0;

	virtual RID multimesh_get_mesh(RID p_multimesh) const = 0;

	virtual Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const = 0;
	virtual Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const = 0;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const = 0;

	virtual void multimesh_set_visible_instances(RID p_multimesh, int p_visible) = 0;
	virtual int multimesh_get_visible_instances(RID p_multimesh) const = 0;

	virtual Rect3 multimesh_get_aabb(RID p_multimesh) const = 0;

	/* IMMEDIATE API */

	virtual RID immediate_create() = 0;
	virtual void immediate_begin(RID p_immediate, VS::PrimitiveType p_rimitive, RID p_texture = RID()) = 0;
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
	virtual Rect3 immediate_get_aabb(RID p_immediate) const = 0;

	/* SKELETON API */

	virtual RID skeleton_create() = 0;
	virtual void skeleton_allocate(RID p_skeleton, int p_bones, bool p_2d_skeleton = false) = 0;
	virtual int skeleton_get_bone_count(RID p_skeleton) const = 0;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform) = 0;
	virtual Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone) const = 0;
	virtual void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) = 0;
	virtual Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const = 0;

	/* Light API */

	virtual RID light_create(VS::LightType p_type) = 0;

	virtual void light_set_color(RID p_light, const Color &p_color) = 0;
	virtual void light_set_param(RID p_light, VS::LightParam p_param, float p_value) = 0;
	virtual void light_set_shadow(RID p_light, bool p_enabled) = 0;
	virtual void light_set_shadow_color(RID p_light, const Color &p_color) = 0;
	virtual void light_set_projector(RID p_light, RID p_texture) = 0;
	virtual void light_set_negative(RID p_light, bool p_enable) = 0;
	virtual void light_set_cull_mask(RID p_light, uint32_t p_mask) = 0;
	virtual void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) = 0;

	virtual void light_omni_set_shadow_mode(RID p_light, VS::LightOmniShadowMode p_mode) = 0;
	virtual void light_omni_set_shadow_detail(RID p_light, VS::LightOmniShadowDetail p_detail) = 0;

	virtual void light_directional_set_shadow_mode(RID p_light, VS::LightDirectionalShadowMode p_mode) = 0;
	virtual void light_directional_set_blend_splits(RID p_light, bool p_enable) = 0;
	virtual bool light_directional_get_blend_splits(RID p_light) const = 0;
	virtual void light_directional_set_shadow_depth_range_mode(RID p_light, VS::LightDirectionalShadowDepthRangeMode p_range_mode) = 0;
	virtual VS::LightDirectionalShadowDepthRangeMode light_directional_get_shadow_depth_range_mode(RID p_light) const = 0;

	virtual VS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) = 0;
	virtual VS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) = 0;

	virtual bool light_has_shadow(RID p_light) const = 0;

	virtual VS::LightType light_get_type(RID p_light) const = 0;
	virtual Rect3 light_get_aabb(RID p_light) const = 0;
	virtual float light_get_param(RID p_light, VS::LightParam p_param) = 0;
	virtual Color light_get_color(RID p_light) = 0;
	virtual uint64_t light_get_version(RID p_light) const = 0;

	/* PROBE API */

	virtual RID reflection_probe_create() = 0;

	virtual void reflection_probe_set_update_mode(RID p_probe, VS::ReflectionProbeUpdateMode p_mode) = 0;
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

	virtual Rect3 reflection_probe_get_aabb(RID p_probe) const = 0;
	virtual VS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const = 0;
	virtual uint32_t reflection_probe_get_cull_mask(RID p_probe) const = 0;
	virtual Vector3 reflection_probe_get_extents(RID p_probe) const = 0;
	virtual Vector3 reflection_probe_get_origin_offset(RID p_probe) const = 0;
	virtual float reflection_probe_get_origin_max_distance(RID p_probe) const = 0;
	virtual bool reflection_probe_renders_shadows(RID p_probe) const = 0;

	virtual void instance_add_skeleton(RID p_skeleton, RasterizerScene::InstanceBase *p_instance) = 0;
	virtual void instance_remove_skeleton(RID p_skeleton, RasterizerScene::InstanceBase *p_instance) = 0;

	virtual void instance_add_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) = 0;
	virtual void instance_remove_dependency(RID p_base, RasterizerScene::InstanceBase *p_instance) = 0;

	/* GI PROBE API */

	virtual RID gi_probe_create() = 0;

	virtual void gi_probe_set_bounds(RID p_probe, const Rect3 &p_bounds) = 0;
	virtual Rect3 gi_probe_get_bounds(RID p_probe) const = 0;

	virtual void gi_probe_set_cell_size(RID p_probe, float p_range) = 0;
	virtual float gi_probe_get_cell_size(RID p_probe) const = 0;

	virtual void gi_probe_set_to_cell_xform(RID p_probe, const Transform &p_xform) = 0;
	virtual Transform gi_probe_get_to_cell_xform(RID p_probe) const = 0;

	virtual void gi_probe_set_dynamic_data(RID p_probe, const PoolVector<int> &p_data) = 0;
	virtual PoolVector<int> gi_probe_get_dynamic_data(RID p_probe) const = 0;

	virtual void gi_probe_set_dynamic_range(RID p_probe, int p_range) = 0;
	virtual int gi_probe_get_dynamic_range(RID p_probe) const = 0;

	virtual void gi_probe_set_energy(RID p_probe, float p_range) = 0;
	virtual float gi_probe_get_energy(RID p_probe) const = 0;

	virtual void gi_probe_set_bias(RID p_probe, float p_range) = 0;
	virtual float gi_probe_get_bias(RID p_probe) const = 0;

	virtual void gi_probe_set_normal_bias(RID p_probe, float p_range) = 0;
	virtual float gi_probe_get_normal_bias(RID p_probe) const = 0;

	virtual void gi_probe_set_propagation(RID p_probe, float p_range) = 0;
	virtual float gi_probe_get_propagation(RID p_probe) const = 0;

	virtual void gi_probe_set_interior(RID p_probe, bool p_enable) = 0;
	virtual bool gi_probe_is_interior(RID p_probe) const = 0;

	virtual void gi_probe_set_compress(RID p_probe, bool p_enable) = 0;
	virtual bool gi_probe_is_compressed(RID p_probe) const = 0;

	virtual uint32_t gi_probe_get_version(RID p_probe) = 0;

	enum GIProbeCompression {
		GI_PROBE_UNCOMPRESSED,
		GI_PROBE_S3TC,
		GI_PROBE_ETC2
	};

	virtual GIProbeCompression gi_probe_get_dynamic_data_get_preferred_compression() const = 0;
	virtual RID gi_probe_dynamic_data_create(int p_width, int p_height, int p_depth, GIProbeCompression p_compression) = 0;
	virtual void gi_probe_dynamic_data_update(RID p_gi_probe_data, int p_depth_slice, int p_slice_count, int p_mipmap, const void *p_data) = 0;

	/* PARTICLES */

	virtual RID particles_create() = 0;

	virtual void particles_set_emitting(RID p_particles, bool p_emitting) = 0;
	virtual void particles_set_amount(RID p_particles, int p_amount) = 0;
	virtual void particles_set_lifetime(RID p_particles, float p_lifetime) = 0;
	virtual void particles_set_one_shot(RID p_particles, bool p_one_shot) = 0;
	virtual void particles_set_pre_process_time(RID p_particles, float p_time) = 0;
	virtual void particles_set_explosiveness_ratio(RID p_particles, float p_ratio) = 0;
	virtual void particles_set_randomness_ratio(RID p_particles, float p_ratio) = 0;
	virtual void particles_set_custom_aabb(RID p_particles, const Rect3 &p_aabb) = 0;
	virtual void particles_set_speed_scale(RID p_particles, float p_scale) = 0;
	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable) = 0;
	virtual void particles_set_process_material(RID p_particles, RID p_material) = 0;
	virtual void particles_set_fixed_fps(RID p_particles, int p_fps) = 0;
	virtual void particles_set_fractional_delta(RID p_particles, bool p_enable) = 0;
	virtual void particles_restart(RID p_particles) = 0;

	virtual void particles_set_draw_order(RID p_particles, VS::ParticlesDrawOrder p_order) = 0;

	virtual void particles_set_draw_passes(RID p_particles, int p_count) = 0;
	virtual void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) = 0;

	virtual void particles_request_process(RID p_particles) = 0;
	virtual Rect3 particles_get_current_aabb(RID p_particles) = 0;
	virtual Rect3 particles_get_aabb(RID p_particles) const = 0;

	virtual void particles_set_emission_transform(RID p_particles, const Transform &p_transform) = 0;

	virtual int particles_get_draw_passes(RID p_particles) const = 0;
	virtual RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const = 0;

	/* RENDER TARGET */

	enum RenderTargetFlags {
		RENDER_TARGET_VFLIP,
		RENDER_TARGET_TRANSPARENT,
		RENDER_TARGET_NO_3D_EFFECTS,
		RENDER_TARGET_NO_3D,
		RENDER_TARGET_NO_SAMPLING,
		RENDER_TARGET_HDR,
		RENDER_TARGET_FLAG_MAX
	};

	virtual RID render_target_create() = 0;
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height) = 0;
	virtual RID render_target_get_texture(RID p_render_target) const = 0;
	virtual void render_target_set_flag(RID p_render_target, RenderTargetFlags p_flag, bool p_value) = 0;
	virtual bool render_target_was_used(RID p_render_target) = 0;
	virtual void render_target_clear_used(RID p_render_target) = 0;
	virtual void render_target_set_msaa(RID p_render_target, VS::ViewportMSAA p_msaa) = 0;

	/* CANVAS SHADOW */

	virtual RID canvas_light_shadow_buffer_create(int p_width) = 0;

	/* LIGHT SHADOW MAPPING */

	virtual RID canvas_light_occluder_create() = 0;
	virtual void canvas_light_occluder_set_polylines(RID p_occluder, const PoolVector<Vector2> &p_lines) = 0;

	virtual VS::InstanceType get_base_type(RID p_rid) const = 0;
	virtual bool free(RID p_rid) = 0;

	virtual bool has_os_feature(const String &p_feature) const = 0;

	virtual void update_dirty_resources() = 0;

	virtual void set_debug_generate_wireframes(bool p_generate) = 0;

	virtual void render_info_begin_capture() = 0;
	virtual void render_info_end_capture() = 0;
	virtual int get_captured_render_info(VS::RenderInfo p_info) = 0;

	virtual int get_render_info(VS::RenderInfo p_info) = 0;

	static RasterizerStorage *base_singleton;
	RasterizerStorage();
	virtual ~RasterizerStorage() {}
};

class RasterizerCanvas {
public:
	enum CanvasRectFlags {

		CANVAS_RECT_REGION = 1,
		CANVAS_RECT_TILE = 2,
		CANVAS_RECT_FLIP_H = 4,
		CANVAS_RECT_FLIP_V = 8,
		CANVAS_RECT_TRANSPOSE = 16,
		CANVAS_RECT_CLIP_UV = 32
	};

	struct Light : public RID_Data {

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
		VS::CanvasLightMode mode;
		RID texture;
		Vector2 texture_offset;
		RID canvas;
		RID shadow_buffer;
		int shadow_buffer_size;
		float shadow_gradient_length;
		VS::CanvasLightShadowFilter shadow_filter;
		Color shadow_color;
		float shadow_smooth;

		void *texture_cache; // implementation dependent
		Rect2 rect_cache;
		Transform2D xform_cache;
		float radius_cache; //used for shadow far plane
		CameraMatrix shadow_matrix_cache;

		Transform2D light_shader_xform;
		Vector2 light_shader_pos;

		Light *shadows_next_ptr;
		Light *filter_next_ptr;
		Light *next_ptr;
		Light *mask_next_ptr;

		RID light_internal;

		Light() {
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
			mode = VS::CANVAS_LIGHT_MODE_ADD;
			texture_cache = NULL;
			next_ptr = NULL;
			mask_next_ptr = NULL;
			filter_next_ptr = NULL;
			shadow_buffer_size = 256;
			shadow_gradient_length = 0;
			shadow_filter = VS::CANVAS_LIGHT_FILTER_NONE;
			shadow_smooth = 0.0;
		}
	};

	virtual RID light_internal_create() = 0;
	virtual void light_internal_update(RID p_rid, Light *p_light) = 0;
	virtual void light_internal_free(RID p_rid) = 0;

	struct Item : public RID_Data {

		struct Command {

			enum Type {

				TYPE_LINE,
				TYPE_POLYLINE,
				TYPE_RECT,
				TYPE_NINEPATCH,
				TYPE_PRIMITIVE,
				TYPE_POLYGON,
				TYPE_MESH,
				TYPE_MULTIMESH,
				TYPE_PARTICLES,
				TYPE_CIRCLE,
				TYPE_TRANSFORM,
				TYPE_CLIP_IGNORE,
			};

			Type type;
			virtual ~Command() {}
		};

		struct CommandLine : public Command {

			Point2 from, to;
			Color color;
			float width;
			bool antialiased;
			CommandLine() { type = TYPE_LINE; }
		};
		struct CommandPolyLine : public Command {

			bool antialiased;
			Vector<Point2> triangles;
			Vector<Color> triangle_colors;
			Vector<Point2> lines;
			Vector<Color> line_colors;
			CommandPolyLine() {
				type = TYPE_POLYLINE;
				antialiased = false;
			}
		};

		struct CommandRect : public Command {

			Rect2 rect;
			RID texture;
			RID normal_map;
			Color modulate;
			Rect2 source;
			uint8_t flags;

			CommandRect() {
				flags = 0;
				type = TYPE_RECT;
			}
		};

		struct CommandNinePatch : public Command {

			Rect2 rect;
			Rect2 source;
			RID texture;
			RID normal_map;
			float margin[4];
			bool draw_center;
			Color color;
			VS::NinePatchAxisMode axis_x;
			VS::NinePatchAxisMode axis_y;
			CommandNinePatch() {
				draw_center = true;
				type = TYPE_NINEPATCH;
			}
		};

		struct CommandPrimitive : public Command {

			Vector<Point2> points;
			Vector<Point2> uvs;
			Vector<Color> colors;
			RID texture;
			RID normal_map;
			float width;

			CommandPrimitive() {
				type = TYPE_PRIMITIVE;
				width = 1;
			}
		};

		struct CommandPolygon : public Command {

			Vector<int> indices;
			Vector<Point2> points;
			Vector<Point2> uvs;
			Vector<Color> colors;
			RID texture;
			RID normal_map;
			int count;
			bool antialiased;

			CommandPolygon() {
				type = TYPE_POLYGON;
				count = 0;
			}
		};

		struct CommandMesh : public Command {

			RID mesh;
			RID skeleton;
			CommandMesh() { type = TYPE_MESH; }
		};

		struct CommandMultiMesh : public Command {

			RID multimesh;
			RID skeleton;
			CommandMultiMesh() { type = TYPE_MULTIMESH; }
		};

		struct CommandParticles : public Command {

			RID particles;
			RID texture;
			RID normal_map;
			int h_frames;
			int v_frames;
			CommandParticles() { type = TYPE_PARTICLES; }
		};

		struct CommandCircle : public Command {

			Point2 pos;
			float radius;
			Color color;
			CommandCircle() { type = TYPE_CIRCLE; }
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
			VisualServer *owner;
			void *udata;
			Rect2 rect;
		};

		Transform2D xform;
		bool clip;
		bool visible;
		bool behind;
		//VS::MaterialBlendMode blend_mode;
		int light_mask;
		Vector<Command *> commands;
		mutable bool custom_rect;
		mutable bool rect_dirty;
		mutable Rect2 rect;
		RID material;
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
			if (custom_rect || !rect_dirty)
				return rect;

			//must update rect
			int s = commands.size();
			if (s == 0) {

				rect = Rect2();
				rect_dirty = false;
				return rect;
			}

			Transform2D xf;
			bool found_xform = false;
			bool first = true;

			const Item::Command *const *cmd = &commands[0];

			for (int i = 0; i < s; i++) {

				const Item::Command *c = cmd[i];
				Rect2 r;

				switch (c->type) {
					case Item::Command::TYPE_LINE: {

						const Item::CommandLine *line = static_cast<const Item::CommandLine *>(c);
						r.position = line->from;
						r.expand_to(line->to);
					} break;
					case Item::Command::TYPE_POLYLINE: {

						const Item::CommandPolyLine *pline = static_cast<const Item::CommandPolyLine *>(c);
						if (pline->triangles.size()) {
							for (int j = 0; j < pline->triangles.size(); j++) {

								if (j == 0) {
									r.position = pline->triangles[j];
								} else {
									r.expand_to(pline->triangles[j]);
								}
							}
						} else {

							for (int j = 0; j < pline->lines.size(); j++) {

								if (j == 0) {
									r.position = pline->lines[j];
								} else {
									r.expand_to(pline->lines[j]);
								}
							}
						}

					} break;
					case Item::Command::TYPE_RECT: {

						const Item::CommandRect *crect = static_cast<const Item::CommandRect *>(c);
						r = crect->rect;

					} break;
					case Item::Command::TYPE_NINEPATCH: {

						const Item::CommandNinePatch *style = static_cast<const Item::CommandNinePatch *>(c);
						r = style->rect;
					} break;
					case Item::Command::TYPE_PRIMITIVE: {

						const Item::CommandPrimitive *primitive = static_cast<const Item::CommandPrimitive *>(c);
						r.position = primitive->points[0];
						for (int i = 1; i < primitive->points.size(); i++) {

							r.expand_to(primitive->points[i]);
						}
					} break;
					case Item::Command::TYPE_POLYGON: {

						const Item::CommandPolygon *polygon = static_cast<const Item::CommandPolygon *>(c);
						int l = polygon->points.size();
						const Point2 *pp = &polygon->points[0];
						r.position = pp[0];
						for (int i = 1; i < l; i++) {

							r.expand_to(pp[i]);
						}
					} break;
					case Item::Command::TYPE_MESH: {

						const Item::CommandMesh *mesh = static_cast<const Item::CommandMesh *>(c);
						Rect3 aabb = RasterizerStorage::base_singleton->mesh_get_aabb(mesh->mesh, mesh->skeleton);

						r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);

					} break;
					case Item::Command::TYPE_MULTIMESH: {

						const Item::CommandMultiMesh *multimesh = static_cast<const Item::CommandMultiMesh *>(c);
						Rect3 aabb = RasterizerStorage::base_singleton->multimesh_get_aabb(multimesh->multimesh);

						r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);

					} break;
					case Item::Command::TYPE_PARTICLES: {

						const Item::CommandParticles *particles_cmd = static_cast<const Item::CommandParticles *>(c);
						if (particles_cmd->particles.is_valid()) {
							Rect3 aabb = RasterizerStorage::base_singleton->particles_get_aabb(particles_cmd->particles);
							r = Rect2(aabb.position.x, aabb.position.y, aabb.size.x, aabb.size.y);
						}

					} break;
					case Item::Command::TYPE_CIRCLE: {

						const Item::CommandCircle *circle = static_cast<const Item::CommandCircle *>(c);
						r.position = Point2(-circle->radius, -circle->radius) + circle->pos;
						r.size = Point2(circle->radius * 2.0, circle->radius * 2.0);
					} break;
					case Item::Command::TYPE_TRANSFORM: {

						const Item::CommandTransform *transform = static_cast<const Item::CommandTransform *>(c);
						xf = transform->xform;
						found_xform = true;
						continue;
					} break;

					case Item::Command::TYPE_CLIP_IGNORE: {

					} break;
				}

				if (found_xform) {
					r = xf.xform(r);
					found_xform = false;
				}

				if (first) {
					rect = r;
					first = false;
				} else
					rect = rect.merge(r);
			}

			rect_dirty = false;
			return rect;
		}

		void clear() {
			for (int i = 0; i < commands.size(); i++)
				memdelete(commands[i]);
			commands.clear();
			clip = false;
			rect_dirty = true;
			final_clip_owner = NULL;
			material_owner = NULL;
			light_masked = false;
		}
		Item() {
			light_mask = 1;
			vp_render = NULL;
			next = NULL;
			final_clip_owner = NULL;
			clip = false;
			final_modulate = Color(1, 1, 1, 1);
			visible = true;
			rect_dirty = true;
			custom_rect = false;
			behind = false;
			material_owner = NULL;
			copy_back_buffer = NULL;
			distance_field = false;
			light_masked = false;
		}
		virtual ~Item() {
			clear();
			if (copy_back_buffer) memdelete(copy_back_buffer);
		}
	};

	virtual void canvas_begin() = 0;
	virtual void canvas_end() = 0;

	virtual void canvas_render_items(Item *p_item_list, int p_z, const Color &p_modulate, Light *p_light) = 0;
	virtual void canvas_debug_viewport_shadows(Light *p_lights_with_shadow) = 0;

	struct LightOccluderInstance : public RID_Data {

		bool enabled;
		RID canvas;
		RID polygon;
		RID polygon_buffer;
		Rect2 aabb_cache;
		Transform2D xform;
		Transform2D xform_cache;
		int light_mask;
		VS::CanvasOccluderPolygonCullMode cull_cache;

		LightOccluderInstance *next;

		LightOccluderInstance() {
			enabled = true;
			next = NULL;
			light_mask = 1;
			cull_cache = VS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;
		}
	};

	virtual void canvas_light_shadow_buffer_update(RID p_buffer, const Transform2D &p_light_xform, int p_light_mask, float p_near, float p_far, LightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) = 0;

	virtual void reset_canvas() = 0;

	virtual void draw_window_margins(int *p_margins, RID *p_margin_textures) = 0;

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

	virtual void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale) = 0;

	virtual void initialize() = 0;
	virtual void begin_frame() = 0;
	virtual void set_current_render_target(RID p_render_target) = 0;
	virtual void restore_render_target() = 0;
	virtual void clear_render_target(const Color &p_color) = 0;
	virtual void blit_render_target_to_screen(RID p_render_target, const Rect2 &p_screen_rect, int p_screen = 0) = 0;
	virtual void end_frame() = 0;
	virtual void finalize() = 0;

	virtual ~Rasterizer() {}
};

#endif // RASTERIZER_H
