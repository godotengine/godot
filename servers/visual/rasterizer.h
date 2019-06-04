/*************************************************************************/
/*  rasterizer.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
#include "camera_matrix.h"
#include "map.h"
#include "self_list.h"
#include "servers/visual_server.h"

class Rasterizer {
protected:
	typedef void (*CanvasItemDrawViewportFunc)(VisualServer *owner, void *ud, const Rect2 &p_rect);

	RID create_default_material();
	RID create_overdraw_debug_material();

	/* Fixed Material Shader API */

	union FixedMaterialShaderKey {

		struct {
			uint16_t texcoord_mask;
			uint8_t texture_mask;
			uint8_t light_shader : 2;
			bool use_alpha : 1;
			bool use_color_array : 1;
			bool use_pointsize : 1;
			bool discard_alpha : 1;
			bool use_xy_normalmap : 1;
			bool valid : 1;
		};

		uint32_t key;

		_FORCE_INLINE_ bool operator<(const FixedMaterialShaderKey &p_key) const { return key < p_key.key; }
	};

	struct FixedMaterialShader {

		int refcount;
		RID shader;
	};

	Map<FixedMaterialShaderKey, FixedMaterialShader> fixed_material_shaders;

	RID _create_shader(const FixedMaterialShaderKey &p_key);
	void _free_shader(const FixedMaterialShaderKey &p_key);

	struct FixedMaterial {

		RID self;
		bool use_alpha;
		bool use_color_array;
		bool discard_alpha;
		bool use_pointsize;
		bool use_xy_normalmap;
		float point_size;
		Transform uv_xform;
		VS::FixedMaterialLightShader light_shader;
		RID texture[VS::FIXED_MATERIAL_PARAM_MAX];
		Variant param[VS::FIXED_MATERIAL_PARAM_MAX];
		VS::FixedMaterialTexCoordMode texture_tc[VS::FIXED_MATERIAL_PARAM_MAX];

		SelfList<FixedMaterial> dirty_list;

		FixedMaterialShaderKey current_key;

		_FORCE_INLINE_ FixedMaterialShaderKey get_key() const {

			FixedMaterialShaderKey k;
			k.key = 0;
			k.use_alpha = use_alpha;
			k.use_color_array = use_color_array;
			k.use_pointsize = use_pointsize;
			k.use_xy_normalmap = use_xy_normalmap;
			k.discard_alpha = discard_alpha;
			k.light_shader = light_shader;
			k.valid = true;
			for (int i = 0; i < VS::FIXED_MATERIAL_PARAM_MAX; i++) {
				if (texture[i].is_valid()) {
					//print_line("valid: "+itos(i));
					k.texture_mask |= (1 << i);
					k.texcoord_mask |= (texture_tc[i]) << (i * 2);
				}
			}

			return k;
		}

		FixedMaterial() :
				dirty_list(this) {

			use_alpha = false;
			use_color_array = false;
			use_pointsize = false;
			discard_alpha = false;
			use_xy_normalmap = false;
			point_size = 1.0;
			light_shader = VS::FIXED_MATERIAL_LIGHT_SHADER_LAMBERT;
			for (int i = 0; i < VS::FIXED_MATERIAL_PARAM_MAX; i++) {
				texture_tc[i] = VS::FIXED_MATERIAL_TEXCOORD_UV;
			}
			param[VS::FIXED_MATERIAL_PARAM_DIFFUSE] = Color(1, 1, 1);
			param[VS::FIXED_MATERIAL_PARAM_DETAIL] = 1.0;
			param[VS::FIXED_MATERIAL_PARAM_EMISSION] = Color(0, 0, 0);
			param[VS::FIXED_MATERIAL_PARAM_GLOW] = 0;
			param[VS::FIXED_MATERIAL_PARAM_SHADE_PARAM] = 0;
			param[VS::FIXED_MATERIAL_PARAM_SPECULAR] = Color(0.0, 0.0, 0.0);
			param[VS::FIXED_MATERIAL_PARAM_SPECULAR_EXP] = 40;
			param[VS::FIXED_MATERIAL_PARAM_NORMAL] = 1;

			current_key.key = 0;
		}
	};

	StringName _fixed_material_param_names[VS::FIXED_MATERIAL_PARAM_MAX];
	StringName _fixed_material_tex_names[VS::FIXED_MATERIAL_PARAM_MAX];
	StringName _fixed_material_uv_xform_name;
	StringName _fixed_material_point_size_name;

	Map<RID, FixedMaterial *> fixed_materials;

	SelfList<FixedMaterial>::List fixed_material_dirty_list;

protected:
	void _update_fixed_materials();
	void _free_fixed_material(const RID &p_material);

public:
	enum ShadowFilterTechnique {
		SHADOW_FILTER_NONE,
		SHADOW_FILTER_PCF5,
		SHADOW_FILTER_PCF13,
		SHADOW_FILTER_ESM,
		SHADOW_FILTER_VSM,
	};

	/* TEXTURE API */

	virtual RID texture_create() = 0;
	RID texture_create_from_image(const Image &p_image, uint32_t p_flags = VS::TEXTURE_FLAGS_DEFAULT); // helper
	virtual void texture_allocate(RID p_texture, int p_width, int p_height, Image::Format p_format, uint32_t p_flags = VS::TEXTURE_FLAGS_DEFAULT) = 0;
	virtual void texture_set_data(RID p_texture, const Image &p_image, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT) = 0;
	virtual Image texture_get_data(RID p_texture, VS::CubeMapSide p_cube_side = VS::CUBEMAP_LEFT) const = 0;
	virtual void texture_set_flags(RID p_texture, uint32_t p_flags) = 0;
	virtual uint32_t texture_get_flags(RID p_texture) const = 0;
	virtual Image::Format texture_get_format(RID p_texture) const = 0;
	virtual uint32_t texture_get_width(RID p_texture) const = 0;
	virtual uint32_t texture_get_height(RID p_texture) const = 0;
	virtual bool texture_has_alpha(RID p_texture) const = 0;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) = 0;

	virtual void texture_set_reload_hook(RID p_texture, ObjectID p_owner, const StringName &p_function) const = 0;

	virtual void texture_set_path(RID p_texture, const String &p_path) = 0;
	virtual String texture_get_path(RID p_texture) const = 0;
	virtual void texture_debug_usage(List<VS::TextureInfo> *r_info) = 0;

	virtual void texture_set_shrink_all_x2_on_set_data(bool p_enable) = 0;

	/* SHADER API */

	virtual RID shader_create(VS::ShaderMode p_mode = VS::SHADER_MATERIAL) = 0;

	virtual void shader_set_mode(RID p_shader, VS::ShaderMode p_mode) = 0;
	virtual VS::ShaderMode shader_get_mode(RID p_shader) const = 0;

	virtual void shader_set_code(RID p_shader, const String &p_vertex, const String &p_fragment, const String &p_light, int p_vertex_ofs = 0, int p_fragment_ofs = 0, int p_light_ofs = 0) = 0;
	virtual String shader_get_fragment_code(RID p_shader) const = 0;
	virtual String shader_get_vertex_code(RID p_shader) const = 0;
	virtual String shader_get_light_code(RID p_shader) const = 0;

	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const = 0;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture) = 0;
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name) const = 0;

	virtual Variant shader_get_default_param(RID p_shader, const StringName &p_name) = 0;

	/* COMMON MATERIAL API */

	virtual RID material_create() = 0;

	virtual void material_set_shader(RID p_shader_material, RID p_shader) = 0;
	virtual RID material_get_shader(RID p_shader_material) const = 0;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) = 0;
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const = 0;

	virtual void material_set_flag(RID p_material, VS::MaterialFlag p_flag, bool p_enabled) = 0;
	virtual bool material_get_flag(RID p_material, VS::MaterialFlag p_flag) const = 0;

	virtual void material_set_depth_draw_mode(RID p_material, VS::MaterialDepthDrawMode p_mode) = 0;
	virtual VS::MaterialDepthDrawMode material_get_depth_draw_mode(RID p_material) const = 0;

	virtual void material_set_blend_mode(RID p_material, VS::MaterialBlendMode p_mode) = 0;
	virtual VS::MaterialBlendMode material_get_blend_mode(RID p_material) const = 0;

	virtual void material_set_line_width(RID p_material, float p_line_width) = 0;
	virtual float material_get_line_width(RID p_material) const = 0;

	/* FIXED MATERIAL */

	virtual RID fixed_material_create();

	virtual void fixed_material_set_flag(RID p_material, VS::FixedMaterialFlags p_flag, bool p_enabled);
	virtual bool fixed_material_get_flag(RID p_material, VS::FixedMaterialFlags p_flag) const;

	virtual void fixed_material_set_parameter(RID p_material, VS::FixedMaterialParam p_parameter, const Variant &p_value);
	virtual Variant fixed_material_get_parameter(RID p_material, VS::FixedMaterialParam p_parameter) const;

	virtual void fixed_material_set_texture(RID p_material, VS::FixedMaterialParam p_parameter, RID p_texture);
	virtual RID fixed_material_get_texture(RID p_material, VS::FixedMaterialParam p_parameter) const;

	virtual void fixed_material_set_texcoord_mode(RID p_material, VS::FixedMaterialParam p_parameter, VS::FixedMaterialTexCoordMode p_mode);
	virtual VS::FixedMaterialTexCoordMode fixed_material_get_texcoord_mode(RID p_material, VS::FixedMaterialParam p_parameter) const;

	virtual void fixed_material_set_uv_transform(RID p_material, const Transform &p_transform);
	virtual Transform fixed_material_get_uv_transform(RID p_material) const;

	virtual void fixed_material_set_light_shader(RID p_material, VS::FixedMaterialLightShader p_shader);
	virtual VS::FixedMaterialLightShader fixed_material_get_light_shader(RID p_material) const;

	virtual void fixed_material_set_point_size(RID p_material, float p_size);
	virtual float fixed_material_get_point_size(RID p_material) const;

	/* MESH API */

	virtual RID mesh_create() = 0;

	virtual void mesh_add_surface(RID p_mesh, VS::PrimitiveType p_primitive, const Array &p_arrays, const Array &p_blend_shapes = Array(), bool p_alpha_sort = false) = 0;
	virtual Array mesh_get_surface_arrays(RID p_mesh, int p_surface) const = 0;
	virtual Array mesh_get_surface_morph_arrays(RID p_mesh, int p_surface) const = 0;

	virtual void mesh_add_custom_surface(RID p_mesh, const Variant &p_dat) = 0;

	virtual void mesh_set_morph_target_count(RID p_mesh, int p_amount) = 0;
	virtual int mesh_get_morph_target_count(RID p_mesh) const = 0;

	virtual void mesh_set_morph_target_mode(RID p_mesh, VS::MorphTargetMode p_mode) = 0;
	virtual VS::MorphTargetMode mesh_get_morph_target_mode(RID p_mesh) const = 0;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material, bool p_owned = false) = 0;
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const = 0;

	virtual int mesh_surface_get_array_len(RID p_mesh, int p_surface) const = 0;
	virtual int mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const = 0;
	virtual uint32_t mesh_surface_get_format(RID p_mesh, int p_surface) const = 0;
	virtual VS::PrimitiveType mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const = 0;

	virtual void mesh_remove_surface(RID p_mesh, int p_index) = 0;
	virtual int mesh_get_surface_count(RID p_mesh) const = 0;

	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID()) const = 0;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) = 0;
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const = 0;

	/* MULTIMESH API */

	virtual RID multimesh_create() = 0;

	virtual void multimesh_set_instance_count(RID p_multimesh, int p_count) = 0;
	virtual int multimesh_get_instance_count(RID p_multimesh) const = 0;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh) = 0;
	virtual void multimesh_set_aabb(RID p_multimesh, const AABB &p_aabb) = 0;
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform &p_transform) = 0;
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) = 0;

	virtual RID multimesh_get_mesh(RID p_multimesh) const = 0;
	virtual AABB multimesh_get_aabb(RID p_multimesh) const = 0;

	virtual Transform multimesh_instance_get_transform(RID p_multimesh, int p_index) const = 0;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const = 0;

	virtual void multimesh_set_visible_instances(RID p_multimesh, int p_visible) = 0;
	virtual int multimesh_get_visible_instances(RID p_multimesh) const = 0;

	/* BAKED LIGHT */

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
	virtual AABB immediate_get_aabb(RID p_immediate) const = 0;
	virtual void immediate_set_material(RID p_immediate, RID p_material) = 0;
	virtual RID immediate_get_material(RID p_immediate) const = 0;

	/* PARTICLES API */

	virtual RID particles_create() = 0;

	virtual void particles_set_amount(RID p_particles, int p_amount) = 0;
	virtual int particles_get_amount(RID p_particles) const = 0;

	virtual void particles_set_emitting(RID p_particles, bool p_emitting) = 0;
	virtual bool particles_is_emitting(RID p_particles) const = 0;

	virtual void particles_set_visibility_aabb(RID p_particles, const AABB &p_visibility) = 0;
	virtual AABB particles_get_visibility_aabb(RID p_particles) const = 0;

	virtual void particles_set_emission_half_extents(RID p_particles, const Vector3 &p_half_extents) = 0;
	virtual Vector3 particles_get_emission_half_extents(RID p_particles) const = 0;

	virtual void particles_set_emission_base_velocity(RID p_particles, const Vector3 &p_base_velocity) = 0;
	virtual Vector3 particles_get_emission_base_velocity(RID p_particles) const = 0;

	virtual void particles_set_emission_points(RID p_particles, const DVector<Vector3> &p_points) = 0;
	virtual DVector<Vector3> particles_get_emission_points(RID p_particles) const = 0;

	virtual void particles_set_gravity_normal(RID p_particles, const Vector3 &p_normal) = 0;
	virtual Vector3 particles_get_gravity_normal(RID p_particles) const = 0;

	virtual void particles_set_variable(RID p_particles, VS::ParticleVariable p_variable, float p_value) = 0;
	virtual float particles_get_variable(RID p_particles, VS::ParticleVariable p_variable) const = 0;

	virtual void particles_set_randomness(RID p_particles, VS::ParticleVariable p_variable, float p_randomness) = 0;
	virtual float particles_get_randomness(RID p_particles, VS::ParticleVariable p_variable) const = 0;

	virtual void particles_set_color_phase_pos(RID p_particles, int p_phase, float p_pos) = 0;
	virtual float particles_get_color_phase_pos(RID p_particles, int p_phase) const = 0;

	virtual void particles_set_color_phases(RID p_particles, int p_phases) = 0;
	virtual int particles_get_color_phases(RID p_particles) const = 0;

	virtual void particles_set_color_phase_color(RID p_particles, int p_phase, const Color &p_color) = 0;
	virtual Color particles_get_color_phase_color(RID p_particles, int p_phase) const = 0;

	virtual void particles_set_attractors(RID p_particles, int p_attractors) = 0;
	virtual int particles_get_attractors(RID p_particles) const = 0;

	virtual void particles_set_attractor_pos(RID p_particles, int p_attractor, const Vector3 &p_pos) = 0;
	virtual Vector3 particles_get_attractor_pos(RID p_particles, int p_attractor) const = 0;

	virtual void particles_set_attractor_strength(RID p_particles, int p_attractor, float p_force) = 0;
	virtual float particles_get_attractor_strength(RID p_particles, int p_attractor) const = 0;

	virtual void particles_set_material(RID p_particles, RID p_material, bool p_owned = false) = 0;
	virtual RID particles_get_material(RID p_particles) const = 0;

	virtual AABB particles_get_aabb(RID p_particles) const = 0;

	virtual void particles_set_height_from_velocity(RID p_particles, bool p_enable) = 0;
	virtual bool particles_has_height_from_velocity(RID p_particles) const = 0;

	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable) = 0;
	virtual bool particles_is_using_local_coordinates(RID p_particles) const = 0;

	/* SKELETON API */

	virtual RID skeleton_create() = 0;
	virtual void skeleton_resize(RID p_skeleton, int p_bones) = 0;
	virtual int skeleton_get_bone_count(RID p_skeleton) const = 0;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform &p_transform) = 0;
	virtual Transform skeleton_bone_get_transform(RID p_skeleton, int p_bone) = 0;

	/* LIGHT API */

	virtual RID light_create(VS::LightType p_type) = 0;
	virtual VS::LightType light_get_type(RID p_light) const = 0;

	virtual void light_set_color(RID p_light, VS::LightColor p_type, const Color &p_color) = 0;
	virtual Color light_get_color(RID p_light, VS::LightColor p_type) const = 0;

	virtual void light_set_shadow(RID p_light, bool p_enabled) = 0;
	virtual bool light_has_shadow(RID p_light) const = 0;

	virtual void light_set_volumetric(RID p_light, bool p_enabled) = 0;
	virtual bool light_is_volumetric(RID p_light) const = 0;

	virtual void light_set_projector(RID p_light, RID p_texture) = 0;
	virtual RID light_get_projector(RID p_light) const = 0;

	virtual void light_set_var(RID p_light, VS::LightParam p_var, float p_value) = 0;
	virtual float light_get_var(RID p_light, VS::LightParam p_var) const = 0;

	virtual void light_set_operator(RID p_light, VS::LightOp p_op) = 0;
	virtual VS::LightOp light_get_operator(RID p_light) const = 0;

	virtual void light_omni_set_shadow_mode(RID p_light, VS::LightOmniShadowMode p_mode) = 0;
	virtual VS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) const = 0;

	virtual void light_directional_set_shadow_mode(RID p_light, VS::LightDirectionalShadowMode p_mode) = 0;
	virtual VS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) const = 0;
	virtual void light_directional_set_shadow_param(RID p_light, VS::LightDirectionalShadowParam p_param, float p_value) = 0;
	virtual float light_directional_get_shadow_param(RID p_light, VS::LightDirectionalShadowParam p_param) const = 0;

	virtual AABB light_get_aabb(RID p_poly) const = 0;

	virtual RID light_instance_create(RID p_light) = 0;
	virtual void light_instance_set_transform(RID p_light_instance, const Transform &p_transform) = 0;

	enum ShadowType {
		SHADOW_NONE,
		SHADOW_SIMPLE,
		SHADOW_ORTHOGONAL,
		SHADOW_DUAL_PARABOLOID,
		SHADOW_CUBE,
		SHADOW_PSSM, //parallel split shadow map
		SHADOW_PSM //perspective shadow map
	};

	enum ShadowPass {
		PASS_DUAL_PARABOLOID_FRONT = 0,
		PASS_DUAL_PARABOLOID_BACK = 1,
		PASS_CUBE_FRONT = 0,
		PASS_CUBE_BACK = 1,
		PASS_CUBE_TOP = 2,
		PASS_CUBE_BOTTOM = 3,
		PASS_CUBE_LEFT = 4,
		PASS_CUBE_RIGHT = 5,
	};

	virtual ShadowType light_instance_get_shadow_type(RID p_light_instance, bool p_far = false) const = 0;
	virtual int light_instance_get_shadow_passes(RID p_light_instance) const = 0;
	virtual void light_instance_set_shadow_transform(RID p_light_instance, int p_index, const CameraMatrix &p_camera, const Transform &p_transform, float p_split_near = 0, float p_split_far = 0) = 0;
	virtual int light_instance_get_shadow_size(RID p_light_instance, int p_index = 0) const = 0;
	virtual bool light_instance_get_pssm_shadow_overlap(RID p_light_instance) const = 0;

	/* SHADOWS */

	virtual void shadow_clear_near() = 0;
	virtual bool shadow_allocate_near(RID p_light) = 0; //true on successful alloc
	virtual bool shadow_allocate_far(RID p_light) = 0; //true on successful alloc

	/* PARTICLES INSTANCE */

	virtual RID particles_instance_create(RID p_particles) = 0;
	virtual void particles_instance_set_transform(RID p_particles_instance, const Transform &p_transform) = 0;

	/* RENDER API */
	/* all calls (inside begin/end shadow) are always warranted to be in the following order: */

	/* VIEWPORT API */

	virtual RID viewport_data_create() = 0;

	virtual RID render_target_create() = 0;
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height) = 0;
	virtual RID render_target_get_texture(RID p_render_target) const = 0;
	virtual bool render_target_renedered_in_frame(RID p_render_target) = 0;

	virtual void begin_frame() = 0;

	virtual void set_viewport(const VS::ViewportRect &p_viewport) = 0;
	virtual void set_render_target(RID p_render_target, bool p_transparent_bg = false, bool p_vflip = false) = 0;
	virtual void clear_viewport(const Color &p_color) = 0;
	virtual void capture_viewport(Image *r_capture) = 0;

	virtual void begin_scene(RID p_viewport_data, RID p_env, VS::ScenarioDebugMode p_debug) = 0;
	virtual void begin_shadow_map(RID p_light_instance, int p_shadow_pass) = 0;

	virtual void set_camera(const Transform &p_world, const CameraMatrix &p_projection, bool p_ortho_hint) = 0;

	virtual void add_light(RID p_light_instance) = 0; ///< all "add_light" calls happen before add_geometry calls

	typedef Map<StringName, Variant> ParamOverrideMap;

	struct BakedLightData {

		VS::BakedLightMode mode;
		RID octree_texture;
		RID light_texture;
		float color_multiplier; //used for both lightmaps and octree
		Transform octree_transform;
		Map<int, RID> lightmaps;
		//cache

		float octree_lattice_size;
		float octree_lattice_divide;
		float texture_multiplier;
		float lightmap_multiplier;
		int octree_steps;
		Vector2 octree_tex_pixel_size;
		Vector2 light_tex_pixel_size;

		bool realtime_color_enabled;
		Color realtime_color;
		float realtime_energy;
	};

	struct InstanceData {

		Transform transform;
		RID skeleton;
		RID material_override;
		RID sampled_light;
		Vector<RID> materials;
		Vector<RID> light_instances;
		Vector<float> morph_values;
		BakedLightData *baked_light;
		VS::ShadowCastingSetting cast_shadows;
		Transform *baked_light_octree_xform;
		int baked_lightmap_id;
		bool mirror : 8;
		bool depth_scale : 8;
		bool billboard : 8;
		bool billboard_y : 8;
		bool receive_shadows : 8;
	};

	virtual void add_mesh(const RID &p_mesh, const InstanceData *p_data) = 0;
	virtual void add_multimesh(const RID &p_multimesh, const InstanceData *p_data) = 0;
	virtual void add_immediate(const RID &p_immediate, const InstanceData *p_data) = 0;
	virtual void add_particles(const RID &p_particle_instance, const InstanceData *p_data) = 0;

	virtual void end_scene() = 0;
	virtual void end_shadow_map() = 0;

	virtual void end_frame() = 0;
	virtual void flush_frame(); //not necesary in most cases

	/* CANVAS API */

	enum CanvasRectFlags {

		CANVAS_RECT_REGION = 1,
		CANVAS_RECT_TILE = 2,
		CANVAS_RECT_FLIP_H = 4,
		CANVAS_RECT_FLIP_V = 8,
		CANVAS_RECT_TRANSPOSE = 16
	};

	struct CanvasLight {

		bool enabled;
		Color color;
		Matrix32 xform;
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
		float shadow_esm_mult;
		Color shadow_color;

		void *texture_cache; // implementation dependent
		Rect2 rect_cache;
		Matrix32 xform_cache;
		float radius_cache; //used for shadow far plane
		CameraMatrix shadow_matrix_cache;

		Matrix32 light_shader_xform;
		Vector2 light_shader_pos;

		CanvasLight *shadows_next_ptr;
		CanvasLight *filter_next_ptr;
		CanvasLight *next_ptr;
		CanvasLight *mask_next_ptr;

		CanvasLight() {
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
			shadow_buffer_size = 2048;
			shadow_esm_mult = 80;
		}
	};

	struct CanvasItem;

	struct CanvasItemMaterial {

		RID shader;
		Map<StringName, Variant> shader_param;
		uint32_t shader_version;
		Set<CanvasItem *> owners;
		VS::CanvasItemShadingMode shading_mode;

		CanvasItemMaterial() {
			shading_mode = VS::CANVAS_ITEM_SHADING_NORMAL;
			shader_version = 0;
		}
	};

	struct CanvasItem {

		struct Command {

			enum Type {

				TYPE_LINE,
				TYPE_RECT,
				TYPE_STYLE,
				TYPE_PRIMITIVE,
				TYPE_POLYGON,
				TYPE_POLYGON_PTR,
				TYPE_CIRCLE,
				TYPE_TRANSFORM,
				TYPE_BLEND_MODE,
				TYPE_CLIP_IGNORE,
			};

			Type type;
			virtual ~Command() {}
		};

		struct CommandLine : public Command {

			Point2 from, to;
			Color color;
			float width;
			CommandLine() { type = TYPE_LINE; }
		};

		struct CommandRect : public Command {

			Rect2 rect;
			RID texture;
			Color modulate;
			Rect2 source;
			uint8_t flags;

			CommandRect() {
				flags = 0;
				type = TYPE_RECT;
			}
		};

		struct CommandStyle : public Command {

			Rect2 rect;
			Rect2 source;
			RID texture;
			float margin[4];
			bool draw_center;
			Color color;
			CommandStyle() {
				draw_center = true;
				type = TYPE_STYLE;
			}
		};

		struct CommandPrimitive : public Command {

			Vector<Point2> points;
			Vector<Point2> uvs;
			Vector<Color> colors;
			RID texture;
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
			int count;

			CommandPolygon() {
				type = TYPE_POLYGON;
				count = 0;
			}
		};

		struct CommandPolygonPtr : public Command {

			const int *indices;
			const Point2 *points;
			const Point2 *uvs;
			const Color *colors;
			RID texture;
			int count;

			CommandPolygonPtr() {
				type = TYPE_POLYGON_PTR;
				count = 0;
			}
		};

		struct CommandCircle : public Command {

			Point2 pos;
			float radius;
			Color color;
			CommandCircle() { type = TYPE_CIRCLE; }
		};

		struct CommandTransform : public Command {

			Matrix32 xform;
			CommandTransform() { type = TYPE_TRANSFORM; }
		};

		struct CommandBlendMode : public Command {

			VS::MaterialBlendMode blend_mode;
			CommandBlendMode() {
				type = TYPE_BLEND_MODE;
				blend_mode = VS::MATERIAL_BLEND_MODE_MIX;
			}
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

		Matrix32 xform;
		bool clip;
		bool visible;
		bool ontop;
		VS::MaterialBlendMode blend_mode;
		int light_mask;
		Vector<Command *> commands;
		mutable bool custom_rect;
		mutable bool rect_dirty;
		mutable Rect2 rect;
		CanvasItem *next;
		CanvasItemMaterial *material;
		struct CopyBackBuffer {
			Rect2 rect;
			Rect2 screen_rect;
			bool full;
		};
		CopyBackBuffer *copy_back_buffer;

		float final_opacity;
		Matrix32 final_transform;
		Rect2 final_clip_rect;
		CanvasItem *final_clip_owner;
		CanvasItem *material_owner;
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

			Matrix32 xf;
			bool found_xform = false;
			bool first = true;

			const CanvasItem::Command *const *cmd = &commands[0];

			for (int i = 0; i < s; i++) {

				const CanvasItem::Command *c = cmd[i];
				Rect2 r;

				switch (c->type) {
					case CanvasItem::Command::TYPE_LINE: {

						const CanvasItem::CommandLine *line = static_cast<const CanvasItem::CommandLine *>(c);
						r.pos = line->from;
						r.expand_to(line->to);
					} break;
					case CanvasItem::Command::TYPE_RECT: {

						const CanvasItem::CommandRect *crect = static_cast<const CanvasItem::CommandRect *>(c);
						r = crect->rect;

					} break;
					case CanvasItem::Command::TYPE_STYLE: {

						const CanvasItem::CommandStyle *style = static_cast<const CanvasItem::CommandStyle *>(c);
						r = style->rect;
					} break;
					case CanvasItem::Command::TYPE_PRIMITIVE: {

						const CanvasItem::CommandPrimitive *primitive = static_cast<const CanvasItem::CommandPrimitive *>(c);
						r.pos = primitive->points[0];
						for (int i = 1; i < primitive->points.size(); i++) {

							r.expand_to(primitive->points[i]);
						}
					} break;
					case CanvasItem::Command::TYPE_POLYGON: {

						const CanvasItem::CommandPolygon *polygon = static_cast<const CanvasItem::CommandPolygon *>(c);
						int l = polygon->points.size();
						const Point2 *pp = &polygon->points[0];
						r.pos = pp[0];
						for (int i = 1; i < l; i++) {

							r.expand_to(pp[i]);
						}
					} break;

					case CanvasItem::Command::TYPE_POLYGON_PTR: {

						const CanvasItem::CommandPolygonPtr *polygon = static_cast<const CanvasItem::CommandPolygonPtr *>(c);
						int l = polygon->count;
						if (polygon->indices != NULL) {

							r.pos = polygon->points[polygon->indices[0]];
							for (int i = 1; i < l; i++) {

								r.expand_to(polygon->points[polygon->indices[i]]);
							}
						} else {
							r.pos = polygon->points[0];
							for (int i = 1; i < l; i++) {

								r.expand_to(polygon->points[i]);
							}
						}
					} break;
					case CanvasItem::Command::TYPE_CIRCLE: {

						const CanvasItem::CommandCircle *circle = static_cast<const CanvasItem::CommandCircle *>(c);
						r.pos = Point2(-circle->radius, -circle->radius) + circle->pos;
						r.size = Point2(circle->radius * 2.0, circle->radius * 2.0);
					} break;
					case CanvasItem::Command::TYPE_TRANSFORM: {

						const CanvasItem::CommandTransform *transform = static_cast<const CanvasItem::CommandTransform *>(c);
						xf = transform->xform;
						found_xform = true;
						continue;
					} break;
					case CanvasItem::Command::TYPE_BLEND_MODE: {

					} break;
					case CanvasItem::Command::TYPE_CLIP_IGNORE: {

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
		CanvasItem() {
			light_mask = 1;
			vp_render = NULL;
			next = NULL;
			final_clip_owner = NULL;
			clip = false;
			final_opacity = 1;
			blend_mode = VS::MATERIAL_BLEND_MODE_MIX;
			visible = true;
			rect_dirty = true;
			custom_rect = false;
			ontop = true;
			material_owner = NULL;
			material = NULL;
			copy_back_buffer = NULL;
			distance_field = false;
			light_masked = false;
		}
		virtual ~CanvasItem() {
			clear();
			if (copy_back_buffer) memdelete(copy_back_buffer);
		}
	};

	CanvasItemDrawViewportFunc draw_viewport_func;

	virtual void begin_canvas_bg() = 0;
	virtual void canvas_begin() = 0;
	virtual void canvas_disable_blending() = 0;
	virtual void canvas_set_opacity(float p_opacity) = 0;
	virtual void canvas_set_blend_mode(VS::MaterialBlendMode p_mode) = 0;
	virtual void canvas_begin_rect(const Matrix32 &p_transform) = 0;
	virtual void canvas_set_clip(bool p_clip, const Rect2 &p_rect) = 0;
	virtual void canvas_end_rect() = 0;
	virtual void canvas_draw_line(const Point2 &p_from, const Point2 &p_to, const Color &p_color, float p_width) = 0;
	virtual void canvas_draw_rect(const Rect2 &p_rect, int p_flags, const Rect2 &p_source, RID p_texture, const Color &p_modulate) = 0;
	virtual void canvas_draw_style_box(const Rect2 &p_rect, const Rect2 &p_src_region, RID p_texture, const float *p_margins, bool p_draw_center = true, const Color &p_modulate = Color(1, 1, 1)) = 0;
	virtual void canvas_draw_primitive(const Vector<Point2> &p_points, const Vector<Color> &p_colors, const Vector<Point2> &p_uvs, RID p_texture, float p_width) = 0;
	virtual void canvas_draw_polygon(int p_vertex_count, const int *p_indices, const Vector2 *p_vertices, const Vector2 *p_uvs, const Color *p_colors, const RID &p_texture, bool p_singlecolor) = 0;
	virtual void canvas_set_transform(const Matrix32 &p_transform) = 0;

	virtual void canvas_render_items(CanvasItem *p_item_list, int p_z, const Color &p_modulate, CanvasLight *p_light) = 0;
	virtual void canvas_debug_viewport_shadows(CanvasLight *p_lights_with_shadow) = 0;
	/* LIGHT SHADOW MAPPING */

	virtual RID canvas_light_occluder_create() = 0;
	virtual void canvas_light_occluder_set_polylines(RID p_occluder, const DVector<Vector2> &p_lines) = 0;

	virtual RID canvas_light_shadow_buffer_create(int p_width) = 0;

	struct CanvasLightOccluderInstance {

		bool enabled;
		RID canvas;
		RID polygon;
		RID polygon_buffer;
		Rect2 aabb_cache;
		Matrix32 xform;
		Matrix32 xform_cache;
		int light_mask;
		VS::CanvasOccluderPolygonCullMode cull_cache;

		CanvasLightOccluderInstance *next;

		CanvasLightOccluderInstance() {
			enabled = true;
			next = NULL;
			light_mask = 1;
			cull_cache = VS::CANVAS_OCCLUDER_POLYGON_CULL_DISABLED;
		}
	};

	virtual void canvas_light_shadow_buffer_update(RID p_buffer, const Matrix32 &p_light_xform, int p_light_mask, float p_near, float p_far, CanvasLightOccluderInstance *p_occluders, CameraMatrix *p_xform_cache) = 0;

	/* ENVIRONMENT */

	virtual RID environment_create() = 0;

	virtual void environment_set_background(RID p_env, VS::EnvironmentBG p_bg) = 0;
	virtual VS::EnvironmentBG environment_get_background(RID p_env) const = 0;

	virtual void environment_set_background_param(RID p_env, VS::EnvironmentBGParam p_param, const Variant &p_value) = 0;
	virtual Variant environment_get_background_param(RID p_env, VS::EnvironmentBGParam p_param) const = 0;

	virtual void environment_set_enable_fx(RID p_env, VS::EnvironmentFx p_effect, bool p_enabled) = 0;
	virtual bool environment_is_fx_enabled(RID p_env, VS::EnvironmentFx p_effect) const = 0;

	virtual void environment_fx_set_param(RID p_env, VS::EnvironmentFxParam p_param, const Variant &p_value) = 0;
	virtual Variant environment_fx_get_param(RID p_env, VS::EnvironmentFxParam p_param) const = 0;

	/* SAMPLED LIGHT */
	virtual RID sampled_light_dp_create(int p_width, int p_height) = 0;
	virtual void sampled_light_dp_update(RID p_sampled_light, const Color *p_data, float p_multiplier) = 0;

	/*MISC*/

	virtual bool is_texture(const RID &p_rid) const = 0;
	virtual bool is_material(const RID &p_rid) const = 0;
	virtual bool is_mesh(const RID &p_rid) const = 0;
	virtual bool is_multimesh(const RID &p_rid) const = 0;
	virtual bool is_immediate(const RID &p_rid) const = 0;
	virtual bool is_particles(const RID &p_beam) const = 0;

	virtual bool is_light(const RID &p_rid) const = 0;
	virtual bool is_light_instance(const RID &p_rid) const = 0;
	virtual bool is_particles_instance(const RID &p_rid) const = 0;
	virtual bool is_skeleton(const RID &p_rid) const = 0;
	virtual bool is_environment(const RID &p_rid) const = 0;
	virtual bool is_shader(const RID &p_rid) const = 0;

	virtual bool is_canvas_light_occluder(const RID &p_rid) const = 0;

	virtual void set_time_scale(float p_scale) = 0;

	virtual void free(const RID &p_rid) = 0;

	virtual void init() = 0;
	virtual void finish() = 0;

	virtual bool needs_to_draw_next_frame() const = 0;

	virtual void reload_vram() {}

	virtual bool has_feature(VS::Features p_feature) const = 0;

	virtual void restore_framebuffer() = 0;

	virtual int get_render_info(VS::RenderInfo p_info) = 0;

	virtual void set_force_16_bits_fbo(bool p_force) {}

	Rasterizer();
	virtual ~Rasterizer() {}
};

#endif
