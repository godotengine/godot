/*************************************************************************/
/*  renderer_storage.h                                                   */
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

#ifndef RENDERINGSERVERSTORAGE_H
#define RENDERINGSERVERSTORAGE_H

#include "servers/rendering_server.h"

class RendererStorage {
	Color default_clear_color;

public:
	enum DependencyChangedNotification {
		DEPENDENCY_CHANGED_AABB,
		DEPENDENCY_CHANGED_MATERIAL,
		DEPENDENCY_CHANGED_MESH,
		DEPENDENCY_CHANGED_MULTIMESH,
		DEPENDENCY_CHANGED_MULTIMESH_VISIBLE_INSTANCES,
		DEPENDENCY_CHANGED_PARTICLES,
		DEPENDENCY_CHANGED_DECAL,
		DEPENDENCY_CHANGED_SKELETON_DATA,
		DEPENDENCY_CHANGED_SKELETON_BONES,
		DEPENDENCY_CHANGED_LIGHT,
		DEPENDENCY_CHANGED_LIGHT_SOFT_SHADOW_AND_PROJECTOR,
		DEPENDENCY_CHANGED_REFLECTION_PROBE,
	};

	struct DependencyTracker;

protected:
	struct Dependency {
		void changed_notify(DependencyChangedNotification p_notification);
		void deleted_notify(const RID &p_rid);

		~Dependency();

	private:
		friend struct DependencyTracker;
		Map<DependencyTracker *, uint32_t> instances;
	};

public:
	struct DependencyTracker {
		void *userdata = nullptr;
		typedef void (*ChangedCallback)(DependencyChangedNotification, DependencyTracker *);
		typedef void (*DeletedCallback)(const RID &, DependencyTracker *);

		ChangedCallback changed_callback = nullptr;
		DeletedCallback deleted_callback = nullptr;

		void update_begin() { // call before updating dependencies
			instance_version++;
		}

		void update_dependency(Dependency *p_dependency) { //called internally, can't be used directly, use update functions in Storage
			dependencies.insert(p_dependency);
			p_dependency->instances[this] = instance_version;
		}

		void update_end() { //call after updating dependencies
			List<Pair<Dependency *, Map<DependencyTracker *, uint32_t>::Element *>> to_clean_up;
			for (Set<Dependency *>::Element *E = dependencies.front(); E; E = E->next()) {
				Dependency *dep = E->get();
				Map<DependencyTracker *, uint32_t>::Element *F = dep->instances.find(this);
				ERR_CONTINUE(!F);
				if (F->get() != instance_version) {
					Pair<Dependency *, Map<DependencyTracker *, uint32_t>::Element *> p;
					p.first = dep;
					p.second = F;
					to_clean_up.push_back(p);
				}
			}

			while (to_clean_up.size()) {
				to_clean_up.front()->get().first->instances.erase(to_clean_up.front()->get().second);
				dependencies.erase(to_clean_up.front()->get().first);
				to_clean_up.pop_front();
			}
		}

		void clear() { // clear all dependencies
			for (Set<Dependency *>::Element *E = dependencies.front(); E; E = E->next()) {
				Dependency *dep = E->get();
				dep->instances.erase(this);
			}
			dependencies.clear();
		}

		~DependencyTracker() { clear(); }

	private:
		friend struct Dependency;
		uint32_t instance_version = 0;
		Set<Dependency *> dependencies;
	};

	virtual bool can_create_resources_async() const = 0;
	/* TEXTURE API */

	virtual RID texture_allocate() = 0;

	virtual void texture_2d_initialize(RID p_texture, const Ref<Image> &p_image) = 0;
	virtual void texture_2d_layered_initialize(RID p_texture, const Vector<Ref<Image>> &p_layers, RS::TextureLayeredType p_layered_type) = 0;
	virtual void texture_3d_initialize(RID p_texture, Image::Format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) = 0;
	virtual void texture_proxy_initialize(RID p_texture, RID p_base) = 0; //all slices, then all the mipmaps, must be coherent

	virtual void texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer = 0) = 0;
	virtual void texture_3d_update(RID p_texture, const Vector<Ref<Image>> &p_data) = 0;
	virtual void texture_proxy_update(RID p_proxy, RID p_base) = 0;

	//these two APIs can be used together or in combination with the others.
	virtual void texture_2d_placeholder_initialize(RID p_texture) = 0;
	virtual void texture_2d_layered_placeholder_initialize(RID p_texture, RenderingServer::TextureLayeredType p_layered_type) = 0;
	virtual void texture_3d_placeholder_initialize(RID p_texture) = 0;

	virtual Ref<Image> texture_2d_get(RID p_texture) const = 0;
	virtual Ref<Image> texture_2d_layer_get(RID p_texture, int p_layer) const = 0;
	virtual Vector<Ref<Image>> texture_3d_get(RID p_texture) const = 0;

	virtual void texture_replace(RID p_texture, RID p_by_texture) = 0;
	virtual void texture_set_size_override(RID p_texture, int p_width, int p_height) = 0;

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

	/* CANVAS TEXTURE API */

	virtual RID canvas_texture_allocate() = 0;
	virtual void canvas_texture_initialize(RID p_rid) = 0;

	virtual void canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) = 0;
	virtual void canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_base_color, float p_shininess) = 0;

	virtual void canvas_texture_set_texture_filter(RID p_item, RS::CanvasItemTextureFilter p_filter) = 0;
	virtual void canvas_texture_set_texture_repeat(RID p_item, RS::CanvasItemTextureRepeat p_repeat) = 0;

	/* SHADER API */

	virtual RID shader_allocate() = 0;
	virtual void shader_initialize(RID p_rid) = 0;

	virtual void shader_set_code(RID p_shader, const String &p_code) = 0;
	virtual String shader_get_code(RID p_shader) const = 0;
	virtual void shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const = 0;

	virtual void shader_set_default_texture_param(RID p_shader, const StringName &p_name, RID p_texture, int p_index) = 0;
	virtual RID shader_get_default_texture_param(RID p_shader, const StringName &p_name, int p_index) const = 0;
	virtual Variant shader_get_param_default(RID p_material, const StringName &p_param) const = 0;

	virtual RS::ShaderNativeSourceCode shader_get_native_source_code(RID p_shader) const = 0;

	/* COMMON MATERIAL API */

	virtual RID material_allocate() = 0;
	virtual void material_initialize(RID p_rid) = 0;

	virtual void material_set_render_priority(RID p_material, int priority) = 0;
	virtual void material_set_shader(RID p_shader_material, RID p_shader) = 0;

	virtual void material_set_param(RID p_material, const StringName &p_param, const Variant &p_value) = 0;
	virtual Variant material_get_param(RID p_material, const StringName &p_param) const = 0;

	virtual void material_set_next_pass(RID p_material, RID p_next_material) = 0;

	virtual bool material_is_animated(RID p_material) = 0;
	virtual bool material_casts_shadows(RID p_material) = 0;

	struct InstanceShaderParam {
		PropertyInfo info;
		int index;
		Variant default_value;
	};

	virtual void material_get_instance_shader_parameters(RID p_material, List<InstanceShaderParam> *r_parameters) = 0;

	virtual void material_update_dependency(RID p_material, DependencyTracker *p_instance) = 0;

	/* MESH API */

	virtual RID mesh_allocate() = 0;
	virtual void mesh_initialize(RID p_rid) = 0;

	virtual void mesh_set_blend_shape_count(RID p_mesh, int p_blend_shape_count) = 0;

	/// Returns stride
	virtual void mesh_add_surface(RID p_mesh, const RS::SurfaceData &p_surface) = 0;

	virtual int mesh_get_blend_shape_count(RID p_mesh) const = 0;

	virtual void mesh_set_blend_shape_mode(RID p_mesh, RS::BlendShapeMode p_mode) = 0;
	virtual RS::BlendShapeMode mesh_get_blend_shape_mode(RID p_mesh) const = 0;

	virtual void mesh_surface_update_vertex_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) = 0;
	virtual void mesh_surface_update_attribute_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) = 0;
	virtual void mesh_surface_update_skin_region(RID p_mesh, int p_surface, int p_offset, const Vector<uint8_t> &p_data) = 0;

	virtual void mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material) = 0;
	virtual RID mesh_surface_get_material(RID p_mesh, int p_surface) const = 0;

	virtual RS::SurfaceData mesh_get_surface(RID p_mesh, int p_surface) const = 0;

	virtual int mesh_get_surface_count(RID p_mesh) const = 0;

	virtual void mesh_set_custom_aabb(RID p_mesh, const AABB &p_aabb) = 0;
	virtual AABB mesh_get_custom_aabb(RID p_mesh) const = 0;

	virtual AABB mesh_get_aabb(RID p_mesh, RID p_skeleton = RID()) = 0;

	virtual void mesh_set_shadow_mesh(RID p_mesh, RID p_shadow_mesh) = 0;

	virtual void mesh_clear(RID p_mesh) = 0;

	virtual bool mesh_needs_instance(RID p_mesh, bool p_has_skeleton) = 0;

	/* MESH INSTANCE */

	virtual RID mesh_instance_create(RID p_base) = 0;
	virtual void mesh_instance_set_skeleton(RID p_mesh_instance, RID p_skeleton) = 0;
	virtual void mesh_instance_set_blend_shape_weight(RID p_mesh_instance, int p_shape, float p_weight) = 0;
	virtual void mesh_instance_check_for_update(RID p_mesh_instance) = 0;
	virtual void update_mesh_instances() = 0;

	/* MULTIMESH API */

	virtual RID multimesh_allocate() = 0;
	virtual void multimesh_initialize(RID p_rid) = 0;

	virtual void multimesh_allocate_data(RID p_multimesh, int p_instances, RS::MultimeshTransformFormat p_transform_format, bool p_use_colors = false, bool p_use_custom_data = false) = 0;

	virtual int multimesh_get_instance_count(RID p_multimesh) const = 0;

	virtual void multimesh_set_mesh(RID p_multimesh, RID p_mesh) = 0;
	virtual void multimesh_instance_set_transform(RID p_multimesh, int p_index, const Transform3D &p_transform) = 0;
	virtual void multimesh_instance_set_transform_2d(RID p_multimesh, int p_index, const Transform2D &p_transform) = 0;
	virtual void multimesh_instance_set_color(RID p_multimesh, int p_index, const Color &p_color) = 0;
	virtual void multimesh_instance_set_custom_data(RID p_multimesh, int p_index, const Color &p_color) = 0;

	virtual RID multimesh_get_mesh(RID p_multimesh) const = 0;

	virtual Transform3D multimesh_instance_get_transform(RID p_multimesh, int p_index) const = 0;
	virtual Transform2D multimesh_instance_get_transform_2d(RID p_multimesh, int p_index) const = 0;
	virtual Color multimesh_instance_get_color(RID p_multimesh, int p_index) const = 0;
	virtual Color multimesh_instance_get_custom_data(RID p_multimesh, int p_index) const = 0;

	virtual void multimesh_set_buffer(RID p_multimesh, const Vector<float> &p_buffer) = 0;
	virtual Vector<float> multimesh_get_buffer(RID p_multimesh) const = 0;

	virtual void multimesh_set_visible_instances(RID p_multimesh, int p_visible) = 0;
	virtual int multimesh_get_visible_instances(RID p_multimesh) const = 0;

	virtual AABB multimesh_get_aabb(RID p_multimesh) const = 0;

	/* SKELETON API */

	virtual RID skeleton_allocate() = 0;
	virtual void skeleton_initialize(RID p_rid) = 0;

	virtual void skeleton_allocate_data(RID p_skeleton, int p_bones, bool p_2d_skeleton = false) = 0;
	virtual int skeleton_get_bone_count(RID p_skeleton) const = 0;
	virtual void skeleton_bone_set_transform(RID p_skeleton, int p_bone, const Transform3D &p_transform) = 0;
	virtual Transform3D skeleton_bone_get_transform(RID p_skeleton, int p_bone) const = 0;
	virtual void skeleton_bone_set_transform_2d(RID p_skeleton, int p_bone, const Transform2D &p_transform) = 0;
	virtual Transform2D skeleton_bone_get_transform_2d(RID p_skeleton, int p_bone) const = 0;
	virtual void skeleton_set_base_transform_2d(RID p_skeleton, const Transform2D &p_base_transform) = 0;

	/* Light API */

	virtual RID directional_light_allocate() = 0;
	virtual void directional_light_initialize(RID p_rid) = 0;

	virtual RID omni_light_allocate() = 0;
	virtual void omni_light_initialize(RID p_rid) = 0;

	virtual RID spot_light_allocate() = 0;
	virtual void spot_light_initialize(RID p_rid) = 0;

	virtual void light_set_color(RID p_light, const Color &p_color) = 0;
	virtual void light_set_param(RID p_light, RS::LightParam p_param, float p_value) = 0;
	virtual void light_set_shadow(RID p_light, bool p_enabled) = 0;
	virtual void light_set_shadow_color(RID p_light, const Color &p_color) = 0;
	virtual void light_set_projector(RID p_light, RID p_texture) = 0;
	virtual void light_set_negative(RID p_light, bool p_enable) = 0;
	virtual void light_set_cull_mask(RID p_light, uint32_t p_mask) = 0;
	virtual void light_set_reverse_cull_face_mode(RID p_light, bool p_enabled) = 0;
	virtual void light_set_bake_mode(RID p_light, RS::LightBakeMode p_bake_mode) = 0;
	virtual void light_set_max_sdfgi_cascade(RID p_light, uint32_t p_cascade) = 0;

	virtual void light_omni_set_shadow_mode(RID p_light, RS::LightOmniShadowMode p_mode) = 0;

	virtual void light_directional_set_shadow_mode(RID p_light, RS::LightDirectionalShadowMode p_mode) = 0;
	virtual void light_directional_set_blend_splits(RID p_light, bool p_enable) = 0;
	virtual bool light_directional_get_blend_splits(RID p_light) const = 0;
	virtual void light_directional_set_sky_only(RID p_light, bool p_sky_only) = 0;
	virtual bool light_directional_is_sky_only(RID p_light) const = 0;

	virtual RS::LightDirectionalShadowMode light_directional_get_shadow_mode(RID p_light) = 0;
	virtual RS::LightOmniShadowMode light_omni_get_shadow_mode(RID p_light) = 0;

	virtual bool light_has_shadow(RID p_light) const = 0;

	virtual bool light_has_projector(RID p_light) const = 0;

	virtual RS::LightType light_get_type(RID p_light) const = 0;
	virtual AABB light_get_aabb(RID p_light) const = 0;
	virtual float light_get_param(RID p_light, RS::LightParam p_param) = 0;
	virtual Color light_get_color(RID p_light) = 0;
	virtual RS::LightBakeMode light_get_bake_mode(RID p_light) = 0;
	virtual uint32_t light_get_max_sdfgi_cascade(RID p_light) = 0;
	virtual uint64_t light_get_version(RID p_light) const = 0;

	/* PROBE API */

	virtual RID reflection_probe_allocate() = 0;
	virtual void reflection_probe_initialize(RID p_rid) = 0;

	virtual void reflection_probe_set_update_mode(RID p_probe, RS::ReflectionProbeUpdateMode p_mode) = 0;
	virtual void reflection_probe_set_resolution(RID p_probe, int p_resolution) = 0;
	virtual void reflection_probe_set_intensity(RID p_probe, float p_intensity) = 0;
	virtual void reflection_probe_set_ambient_mode(RID p_probe, RS::ReflectionProbeAmbientMode p_mode) = 0;
	virtual void reflection_probe_set_ambient_color(RID p_probe, const Color &p_color) = 0;
	virtual void reflection_probe_set_ambient_energy(RID p_probe, float p_energy) = 0;
	virtual void reflection_probe_set_max_distance(RID p_probe, float p_distance) = 0;
	virtual void reflection_probe_set_extents(RID p_probe, const Vector3 &p_extents) = 0;
	virtual void reflection_probe_set_origin_offset(RID p_probe, const Vector3 &p_offset) = 0;
	virtual void reflection_probe_set_as_interior(RID p_probe, bool p_enable) = 0;
	virtual void reflection_probe_set_enable_box_projection(RID p_probe, bool p_enable) = 0;
	virtual void reflection_probe_set_enable_shadows(RID p_probe, bool p_enable) = 0;
	virtual void reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers) = 0;
	virtual void reflection_probe_set_mesh_lod_threshold(RID p_probe, float p_ratio) = 0;

	virtual AABB reflection_probe_get_aabb(RID p_probe) const = 0;
	virtual RS::ReflectionProbeUpdateMode reflection_probe_get_update_mode(RID p_probe) const = 0;
	virtual uint32_t reflection_probe_get_cull_mask(RID p_probe) const = 0;
	virtual Vector3 reflection_probe_get_extents(RID p_probe) const = 0;
	virtual Vector3 reflection_probe_get_origin_offset(RID p_probe) const = 0;
	virtual float reflection_probe_get_origin_max_distance(RID p_probe) const = 0;
	virtual bool reflection_probe_renders_shadows(RID p_probe) const = 0;
	virtual float reflection_probe_get_mesh_lod_threshold(RID p_probe) const = 0;

	virtual void base_update_dependency(RID p_base, DependencyTracker *p_instance) = 0;
	virtual void skeleton_update_dependency(RID p_base, DependencyTracker *p_instance) = 0;

	/* DECAL API */

	virtual RID decal_allocate() = 0;
	virtual void decal_initialize(RID p_rid) = 0;

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

	/* VOXEL GI API */

	virtual RID voxel_gi_allocate() = 0;
	virtual void voxel_gi_initialize(RID p_rid) = 0;

	virtual void voxel_gi_allocate_data(RID p_voxel_gi, const Transform3D &p_to_cell_xform, const AABB &p_aabb, const Vector3i &p_octree_size, const Vector<uint8_t> &p_octree_cells, const Vector<uint8_t> &p_data_cells, const Vector<uint8_t> &p_distance_field, const Vector<int> &p_level_counts) = 0;

	virtual AABB voxel_gi_get_bounds(RID p_voxel_gi) const = 0;
	virtual Vector3i voxel_gi_get_octree_size(RID p_voxel_gi) const = 0;
	virtual Vector<uint8_t> voxel_gi_get_octree_cells(RID p_voxel_gi) const = 0;
	virtual Vector<uint8_t> voxel_gi_get_data_cells(RID p_voxel_gi) const = 0;
	virtual Vector<uint8_t> voxel_gi_get_distance_field(RID p_voxel_gi) const = 0;

	virtual Vector<int> voxel_gi_get_level_counts(RID p_voxel_gi) const = 0;
	virtual Transform3D voxel_gi_get_to_cell_xform(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_dynamic_range(RID p_voxel_gi, float p_range) = 0;
	virtual float voxel_gi_get_dynamic_range(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_propagation(RID p_voxel_gi, float p_range) = 0;
	virtual float voxel_gi_get_propagation(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_energy(RID p_voxel_gi, float p_energy) = 0;
	virtual float voxel_gi_get_energy(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_bias(RID p_voxel_gi, float p_bias) = 0;
	virtual float voxel_gi_get_bias(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_normal_bias(RID p_voxel_gi, float p_range) = 0;
	virtual float voxel_gi_get_normal_bias(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_interior(RID p_voxel_gi, bool p_enable) = 0;
	virtual bool voxel_gi_is_interior(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_use_two_bounces(RID p_voxel_gi, bool p_enable) = 0;
	virtual bool voxel_gi_is_using_two_bounces(RID p_voxel_gi) const = 0;

	virtual void voxel_gi_set_anisotropy_strength(RID p_voxel_gi, float p_strength) = 0;
	virtual float voxel_gi_get_anisotropy_strength(RID p_voxel_gi) const = 0;

	virtual uint32_t voxel_gi_get_version(RID p_probe) = 0;

	/* LIGHTMAP  */

	virtual RID lightmap_allocate() = 0;
	virtual void lightmap_initialize(RID p_rid) = 0;

	virtual void lightmap_set_textures(RID p_lightmap, RID p_light, bool p_uses_spherical_haromics) = 0;
	virtual void lightmap_set_probe_bounds(RID p_lightmap, const AABB &p_bounds) = 0;
	virtual void lightmap_set_probe_interior(RID p_lightmap, bool p_interior) = 0;
	virtual void lightmap_set_probe_capture_data(RID p_lightmap, const PackedVector3Array &p_points, const PackedColorArray &p_point_sh, const PackedInt32Array &p_tetrahedra, const PackedInt32Array &p_bsp_tree) = 0;
	virtual PackedVector3Array lightmap_get_probe_capture_points(RID p_lightmap) const = 0;
	virtual PackedColorArray lightmap_get_probe_capture_sh(RID p_lightmap) const = 0;
	virtual PackedInt32Array lightmap_get_probe_capture_tetrahedra(RID p_lightmap) const = 0;
	virtual PackedInt32Array lightmap_get_probe_capture_bsp_tree(RID p_lightmap) const = 0;
	virtual AABB lightmap_get_aabb(RID p_lightmap) const = 0;
	virtual void lightmap_tap_sh_light(RID p_lightmap, const Vector3 &p_point, Color *r_sh) = 0;
	virtual bool lightmap_is_interior(RID p_lightmap) const = 0;
	virtual void lightmap_set_probe_capture_update_speed(float p_speed) = 0;
	virtual float lightmap_get_probe_capture_update_speed() const = 0;

	/* PARTICLES */

	virtual RID particles_allocate() = 0;
	virtual void particles_initialize(RID p_rid) = 0;
	virtual void particles_set_mode(RID p_particles, RS::ParticlesMode p_mode) = 0;

	virtual void particles_set_emitting(RID p_particles, bool p_emitting) = 0;
	virtual bool particles_get_emitting(RID p_particles) = 0;

	virtual void particles_set_amount(RID p_particles, int p_amount) = 0;
	virtual void particles_set_lifetime(RID p_particles, double p_lifetime) = 0;
	virtual void particles_set_one_shot(RID p_particles, bool p_one_shot) = 0;
	virtual void particles_set_pre_process_time(RID p_particles, double p_time) = 0;
	virtual void particles_set_explosiveness_ratio(RID p_particles, real_t p_ratio) = 0;
	virtual void particles_set_randomness_ratio(RID p_particles, real_t p_ratio) = 0;
	virtual void particles_set_custom_aabb(RID p_particles, const AABB &p_aabb) = 0;
	virtual void particles_set_speed_scale(RID p_particles, double p_scale) = 0;
	virtual void particles_set_use_local_coordinates(RID p_particles, bool p_enable) = 0;
	virtual void particles_set_process_material(RID p_particles, RID p_material) = 0;
	virtual void particles_set_fixed_fps(RID p_particles, int p_fps) = 0;
	virtual void particles_set_interpolate(RID p_particles, bool p_enable) = 0;
	virtual void particles_set_fractional_delta(RID p_particles, bool p_enable) = 0;
	virtual void particles_set_collision_base_size(RID p_particles, real_t p_size) = 0;

	virtual void particles_set_transform_align(RID p_particles, RS::ParticlesTransformAlign p_transform_align) = 0;

	virtual void particles_set_trails(RID p_particles, bool p_enable, double p_length) = 0;
	virtual void particles_set_trail_bind_poses(RID p_particles, const Vector<Transform3D> &p_bind_poses) = 0;

	virtual void particles_restart(RID p_particles) = 0;
	virtual void particles_emit(RID p_particles, const Transform3D &p_transform, const Vector3 &p_velocity, const Color &p_color, const Color &p_custom, uint32_t p_emit_flags) = 0;
	virtual void particles_set_subemitter(RID p_particles, RID p_subemitter_particles) = 0;

	virtual bool particles_is_inactive(RID p_particles) const = 0;

	virtual void particles_set_draw_order(RID p_particles, RS::ParticlesDrawOrder p_order) = 0;

	virtual void particles_set_draw_passes(RID p_particles, int p_count) = 0;
	virtual void particles_set_draw_pass_mesh(RID p_particles, int p_pass, RID p_mesh) = 0;

	virtual void particles_request_process(RID p_particles) = 0;
	virtual AABB particles_get_current_aabb(RID p_particles) = 0;
	virtual AABB particles_get_aabb(RID p_particles) const = 0;

	virtual void particles_set_emission_transform(RID p_particles, const Transform3D &p_transform) = 0;

	virtual int particles_get_draw_passes(RID p_particles) const = 0;
	virtual RID particles_get_draw_pass_mesh(RID p_particles, int p_pass) const = 0;

	virtual void particles_set_view_axis(RID p_particles, const Vector3 &p_axis, const Vector3 &p_up_axis) = 0;

	virtual void particles_add_collision(RID p_particles, RID p_particles_collision_instance) = 0;
	virtual void particles_remove_collision(RID p_particles, RID p_particles_collision_instance) = 0;

	virtual void particles_set_canvas_sdf_collision(RID p_particles, bool p_enable, const Transform2D &p_xform, const Rect2 &p_to_screen, RID p_texture) = 0;

	virtual void update_particles() = 0;

	/* PARTICLES COLLISION */

	virtual RID particles_collision_allocate() = 0;
	virtual void particles_collision_initialize(RID p_rid) = 0;

	virtual void particles_collision_set_collision_type(RID p_particles_collision, RS::ParticlesCollisionType p_type) = 0;
	virtual void particles_collision_set_cull_mask(RID p_particles_collision, uint32_t p_cull_mask) = 0;
	virtual void particles_collision_set_sphere_radius(RID p_particles_collision, real_t p_radius) = 0; //for spheres
	virtual void particles_collision_set_box_extents(RID p_particles_collision, const Vector3 &p_extents) = 0; //for non-spheres
	virtual void particles_collision_set_attractor_strength(RID p_particles_collision, real_t p_strength) = 0;
	virtual void particles_collision_set_attractor_directionality(RID p_particles_collision, real_t p_directionality) = 0;
	virtual void particles_collision_set_attractor_attenuation(RID p_particles_collision, real_t p_curve) = 0;
	virtual void particles_collision_set_field_texture(RID p_particles_collision, RID p_texture) = 0; //for SDF and vector field, heightfield is dynamic
	virtual void particles_collision_height_field_update(RID p_particles_collision) = 0; //for SDF and vector field
	virtual void particles_collision_set_height_field_resolution(RID p_particles_collision, RS::ParticlesCollisionHeightfieldResolution p_resolution) = 0; //for SDF and vector field
	virtual AABB particles_collision_get_aabb(RID p_particles_collision) const = 0;
	virtual bool particles_collision_is_heightfield(RID p_particles_collision) const = 0;
	virtual RID particles_collision_get_heightfield_framebuffer(RID p_particles_collision) const = 0;

	/* FOG VOLUMES */

	virtual RID fog_volume_allocate() = 0;
	virtual void fog_volume_initialize(RID p_rid) = 0;

	virtual void fog_volume_set_shape(RID p_fog_volume, RS::FogVolumeShape p_shape) = 0;
	virtual void fog_volume_set_extents(RID p_fog_volume, const Vector3 &p_extents) = 0;
	virtual void fog_volume_set_material(RID p_fog_volume, RID p_material) = 0;
	virtual AABB fog_volume_get_aabb(RID p_fog_volume) const = 0;
	virtual RS::FogVolumeShape fog_volume_get_shape(RID p_fog_volume) const = 0;

	/* VISIBILITY NOTIFIER */

	virtual RID visibility_notifier_allocate() = 0;
	virtual void visibility_notifier_initialize(RID p_notifier) = 0;
	virtual void visibility_notifier_set_aabb(RID p_notifier, const AABB &p_aabb) = 0;
	virtual void visibility_notifier_set_callbacks(RID p_notifier, const Callable &p_enter_callbable, const Callable &p_exit_callable) = 0;

	virtual AABB visibility_notifier_get_aabb(RID p_notifier) const = 0;
	virtual void visibility_notifier_call(RID p_notifier, bool p_enter, bool p_deferred) = 0;

	//used from 2D and 3D
	virtual RID particles_collision_instance_create(RID p_collision) = 0;
	virtual void particles_collision_instance_set_transform(RID p_collision_instance, const Transform3D &p_transform) = 0;
	virtual void particles_collision_instance_set_active(RID p_collision_instance, bool p_active) = 0;

	/* GLOBAL VARIABLES */

	virtual void global_variable_add(const StringName &p_name, RS::GlobalVariableType p_type, const Variant &p_value) = 0;
	virtual void global_variable_remove(const StringName &p_name) = 0;
	virtual Vector<StringName> global_variable_get_list() const = 0;

	virtual void global_variable_set(const StringName &p_name, const Variant &p_value) = 0;
	virtual void global_variable_set_override(const StringName &p_name, const Variant &p_value) = 0;
	virtual Variant global_variable_get(const StringName &p_name) const = 0;
	virtual RS::GlobalVariableType global_variable_get_type(const StringName &p_name) const = 0;

	virtual void global_variables_load_settings(bool p_load_textures = true) = 0;
	virtual void global_variables_clear() = 0;

	virtual int32_t global_variables_instance_allocate(RID p_instance) = 0;
	virtual void global_variables_instance_free(RID p_instance) = 0;
	virtual void global_variables_instance_update(RID p_instance, int p_index, const Variant &p_value) = 0;

	/* RENDER TARGET */

	enum RenderTargetFlags {
		RENDER_TARGET_TRANSPARENT,
		RENDER_TARGET_DIRECT_TO_SCREEN,
		RENDER_TARGET_FLAG_MAX
	};

	virtual RID render_target_create() = 0;
	virtual void render_target_set_position(RID p_render_target, int p_x, int p_y) = 0;
	virtual void render_target_set_size(RID p_render_target, int p_width, int p_height, uint32_t p_view_count) = 0;
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

	virtual void render_target_set_sdf_size_and_scale(RID p_render_target, RS::ViewportSDFOversize p_size, RS::ViewportSDFScale p_scale) = 0;
	virtual Rect2i render_target_get_sdf_rect(RID p_render_target) const = 0;
	virtual void render_target_mark_sdf_enabled(RID p_render_target, bool p_enabled) = 0;

	virtual RS::InstanceType get_base_type(RID p_rid) const = 0;
	virtual bool free(RID p_rid) = 0;

	virtual bool has_os_feature(const String &p_feature) const = 0;

	virtual void update_dirty_resources() = 0;

	virtual void set_debug_generate_wireframes(bool p_generate) = 0;

	virtual void update_memory_info() = 0;

	virtual uint64_t get_rendering_info(RS::RenderingInfo p_info) = 0;
	virtual String get_video_adapter_name() const = 0;
	virtual String get_video_adapter_vendor() const = 0;
	virtual RenderingDevice::DeviceType get_video_adapter_type() const = 0;

	static RendererStorage *base_singleton;

	void set_default_clear_color(const Color &p_color) {
		default_clear_color = p_color;
	}

	Color get_default_clear_color() const {
		return default_clear_color;
	}
#define TIMESTAMP_BEGIN()                             \
	{                                                 \
		if (RSG::storage->capturing_timestamps)       \
			RSG::storage->capture_timestamps_begin(); \
	}

#define RENDER_TIMESTAMP(m_text)                     \
	{                                                \
		if (RSG::storage->capturing_timestamps)      \
			RSG::storage->capture_timestamp(m_text); \
	}

	bool capturing_timestamps = false;

	virtual void capture_timestamps_begin() = 0;
	virtual void capture_timestamp(const String &p_name) = 0;
	virtual uint32_t get_captured_timestamps_count() const = 0;
	virtual uint64_t get_captured_timestamps_frame() const = 0;
	virtual uint64_t get_captured_timestamp_gpu_time(uint32_t p_index) const = 0;
	virtual uint64_t get_captured_timestamp_cpu_time(uint32_t p_index) const = 0;
	virtual String get_captured_timestamp_name(uint32_t p_index) const = 0;

	RendererStorage();
	virtual ~RendererStorage() {}
};

#endif // RENDERINGSERVERSTORAGE_H
