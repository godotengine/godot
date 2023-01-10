/**************************************************************************/
/*  gltf_document.h                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GLTF_DOCUMENT_H
#define GLTF_DOCUMENT_H

#include "extensions/gltf_document_extension.h"
#include "gltf_defines.h"
#include "structures/gltf_animation.h"

#include "scene/3d/bone_attachment.h"
#include "scene/3d/camera.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"

#include "modules/modules_enabled.gen.h" // For csg, gridmap.

#ifdef MODULE_CSG_ENABLED
class CSGShape;
#endif // MODULE_CSG_ENABLED
#ifdef MODULE_GRIDMAP_ENABLED
class GridMap;
#endif // MODULE_GRIDMAP_ENABLED

class GLTFDocument : public Resource {
	GDCLASS(GLTFDocument, Resource);
	static Vector<Ref<GLTFDocumentExtension>> all_document_extensions;
	Vector<Ref<GLTFDocumentExtension>> document_extensions;

private:
	const float BAKE_FPS = 30.0f;

public:
	const int32_t JOINT_GROUP_SIZE = 4;

	enum {
		ARRAY_BUFFER = 34962,
		ELEMENT_ARRAY_BUFFER = 34963,

		TYPE_BYTE = 5120,
		TYPE_UNSIGNED_BYTE = 5121,
		TYPE_SHORT = 5122,
		TYPE_UNSIGNED_SHORT = 5123,
		TYPE_UNSIGNED_INT = 5125,
		TYPE_FLOAT = 5126,

		COMPONENT_TYPE_BYTE = 5120,
		COMPONENT_TYPE_UNSIGNED_BYTE = 5121,
		COMPONENT_TYPE_SHORT = 5122,
		COMPONENT_TYPE_UNSIGNED_SHORT = 5123,
		COMPONENT_TYPE_INT = 5125,
		COMPONENT_TYPE_FLOAT = 5126,
	};

protected:
	static void _bind_methods();

public:
	void _register_gltf_document_extension(Ref<GLTFDocumentExtension> p_extension, bool p_first_priority = false);
	void _unregister_gltf_document_extension(Ref<GLTFDocumentExtension> p_extension);
	void _unregister_all_gltf_document_extensions();
	static void register_gltf_document_extension(Ref<GLTFDocumentExtension> p_extension, bool p_first_priority = false);
	static void unregister_gltf_document_extension(Ref<GLTFDocumentExtension> p_extension);
	static void unregister_all_gltf_document_extensions();

private:
	double _filter_number(double p_float);
	String _get_component_type_name(const uint32_t p_component);
	int _get_component_type_size(const int p_component_type);
	Error _parse_scenes(Ref<GLTFState> p_state);
	Error _parse_nodes(Ref<GLTFState> p_state);
	String _get_type_name(const GLTFType p_component);
	String _get_accessor_type_name(const GLTFType p_type);
	String _gen_unique_name(Ref<GLTFState> p_state, const String &p_name);
	String _sanitize_animation_name(const String &p_name);
	String _gen_unique_animation_name(Ref<GLTFState> p_state, const String &p_name);
	String _sanitize_bone_name(Ref<GLTFState> p_state, const String &p_name);
	String _gen_unique_bone_name(Ref<GLTFState> p_state,
			const GLTFSkeletonIndex p_skel_i,
			const String &p_name);
	GLTFTextureIndex _set_texture(Ref<GLTFState> p_state, Ref<Texture> p_texture);
	Ref<Texture> _get_texture(Ref<GLTFState> p_state,
			const GLTFTextureIndex p_texture);
	GLTFTextureSamplerIndex _set_sampler_for_mode(Ref<GLTFState> p_state,
			uint32_t p_mode);
	Ref<GLTFTextureSampler> _get_sampler_for_texture(Ref<GLTFState> p_state,
			const GLTFTextureIndex p_texture);
	Error _parse_json(const String &p_path, Ref<GLTFState> p_state);
	Error _parse_glb(const String &p_path, Ref<GLTFState> p_state);
	void _compute_node_heights(Ref<GLTFState> p_state);
	Error _parse_buffers(Ref<GLTFState> p_state, const String &p_base_path);
	Error _parse_buffer_views(Ref<GLTFState> p_state);
	GLTFType _get_type_from_str(const String &p_string);
	Error _parse_accessors(Ref<GLTFState> p_state);
	Error _decode_buffer_view(Ref<GLTFState> p_state, double *p_dst,
			const GLTFBufferViewIndex p_buffer_view,
			const int p_skip_every, const int p_skip_bytes,
			const int p_element_size, const int p_count,
			const GLTFType p_type, const int p_component_count,
			const int p_component_type, const int p_component_size,
			const bool p_normalized, const int p_byte_offset,
			const bool p_for_vertex);
	Vector<double> _decode_accessor(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<float> _decode_accessor_as_floats(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<int> _decode_accessor_as_ints(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Vector2> _decode_accessor_as_vec2(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Vector3> _decode_accessor_as_vec3(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Color> _decode_accessor_as_color(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Quat> _decode_accessor_as_quat(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Transform2D> _decode_accessor_as_xform2d(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Basis> _decode_accessor_as_basis(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Transform> _decode_accessor_as_xform(Ref<GLTFState> p_state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Error _parse_meshes(Ref<GLTFState> p_state);
	Error _serialize_textures(Ref<GLTFState> p_state);
	Error _serialize_texture_samplers(Ref<GLTFState> p_state);
	Error _serialize_images(Ref<GLTFState> p_state, const String &p_path);
	Error _serialize_lights(Ref<GLTFState> p_state);
	Error _parse_images(Ref<GLTFState> p_state, const String &p_base_path);
	Error _parse_textures(Ref<GLTFState> p_state);
	Error _parse_texture_samplers(Ref<GLTFState> p_state);
	Error _parse_materials(Ref<GLTFState> p_state);
	void _set_texture_transform_uv1(const Dictionary &p_dict, Ref<SpatialMaterial> p_material);
	void spec_gloss_to_rough_metal(Ref<GLTFSpecGloss> r_spec_gloss,
			Ref<SpatialMaterial> p_material);
	static void spec_gloss_to_metal_base_color(const Color &p_specular_factor,
			const Color &p_diffuse,
			Color &r_base_color,
			float &r_metallic);
	GLTFNodeIndex _find_highest_node(Ref<GLTFState> p_state,
			const Vector<GLTFNodeIndex> &p_subset);
	bool _capture_nodes_in_skin(Ref<GLTFState> p_state, Ref<GLTFSkin> p_skin,
			const GLTFNodeIndex p_node_index);
	void _capture_nodes_for_multirooted_skin(Ref<GLTFState> p_state, Ref<GLTFSkin> p_skin);
	Error _expand_skin(Ref<GLTFState> p_state, Ref<GLTFSkin> p_skin);
	Error _verify_skin(Ref<GLTFState> p_state, Ref<GLTFSkin> p_skin);
	Error _parse_skins(Ref<GLTFState> p_state);
	Error _determine_skeletons(Ref<GLTFState> p_state);
	Error _reparent_non_joint_skeleton_subtrees(
			Ref<GLTFState> p_state, Ref<GLTFSkeleton> p_skeleton,
			const Vector<GLTFNodeIndex> &p_non_joints);
	Error _determine_skeleton_roots(Ref<GLTFState> p_state,
			const GLTFSkeletonIndex p_skel_i);
	Error _create_skeletons(Ref<GLTFState> p_state);
	Error _map_skin_joints_indices_to_skeleton_bone_indices(Ref<GLTFState> p_state);
	Error _serialize_skins(Ref<GLTFState> p_state);
	Error _create_skins(Ref<GLTFState> p_state);
	bool _skins_are_same(const Ref<Skin> p_skin_a, const Ref<Skin> p_skin_b);
	void _remove_duplicate_skins(Ref<GLTFState> p_state);
	Error _serialize_cameras(Ref<GLTFState> p_state);
	Error _parse_cameras(Ref<GLTFState> p_state);
	Error _parse_lights(Ref<GLTFState> p_state);
	Error _parse_animations(Ref<GLTFState> p_state);
	Error _serialize_animations(Ref<GLTFState> p_state);
	BoneAttachment *_generate_bone_attachment(Ref<GLTFState> p_state,
			Skeleton *p_skeleton,
			const GLTFNodeIndex p_node_index,
			const GLTFNodeIndex p_bone_index);
	Spatial *_generate_mesh_instance(Ref<GLTFState> p_state, Node *p_scene_parent, const GLTFNodeIndex p_node_index);
	Camera *_generate_camera(Ref<GLTFState> p_state, Node *p_scene_parent,
			const GLTFNodeIndex p_node_index);
	Spatial *_generate_light(Ref<GLTFState> p_state, Node *p_scene_parent, const GLTFNodeIndex p_node_index);
	Spatial *_generate_spatial(Ref<GLTFState> p_state, Node *p_scene_parent,
			const GLTFNodeIndex p_node_index);
	void _assign_scene_names(Ref<GLTFState> p_state);
	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values,
			const float p_time,
			const GLTFAnimation::Interpolation p_interp);
	GLTFAccessorIndex _encode_accessor_as_quats(Ref<GLTFState> p_state,
			const Vector<Quat> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_weights(Ref<GLTFState> p_state,
			const Vector<Color> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_joints(Ref<GLTFState> p_state,
			const Vector<Color> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_floats(Ref<GLTFState> p_state,
			const Vector<real_t> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_vec2(Ref<GLTFState> p_state,
			const Vector<Vector2> p_attribs,
			const bool p_for_vertex);

	void _calc_accessor_vec2_min_max(int p_i, const int p_element_count, Vector<double> &p_type_max, Vector2 p_attribs, Vector<double> &p_type_min) {
		if (p_i == 0) {
			for (int32_t type_i = 0; type_i < p_element_count; type_i++) {
				p_type_max.write[type_i] = p_attribs[(p_i * p_element_count) + type_i];
				p_type_min.write[type_i] = p_attribs[(p_i * p_element_count) + type_i];
			}
		}
		for (int32_t type_i = 0; type_i < p_element_count; type_i++) {
			p_type_max.write[type_i] = MAX(p_attribs[(p_i * p_element_count) + type_i], p_type_max[type_i]);
			p_type_min.write[type_i] = MIN(p_attribs[(p_i * p_element_count) + type_i], p_type_min[type_i]);
			p_type_max.write[type_i] = _filter_number(p_type_max.write[type_i]);
			p_type_min.write[type_i] = _filter_number(p_type_min.write[type_i]);
		}
	}

	GLTFAccessorIndex _encode_accessor_as_vec3(Ref<GLTFState> p_state,
			const Vector<Vector3> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_color(Ref<GLTFState> p_state,
			const Vector<Color> p_attribs,
			const bool p_for_vertex);

	void _calc_accessor_min_max(int p_i, const int p_element_count, Vector<double> &p_type_max, Vector<double> p_attribs, Vector<double> &p_type_min);

	GLTFAccessorIndex _encode_accessor_as_ints(Ref<GLTFState> p_state,
			const Vector<int32_t> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_xform(Ref<GLTFState> p_state,
			const Vector<Transform> p_attribs,
			const bool p_for_vertex);
	Error _encode_buffer_view(Ref<GLTFState> p_state, const double *p_src,
			const int p_count, const GLTFType p_type,
			const int p_component_type, const bool p_normalized,
			const int p_byte_offset, const bool p_for_vertex,
			GLTFBufferViewIndex &r_accessor);
	Error _encode_accessors(Ref<GLTFState> p_state);
	Error _encode_buffer_views(Ref<GLTFState> p_state);
	Error _serialize_materials(Ref<GLTFState> p_state);
	Error _serialize_meshes(Ref<GLTFState> p_state);
	Error _serialize_nodes(Ref<GLTFState> p_state);
	Error _serialize_scenes(Ref<GLTFState> p_state);
	String interpolation_to_string(const GLTFAnimation::Interpolation p_interp);
	GLTFAnimation::Track _convert_animation_track(Ref<GLTFState> p_state,
			GLTFAnimation::Track p_track,
			Ref<Animation> p_animation, Transform p_bone_rest,
			int32_t p_track_i,
			GLTFNodeIndex p_node_i);
	Error _encode_buffer_bins(Ref<GLTFState> p_state, const String &p_path);
	Error _encode_buffer_glb(Ref<GLTFState> p_state, const String &p_path);
	Dictionary _serialize_texture_transform_uv1(Ref<SpatialMaterial> p_material);
	Dictionary _serialize_texture_transform_uv2(Ref<SpatialMaterial> p_material);
	Error _serialize_version(Ref<GLTFState> p_state);
	Error _serialize_file(Ref<GLTFState> p_state, const String p_path);
	Error _serialize_gltf_extensions(Ref<GLTFState> p_state) const;

public:
	// http://www.itu.int/rec/R-REC-BT.601
	// http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
	static constexpr float R_BRIGHTNESS_COEFF = 0.299f;
	static constexpr float G_BRIGHTNESS_COEFF = 0.587f;
	static constexpr float B_BRIGHTNESS_COEFF = 0.114f;

private:
	// https://github.com/microsoft/glTF-SDK/blob/master/GLTFSDK/Source/PBRUtils.cpp#L9
	// https://bghgary.github.io/glTF/convert-between-workflows-bjs/js/babylon.pbrUtilities.js
	static float solve_metallic(float p_dielectric_specular, float p_diffuse,
			float p_specular,
			float p_one_minus_specular_strength);
	static float get_perceived_brightness(const Color p_color);
	static float get_max_component(const Color &p_color);

public:
	void extension_generate_scene(Ref<GLTFState> p_state);

	String _sanitize_scene_name(Ref<GLTFState> p_state, const String &p_name);
	String _legacy_validate_node_name(const String &p_name);

	Error _parse_gltf_extensions(Ref<GLTFState> p_state);

	void _process_mesh_instances(Ref<GLTFState> p_state, Node *p_scene_root);
	void _generate_scene_node(Ref<GLTFState> p_state, Node *p_scene_parent,
			Spatial *p_scene_root,
			const GLTFNodeIndex p_node_index);
	void _generate_skeleton_bone_node(Ref<GLTFState> p_state, Node *p_scene_parent, Spatial *p_scene_root, const GLTFNodeIndex p_node_index);
	void _import_animation(Ref<GLTFState> p_state, AnimationPlayer *p_animation_player,
			const GLTFAnimationIndex p_index, const int p_bake_fps);
	void _convert_mesh_instances(Ref<GLTFState> p_state);
	GLTFCameraIndex _convert_camera(Ref<GLTFState> p_state, Camera *p_camera);
	void _convert_light_to_gltf(Light *p_light, Ref<GLTFState> p_state, Ref<GLTFNode> p_gltf_node);
	GLTFLightIndex _convert_light(Ref<GLTFState> p_state, Light *p_light);
	void _convert_spatial(Ref<GLTFState> p_state, Spatial *p_spatial, Ref<GLTFNode> p_node);
	void _convert_scene_node(Ref<GLTFState> p_state, Node *p_current,
			const GLTFNodeIndex p_gltf_current,
			const GLTFNodeIndex p_gltf_root);

#ifdef MODULE_CSG_ENABLED
	void _convert_csg_shape_to_gltf(CSGShape *p_current, GLTFNodeIndex p_gltf_parent, Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state);
#endif // MODULE_CSG_ENABLED

	void _create_gltf_node(Ref<GLTFState> p_state,
			Node *p_scene_parent,
			GLTFNodeIndex p_current_node_i,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_gltf_node,
			Ref<GLTFNode> p_gltf_node);
	void _convert_animation_player_to_gltf(
			AnimationPlayer *p_animation_player, Ref<GLTFState> p_state,
			GLTFNodeIndex p_gltf_current,
			GLTFNodeIndex p_gltf_root_index,
			Ref<GLTFNode> p_gltf_node, Node *p_scene_parent);
	void _check_visibility(Node *p_node, bool &r_retflag);
	void _convert_camera_to_gltf(Camera *p_camera, Ref<GLTFState> p_state,
			Ref<GLTFNode> p_gltf_node);
#ifdef MODULE_GRIDMAP_ENABLED
	void _convert_grid_map_to_gltf(
			GridMap *p_grid_map,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_node_index,
			Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state);
#endif // MODULE_GRIDMAP_ENABLED
	void _convert_multi_mesh_instance_to_gltf(
			MultiMeshInstance *p_scene_parent,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_node_index,
			Ref<GLTFNode> p_gltf_node, Ref<GLTFState> p_state);
	void _convert_skeleton_to_gltf(
			Skeleton *p_scene_parent, Ref<GLTFState> p_state,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_node_index,
			Ref<GLTFNode> p_gltf_node);
	void _convert_bone_attachment_to_gltf(BoneAttachment *p_bone_attachment,
			Ref<GLTFState> p_state,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_node_index,
			Ref<GLTFNode> p_gltf_node);
	void _convert_mesh_instance_to_gltf(MeshInstance *p_mesh_instance,
			Ref<GLTFState> p_state,
			Ref<GLTFNode> p_gltf_node);
	GLTFMeshIndex _convert_mesh_to_gltf(Ref<GLTFState> p_state,
			MeshInstance *p_mesh_instance);
	void _convert_animation(Ref<GLTFState> p_state, AnimationPlayer *p_animation_player,
			String p_animation_track_name);
	Error serialize(Ref<GLTFState> p_state, Node *p_root, const String &p_path);
	Error parse(Ref<GLTFState> p_state, String p_paths, bool p_read_binary = false);
};

#endif // GLTF_DOCUMENT_H
