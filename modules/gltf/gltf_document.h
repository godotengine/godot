/*************************************************************************/
/*  gltf_document.h                                                      */
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

#ifndef GLTF_DOCUMENT_H
#define GLTF_DOCUMENT_H

#include "editor/import/resource_importer_scene.h"
#include "gltf_animation.h"
#include "scene/2d/node_2d.h"
#include "scene/3d/bone_attachment.h"
#include "scene/3d/light.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/material.h"
#include "scene/resources/texture.h"

class GLTFState;
class GLTFSkin;
class GLTFNode;
class GLTFSpecGloss;
class GLTFSkeleton;
class CSGShape;
class GridMap;
class MultiMeshInstance;

using GLTFAccessorIndex = int;
using GLTFAnimationIndex = int;
using GLTFBufferIndex = int;
using GLTFBufferViewIndex = int;
using GLTFCameraIndex = int;
using GLTFImageIndex = int;
using GLTFMaterialIndex = int;
using GLTFMeshIndex = int;
using GLTFLightIndex = int;
using GLTFNodeIndex = int;
using GLTFSkeletonIndex = int;
using GLTFSkinIndex = int;
using GLTFTextureIndex = int;

class GLTFDocument : public Resource {
	GDCLASS(GLTFDocument, Resource);
	friend class GLTFState;
	friend class GLTFSkin;
	friend class GLTFSkeleton;

private:
	const float BAKE_FPS = 30.0f;

public:
	const int32_t JOINT_GROUP_SIZE = 4;
	enum GLTFType {
		TYPE_SCALAR,
		TYPE_VEC2,
		TYPE_VEC3,
		TYPE_VEC4,
		TYPE_MAT2,
		TYPE_MAT3,
		TYPE_MAT4,
	};

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

private:
	template <class T>
	static Array to_array(const Vector<T> &p_inp) {
		Array ret;
		for (int i = 0; i < p_inp.size(); i++) {
			ret.push_back(p_inp[i]);
		}
		return ret;
	}

	template <class T>
	static Array to_array(const Set<T> &p_inp) {
		Array ret;
		typename Set<T>::Element *elem = p_inp.front();
		while (elem) {
			ret.push_back(elem->get());
			elem = elem->next();
		}
		return ret;
	}

	template <class T>
	static void set_from_array(Vector<T> &r_out, const Array &p_inp) {
		r_out.clear();
		for (int i = 0; i < p_inp.size(); i++) {
			r_out.push_back(p_inp[i]);
		}
	}

	template <class T>
	static void set_from_array(Set<T> &r_out, const Array &p_inp) {
		r_out.clear();
		for (int i = 0; i < p_inp.size(); i++) {
			r_out.insert(p_inp[i]);
		}
	}
	template <class K, class V>
	static Dictionary to_dict(const Map<K, V> &p_inp) {
		Dictionary ret;
		for (typename Map<K, V>::Element *E = p_inp.front(); E; E = E->next()) {
			ret[E->key()] = E->value();
		}
		return ret;
	}

	template <class K, class V>
	static void set_from_dict(Map<K, V> &r_out, const Dictionary &p_inp) {
		r_out.clear();
		Array keys = p_inp.keys();
		for (int i = 0; i < keys.size(); i++) {
			r_out[keys[i]] = p_inp[keys[i]];
		}
	}
	double _filter_number(double p_float);
	String _get_component_type_name(const uint32_t p_component);
	int _get_component_type_size(const int component_type);
	Error _parse_scenes(Ref<GLTFState> state);
	Error _parse_nodes(Ref<GLTFState> state);
	String _get_type_name(const GLTFType p_component);
	String _get_accessor_type_name(const GLTFDocument::GLTFType p_type);
	String _gen_unique_name(Ref<GLTFState> state, const String &p_name);
	String _sanitize_animation_name(const String &name);
	String _gen_unique_animation_name(Ref<GLTFState> state, const String &p_name);
	String _sanitize_bone_name(Ref<GLTFState> state, const String &name);
	String _gen_unique_bone_name(Ref<GLTFState> state,
			const GLTFSkeletonIndex skel_i,
			const String &p_name);
	GLTFTextureIndex _set_texture(Ref<GLTFState> state, Ref<Texture> p_texture);
	Ref<Texture> _get_texture(Ref<GLTFState> state,
			const GLTFTextureIndex p_texture);
	Error _parse_json(const String &p_path, Ref<GLTFState> state);
	Error _parse_glb(const String &p_path, Ref<GLTFState> state);
	void _compute_node_heights(Ref<GLTFState> state);
	Error _parse_buffers(Ref<GLTFState> state, const String &p_base_path);
	Error _parse_buffer_views(Ref<GLTFState> state);
	GLTFType _get_type_from_str(const String &p_string);
	Error _parse_accessors(Ref<GLTFState> state);
	Error _decode_buffer_view(Ref<GLTFState> state, double *dst,
			const GLTFBufferViewIndex p_buffer_view,
			const int skip_every, const int skip_bytes,
			const int element_size, const int count,
			const GLTFType type, const int component_count,
			const int component_type, const int component_size,
			const bool normalized, const int byte_offset,
			const bool for_vertex);
	Vector<double> _decode_accessor(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<float> _decode_accessor_as_floats(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<int> _decode_accessor_as_ints(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Vector2> _decode_accessor_as_vec2(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Vector3> _decode_accessor_as_vec3(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Color> _decode_accessor_as_color(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Quat> _decode_accessor_as_quat(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Transform2D> _decode_accessor_as_xform2d(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Basis> _decode_accessor_as_basis(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Vector<Transform> _decode_accessor_as_xform(Ref<GLTFState> state,
			const GLTFAccessorIndex p_accessor,
			const bool p_for_vertex);
	Error _parse_meshes(Ref<GLTFState> state);
	Error _serialize_textures(Ref<GLTFState> state);
	Error _serialize_images(Ref<GLTFState> state, const String &p_path);
	Error _serialize_lights(Ref<GLTFState> state);
	Error _parse_images(Ref<GLTFState> state, const String &p_base_path);
	Error _parse_textures(Ref<GLTFState> state);
	Error _parse_materials(Ref<GLTFState> state);
	void _set_texture_transform_uv1(const Dictionary &d, Ref<SpatialMaterial> material);
	void spec_gloss_to_rough_metal(Ref<GLTFSpecGloss> r_spec_gloss,
			Ref<SpatialMaterial> p_material);
	static void spec_gloss_to_metal_base_color(const Color &p_specular_factor,
			const Color &p_diffuse,
			Color &r_base_color,
			float &r_metallic);
	GLTFNodeIndex _find_highest_node(Ref<GLTFState> state,
			const Vector<GLTFNodeIndex> &subset);
	bool _capture_nodes_in_skin(Ref<GLTFState> state, Ref<GLTFSkin> skin,
			const GLTFNodeIndex node_index);
	void _capture_nodes_for_multirooted_skin(Ref<GLTFState> state, Ref<GLTFSkin> skin);
	Error _expand_skin(Ref<GLTFState> state, Ref<GLTFSkin> skin);
	Error _verify_skin(Ref<GLTFState> state, Ref<GLTFSkin> skin);
	Error _parse_skins(Ref<GLTFState> state);
	Error _determine_skeletons(Ref<GLTFState> state);
	Error _reparent_non_joint_skeleton_subtrees(
			Ref<GLTFState> state, Ref<GLTFSkeleton> skeleton,
			const Vector<GLTFNodeIndex> &non_joints);
	Error _reparent_to_fake_joint(Ref<GLTFState> state, Ref<GLTFSkeleton> skeleton,
			const GLTFNodeIndex node_index);
	Error _determine_skeleton_roots(Ref<GLTFState> state,
			const GLTFSkeletonIndex skel_i);
	Error _create_skeletons(Ref<GLTFState> state);
	Error _map_skin_joints_indices_to_skeleton_bone_indices(Ref<GLTFState> state);
	Error _serialize_skins(Ref<GLTFState> state);
	Error _create_skins(Ref<GLTFState> state);
	bool _skins_are_same(const Ref<Skin> skin_a, const Ref<Skin> skin_b);
	void _remove_duplicate_skins(Ref<GLTFState> state);
	Error _serialize_cameras(Ref<GLTFState> state);
	Error _parse_cameras(Ref<GLTFState> state);
	Error _parse_lights(Ref<GLTFState> state);
	Error _parse_animations(Ref<GLTFState> state);
	Error _serialize_animations(Ref<GLTFState> state);
	BoneAttachment *_generate_bone_attachment(Ref<GLTFState> state,
			Skeleton *skeleton,
			const GLTFNodeIndex node_index,
			const GLTFNodeIndex bone_index);
	Spatial *_generate_mesh_instance(Ref<GLTFState> state, Node *scene_parent, const GLTFNodeIndex node_index);
	Camera *_generate_camera(Ref<GLTFState> state, Node *scene_parent,
			const GLTFNodeIndex node_index);
	Spatial *_generate_light(Ref<GLTFState> state, Node *scene_parent, const GLTFNodeIndex node_index);
	Spatial *_generate_spatial(Ref<GLTFState> state, Node *scene_parent,
			const GLTFNodeIndex node_index);
	void _assign_scene_names(Ref<GLTFState> state);
	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values,
			const float p_time,
			const GLTFAnimation::Interpolation p_interp);
	GLTFAccessorIndex _encode_accessor_as_quats(Ref<GLTFState> state,
			const Vector<Quat> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_weights(Ref<GLTFState> state,
			const Vector<Color> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_joints(Ref<GLTFState> state,
			const Vector<Color> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_floats(Ref<GLTFState> state,
			const Vector<real_t> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_vec2(Ref<GLTFState> state,
			const Vector<Vector2> p_attribs,
			const bool p_for_vertex);

	void _calc_accessor_vec2_min_max(int i, const int element_count, Vector<double> &type_max, Vector2 attribs, Vector<double> &type_min) {
		if (i == 0) {
			for (int32_t type_i = 0; type_i < element_count; type_i++) {
				type_max.write[type_i] = attribs[(i * element_count) + type_i];
				type_min.write[type_i] = attribs[(i * element_count) + type_i];
			}
		}
		for (int32_t type_i = 0; type_i < element_count; type_i++) {
			type_max.write[type_i] = MAX(attribs[(i * element_count) + type_i], type_max[type_i]);
			type_min.write[type_i] = MIN(attribs[(i * element_count) + type_i], type_min[type_i]);
			type_max.write[type_i] = _filter_number(type_max.write[type_i]);
			type_min.write[type_i] = _filter_number(type_min.write[type_i]);
		}
	}

	GLTFAccessorIndex _encode_accessor_as_vec3(Ref<GLTFState> state,
			const Vector<Vector3> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_color(Ref<GLTFState> state,
			const Vector<Color> p_attribs,
			const bool p_for_vertex);

	void _calc_accessor_min_max(int p_i, const int p_element_count, Vector<double> &p_type_max, Vector<double> p_attribs, Vector<double> &p_type_min);

	GLTFAccessorIndex _encode_accessor_as_ints(Ref<GLTFState> state,
			const Vector<int32_t> p_attribs,
			const bool p_for_vertex);
	GLTFAccessorIndex _encode_accessor_as_xform(Ref<GLTFState> state,
			const Vector<Transform> p_attribs,
			const bool p_for_vertex);
	Error _encode_buffer_view(Ref<GLTFState> state, const double *src,
			const int count, const GLTFType type,
			const int component_type, const bool normalized,
			const int byte_offset, const bool for_vertex,
			GLTFBufferViewIndex &r_accessor);
	Error _encode_accessors(Ref<GLTFState> state);
	Error _encode_buffer_views(Ref<GLTFState> state);
	Error _serialize_materials(Ref<GLTFState> state);
	Error _serialize_meshes(Ref<GLTFState> state);
	Error _serialize_nodes(Ref<GLTFState> state);
	Error _serialize_scenes(Ref<GLTFState> state);
	String interpolation_to_string(const GLTFAnimation::Interpolation p_interp);
	GLTFAnimation::Track _convert_animation_track(Ref<GLTFState> state,
			GLTFAnimation::Track p_track,
			Ref<Animation> p_animation, Transform p_bone_rest,
			int32_t p_track_i,
			GLTFNodeIndex p_node_i);
	Error _encode_buffer_bins(Ref<GLTFState> state, const String &p_path);
	Error _encode_buffer_glb(Ref<GLTFState> state, const String &p_path);
	Dictionary _serialize_texture_transform_uv1(Ref<SpatialMaterial> p_material);
	Dictionary _serialize_texture_transform_uv2(Ref<SpatialMaterial> p_material);
	Error _serialize_version(Ref<GLTFState> state);
	Error _serialize_file(Ref<GLTFState> state, const String p_path);
	Error _serialize_extensions(Ref<GLTFState> state) const;

public:
	// http://www.itu.int/rec/R-REC-BT.601
	// http://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.601-7-201103-I!!PDF-E.pdf
	static constexpr float R_BRIGHTNESS_COEFF = 0.299f;
	static constexpr float G_BRIGHTNESS_COEFF = 0.587f;
	static constexpr float B_BRIGHTNESS_COEFF = 0.114f;

private:
	// https://github.com/microsoft/glTF-SDK/blob/master/GLTFSDK/Source/PBRUtils.cpp#L9
	// https://bghgary.github.io/glTF/convert-between-workflows-bjs/js/babylon.pbrUtilities.js
	static float solve_metallic(float p_dielectric_specular, float diffuse,
			float specular,
			float p_one_minus_specular_strength);
	static float get_perceived_brightness(const Color p_color);
	static float get_max_component(const Color &p_color);

public:
	String _sanitize_scene_name(Ref<GLTFState> state, const String &p_name);
	String _legacy_validate_node_name(const String &p_name);

	void _process_mesh_instances(Ref<GLTFState> state, Node *scene_root);
	void _generate_scene_node(Ref<GLTFState> state, Node *scene_parent,
			Spatial *scene_root,
			const GLTFNodeIndex node_index);
	void _generate_skeleton_bone_node(Ref<GLTFState> state, Node *scene_parent, Spatial *scene_root, const GLTFNodeIndex node_index);
	void _import_animation(Ref<GLTFState> state, AnimationPlayer *ap,
			const GLTFAnimationIndex index, const int bake_fps);
	void _convert_mesh_instances(Ref<GLTFState> state);
	GLTFCameraIndex _convert_camera(Ref<GLTFState> state, Camera *p_camera);
	void _convert_light_to_gltf(Light *light, Ref<GLTFState> state, Ref<GLTFNode> gltf_node);
	GLTFLightIndex _convert_light(Ref<GLTFState> state, Light *p_light);
	void _convert_spatial(Ref<GLTFState> state, Spatial *p_spatial, Ref<GLTFNode> p_node);
	void _convert_scene_node(Ref<GLTFState> state, Node *p_current,
			const GLTFNodeIndex p_gltf_current,
			const GLTFNodeIndex p_gltf_root);

#ifdef MODULE_CSG_ENABLED
	void _convert_csg_shape_to_gltf(CSGShape *p_current, GLTFNodeIndex p_gltf_parent, Ref<GLTFNode> gltf_node, Ref<GLTFState> state);
#endif // MODULE_CSG_ENABLED

	void _create_gltf_node(Ref<GLTFState> state,
			Node *p_scene_parent,
			GLTFNodeIndex current_node_i,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_gltf_node,
			Ref<GLTFNode> gltf_node);
	void _convert_animation_player_to_gltf(
			AnimationPlayer *animation_player, Ref<GLTFState> state,
			GLTFNodeIndex p_gltf_current,
			GLTFNodeIndex p_gltf_root_index,
			Ref<GLTFNode> p_gltf_node, Node *p_scene_parent);
	void _check_visibility(Node *p_node, bool &retflag);
	void _convert_camera_to_gltf(Camera *camera, Ref<GLTFState> state,
			Ref<GLTFNode> gltf_node);
#ifdef MODULE_GRIDMAP_ENABLED
	void _convert_grid_map_to_gltf(
			GridMap *p_grid_map,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_node_index,
			Ref<GLTFNode> gltf_node, Ref<GLTFState> state);
#endif // MODULE_GRIDMAP_ENABLED
	void _convert_mult_mesh_instance_to_gltf(
			MultiMeshInstance *p_scene_parent,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_node_index,
			Ref<GLTFNode> gltf_node, Ref<GLTFState> state);
	void _convert_skeleton_to_gltf(
			Skeleton *p_scene_parent, Ref<GLTFState> state,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_node_index,
			Ref<GLTFNode> gltf_node);
	void _convert_bone_attachment_to_gltf(BoneAttachment *p_bone_attachment,
			Ref<GLTFState> state,
			GLTFNodeIndex p_parent_node_index,
			GLTFNodeIndex p_root_node_index,
			Ref<GLTFNode> gltf_node);
	void _convert_mesh_instance_to_gltf(MeshInstance *p_mesh_instance,
			Ref<GLTFState> state,
			Ref<GLTFNode> gltf_node);
	GLTFMeshIndex _convert_mesh_to_gltf(Ref<GLTFState> state,
			MeshInstance *p_mesh_instance);
	void _convert_animation(Ref<GLTFState> state, AnimationPlayer *ap,
			String p_animation_track_name);
	Error serialize(Ref<GLTFState> state, Node *p_root, const String &p_path);
	Error parse(Ref<GLTFState> state, String p_paths, bool p_read_binary = false);
};

#endif // GLTF_DOCUMENT_H
