/*************************************************************************/
/*  editor_scene_importer_gltf.h                                         */
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

#ifndef EDITOR_SCENE_IMPORTER_GLTF_H
#define EDITOR_SCENE_IMPORTER_GLTF_H

#include "editor/import/resource_importer_scene.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"

class AnimationPlayer;
class BoneAttachment3D;
class MeshInstance3D;

class EditorSceneImporterGLTF : public EditorSceneImporter {
	GDCLASS(EditorSceneImporterGLTF, EditorSceneImporter);

	typedef int GLTFAccessorIndex;
	typedef int GLTFAnimationIndex;
	typedef int GLTFBufferIndex;
	typedef int GLTFBufferViewIndex;
	typedef int GLTFCameraIndex;
	typedef int GLTFImageIndex;
	typedef int GLTFMaterialIndex;
	typedef int GLTFMeshIndex;
	typedef int GLTFLightIndex;
	typedef int GLTFNodeIndex;
	typedef int GLTFSkeletonIndex;
	typedef int GLTFSkinIndex;
	typedef int GLTFTextureIndex;

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

	String _get_component_type_name(const uint32_t p_component);
	int _get_component_type_size(const int component_type);

	enum GLTFType {
		TYPE_SCALAR,
		TYPE_VEC2,
		TYPE_VEC3,
		TYPE_VEC4,
		TYPE_MAT2,
		TYPE_MAT3,
		TYPE_MAT4,
	};

	String _get_type_name(const GLTFType p_component);

	struct GLTFNode {
		//matrices need to be transformed to this
		GLTFNodeIndex parent = -1;
		int height = -1;

		Transform xform;
		String name;

		GLTFMeshIndex mesh = -1;
		GLTFCameraIndex camera = -1;
		GLTFSkinIndex skin = -1;

		GLTFSkeletonIndex skeleton = -1;
		bool joint = false;

		Vector3 translation;
		Quat rotation;
		Vector3 scale = Vector3(1, 1, 1);

		Vector<int> children;

		GLTFNodeIndex fake_joint_parent = -1;

		GLTFLightIndex light = -1;

		GLTFNode() {}
	};

	struct GLTFBufferView {
		GLTFBufferIndex buffer = -1;
		int byte_offset = 0;
		int byte_length = 0;
		int byte_stride = 0;
		bool indices = false;
		//matrices need to be transformed to this

		GLTFBufferView() {}
	};

	struct GLTFAccessor {
		GLTFBufferViewIndex buffer_view = 0;
		int byte_offset = 0;
		int component_type = 0;
		bool normalized = false;
		int count = 0;
		GLTFType type;
		float min = 0;
		float max = 0;
		int sparse_count = 0;
		int sparse_indices_buffer_view = 0;
		int sparse_indices_byte_offset = 0;
		int sparse_indices_component_type = 0;
		int sparse_values_buffer_view = 0;
		int sparse_values_byte_offset = 0;

		GLTFAccessor() {}
	};
	struct GLTFTexture {
		GLTFImageIndex src_image;
	};

	struct GLTFSkeleton {
		// The *synthesized* skeletons joints
		Vector<GLTFNodeIndex> joints;

		// The roots of the skeleton. If there are multiple, each root must have the same parent
		// (ie roots are siblings)
		Vector<GLTFNodeIndex> roots;

		// The created Skeleton for the scene
		Skeleton3D *godot_skeleton = nullptr;

		// Set of unique bone names for the skeleton
		Set<String> unique_names;

		GLTFSkeleton() {}
	};

	struct GLTFSkin {
		String name;

		// The "skeleton" property defined in the gltf spec. -1 = Scene Root
		GLTFNodeIndex skin_root = -1;

		Vector<GLTFNodeIndex> joints_original;
		Vector<Transform> inverse_binds;

		// Note: joints + non_joints should form a complete subtree, or subtrees with a common parent

		// All nodes that are skins that are caught in-between the original joints
		// (inclusive of joints_original)
		Vector<GLTFNodeIndex> joints;

		// All Nodes that are caught in-between skin joint nodes, and are not defined
		// as joints by any skin
		Vector<GLTFNodeIndex> non_joints;

		// The roots of the skin. In the case of multiple roots, their parent *must*
		// be the same (the roots must be siblings)
		Vector<GLTFNodeIndex> roots;

		// The GLTF Skeleton this Skin points to (after we determine skeletons)
		GLTFSkeletonIndex skeleton = -1;

		// A mapping from the joint indices (in the order of joints_original) to the
		// Godot Skeleton's bone_indices
		Map<int, int> joint_i_to_bone_i;
		Map<int, StringName> joint_i_to_name;

		// The Actual Skin that will be created as a mapping between the IBM's of this skin
		// to the generated skeleton for the mesh instances.
		Ref<Skin> godot_skin;

		GLTFSkin() {}
	};

	struct GLTFMesh {
		Ref<ArrayMesh> mesh;
		Vector<float> blend_weights;
	};

	struct GLTFCamera {
		bool perspective = true;
		float fov_size = 64;
		float zfar = 500;
		float znear = 0.1;

		GLTFCamera() {}
	};

	struct GLTFLight {
		Color color = Color(1.0f, 1.0f, 1.0f);
		float intensity = 1.0f;
		String type = "";
		float range = Math_INF;
		float inner_cone_angle = 0.0f;
		float outer_cone_angle = Math_PI / 4.0;

		GLTFLight() {}
	};

	struct GLTFAnimation {
		bool loop = false;

		enum Interpolation {
			INTERP_LINEAR,
			INTERP_STEP,
			INTERP_CATMULLROMSPLINE,
			INTERP_CUBIC_SPLINE
		};

		template <class T>
		struct Channel {
			Interpolation interpolation;
			Vector<float> times;
			Vector<T> values;
		};

		struct Track {
			Channel<Vector3> translation_track;
			Channel<Quat> rotation_track;
			Channel<Vector3> scale_track;
			Vector<Channel<float>> weight_tracks;
		};

		String name;

		Map<int, Track> tracks;
	};

	struct GLTFState {
		Dictionary json;
		int major_version;
		int minor_version;
		Vector<uint8_t> glb_data;

		bool use_named_skin_binds;

		Vector<GLTFNode *> nodes;
		Vector<Vector<uint8_t>> buffers;
		Vector<GLTFBufferView> buffer_views;
		Vector<GLTFAccessor> accessors;

		Vector<GLTFMesh> meshes; //meshes are loaded directly, no reason not to.
		Vector<Ref<Material>> materials;

		String scene_name;
		Vector<int> root_nodes;

		Vector<GLTFTexture> textures;
		Vector<Ref<Texture2D>> images;

		Vector<GLTFSkin> skins;
		Vector<GLTFCamera> cameras;
		Vector<GLTFLight> lights;

		Set<String> unique_names;

		Vector<GLTFSkeleton> skeletons;
		Vector<GLTFAnimation> animations;

		Map<GLTFNodeIndex, Node *> scene_nodes;

		// EditorSceneImporter::ImportFlags
		uint32_t import_flags;

		~GLTFState() {
			for (int i = 0; i < nodes.size(); i++) {
				memdelete(nodes[i]);
			}
		}
	};

	String _sanitize_scene_name(const String &name);
	String _gen_unique_name(GLTFState &state, const String &p_name);

	String _sanitize_bone_name(const String &name);
	String _gen_unique_bone_name(GLTFState &state, const GLTFSkeletonIndex skel_i, const String &p_name);

	Ref<Texture2D> _get_texture(GLTFState &state, const GLTFTextureIndex p_texture);

	Error _parse_json(const String &p_path, GLTFState &state);
	Error _parse_glb(const String &p_path, GLTFState &state);

	Error _parse_scenes(GLTFState &state);
	Error _parse_nodes(GLTFState &state);

	void _compute_node_heights(GLTFState &state);

	Error _parse_buffers(GLTFState &state, const String &p_base_path);
	Error _parse_buffer_views(GLTFState &state);
	GLTFType _get_type_from_str(const String &p_string);
	Error _parse_accessors(GLTFState &state);
	Error _decode_buffer_view(GLTFState &state, double *dst, const GLTFBufferViewIndex p_buffer_view, const int skip_every, const int skip_bytes, const int element_size, const int count, const GLTFType type, const int component_count, const int component_type, const int component_size, const bool normalized, const int byte_offset, const bool for_vertex);

	Vector<double> _decode_accessor(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<float> _decode_accessor_as_floats(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<int> _decode_accessor_as_ints(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<Vector2> _decode_accessor_as_vec2(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<Vector3> _decode_accessor_as_vec3(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<Color> _decode_accessor_as_color(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<Quat> _decode_accessor_as_quat(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<Transform2D> _decode_accessor_as_xform2d(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<Basis> _decode_accessor_as_basis(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);
	Vector<Transform> _decode_accessor_as_xform(GLTFState &state, const GLTFAccessorIndex p_accessor, const bool p_for_vertex);

	Error _parse_meshes(GLTFState &state);
	Error _parse_images(GLTFState &state, const String &p_base_path);
	Error _parse_textures(GLTFState &state);

	Error _parse_materials(GLTFState &state);

	GLTFNodeIndex _find_highest_node(GLTFState &state, const Vector<GLTFNodeIndex> &subset);

	bool _capture_nodes_in_skin(GLTFState &state, GLTFSkin &skin, const GLTFNodeIndex node_index);
	void _capture_nodes_for_multirooted_skin(GLTFState &state, GLTFSkin &skin);
	Error _expand_skin(GLTFState &state, GLTFSkin &skin);
	Error _verify_skin(GLTFState &state, GLTFSkin &skin);
	Error _parse_skins(GLTFState &state);

	Error _determine_skeletons(GLTFState &state);
	Error _reparent_non_joint_skeleton_subtrees(GLTFState &state, GLTFSkeleton &skeleton, const Vector<GLTFNodeIndex> &non_joints);
	Error _reparent_to_fake_joint(GLTFState &state, GLTFSkeleton &skeleton, const GLTFNodeIndex node_index);
	Error _determine_skeleton_roots(GLTFState &state, const GLTFSkeletonIndex skel_i);

	Error _create_skeletons(GLTFState &state);
	Error _map_skin_joints_indices_to_skeleton_bone_indices(GLTFState &state);

	Error _create_skins(GLTFState &state);
	bool _skins_are_same(const Ref<Skin> &skin_a, const Ref<Skin> &skin_b);
	void _remove_duplicate_skins(GLTFState &state);

	Error _parse_cameras(GLTFState &state);
	Error _parse_lights(GLTFState &state);
	Error _parse_animations(GLTFState &state);

	BoneAttachment3D *_generate_bone_attachment(GLTFState &state, Skeleton3D *skeleton, const GLTFNodeIndex node_index);
	MeshInstance3D *_generate_mesh_instance(GLTFState &state, Node *scene_parent, const GLTFNodeIndex node_index);
	Camera3D *_generate_camera(GLTFState &state, Node *scene_parent, const GLTFNodeIndex node_index);
	Light3D *_generate_light(GLTFState &state, Node *scene_parent, const GLTFNodeIndex node_index);
	Node3D *_generate_spatial(GLTFState &state, Node *scene_parent, const GLTFNodeIndex node_index);

	void _generate_scene_node(GLTFState &state, Node *scene_parent, Node3D *scene_root, const GLTFNodeIndex node_index);
	Node3D *_generate_scene(GLTFState &state, const int p_bake_fps);

	void _process_mesh_instances(GLTFState &state, Node3D *scene_root);

	void _assign_scene_names(GLTFState &state);

	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, const float p_time, const GLTFAnimation::Interpolation p_interp);

	void _import_animation(GLTFState &state, AnimationPlayer *ap, const GLTFAnimationIndex index, const int bake_fps);

public:
	virtual uint32_t get_import_flags() const override;
	virtual void get_extensions(List<String> *r_extensions) const override;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps = nullptr, Error *r_err = nullptr) override;
	virtual Ref<Animation> import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps) override;

	EditorSceneImporterGLTF();
};

#endif // EDITOR_SCENE_IMPORTER_GLTF_H
