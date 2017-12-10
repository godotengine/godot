#ifndef EDITOR_SCENE_IMPORTER_GLTF_H
#define EDITOR_SCENE_IMPORTER_GLTF_H

#include "editor/import/resource_importer_scene.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"

class AnimationPlayer;

class EditorSceneImporterGLTF : public EditorSceneImporter {

	GDCLASS(EditorSceneImporterGLTF, EditorSceneImporter);

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

	String _get_component_type_name(uint32_t p_component);
	int _get_component_type_size(int component_type);

	enum GLTFType {
		TYPE_SCALAR,
		TYPE_VEC2,
		TYPE_VEC3,
		TYPE_VEC4,
		TYPE_MAT2,
		TYPE_MAT3,
		TYPE_MAT4,
	};

	String _get_type_name(GLTFType p_component);

	struct GLTFNode {
		//matrices need to be transformed to this
		int parent;

		Transform xform;
		String name;
		Node *godot_node;
		int godot_bone_index;

		int mesh;
		int camera;
		int skin;
		int skeleton_skin;
		int child_of_skeleton; // put as children of skeleton
		Vector<int> skeleton_children; //skeleton put as children of this

		int joint_skin;
		int joint_bone;

		//keep them for animation
		Vector3 translation;
		Quat rotation;
		Vector3 scale;

		Vector<int> children;

		GLTFNode() {
			godot_node = NULL;
			godot_bone_index = -1;
			joint_skin = -1;
			joint_bone = -1;
			child_of_skeleton = -1;
			skeleton_skin = -1;
			mesh = -1;
			camera = -1;
			parent = -1;
			scale = Vector3(1, 1, 1);
		}
	};

	struct GLTFBufferView {

		int buffer;
		int byte_offset;
		int byte_length;
		int byte_stride;
		bool indices;
		//matrices need to be transformed to this

		GLTFBufferView() {
			buffer = 0;
			byte_offset = 0;
			byte_length = 0;
			byte_stride = 0;
			indices = false;
		}
	};

	struct GLTFAccessor {

		int buffer_view;
		int byte_offset;
		int component_type;
		bool normalized;
		int count;
		GLTFType type;
		float min;
		float max;
		int sparse_count;
		int sparse_indices_buffer_view;
		int sparse_indices_byte_offset;
		int sparse_indices_component_type;
		int sparse_values_buffer_view;
		int sparse_values_byte_offset;

		//matrices need to be transformed to this

		GLTFAccessor() {
			buffer_view = 0;
			byte_offset = 0;
			component_type = 0;
			normalized = false;
			count = 0;
			min = 0;
			max = 0;
			sparse_count = 0;
			sparse_indices_byte_offset = 0;
			sparse_values_byte_offset = 0;
		}
	};
	struct GLTFTexture {
		int src_image;
	};

	struct GLTFSkin {

		String name;
		struct Bone {
			Transform inverse_bind;
			int node;
		};

		int skeleton;
		Vector<Bone> bones;

		//matrices need to be transformed to this

		GLTFSkin() {
			skeleton = -1;
		}
	};

	struct GLTFMesh {
		Ref<ArrayMesh> mesh;
		Vector<float> blend_weights;
	};

	struct GLTFCamera {

		bool perspective;
		float fov_size;
		float zfar;
		float znear;

		GLTFCamera() {
			perspective = true;
			fov_size = 65;
			zfar = 500;
			znear = 0.1;
		}
	};

	struct GLTFAnimation {

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
			Vector<Channel<float> > weight_tracks;
		};

		String name;

		Map<int, Track> tracks;
	};

	struct GLTFState {

		Dictionary json;
		int major_version;
		int minor_version;
		Vector<uint8_t> glb_data;

		Vector<GLTFNode *> nodes;
		Vector<Vector<uint8_t> > buffers;
		Vector<GLTFBufferView> buffer_views;
		Vector<GLTFAccessor> accessors;

		Vector<GLTFMesh> meshes; //meshes are loaded directly, no reason not to.
		Vector<Ref<Material> > materials;

		String scene_name;
		Vector<int> root_nodes;

		Vector<GLTFTexture> textures;
		Vector<Ref<Texture> > images;

		Vector<GLTFSkin> skins;
		Vector<GLTFCamera> cameras;

		Set<String> unique_names;

		Vector<GLTFAnimation> animations;

		Map<int, Vector<int> > skin_users; //cache skin users

		~GLTFState() {
			for (int i = 0; i < nodes.size(); i++) {
				memdelete(nodes[i]);
			}
		}
	};

	String _gen_unique_name(GLTFState &state, const String &p_name);

	Ref<Texture> _get_texture(GLTFState &state, int p_texture);

	Error _parse_json(const String &p_path, GLTFState &state);
	Error _parse_glb(const String &p_path, GLTFState &state);

	Error _parse_scenes(GLTFState &state);
	Error _parse_nodes(GLTFState &state);
	Error _parse_buffers(GLTFState &state, const String &p_base_path);
	Error _parse_buffer_views(GLTFState &state);
	GLTFType _get_type_from_str(const String &p_string);
	Error _parse_accessors(GLTFState &state);
	Error _decode_buffer_view(GLTFState &state, int p_buffer_view, double *dst, int skip_every, int skip_bytes, int element_size, int count, GLTFType type, int component_count, int component_type, int component_size, bool normalized, int byte_offset, bool for_vertex);
	Vector<double> _decode_accessor(GLTFState &state, int p_accessor, bool p_for_vertex);
	PoolVector<float> _decode_accessor_as_floats(GLTFState &state, int p_accessor, bool p_for_vertex);
	PoolVector<int> _decode_accessor_as_ints(GLTFState &state, int p_accessor, bool p_for_vertex);
	PoolVector<Vector2> _decode_accessor_as_vec2(GLTFState &state, int p_accessor, bool p_for_vertex);
	PoolVector<Vector3> _decode_accessor_as_vec3(GLTFState &state, int p_accessor, bool p_for_vertex);
	PoolVector<Color> _decode_accessor_as_color(GLTFState &state, int p_accessor, bool p_for_vertex);
	Vector<Quat> _decode_accessor_as_quat(GLTFState &state, int p_accessor, bool p_for_vertex);
	Vector<Transform2D> _decode_accessor_as_xform2d(GLTFState &state, int p_accessor, bool p_for_vertex);
	Vector<Basis> _decode_accessor_as_basis(GLTFState &state, int p_accessor, bool p_for_vertex);
	Vector<Transform> _decode_accessor_as_xform(GLTFState &state, int p_accessor, bool p_for_vertex);

	void _generate_bone(GLTFState &state, int p_node, Vector<Skeleton *> &skeletons, int p_parent_bone);
	void _generate_node(GLTFState &state, int p_node, Node *p_parent, Node *p_owner, Vector<Skeleton *> &skeletons);
	void _import_animation(GLTFState &state, AnimationPlayer *ap, int index, int bake_fps, Vector<Skeleton *> skeletons);

	Spatial *_generate_scene(GLTFState &state, int p_bake_fps);

	Error _parse_meshes(GLTFState &state);
	Error _parse_images(GLTFState &state, const String &p_base_path);
	Error _parse_textures(GLTFState &state);

	Error _parse_materials(GLTFState &state);

	Error _parse_skins(GLTFState &state);

	Error _parse_cameras(GLTFState &state);

	Error _parse_animations(GLTFState &state);

	void _assign_scene_names(GLTFState &state);

	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time, GLTFAnimation::Interpolation p_interp);

public:
	virtual uint32_t get_import_flags() const;
	virtual void get_extensions(List<String> *r_extensions) const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps = NULL, Error *r_err = NULL);
	virtual Ref<Animation> import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps);

	EditorSceneImporterGLTF();
};

#endif // EDITOR_SCENE_IMPORTER_GLTF_H
