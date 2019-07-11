/*************************************************************************/
/*  editor_scene_importer_assimp.h                                       */
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

#ifndef EDITOR_SCENE_IMPORTER_ASSIMP_H
#define EDITOR_SCENE_IMPORTER_ASSIMP_H

#ifdef TOOLS_ENABLED
#include "core/bind/core_bind.h"
#include "core/io/resource_importer.h"
#include "core/vector.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"

#include "assimp/DefaultLogger.hpp"
#include "assimp/LogStream.hpp"
#include "assimp/Logger.hpp"
#include "assimp/matrix4x4.h"
#include "assimp/scene.h"
#include "assimp/types.h"

class AssimpStream : public Assimp::LogStream {
public:
	// Constructor
	AssimpStream();

	// Destructor
	~AssimpStream();
	// Write something using your own functionality
	void write(const char *message);
};

#define AI_MATKEY_FBX_MAYA_BASE_COLOR_FACTOR "$raw.Maya|baseColor", 0, 0
#define AI_MATKEY_FBX_MAYA_METALNESS_FACTOR "$raw.Maya|metalness", 0, 0
#define AI_MATKEY_FBX_MAYA_DIFFUSE_ROUGHNESS_FACTOR "$raw.Maya|diffuseRoughness", 0, 0

#define AI_MATKEY_FBX_MAYA_METALNESS_TEXTURE "$raw.Maya|metalness|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_METALNESS_UV_XFORM "$raw.Maya|metalness|uvtrafo", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_DIFFUSE_ROUGHNESS_TEXTURE "$raw.Maya|diffuseRoughness|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_DIFFUSE_ROUGHNESS_UV_XFORM "$raw.Maya|diffuseRoughness|uvtrafo", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_BASE_COLOR_TEXTURE "$raw.Maya|baseColor|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_BASE_COLOR_UV_XFORM "$raw.Maya|baseColor|uvtrafo", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_NORMAL_TEXTURE "$raw.Maya|normalCamera|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_NORMAL_UV_XFORM "$raw.Maya|normalCamera|uvtrafo", aiTextureType_UNKNOWN, 0

#define AI_MATKEY_FBX_NORMAL_TEXTURE "$raw.Maya|normalCamera|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_NORMAL_UV_XFORM "$raw.Maya|normalCamera|uvtrafo", aiTextureType_UNKNOWN, 0

#define AI_MATKEY_FBX_MAYA_STINGRAY_DISPLACEMENT_SCALING_FACTOR "$raw.Maya|displacementscaling", 0, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_BASE_COLOR_FACTOR "$raw.Maya|base_color", 0, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_FACTOR "$raw.Maya|emissive", 0, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_METALLIC_FACTOR "$raw.Maya|metallic", 0, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_ROUGHNESS_FACTOR "$raw.Maya|roughness", 0, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_INTENSITY_FACTOR "$raw.Maya|emissive_intensity", 0, 0

#define AI_MATKEY_FBX_MAYA_STINGRAY_NORMAL_TEXTURE "$raw.Maya|TEX_normal_map|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_NORMAL_UV_XFORM "$raw.Maya|TEX_normal_map|uvtrafo", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_COLOR_TEXTURE "$raw.Maya|TEX_color_map|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_COLOR_UV_XFORM "$raw.Maya|TEX_color_map|uvtrafo", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_METALLIC_TEXTURE "$raw.Maya|TEX_metallic_map|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_METALLIC_UV_XFORM "$raw.Maya|TEX_metallic_map|uvtrafo", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_ROUGHNESS_TEXTURE "$raw.Maya|TEX_roughness_map|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_ROUGHNESS_UV_XFORM "$raw.Maya|TEX_roughness_map|uvtrafo", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_TEXTURE "$raw.Maya|TEX_emissive_map|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_EMISSIVE_UV_XFORM "$raw.Maya|TEX_emissive_map|uvtrafo", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_AO_TEXTURE "$raw.Maya|TEX_ao_map|file", aiTextureType_UNKNOWN, 0
#define AI_MATKEY_FBX_MAYA_STINGRAY_AO_UV_XFORM "$raw.Maya|TEX_ao_map|uvtrafo", aiTextureType_UNKNOWN, 0

class EditorSceneImporterAssimp : public EditorSceneImporter {
private:
	GDCLASS(EditorSceneImporterAssimp, EditorSceneImporter);
	const String ASSIMP_FBX_KEY = "_$AssimpFbx$";

	struct AssetImportAnimation {
		enum Interpolation {
			INTERP_LINEAR,
			INTERP_STEP,
			INTERP_CATMULLROMSPLINE,
			INTERP_CUBIC_SPLINE
		};
	};

	struct AssetImportFbx {
		enum ETimeMode {
			TIME_MODE_DEFAULT = 0,
			TIME_MODE_120 = 1,
			TIME_MODE_100 = 2,
			TIME_MODE_60 = 3,
			TIME_MODE_50 = 4,
			TIME_MODE_48 = 5,
			TIME_MODE_30 = 6,
			TIME_MODE_30_DROP = 7,
			TIME_MODE_NTSC_DROP_FRAME = 8,
			TIME_MODE_NTSC_FULL_FRAME = 9,
			TIME_MODE_PAL = 10,
			TIME_MODE_CINEMA = 11,
			TIME_MODE_1000 = 12,
			TIME_MODE_CINEMA_ND = 13,
			TIME_MODE_CUSTOM = 14,
			TIME_MODE_TIME_MODE_COUNT = 15
		};
		enum UpAxis {
			UP_VECTOR_AXIS_X = 1,
			UP_VECTOR_AXIS_Y = 2,
			UP_VECTOR_AXIS_Z = 3
		};
		enum FrontAxis {
			FRONT_PARITY_EVEN = 1,
			FRONT_PARITY_ODD = 2,
		};

		enum CoordAxis {
			COORD_RIGHT = 0,
			COORD_LEFT = 1
		};
	};

	struct ImportState {

		String path;
		const aiScene *assimp_scene;
		uint32_t max_bone_weights;
		Spatial *root;
		Map<String, Ref<Mesh> > mesh_cache;
		Map<int, Ref<Material> > material_cache;
		Map<String, int> light_cache;
		Map<String, int> camera_cache;
		Vector<Skeleton *> skeletons;
		Map<String, int> bone_owners; //maps bones to skeleton index owned by
		Map<String, Node *> node_map;
		Map<MeshInstance *, Skeleton *> mesh_skeletons;
		bool fbx; //for some reason assimp does some things different for FBX
		AnimationPlayer *animation_player;
	};

	struct BoneInfo {
		uint32_t bone;
		float weight;
	};

	struct SkeletonHole { //nodes may be part of the skeleton by used by vertex
		String name;
		String parent;
		Transform pose;
		const aiNode *node;
	};

	const Transform _assimp_matrix_transform(const aiMatrix4x4 p_matrix);
	String _assimp_get_string(const aiString p_string) const;
	Transform _get_global_assimp_node_transform(const aiNode *p_current_node);

	void _calc_tangent_from_mesh(const aiMesh *ai_mesh, int i, int tri_index, int index, PoolColorArray::Write &w);
	void _set_texture_mapping_mode(aiTextureMapMode *map_mode, Ref<Texture> texture);
	void _find_texture_path(const String &p_path, String &path, bool &r_found);
	void _find_texture_path(const String &p_path, _Directory &dir, String &path, bool &found, String extension);

	Ref<Texture> _load_texture(ImportState &state, String p_path);
	Ref<Material> _generate_material_from_index(ImportState &state, int p_index, bool p_double_sided);
	Ref<Mesh> _generate_mesh_from_surface_indices(ImportState &state, const Vector<int> &p_surface_indices, Skeleton *p_skeleton = NULL, bool p_double_sided_material = false);
	void _generate_node(ImportState &state, const aiNode *p_assimp_node, Node *p_parent);
	void _generate_bone_groups(ImportState &state, const aiNode *p_assimp_node, Map<String, int> &ownership, Map<String, Transform> &bind_xforms);
	void _fill_node_relationships(ImportState &state, const aiNode *p_assimp_node, Map<String, int> &ownership, Map<int, int> &skeleton_map, int p_skeleton_id, Skeleton *p_skeleton, const String &p_parent_name, int &holecount, const Vector<SkeletonHole> &p_holes, const Map<String, Transform> &bind_xforms);
	void _generate_skeletons(ImportState &state, const aiNode *p_assimp_node, Map<String, int> &ownership, Map<int, int> &skeleton_map, const Map<String, Transform> &bind_xforms);

	void _insert_animation_track(ImportState &scene, const aiAnimation *assimp_anim, int p_track, int p_bake_fps, Ref<Animation> animation, float ticks_per_second, Skeleton *p_skeleton, const NodePath &p_path, const String &p_name);

	void _import_animation(ImportState &state, int p_animation_index, int p_bake_fps);

	Spatial *_generate_scene(const String &p_path, const aiScene *scene, const uint32_t p_flags, int p_bake_fps, const int32_t p_max_bone_weights);

	String _assimp_anim_string_to_string(const aiString p_string) const;
	String _assimp_raw_string_to_string(const aiString p_string) const;
	float _get_fbx_fps(int32_t time_mode, const aiScene *p_scene);
	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time, AssetImportAnimation::Interpolation p_interp);
	void _register_project_setting_import(const String generic, const String import_setting_string, const Vector<String> &exts, List<String> *r_extensions, const bool p_enabled) const;

	struct ImportFormat {
		Vector<String> extensions;
		bool is_default;
	};

protected:
	static void _bind_methods();

public:
	EditorSceneImporterAssimp() {
		Assimp::DefaultLogger::create("", Assimp::Logger::VERBOSE);
		unsigned int severity = Assimp::Logger::Info | Assimp::Logger::Err | Assimp::Logger::Warn;
		Assimp::DefaultLogger::get()->attachStream(new AssimpStream(), severity);
	}
	~EditorSceneImporterAssimp() {
		Assimp::DefaultLogger::kill();
	}

	virtual void get_extensions(List<String> *r_extensions) const;
	virtual uint32_t get_import_flags() const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = NULL);
	virtual Ref<Animation> import_animation(const String &p_path, uint32_t p_flags, int p_bake_fps);
};
#endif
#endif
