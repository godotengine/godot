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

#include <assimp/matrix4x4.h>
#include <assimp/scene.h>
#include <assimp/types.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/LogStream.hpp>
#include <assimp/Logger.hpp>

#include "import_state.h"
#include "import_utils.h"

using namespace AssimpImporter;

class AssimpStream : public Assimp::LogStream {
public:
	// Constructor
	AssimpStream() {}

	// Destructor
	~AssimpStream() {}
	// Write something using your own functionality
	void write(const char *message) {
		print_verbose(String("Open Asset Import: ") + String(message).strip_edges());
	}
};

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

	void _calc_tangent_from_mesh(const aiMesh *ai_mesh, int i, int tri_index, int index, PoolColorArray::Write &w);
	void _set_texture_mapping_mode(aiTextureMapMode *map_mode, Ref<Texture> texture);

	Ref<Mesh> _generate_mesh_from_surface_indices(ImportState &state, const Vector<int> &p_surface_indices, const aiNode *assimp_node, Skeleton *p_skeleton = NULL);

	// utility for node creation
	void attach_new_node(ImportState &state, Spatial *new_node, const aiNode *node, Node *parent_node, String Name, Transform &transform);
	// simple object creation functions
	void create_light(ImportState &state, RecursiveState &recursive_state);
	void create_camera(ImportState &state, RecursiveState &recursive_state);
	void create_bone(ImportState &state, RecursiveState &recursive_state);
	// non recursive - linear so must not use recursive arguments
	void create_mesh(ImportState &state, const aiNode *assimp_node, const String &node_name, Node *current_node, Node *parent_node, Transform node_transform);

	// recursive node generator
	void _generate_node(ImportState &state, Skeleton *skeleton, const aiNode *assimp_node, Node *parent_node);
	// runs after _generate_node as it must then use pre-created godot skeleton.
	void generate_mesh_phase_from_skeletal_mesh(ImportState &state);
	void _insert_animation_track(ImportState &scene, const aiAnimation *assimp_anim, int p_track, int p_bake_fps, Ref<Animation> animation, float ticks_per_second, Skeleton *p_skeleton, const NodePath &p_path, const String &p_name);

	void _import_animation(ImportState &state, int p_animation_index, int p_bake_fps);

	Spatial *_generate_scene(const String &p_path, aiScene *scene, const uint32_t p_flags, int p_bake_fps, const int32_t p_max_bone_weights);

	String _assimp_anim_string_to_string(const aiString &p_string) const;
	String _assimp_raw_string_to_string(const aiString &p_string) const;
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
	Ref<Image> load_image(ImportState &state, const aiScene *p_scene, String p_path);
};
#endif
#endif
