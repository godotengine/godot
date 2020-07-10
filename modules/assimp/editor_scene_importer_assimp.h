/*************************************************************************/
/*  editor_scene_importer_assimp.h                                       */
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

#ifndef EDITOR_SCENE_IMPORTER_ASSIMP_H
#define EDITOR_SCENE_IMPORTER_ASSIMP_H

#ifdef TOOLS_ENABLED
#include "core/bind/core_bind.h"
#include "core/io/resource_importer.h"
#include "core/vector.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"

#include <assimp/matrix4x4.h>
#include <assimp/scene.h>
#include <assimp/types.h>
#include <assimp/DefaultLogger.hpp>
#include <assimp/LogStream.hpp>
#include <assimp/Logger.hpp>
#include <map>

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

	Ref<Mesh> _generate_mesh_from_surface_indices(ImportState &state, const Vector<int> &p_surface_indices,
			const aiNode *assimp_node, Ref<Skin> &skin,
			Skeleton3D *&skeleton_assigned);

	// simple object creation functions
	Node3D *create_light(ImportState &state,
			const String &node_name,
			Transform &look_at_transform);
	Node3D *create_camera(
			ImportState &state,
			const String &node_name,
			Transform &look_at_transform);
	// non recursive - linear so must not use recursive arguments
	MeshInstance3D *create_mesh(ImportState &state, const aiNode *assimp_node, const String &node_name, Node *active_node, Transform node_transform);
	// recursive node generator
	void _generate_node(ImportState &state, const aiNode *assimp_node);
	void _insert_animation_track(ImportState &scene, const aiAnimation *assimp_anim, int track_id,
			int anim_fps, Ref<Animation> animation, float ticks_per_second,
			Skeleton3D *skeleton, const NodePath &node_path,
			const String &node_name, aiBone *track_bone);

	void _import_animation(ImportState &state, int p_animation_index, int p_bake_fps);
	Node *get_node_by_name(ImportState &state, String name);
	aiBone *get_bone_from_stack(ImportState &state, aiString name);
	Node3D *_generate_scene(const String &p_path, aiScene *scene, const uint32_t p_flags, int p_bake_fps, const int32_t p_max_bone_weights);

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

	virtual void get_extensions(List<String> *r_extensions) const override;
	virtual uint32_t get_import_flags() const override;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = nullptr) override;
	Ref<Image> load_image(ImportState &state, const aiScene *p_scene, String p_path);

	static void RegenerateBoneStack(ImportState &state);

	void RegenerateBoneStack(ImportState &state, aiMesh *mesh);
};
#endif
#endif
