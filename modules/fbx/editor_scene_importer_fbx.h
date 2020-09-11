/*************************************************************************/
/*  editor_scene_importer_fbx.h                                          */
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

#ifndef EDITOR_SCENE_FBX_IMPORTER_H
#define EDITOR_SCENE_FBX_IMPORTER_H

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

#include "data/import_state.h"
#include "thirdparty/assimp_fbx/FBXDocument.h"
#include "thirdparty/assimp_fbx/FBXImportSettings.h"
#include "thirdparty/assimp_fbx/FBXMeshGeometry.h"
#include "thirdparty/assimp_fbx/FBXUtil.h"
#include "tools/import_utils.h"

#define CONVERT_FBX_TIME(time) static_cast<double>(time) / 46186158000LL

class EditorSceneImporterFBX : public EditorSceneImporter {
private:
	GDCLASS(EditorSceneImporterFBX, EditorSceneImporter);

	struct AssetImportAnimation {
		enum Interpolation {
			INTERP_LINEAR,
			INTERP_STEP,
			INTERP_CATMULLROMSPLINE,
			INTERP_CUBIC_SPLINE
		};
	};

	// ------------------------------------------------------------------------------------------------
	template <typename T>
	const T *ProcessDOMConnection(
			const Assimp::FBX::Document *doc,
			uint64_t current_element,
			bool reverse_lookup = false) {

		const std::vector<const Assimp::FBX::Connection *> &conns = reverse_lookup ? doc->GetConnectionsByDestinationSequenced(current_element) : doc->GetConnectionsBySourceSequenced(current_element);
		//print_verbose("[doc] looking for " + String(element_to_find));
		// using the temp pattern here so we can debug before it returns
		// in some cases we return too early, with 'deformer object base class' in wrong place
		// in assimp this means we can accidentally return too early...
		const T *return_obj = nullptr;

		for (const Assimp::FBX::Connection *con : conns) {
			const Assimp::FBX::Object *source_object = con->SourceObject();
			const Assimp::FBX::Object *dest_object = con->DestinationObject();
			if (source_object && dest_object != nullptr) {
				//print_verbose("[doc] connection name: " + String(source_object->Name().c_str()) + ", dest: " + String(dest_object->Name().c_str()));
				const T *temp = dynamic_cast<const T *>(reverse_lookup ? source_object : dest_object);
				if (temp) {
					return_obj = temp;
				}
			}
		}

		if (return_obj != nullptr) {
			//print_verbose("[doc] returned valid element");
			//print_verbose("Found object for bone");
			return return_obj;
		}

		// safe to return nothing, need to use nullptr here as nullptr is used internally for FBX document.
		return nullptr;
	}

	void BuildDocumentBones(Ref<FBXBone> p_parent_bone,
			ImportState &state, const Assimp::FBX::Document *p_doc,
			uint64_t p_id);

	void BuildDocumentNodes(Ref<PivotTransform> parent_transform, ImportState &state, const Assimp::FBX::Document *doc, uint64_t id, Ref<FBXNode> fbx_parent);

	Spatial *_generate_scene(const String &p_path, const Assimp::FBX::Document *p_document,
			const uint32_t p_flags,
			int p_bake_fps, const int32_t p_max_bone_weights);

	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time, AssetImportAnimation::Interpolation p_interp);
	void _register_project_setting_import(const String generic, const String import_setting_string, const Vector<String> &exts, List<String> *r_extensions, const bool p_enabled) const;

	struct ImportFormat {
		Vector<String> extensions;
		bool is_default = false;
		ImportFormat(){};
		ImportFormat(const Vector<String> &ext, bool def) :
				extensions(ext),
				is_default(def) {
		}
	};

protected:
	static void _bind_methods();

public:
	EditorSceneImporterFBX() {
	}
	~EditorSceneImporterFBX() {
	}

	virtual void get_extensions(List<String> *r_extensions) const;
	virtual uint32_t get_import_flags() const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = NULL);
	void
	GenFBXWeightInfo(Ref<FBXMeshData> &renderer_mesh_data, const Assimp::FBX::MeshGeometry *mesh_geometry,
			Ref<SurfaceTool> st, size_t vertex_id) const;
	void create_mesh_data_skin(ImportState &state, const Ref<FBXNode> &fbx_node, uint64_t mesh_id);
};

#endif // TOOLS_ENABLED
#endif // EDITOR_SCENE_FBX_IMPORTER_H
