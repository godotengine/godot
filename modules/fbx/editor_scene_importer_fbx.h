/*************************************************************************/
/*  editor_scene_importer_fbx.h                                          */
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

#ifndef EDITOR_SCENE_IMPORTER_FBX_H
#define EDITOR_SCENE_IMPORTER_FBX_H

#ifdef TOOLS_ENABLED

#include "data/import_state.h"
#include "tools/import_utils.h"

#include "core/bind/core_bind.h"
#include "core/dictionary.h"
#include "core/io/resource_importer.h"
#include "core/local_vector.h"
#include "core/ustring.h"
#include "core/vector.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"

#include "fbx_parser/FBXDocument.h"
#include "fbx_parser/FBXImportSettings.h"
#include "fbx_parser/FBXMeshGeometry.h"
#include "fbx_parser/FBXUtil.h"

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
			const FBXDocParser::Document *doc,
			uint64_t current_element,
			bool reverse_lookup = false) {
		const std::vector<const FBXDocParser::Connection *> &conns = reverse_lookup ? doc->GetConnectionsByDestinationSequenced(current_element) : doc->GetConnectionsBySourceSequenced(current_element);
		//print_verbose("[doc] looking for " + String(element_to_find));
		// using the temp pattern here so we can debug before it returns
		// in some cases we return too early, with 'deformer object base class' in wrong place
		// in assimp this means we can accidentally return too early...
		const T *return_obj = nullptr;

		for (const FBXDocParser::Connection *con : conns) {
			const FBXDocParser::Object *source_object = con->SourceObject();
			const FBXDocParser::Object *dest_object = con->DestinationObject();
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
			ImportState &state, const FBXDocParser::Document *p_doc,
			uint64_t p_id);

	void BuildDocumentNodes(Ref<PivotTransform> parent_transform, ImportState &state, const FBXDocParser::Document *doc, uint64_t id, Ref<FBXNode> fbx_parent);

	Spatial *_generate_scene(const String &p_path, const FBXDocParser::Document *p_document,
			const uint32_t p_flags,
			int p_bake_fps,
			const uint32_t p_compress_flags,
			const int32_t p_max_bone_weights,
			bool p_is_blender_fbx);

	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time, AssetImportAnimation::Interpolation p_interp);

public:
	EditorSceneImporterFBX();
	~EditorSceneImporterFBX() {}

	virtual void get_extensions(List<String> *r_extensions) const;
	virtual uint32_t get_import_flags() const;
	virtual Node *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, uint32_t p_compress_flags, List<String> *r_missing_deps, Error *r_err = nullptr);
	void create_mesh_data_skin(ImportState &state, const Ref<FBXNode> &fbx_node, uint64_t mesh_id);
};

#endif // TOOLS_ENABLED
#endif // EDITOR_SCENE_IMPORTER_FBX_H
