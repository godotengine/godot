/*************************************************************************/
/*  editor_scene_importer_fbx.h                                          */
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

#ifndef EDITOR_SCENE_IMPORTER_FBX_H
#define EDITOR_SCENE_IMPORTER_FBX_H

#ifdef TOOLS_ENABLED

#include "data/import_state.h"
#include "tools/import_utils.h"

#include "core/io/resource_importer.h"
#include "core/string/ustring.h"
#include "core/templates/local_vector.h"
#include "core/templates/vector.h"
#include "core/variant/dictionary.h"
#include "editor/import/resource_importer_scene.h"
#include "editor/project_settings_editor.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/animation.h"
#include "scene/resources/surface_tool.h"

#include "fbx_parser/FBXDocument.h"
#include "fbx_parser/FBXImportSettings.h"
#include "fbx_parser/FBXMeshGeometry.h"
#include "fbx_parser/FBXUtil.h"

#define CONVERT_FBX_TIME(time) static_cast<double>(time) / 46186158000LL

class EditorSceneFormatImporterFBX : public EditorSceneFormatImporter {
private:
	GDCLASS(EditorSceneFormatImporterFBX, EditorSceneFormatImporter);

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

	Node3D *_generate_scene(const String &p_path, const FBXDocParser::Document *p_document,
			const uint32_t p_flags,
			int p_bake_fps,
			const int32_t p_max_bone_weights,
			bool p_is_blender_fbx);

	template <class T>
	T _interpolate_track(const Vector<float> &p_times, const Vector<T> &p_values, float p_time, AssetImportAnimation::Interpolation p_interp);
	void _register_project_setting_import(const String generic, const String import_setting_string, const Vector<String> &exts, List<String> *r_extensions, const bool p_enabled) const;

public:
	EditorSceneFormatImporterFBX() {}
	~EditorSceneFormatImporterFBX() {}

	virtual void get_extensions(List<String> *r_extensions) const override;
	virtual uint32_t get_import_flags() const override;
	virtual Node3D *import_scene(const String &p_path, uint32_t p_flags, int p_bake_fps, List<String> *r_missing_deps, Error *r_err = nullptr) override;
};

#endif // TOOLS_ENABLED
#endif // EDITOR_SCENE_IMPORTER_FBX_H
