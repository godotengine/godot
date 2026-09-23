/**************************************************************************/
/*  slime_ai_project_inspector.cpp                                        */
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

#include "slime_ai_project_inspector.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "editor/editor_data.h"
#include "editor/editor_node.h"
#include "editor/script/script_editor_plugin.h"
#include "editor/slime_ai/slime_ai_scene_inspector.h"

namespace SlimeAI {

Dictionary ProjectInspector::inspect(int p_asset_limit) {
	Dictionary result;
	ProjectSettings *settings = ProjectSettings::get_singleton();
	if (!settings) {
		result["error"] = "CAPABILITY_UNAVAILABLE";
		return result;
	}

	const String project_root = settings->get_resource_path();
	result["project_ref"] = "project:" + project_root.sha256_text();
	result["project_name"] = settings->get_setting("application/config/name", "Unnamed Project");
	result["engine"] = Engine::get_singleton()->get_version_info();
	result["project_disk_revision"] = FileAccess::exists("res://project.godot") ? FileAccess::get_sha256("res://project.godot") : String();
	result["runtime_state"] = "not_inspected";
	result["staged_state"] = "not_inspected";

	Array scenes;
	Array unsaved_buffers;
	if (EditorNode *editor = EditorNode::get_singleton()) {
		EditorData &data = EditorNode::get_editor_data();
		for (int i = 0; i < data.get_edited_scene_count(); i++) {
			Dictionary scene;
			Node *root = data.get_edited_scene_root(i);
			scene["scene_ref"] = SceneInspector::scene_ref(root);
			scene["path"] = data.get_scene_path(i);
			scene["unsaved"] = editor->is_scene_unsaved(i);
			scene["selected"] = i == data.get_edited_scene();
			scenes.push_back(scene);
		}
		if (ScriptEditor *script_editor = ScriptEditor::get_singleton()) {
			const PackedStringArray paths = script_editor->get_unsaved_files();
			for (const String &path : paths) {
				unsaved_buffers.push_back(path);
			}
		}
	}
	result["open_scenes"] = scenes;
	result["unsaved_buffers"] = unsaved_buffers;
	result["editor_state"] = EditorNode::get_singleton() ? "live_editor" : "editor_unavailable";

	PackedStringArray root_files = DirAccess::get_files_at("res://");
	root_files.sort();
	const int limit = CLAMP(p_asset_limit, 1, 256);
	Array assets;
	for (int i = 0; i < MIN(root_files.size(), limit); i++) {
		Dictionary asset;
		asset["path"] = "res://" + root_files[i];
		asset["extension"] = root_files[i].get_extension();
		assets.push_back(asset);
	}
	result["root_asset_summaries"] = assets;
	result["assets_truncated"] = root_files.size() > limit;
	result["asset_scope"] = "res:// root files only";
	Array tools;
	tools.push_back("project_inspect");
	tools.push_back("scene_inspect");
	tools.push_back("object_inspect");
	tools.push_back("api_describe");
	result["registered_inspection_tools"] = tools;
	return result;
}

} // namespace SlimeAI
