/**************************************************************************/
/*  script_builtin_scanner.cpp                                            */
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

#include "script_builtin_scanner.h"

#include "core/core_bind.h"
#include "editor/editor_node.h"
#include "editor/plugins/text_editor.h"
#include "modules/gdscript/gdscript.h"
#include "scene/resources/packed_scene.h"

TypedArray<Dictionary> ScriptBuiltinScanner::scan(Ref<FileAccess> p_file, String p_display_path, String p_pattern, bool p_match_case, bool p_whole_words, Size2i p_range) {
	TypedArray<Dictionary> matches;

	Vector<Ref<Script>> found_scripts = _parse_scene(p_file->get_path());
	for (Ref<Script> scr : found_scripts) {
		String source_code = scr->get_source_code();
		String line;
		int line_number = 1; // Line number starts at 1.
		for (int char_idx = 0; char_idx < source_code.length(); char_idx++) {
			line += source_code[char_idx];
			if (source_code[char_idx] == '\n') {
				if (!_scan_line(line, line_number, p_file->get_path(), p_display_path, p_pattern, p_match_case, p_whole_words, p_range, matches)) {
					break;
				}
				line_number++;
				line = "";
			}
		}

		// Update the display_text and file_path to reference the actual scr itself (within the scene file).
		for (int k = 0; k < matches.size(); k++) {
			Dictionary match = matches[k];
			match["display_text"] = scr->get_path();
			match["file_path"] = scr->get_path();
		}
	}

	return matches;
}

bool ScriptBuiltinScanner::replace(Ref<FileAccess> p_file, String p_file_path, String p_display_path, TypedArray<Dictionary> p_locations, bool p_match_case, bool p_whole_words, String p_search_text, String p_new_text, Size2i p_range) {
	Ref<Script> scr = _parse_scene(p_file_path, p_display_path);
	if (scr == nullptr) {
		// Failed to parse the builtin script.
		return false;
	}

	String result;
	bool ok = _replace(scr->get_source_code(), p_file_path, p_display_path, p_locations, p_match_case, p_whole_words, p_search_text, p_new_text, p_range, result);

	if (!ok) {
		// Nothing to replace, sort-of a success.
		return true;
	}

	// TODO: Reload opened builtin scripts.
	scr->set_source_code(result);
	ResourceSaver::save(scr, p_display_path);

	return true;
}

Vector<Ref<Script>> ScriptBuiltinScanner::_parse_scene(String p_file_path) const {
	Vector<Ref<Script>> found_scripts;

	Error err;
	Ref<PackedScene> packed_scene = ResourceLoader::load(p_file_path, "", ResourceFormatLoader::CACHE_MODE_REUSE, &err);
	ERR_FAIL_COND_V_MSG(packed_scene.is_null(), found_scripts, vformat("Failed opening PackedScene with error: %s", err));
	Ref<SceneState> state = packed_scene->get_state();

	for (int i = 0; i < state->get_node_count(); i++) {
		for (int j = 0; j < state->get_node_property_count(i); j++) {
			StringName property_name = state->get_node_property_name(i, j);
			if (property_name != SNAME("script")) {
				continue;
			}
			Variant scr = state->get_node_property_value(i, j);
			Ref<Script> gdscript = Object::cast_to<Script>(scr);
			if (gdscript != nullptr && gdscript->is_built_in()) {
				found_scripts.append(gdscript);
			}
		}
	}

	return found_scripts;
}

Ref<Script> ScriptBuiltinScanner::_parse_scene(String p_file_path, String p_display_path) const {
	for (Ref<Script> scr : _parse_scene(p_file_path)) {
		if (scr->get_path() == p_display_path) {
			return scr;
		}
	}

	return nullptr;
}
