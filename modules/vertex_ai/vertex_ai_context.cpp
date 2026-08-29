/**************************************************************************/
/*  vertex_ai_context.cpp                                                */
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

#include "vertex_ai_context.h"

#include "core/io/dir_access.h"
#include "core/object/class_db.h"

static void _gather_files(const String &p_path, const String &p_root, int p_depth, Dictionary &p_out, int p_max_depth) {
	if (p_depth > p_max_depth) {
		return;
	}
	Ref<DirAccess> dir = DirAccess::open(p_path);
	if (dir.is_null()) {
		return;
	}
	dir->list_dir_begin();
	String name = dir->get_next();
	while (!name.is_empty()) {
		if (name == "." || name == "..") {
			name = dir->get_next();
			continue;
		}
		String full = p_path.ends_with("/") ? p_path + name : p_path + "/" + name;
		String rel = full.replace(p_root, "");
		if (dir->current_is_dir()) {
			_gather_files(full, p_root, p_depth + 1, p_out, p_max_depth);
		} else {
			String ext = name.get_extension().to_lower();
			p_out[rel] = ext;
		}
		name = dir->get_next();
	}
}

void VertexAIContext::build_from_project(const String &p_root) {
	Dictionary structure;
	_gather_files(p_root, p_root, 0, structure, 6);
	data["project_structure"] = structure;
	Dictionary scripts, scenes, shaders, assets;
	for (const Variant *key = structure.next(nullptr); key; key = structure.next(key)) {
		String path = *key;
		String ext = String(structure[*key]);
		if (ext == "gd" || ext == "cs" || ext == "cpp" || ext == "glec" || ext == "swift") {
			scripts[path] = ext;
		} else if (ext == "tscn" || ext == "scn" || ext == "res" || ext == "tres") {
			scenes[path] = ext;
		} else if (ext == "gdshader" || ext == "shader" || ext == "glsl") {
			shaders[path] = ext;
		} else if (ext == "png" || ext == "jpg" || ext == "webp" || ext == "wav" || ext == "ogg" || ext == "mp3") {
			assets[path] = ext;
		}
	}
	data["scripts"] = scripts;
	data["scenes"] = scenes;
	data["shaders"] = shaders;
	data["assets"] = assets;
}

void VertexAIContext::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_data"), &VertexAIContext::get_data);
	ClassDB::bind_method(D_METHOD("set_data", "data"), &VertexAIContext::set_data);
	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data"), "set_data", "get_data");

	ClassDB::bind_method(D_METHOD("build_from_project", "root"), &VertexAIContext::build_from_project, DEFVAL("res://"));
	ClassDB::bind_method(D_METHOD("to_dictionary"), &VertexAIContext::to_dictionary);
}
