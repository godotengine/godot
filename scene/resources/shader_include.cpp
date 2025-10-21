/**************************************************************************/
/*  shader_include.cpp                                                    */
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

#include "shader_include.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "servers/rendering/shader_preprocessor.h"

#ifdef TOOLS_ENABLED
#include "editor/editor_node.h"
#include "editor/shader/shader_editor_plugin.h"
#include "editor/shader/text_shader_editor.h"
#endif // TOOLS_ENABLED

void ShaderInclude::_dependency_changed() {
	emit_changed();
}

void ShaderInclude::set_code(const String &p_code) {
	code = p_code;

	for (const Ref<ShaderInclude> &E : dependencies) {
		E->disconnect_changed(callable_mp(this, &ShaderInclude::_dependency_changed));
	}

	{
		String path = get_path();
		if (path.is_empty()) {
			path = include_path;
		}

		String pp_code;
		HashSet<Ref<ShaderInclude>> new_dependencies;
		ShaderPreprocessor preprocessor;
		Error result = preprocessor.preprocess(p_code, path, pp_code, nullptr, nullptr, nullptr, &new_dependencies);
		if (result == OK) {
			// This ensures previous include resources are not freed and then re-loaded during parse (which would make compiling slower)
			dependencies = new_dependencies;
		}
	}

	for (const Ref<ShaderInclude> &E : dependencies) {
		E->connect_changed(callable_mp(this, &ShaderInclude::_dependency_changed));
	}

	emit_changed();
}

String ShaderInclude::get_code() const {
	return code;
}

void ShaderInclude::set_include_path(const String &p_path) {
	include_path = p_path;
}

void ShaderInclude::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_code", "code"), &ShaderInclude::set_code);
	ClassDB::bind_method(D_METHOD("get_code"), &ShaderInclude::get_code);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "code", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_code", "get_code");
}

// ResourceFormatLoaderShaderInclude

Ref<Resource> ResourceFormatLoaderShaderInclude::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Error error = OK;
	Vector<uint8_t> buffer = FileAccess::get_file_as_bytes(p_path, &error);
	ERR_FAIL_COND_V_MSG(error, nullptr, "Cannot load shader include: " + p_path);

	String str;
	if (buffer.size() > 0) {
		error = str.append_utf8((const char *)buffer.ptr(), buffer.size());
		ERR_FAIL_COND_V_MSG(error, nullptr, "Cannot parse shader include: " + p_path);
	}

	Ref<ShaderInclude> shader_inc;
	shader_inc.instantiate();

	shader_inc->set_include_path(p_path);
	shader_inc->set_code(str);

	if (r_error) {
		*r_error = OK;
	}

	return shader_inc;
}

void ResourceFormatLoaderShaderInclude::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gdshaderinc");
}

bool ResourceFormatLoaderShaderInclude::handles_type(const String &p_type) const {
	return (p_type == "ShaderInclude");
}

String ResourceFormatLoaderShaderInclude::get_resource_type(const String &p_path) const {
	String extension = p_path.get_extension().to_lower();
	if (extension == "gdshaderinc") {
		return "ShaderInclude";
	}
	return "";
}

#define UID_COMMENT_PREFIX "// uid://"
#define UID_COMMENT_SUFFIX "This line is generated, don't modify or remove it."
static ResourceUID::ID extract_uid_from_line(const String &p_line) {
	Vector<String> splits = p_line.strip_edges().substr(3).split(" ", false, 1);
	if (splits.is_empty()) {
		return ResourceUID::INVALID_ID;
	}
	return ResourceUID::get_singleton()->text_to_id(splits[0]);
}

ResourceUID::ID ResourceFormatLoaderShaderInclude::get_resource_uid(const String &p_path) const {
	int64_t uid = ResourceUID::INVALID_ID;

	if (FileAccess::exists(p_path + ".uid")) {
		Ref<FileAccess> file = FileAccess::open(p_path + ".uid", FileAccess::READ);
		if (file.is_valid()) {
			uid = ResourceUID::get_singleton()->text_to_id(file->get_line());
		}
	} else {
		const String extension = p_path.get_extension().to_lower();
		if (extension == "gdshaderinc") {
			Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
			if (file.is_valid()) {
				while (!file->eof_reached()) {
					String line = file->get_line().strip_edges();
					if (!line.is_empty()) {
						if (line.begins_with(UID_COMMENT_PREFIX)) {
							uid = extract_uid_from_line(line);
						}
						break;
					}
				}
			}
		}
	}

	return uid;
}

bool ResourceFormatLoaderShaderInclude::has_custom_uid_support() const {
	return GLOBAL_GET("filesystem/inline_text_resource_uids/shader_include");
}

// ResourceFormatSaverShaderInclude
String ResourceFormatSaverShaderInclude::add_uid_to_source(const String &p_source, const String &p_path, ResourceUID::ID p_uid) const {
	bool need_update = false;
	Vector<String> lines = p_source.split("\n");
	bool uid_comment_valid = false;
	for (Vector<String>::Size i = 0; i < lines.size(); i++) {
		const String &line = lines[i].strip_edges();
		if (line.begins_with(UID_COMMENT_PREFIX)) {
			ResourceUID::ID uid = extract_uid_from_line(line);
			if (uid == ResourceUID::INVALID_ID || p_uid != uid) {
				uid = p_uid == ResourceUID::INVALID_ID ? ResourceSaver::get_resource_id_for_path(p_path, true) : p_uid;
				lines.set(i, vformat("// %s %s", ResourceUID::get_singleton()->id_to_text(uid), UID_COMMENT_SUFFIX));
				if (ResourceUID::get_singleton()->has_id(uid)) {
					ResourceUID::get_singleton()->set_id(uid, p_path);
				} else {
					ResourceUID::get_singleton()->add_id(uid, p_path);
				}
				need_update = true;
			}

			uid_comment_valid = true;
			break;
		} else if (!line.strip_edges().is_empty()) {
			break;
		}
	}

	if (!uid_comment_valid) {
		ResourceUID::ID uid = p_uid == ResourceUID::INVALID_ID ? ResourceSaver::get_resource_id_for_path(p_path, true) : p_uid;
		lines.insert(0, vformat("// %s %s", ResourceUID::get_singleton()->id_to_text(uid), UID_COMMENT_SUFFIX));
		if (ResourceUID::get_singleton()->has_id(uid)) {
			ResourceUID::get_singleton()->set_id(uid, p_path);
		} else {
			ResourceUID::get_singleton()->add_id(uid, p_path);
		}
		need_update = true;
	}

	if (need_update) {
		return String("\n").join(lines);
	} else {
		return p_source;
	}
}

Error ResourceFormatSaverShaderInclude::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<ShaderInclude> shader_inc = p_resource;
	ERR_FAIL_COND_V(shader_inc.is_null(), ERR_INVALID_PARAMETER);

	String source = shader_inc->get_code();

	Error error;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &error);

	ERR_FAIL_COND_V_MSG(error, error, "Cannot save shader include '" + p_path + "'.");

	if (!FileAccess::exists(p_path + ".uid")) {
		if (p_path.begins_with("res://addons/") && GLOBAL_GET("filesystem/inline_text_resource_uids/compatibility/no_inline_text_resource_uids_in_addons")) {
			Ref<FileAccess> f = FileAccess::open(p_path + ".uid", FileAccess::WRITE);
			if (f.is_valid()) {
				ResourceUID::ID uid = ResourceUID::get_singleton()->create_id_for_path(p_path);
				f->store_line(ResourceUID::get_singleton()->id_to_text(uid));
				f->close();
			}
		} else {
			source = add_uid_to_source(source, p_path);
			shader_inc->set_code(source);
#ifdef TOOLS_ENABLED
			if (Engine::get_singleton()->is_editor_hint()) {
				if (ShaderEditorPlugin *shader_editor_plugin = Object::cast_to<ShaderEditorPlugin>(EditorNode::get_editor_data().get_editor_by_name("Shader"))) {
					if (TextShaderEditor *text_shader_editor = Object::cast_to<TextShaderEditor>(shader_editor_plugin->get_shader_editor(shader_inc))) {
						CodeEdit *te = text_shader_editor->get_code_editor()->get_text_editor();
						int column = te->get_caret_column();
						int row = te->get_caret_line();
						int h = te->get_h_scroll();
						int v = te->get_v_scroll();

						te->set_text(source);
						te->set_caret_line(row);
						te->set_caret_column(column);
						te->set_h_scroll(h);
						te->set_v_scroll(v);

						te->tag_saved_version();

						text_shader_editor->get_code_editor()->update_line_and_column();
					}
				}
			}
#endif // TOOLS_ENABLED
		}
	}

	file->store_string(source);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		return ERR_CANT_CREATE;
	}

	return OK;
}

void ResourceFormatSaverShaderInclude::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	const ShaderInclude *shader_inc = Object::cast_to<ShaderInclude>(*p_resource);
	if (shader_inc != nullptr) {
		p_extensions->push_back("gdshaderinc");
	}
}

bool ResourceFormatSaverShaderInclude::recognize(const Ref<Resource> &p_resource) const {
	return p_resource->get_class_name() == "ShaderInclude"; //only shader, not inherited
}

Error ResourceFormatSaverShaderInclude::set_uid(const String &p_path, ResourceUID::ID p_uid) {
	if (FileAccess::exists(p_path + ".uid") || (p_path.begins_with("res://addons/") && GLOBAL_GET("filesystem/inline_text_resource_uids/compatibility/no_inline_text_resource_uids_in_addons"))) {
		Ref<FileAccess> f = FileAccess::open(p_path + ".uid", FileAccess::WRITE);
		if (f.is_valid()) {
			f->store_line(ResourceUID::get_singleton()->id_to_text(p_uid));
			return OK;
		} else {
			return FileAccess::get_open_error();
		}
	} else if (p_path.get_extension().to_lower() == "gdshaderinc") {
		Error err = OK;
		String source = FileAccess::get_file_as_string(p_path, &err);
		if (err != OK) {
			return err;
		}

		source = add_uid_to_source(source, p_path, p_uid);

		Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
		if (f.is_valid()) {
			f->store_string(source);
			err = OK;
		} else {
			err = FileAccess::get_open_error();
		}

		return err;
	}

	return ERR_FILE_UNRECOGNIZED;
}
