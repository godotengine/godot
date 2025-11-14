/**************************************************************************/
/*  shader_template.cpp                                                   */
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

#include "shader_template.h"

#include "core/io/file_access.h"
#include "main/main.h"
#include "servers/rendering/rendering_server.h"
#include "servers/rendering/shader_include_db.h"

void ShaderTemplate::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_mode"), &ShaderTemplate::get_mode);

	ClassDB::bind_method(D_METHOD("set_code", "code"), &ShaderTemplate::set_code);
	ClassDB::bind_method(D_METHOD("get_code"), &ShaderTemplate::get_code);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "code", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_code", "get_code");
}

String ShaderTemplate::_load_include_file(const String &p_path, const String &p_base_path) {
	Error err;

	String include = p_path;
	if (include.is_relative_path()) {
		include = p_base_path.path_join(include);
	}

	Ref<FileAccess> file_inc = FileAccess::open(include, FileAccess::READ, &err);
	if (err != OK) {
		return String();
	}
	return file_inc->get_as_utf8_string();
}

void ShaderTemplate::_update_template() {
	String shader_name = get_path();
	String base_path = "res://"; // We can't guarantee we know folder location, so for now assume paths are relative to project root.

	if (shader_name.is_empty()) {
		uint64_t id = shader_template.get_id();
		shader_name = "resource_" + String::num_uint64(id);
	} else {
		shader_name = shader_name.replace("res://", "");
		shader_name = shader_name.replace(".gdtemplate", "");
		shader_name = shader_name.replace("/", "_");
		shader_name = shader_name.to_camel_case();
	}

	int shader = -1;
	mode = Shader::MODE_MAX;
	String vertex_shader = "";
	String fragment_shader = "";

	// Note, we only split out our vertex and fragment shader here.
	// We do not process includes or compile the shaders.
	// With templates this happens inside of the rendering engine!

	Vector<String> lines = code.split("\n");
	String empty_lines;
	String shader_type;
	for (int lidx = 0; lidx < lines.size(); lidx++) {
		String line = lines[lidx];
		if (line.begins_with("shader_type")) {
			// Extract shader type
			int end_pos = line.find(";");
			if (end_pos != -1) {
				shader_type = line.substr(12, end_pos - 12);
			}
		} else if (line.begins_with("#[")) {
			int pos = line.find("]");
			ERR_FAIL_COND_MSG(pos == -1, "Incomplete tag found in shader template - " + String::num_int64(lidx) + ": " + line);

			String tag = line.substr(2, pos - 2);
			if (tag == "vertex") {
				ERR_FAIL_COND_MSG(shader != -1, "Unexpected vertex shader in shader template - " + String::num_int64(lidx) + ": " + line);
				shader = 0;
				empty_lines = "";
			} else if (tag == "fragment") {
				ERR_FAIL_COND_MSG(shader != 0, "Unexpected fragment shader in shader template - " + String::num_int64(lidx) + ": " + line);
				shader = 1;
				empty_lines = "";
			} else {
				ERR_FAIL_MSG("Unexpected fragment in shader template - " + String::num_int64(lidx) + ": " + line);
			}
		} else if (line.begins_with("#include")) {
			ERR_FAIL_COND_MSG(shader == -1, "Unexpected shader template code - " + String::num_int64(lidx) + ": " + line);

			//process include
			String include = line.replace("#include", "").strip_edges();

			ERR_FAIL_COND_MSG(!include.begins_with("\"") || !include.ends_with("\""), "Malformed #include syntax, expected #include \"<path>\" - " + String::num_int64(lidx) + ": " + line);
			include = include.substr(1, include.length() - 2).strip_edges();

			String include_code;
			if (ShaderIncludeDB::has_built_in_include_file(include)) {
				// Keep the include as is, we'll insert it later in case our shader supports multiple back-ends.
				include_code = line;
			} else {
				include_code = _load_include_file(include, base_path);
			}

			ERR_FAIL_COND_MSG(include_code.is_empty(), "Unexpected include file - " + String::num_int64(lidx) + ": " + line);

			if (shader == 0) {
				vertex_shader += empty_lines + include_code + "\n";
			} else if (shader == 1) {
				fragment_shader += empty_lines + include_code + "\n";
			}
			empty_lines = "";

		} else if (line.is_empty()) {
			// Remove any trailing empty lines
			empty_lines += "\n";
		} else {
			ERR_FAIL_COND_MSG(shader == -1, "Unexpected shader template code - " + String::num_int64(lidx) + ": " + line);

			if (shader == 0) {
				vertex_shader += empty_lines + line + "\n";
			} else if (shader == 1) {
				fragment_shader += empty_lines + line + "\n";
			}
			empty_lines = "";
		}
	}

	if (shader_type == "canvas_item") {
		mode = Shader::MODE_CANVAS_ITEM;
	} else if (shader_type == "particles") {
		mode = Shader::MODE_PARTICLES;
	} else if (shader_type == "sky") {
		mode = Shader::MODE_SKY;
	} else if (shader_type == "fog") {
		mode = Shader::MODE_FOG;
	} else if (shader_type == "spatial") {
		mode = Shader::MODE_SPATIAL;
	} else {
		ERR_FAIL_MSG("Unknown shader type in shader template.");
	}

	ERR_FAIL_COND_MSG(vertex_shader.is_empty(), "No vertex shader code found for this shader template.");
	ERR_FAIL_COND_MSG(fragment_shader.is_empty(), "No vertex shader code found for this shader template.");

	RenderingServer::get_singleton()->shader_template_set_raster_code(shader_template, vertex_shader, fragment_shader, shader_name);
}

void ShaderTemplate::set_path(const String &p_path, bool p_take_over) {
	Resource::set_path(p_path, p_take_over);

	_update_template();
}

Shader::Mode ShaderTemplate::get_mode() const {
	return mode;
}

String ShaderTemplate::get_code() const {
	return code;
}

void ShaderTemplate::set_code(const String &p_code) {
	code = p_code;

	_update_template();
}

RID ShaderTemplate::get_rid() const {
	return shader_template;
}

ShaderTemplate::ShaderTemplate() {
	shader_template = RenderingServer::get_singleton()->shader_template_create();
}

ShaderTemplate::~ShaderTemplate() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RenderingServer::get_singleton()->free_rid(shader_template);
}

////////////

Ref<Resource> ResourceFormatLoaderShaderTemplate::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Error error = OK;
	Vector<uint8_t> buffer = FileAccess::get_file_as_bytes(p_path, &error);
	if (r_error) {
		*r_error = error;
	}
	ERR_FAIL_COND_V_MSG(error, nullptr, "Cannot load shader: " + p_path);

	String str;
	if (buffer.size() > 0) {
		error = str.append_utf8((const char *)buffer.ptr(), buffer.size());
		if (r_error) {
			*r_error = error;
		}
		ERR_FAIL_COND_V_MSG(error, nullptr, "Cannot parse shader: " + p_path);
	}

	Ref<ShaderTemplate> shader_template;
	shader_template.instantiate();
	shader_template->set_code(str);

	if (r_error) {
		*r_error = OK;
	}

	return shader_template;
}

void ResourceFormatLoaderShaderTemplate::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gdtemplate");
}

bool ResourceFormatLoaderShaderTemplate::handles_type(const String &p_type) const {
	return (p_type == "ShaderTemplate");
}

String ResourceFormatLoaderShaderTemplate::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "gdtemplate") {
		return "ShaderTemplate";
	}
	return "";
}

Error ResourceFormatSaverShaderTemplate::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<ShaderTemplate> shader_template = p_resource;
	ERR_FAIL_COND_V(shader_template.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(shader_template->get_mode() >= Shader::MODE_MAX, ERR_INVALID_DATA);

	String code = shader_template->get_code();

	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);

	ERR_FAIL_COND_V_MSG(err, err, "Cannot save shader template '" + p_path + "'.");

	file->store_string(code);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		return ERR_CANT_CREATE;
	}

	return OK;
}

void ResourceFormatSaverShaderTemplate::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<ShaderTemplate>(*p_resource)) {
		p_extensions->push_back("gdtemplate");
	}
}

bool ResourceFormatSaverShaderTemplate::recognize(const Ref<Resource> &p_resource) const {
	return p_resource->get_class_name() == "ShaderTemplate";
}
