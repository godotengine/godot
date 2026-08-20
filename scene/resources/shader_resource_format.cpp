/**************************************************************************/
/*  shader_resource_format.cpp                                            */
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

#include "shader_resource_format.h"

#include "core/io/file_access.h"
#include "scene/resources/shader.h"

Ref<Resource> ResourceFormatLoaderShader::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	Error error = OK;
	Vector<uint8_t> buffer = FileAccess::get_file_as_bytes(p_path, &error);
	ERR_FAIL_COND_V_MSG(error, nullptr, "Cannot load shader: " + p_path);

	String str;
	if (buffer.size() > 0) {
		error = str.append_utf8((const char *)buffer.ptr(), buffer.size());
		ERR_FAIL_COND_V_MSG(error, nullptr, "Cannot parse shader: " + p_path);
	}

	Ref<Shader> shader;
	shader.instantiate();

	shader->set_include_path(p_path);
	shader->set_code(str);

	if (r_error) {
		*r_error = OK;
	}

	return shader;
}

void ResourceFormatLoaderShader::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gdshader");
}

bool ResourceFormatLoaderShader::handles_type(const String &p_type) const {
	return (p_type == "Shader");
}

String ResourceFormatLoaderShader::get_resource_type(const String &p_path) const {
	if (p_path.has_extension("gdshader")) {
		return "Shader";
	}
	return "";
}

Error ResourceFormatSaverShader::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<Shader> shader = p_resource;
	ERR_FAIL_COND_V(shader.is_null(), ERR_INVALID_PARAMETER);

	String source = shader->get_code();

	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);

	ERR_FAIL_COND_V_MSG(err, err, "Cannot save shader '" + p_path + "'.");

	file->store_string(source);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		return ERR_CANT_CREATE;
	}

	return OK;
}

void ResourceFormatSaverShader::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	if (const Shader *shader = Object::cast_to<Shader>(*p_resource)) {
		if (shader->is_text_shader()) {
			p_extensions->push_back("gdshader");
		}
	}
}

bool ResourceFormatSaverShader::recognize(const Ref<Resource> &p_resource) const {
	return p_resource->get_class_name() == "Shader"; //only shader, not inherited
}
