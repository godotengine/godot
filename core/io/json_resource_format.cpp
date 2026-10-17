/**************************************************************************/
/*  json_resource_format.cpp                                              */
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

#include "json_resource_format.h"

#include "core/config/engine.h"
#include "core/io/file_access.h"
#include "core/io/json.h"

Ref<Resource> ResourceFormatLoaderJSON::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	if (r_error) {
		*r_error = ERR_FILE_CANT_OPEN;
	}

	if (!FileAccess::exists(p_path)) {
		if (r_error) {
			*r_error = ERR_FILE_NOT_FOUND;
		}
		return Ref<Resource>();
	}

	Ref<JSON> json;
	json.instantiate();

	Error err = json->parse(FileAccess::get_file_as_string(p_path), Engine::get_singleton()->is_editor_hint());
	if (err != OK) {
		String err_text = "Error parsing JSON file at '" + p_path + "', on line " + itos(json->get_error_line()) + ": " + json->get_error_message();

		if (Engine::get_singleton()->is_editor_hint()) {
			// If running on editor, still allow opening the JSON so the code editor can edit it.
			WARN_PRINT(err_text);
		} else {
			if (r_error) {
				*r_error = err;
			}
			ERR_PRINT(err_text);
			return Ref<Resource>();
		}
	}

	if (r_error) {
		*r_error = OK;
	}

	return json;
}

void ResourceFormatLoaderJSON::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("json");
}

bool ResourceFormatLoaderJSON::handles_type(const String &p_type) const {
	return (p_type == "JSON");
}

String ResourceFormatLoaderJSON::get_resource_type(const String &p_path) const {
	if (p_path.has_extension("json")) {
		return "JSON";
	}
	return "";
}

Error ResourceFormatSaverJSON::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<JSON> json = p_resource;
	ERR_FAIL_COND_V(json.is_null(), ERR_INVALID_PARAMETER);

	String source = json->get_parsed_text().is_empty() ? JSON::stringify(json->get_data(), "\t", false, true) : json->get_parsed_text();

	Error err;
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);

	ERR_FAIL_COND_V_MSG(err, err, vformat("Cannot save json '%s'.", p_path));

	file->store_string(source);
	if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
		return ERR_CANT_CREATE;
	}

	return OK;
}

void ResourceFormatSaverJSON::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	Ref<JSON> json = p_resource;
	if (json.is_valid()) {
		p_extensions->push_back("json");
	}
}

bool ResourceFormatSaverJSON::recognize(const Ref<Resource> &p_resource) const {
	return p_resource->get_class_name() == "JSON"; //only json, not inherited
}
