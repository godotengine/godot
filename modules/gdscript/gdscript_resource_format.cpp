/**************************************************************************/
/*  gdscript_resource_format.cpp                                          */
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

#include "gdscript_resource_format.h"

#include "gdscript_cache.h"
#include "gdscript_parser.h"

#include "core/io/file_access.h"
#include "core/object/class_db.h"

Ref<Resource> ResourceFormatLoaderGDScript::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Error err;
	bool ignoring = p_cache_mode == CACHE_MODE_IGNORE || p_cache_mode == CACHE_MODE_IGNORE_DEEP;
	Ref<GDScript> scr = GDScriptCache::get_full_script(p_original_path, err, "", ignoring);

	if (err && scr.is_valid()) {
		// If !scr.is_valid(), the error was likely from scr->load_source_code(), which already generates an error.
		ERR_PRINT_ED(vformat(R"(Failed to load script "%s" with error "%s".)", p_original_path, error_names[err]));
	}

	if (r_error) {
		// Don't fail loading because of parsing error.
		*r_error = scr.is_valid() ? OK : err;
	}

	return scr;
}

void ResourceFormatLoaderGDScript::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gd");
	p_extensions->push_back("gdc");
}

bool ResourceFormatLoaderGDScript::handles_type(const String &p_type) const {
	return (p_type == "Script" || p_type == "GDScript");
}

String ResourceFormatLoaderGDScript::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "gd" || el == "gdc") {
		return "GDScript";
	}
	return "";
}

void ResourceFormatLoaderGDScript::get_dependencies(const String &p_path, List<String> *p_dependencies, bool p_add_types) {
	Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_MSG(file.is_null(), "Cannot open file '" + p_path + "'.");

	String source = file->get_as_utf8_string();
	if (source.is_empty()) {
		return;
	}

	GDScriptParser parser;
	if (OK != parser.parse(source, p_path, false)) {
		return;
	}

	for (const String &E : parser.get_dependencies()) {
		p_dependencies->push_back(E);
	}
}

void ResourceFormatLoaderGDScript::get_classes_used(const String &p_path, HashSet<StringName> *r_classes) {
	Ref<GDScript> scr = ResourceLoader::load(p_path);
	if (scr.is_null()) {
		return;
	}

	const String source = scr->get_source_code();
	GDScriptTokenizerText tokenizer;
	tokenizer.set_source_code(source);
	GDScriptTokenizer::Token current = tokenizer.scan();
	while (current.type != GDScriptTokenizer::Token::TK_EOF) {
		if (!current.is_identifier()) {
			current = tokenizer.scan();
			continue;
		}

		int insert_idx = 0;
		for (int i = 0; i < current.start_line - 1; i++) {
			insert_idx = source.find("\n", insert_idx) + 1;
		}
		// Insert the "cursor" character, needed for the lookup to work.
		const String source_with_cursor = source.insert(insert_idx + current.start_column, String::chr(0xFFFF));

		ScriptLanguage::LookupResult result;
		if (scr->get_language()->lookup_code(source_with_cursor, current.get_identifier(), p_path, nullptr, result) == OK) {
			if (!result.class_name.is_empty() && ClassDB::class_exists(result.class_name)) {
				r_classes->insert(result.class_name);
			}

			if (result.type == ScriptLanguage::LOOKUP_RESULT_CLASS_PROPERTY) {
				PropertyInfo prop;
				if (ClassDB::get_property_info(result.class_name, result.class_member, &prop)) {
					if (!prop.class_name.is_empty() && ClassDB::class_exists(prop.class_name)) {
						r_classes->insert(prop.class_name);
					}
					if (!prop.hint_string.is_empty() && ClassDB::class_exists(prop.hint_string)) {
						r_classes->insert(prop.hint_string);
					}
				}
			} else if (result.type == ScriptLanguage::LOOKUP_RESULT_CLASS_METHOD) {
				MethodInfo met;
				if (ClassDB::get_method_info(result.class_name, result.class_member, &met)) {
					if (!met.return_val.class_name.is_empty() && ClassDB::class_exists(met.return_val.class_name)) {
						r_classes->insert(met.return_val.class_name);
					}
					if (!met.return_val.hint_string.is_empty() && ClassDB::class_exists(met.return_val.hint_string)) {
						r_classes->insert(met.return_val.hint_string);
					}
				}
			}
		}

		current = tokenizer.scan();
	}
}

Error ResourceFormatSaverGDScript::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	Ref<GDScript> sqscr = p_resource;
	ERR_FAIL_COND_V(sqscr.is_null(), ERR_INVALID_PARAMETER);

	String source = sqscr->get_source_code();

	{
		Error err;
		Ref<FileAccess> file = FileAccess::open(p_path, FileAccess::WRITE, &err);

		ERR_FAIL_COND_V_MSG(err, err, "Cannot save GDScript file '" + p_path + "'.");

		file->store_string(source);
		if (file->get_error() != OK && file->get_error() != ERR_FILE_EOF) {
			return ERR_CANT_CREATE;
		}
	}

	if (ScriptServer::is_reload_scripts_on_save_enabled()) {
		GDScriptLanguage::get_singleton()->reload_tool_script(p_resource, true);
	}

	return OK;
}

void ResourceFormatSaverGDScript::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<GDScript>(*p_resource)) {
		p_extensions->push_back("gd");
	}
}

bool ResourceFormatSaverGDScript::recognize(const Ref<Resource> &p_resource) const {
	return Object::cast_to<GDScript>(*p_resource) != nullptr;
}
