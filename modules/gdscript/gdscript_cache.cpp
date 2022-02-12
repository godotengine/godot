/*************************************************************************/
/*  gdscript_cache.cpp                                                   */
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

#include "gdscript_cache.h"

#include "core/io/file_access.h"
#include "core/templates/vector.h"
#include "gdscript.h"
#include "gdscript_analyzer.h"
#include "gdscript_parser.h"

bool GDScriptParserRef::is_valid() const {
	return parser != nullptr;
}

GDScriptParserRef::Status GDScriptParserRef::get_status() const {
	return status;
}

GDScriptParser *GDScriptParserRef::get_parser() const {
	return parser;
}

Error GDScriptParserRef::raise_status(Status p_new_status) {
	ERR_FAIL_COND_V(parser == nullptr, ERR_INVALID_DATA);

	if (result != OK) {
		return result;
	}

	while (p_new_status > status) {
		switch (status) {
			case EMPTY:
				status = PARSED;
				result = parser->parse(GDScriptCache::get_source_code(path), path, false);
				break;
			case PARSED: {
				analyzer = memnew(GDScriptAnalyzer(parser));
				status = INHERITANCE_SOLVED;
				Error inheritance_result = analyzer->resolve_inheritance();
				if (result == OK) {
					result = inheritance_result;
				}
			} break;
			case INHERITANCE_SOLVED: {
				status = INTERFACE_SOLVED;
				Error interface_result = analyzer->resolve_interface();
				if (result == OK) {
					result = interface_result;
				}
			} break;
			case INTERFACE_SOLVED: {
				status = FULLY_SOLVED;
				Error body_result = analyzer->resolve_body();
				if (result == OK) {
					result = body_result;
				}
			} break;
			case FULLY_SOLVED: {
				return result;
			}
		}
		if (result != OK) {
			return result;
		}
	}

	return result;
}

GDScriptParserRef::~GDScriptParserRef() {
	if (parser != nullptr) {
		memdelete(parser);
	}
	if (analyzer != nullptr) {
		memdelete(analyzer);
	}
	MutexLock lock(GDScriptCache::singleton->lock);
	GDScriptCache::singleton->parser_map.erase(path);
}

GDScriptCache *GDScriptCache::singleton = nullptr;

void GDScriptCache::remove_script(const String &p_path) {
	MutexLock lock(singleton->lock);
	singleton->shallow_gdscript_cache.erase(p_path);
	singleton->full_gdscript_cache.erase(p_path);
}

Ref<GDScriptParserRef> GDScriptCache::get_parser(const String &p_path, GDScriptParserRef::Status p_status, Error &r_error, const String &p_owner) {
	MutexLock lock(singleton->lock);
	Ref<GDScriptParserRef> ref;
	if (!p_owner.is_empty()) {
		singleton->dependencies[p_owner].insert(p_path);
	}
	if (singleton->parser_map.has(p_path)) {
		ref = Ref<GDScriptParserRef>(singleton->parser_map[p_path]);
		if (ref.is_null()) {
			r_error = ERR_INVALID_DATA;
			return ref;
		}
	} else {
		if (!FileAccess::exists(p_path)) {
			r_error = ERR_FILE_NOT_FOUND;
			return ref;
		}
		GDScriptParser *parser = memnew(GDScriptParser);
		ref.instantiate();
		ref->parser = parser;
		ref->path = p_path;
		singleton->parser_map[p_path] = ref.ptr();
	}
	r_error = ref->raise_status(p_status);

	return ref;
}

String GDScriptCache::get_source_code(const String &p_path) {
	Vector<uint8_t> source_file;
	Error err;
	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (err) {
		ERR_FAIL_COND_V(err, "");
	}

	uint64_t len = f->get_length();
	source_file.resize(len + 1);
	uint64_t r = f->get_buffer(source_file.ptrw(), len);
	f->close();
	ERR_FAIL_COND_V(r != len, "");
	source_file.write[len] = 0;

	String source;
	if (source.parse_utf8((const char *)source_file.ptr())) {
		ERR_FAIL_V_MSG("", "Script '" + p_path + "' contains invalid unicode (UTF-8), so it was not loaded. Please ensure that scripts are saved in valid UTF-8 unicode.");
	}
	return source;
}

Ref<GDScript> GDScriptCache::get_shallow_script(const String &p_path, const String &p_owner) {
	MutexLock lock(singleton->lock);
	if (!p_owner.is_empty()) {
		singleton->dependencies[p_owner].insert(p_path);
	}
	if (singleton->full_gdscript_cache.has(p_path)) {
		return singleton->full_gdscript_cache[p_path];
	}
	if (singleton->shallow_gdscript_cache.has(p_path)) {
		return singleton->shallow_gdscript_cache[p_path];
	}

	Ref<GDScript> script;
	script.instantiate();
	script->set_path(p_path, true);
	script->set_script_path(p_path);
	script->load_source_code(p_path);

	singleton->shallow_gdscript_cache[p_path] = script.ptr();
	return script;
}

Ref<GDScript> GDScriptCache::get_full_script(const String &p_path, Error &r_error, const String &p_owner) {
	MutexLock lock(singleton->lock);

	if (!p_owner.is_empty()) {
		singleton->dependencies[p_owner].insert(p_path);
	}

	r_error = OK;
	if (singleton->full_gdscript_cache.has(p_path)) {
		return singleton->full_gdscript_cache[p_path];
	}

	Ref<GDScript> script = get_shallow_script(p_path);
	ERR_FAIL_COND_V(script.is_null(), Ref<GDScript>());

	r_error = script->load_source_code(p_path);

	if (r_error) {
		return script;
	}

	r_error = script->reload();
	if (r_error) {
		return script;
	}

	singleton->full_gdscript_cache[p_path] = script.ptr();
	singleton->shallow_gdscript_cache.erase(p_path);

	return script;
}

Error GDScriptCache::finish_compiling(const String &p_owner) {
	// Mark this as compiled.
	Ref<GDScript> script = get_shallow_script(p_owner);
	singleton->full_gdscript_cache[p_owner] = script.ptr();
	singleton->shallow_gdscript_cache.erase(p_owner);

	Set<String> depends = singleton->dependencies[p_owner];

	Error err = OK;
	for (const Set<String>::Element *E = depends.front(); E != nullptr; E = E->next()) {
		Error this_err = OK;
		// No need to save the script. We assume it's already referenced in the owner.
		get_full_script(E->get(), this_err);

		if (this_err != OK) {
			err = this_err;
		}
	}

	singleton->dependencies.erase(p_owner);

	return err;
}

GDScriptCache::GDScriptCache() {
	singleton = this;
}

GDScriptCache::~GDScriptCache() {
	parser_map.clear();
	shallow_gdscript_cache.clear();
	full_gdscript_cache.clear();
	singleton = nullptr;
}
