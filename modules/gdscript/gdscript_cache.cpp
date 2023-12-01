/**************************************************************************/
/*  gdscript_cache.cpp                                                    */
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

#include "gdscript_cache.h"

#include "gdscript.h"
#include "gdscript_analyzer.h"
#include "gdscript_compiler.h"
#include "gdscript_parser.h"

#include "core/io/file_access.h"
#include "core/templates/vector.h"
#include "scene/resources/packed_scene.h"

bool GDScriptParserRef::is_valid() const {
	return parser != nullptr;
}

GDScriptParserRef::Status GDScriptParserRef::get_status() const {
	return status;
}

GDScriptParser *GDScriptParserRef::get_parser() const {
	return parser;
}

GDScriptAnalyzer *GDScriptParserRef::get_analyzer() {
	if (analyzer == nullptr) {
		analyzer = memnew(GDScriptAnalyzer(parser));
	}
	return analyzer;
}

Error GDScriptParserRef::raise_status(Status p_new_status) {
	ERR_FAIL_NULL_V(parser, ERR_INVALID_DATA);

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
				status = INHERITANCE_SOLVED;
				Error inheritance_result = get_analyzer()->resolve_inheritance();
				if (result == OK) {
					result = inheritance_result;
				}
			} break;
			case INHERITANCE_SOLVED: {
				status = INTERFACE_SOLVED;
				Error interface_result = get_analyzer()->resolve_interface();
				if (result == OK) {
					result = interface_result;
				}
			} break;
			case INTERFACE_SOLVED: {
				status = FULLY_SOLVED;
				Error body_result = get_analyzer()->resolve_body();
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

void GDScriptParserRef::clear() {
	if (cleared) {
		return;
	}
	cleared = true;

	if (parser != nullptr) {
		memdelete(parser);
	}

	if (analyzer != nullptr) {
		memdelete(analyzer);
	}
}

GDScriptParserRef::~GDScriptParserRef() {
	clear();

	MutexLock lock(GDScriptCache::singleton->mutex);
	GDScriptCache::singleton->parser_map.erase(path);
}

GDScriptCache *GDScriptCache::singleton = nullptr;

void GDScriptCache::move_script(const String &p_from, const String &p_to) {
	if (singleton == nullptr || p_from == p_to) {
		return;
	}

	MutexLock lock(singleton->mutex);

	if (singleton->cleared) {
		return;
	}

	for (KeyValue<String, HashSet<String>> &E : singleton->packed_scene_dependencies) {
		if (E.value.has(p_from)) {
			E.value.insert(p_to);
			E.value.erase(p_from);
		}
	}

	if (singleton->parser_map.has(p_from) && !p_from.is_empty()) {
		singleton->parser_map[p_to] = singleton->parser_map[p_from];
	}
	singleton->parser_map.erase(p_from);

	if (singleton->shallow_gdscript_cache.has(p_from) && !p_from.is_empty()) {
		singleton->shallow_gdscript_cache[p_to] = singleton->shallow_gdscript_cache[p_from];
	}
	singleton->shallow_gdscript_cache.erase(p_from);

	if (singleton->full_gdscript_cache.has(p_from) && !p_from.is_empty()) {
		singleton->full_gdscript_cache[p_to] = singleton->full_gdscript_cache[p_from];
	}
	singleton->full_gdscript_cache.erase(p_from);
}

void GDScriptCache::remove_script(const String &p_path) {
	if (singleton == nullptr) {
		return;
	}

	MutexLock lock(singleton->mutex);

	if (singleton->cleared) {
		return;
	}

	for (KeyValue<String, HashSet<String>> &E : singleton->packed_scene_dependencies) {
		if (!E.value.has(p_path)) {
			continue;
		}
		E.value.erase(p_path);
	}

	GDScriptCache::clear_unreferenced_packed_scenes();

	if (singleton->parser_map.has(p_path)) {
		singleton->parser_map[p_path]->clear();
		singleton->parser_map.erase(p_path);
	}

	singleton->dependencies.erase(p_path);
	singleton->shallow_gdscript_cache.erase(p_path);
	singleton->full_gdscript_cache.erase(p_path);
}

Ref<GDScriptParserRef> GDScriptCache::get_parser(const String &p_path, GDScriptParserRef::Status p_status, Error &r_error, const String &p_owner) {
	MutexLock lock(singleton->mutex);
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
	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ, &err);
	ERR_FAIL_COND_V(err, "");

	uint64_t len = f->get_length();
	source_file.resize(len + 1);
	uint64_t r = f->get_buffer(source_file.ptrw(), len);
	ERR_FAIL_COND_V(r != len, "");
	source_file.write[len] = 0;

	String source;
	if (source.parse_utf8((const char *)source_file.ptr()) != OK) {
		ERR_FAIL_V_MSG("", "Script '" + p_path + "' contains invalid unicode (UTF-8), so it was not loaded. Please ensure that scripts are saved in valid UTF-8 unicode.");
	}
	return source;
}

Ref<GDScript> GDScriptCache::get_shallow_script(const String &p_path, Error &r_error, const String &p_owner) {
	MutexLock lock(singleton->mutex);
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
	r_error = script->load_source_code(p_path);

	if (r_error) {
		return Ref<GDScript>(); // Returns null and does not cache when the script fails to load.
	}

	Ref<GDScriptParserRef> parser_ref = get_parser(p_path, GDScriptParserRef::PARSED, r_error);
	if (r_error == OK) {
		GDScriptCompiler::make_scripts(script.ptr(), parser_ref->get_parser()->get_tree(), true);
	}

	singleton->shallow_gdscript_cache[p_path] = script;
	return script;
}

Ref<GDScript> GDScriptCache::get_full_script(const String &p_path, Error &r_error, const String &p_owner, bool p_update_from_disk) {
	MutexLock lock(singleton->mutex);

	if (!p_owner.is_empty()) {
		singleton->dependencies[p_owner].insert(p_path);
	}

	Ref<GDScript> script;
	r_error = OK;
	if (singleton->full_gdscript_cache.has(p_path)) {
		script = singleton->full_gdscript_cache[p_path];
		if (!p_update_from_disk) {
			return script;
		}
	}

	if (script.is_null()) {
		script = get_shallow_script(p_path, r_error);
		// Only exit early if script failed to load, otherwise let reload report errors.
		if (script.is_null()) {
			return script;
		}
	}

	if (p_update_from_disk) {
		r_error = script->load_source_code(p_path);
		if (r_error) {
			return script;
		}
	}

	r_error = script->reload(true);
	if (r_error) {
		return script;
	}

	singleton->full_gdscript_cache[p_path] = script;
	singleton->shallow_gdscript_cache.erase(p_path);

	return script;
}

Ref<GDScript> GDScriptCache::get_cached_script(const String &p_path) {
	MutexLock lock(singleton->mutex);

	if (singleton->full_gdscript_cache.has(p_path)) {
		return singleton->full_gdscript_cache[p_path];
	}

	if (singleton->shallow_gdscript_cache.has(p_path)) {
		return singleton->shallow_gdscript_cache[p_path];
	}

	return Ref<GDScript>();
}

Error GDScriptCache::finish_compiling(const String &p_owner) {
	MutexLock lock(singleton->mutex);

	// Mark this as compiled.
	Ref<GDScript> script = get_cached_script(p_owner);
	singleton->full_gdscript_cache[p_owner] = script;
	singleton->shallow_gdscript_cache.erase(p_owner);

	HashSet<String> depends = singleton->dependencies[p_owner];

	Error err = OK;
	for (const String &E : depends) {
		Error this_err = OK;
		// No need to save the script. We assume it's already referenced in the owner.
		get_full_script(E, this_err);

		if (this_err != OK) {
			err = this_err;
		}
	}

	singleton->dependencies.erase(p_owner);

	return err;
}

void GDScriptCache::add_static_script(Ref<GDScript> p_script) {
	ERR_FAIL_COND_MSG(p_script.is_null(), "Trying to cache empty script as static.");
	ERR_FAIL_COND_MSG(!p_script->is_valid(), "Trying to cache non-compiled script as static.");
	singleton->static_gdscript_cache[p_script->get_fully_qualified_name()] = p_script;
}

void GDScriptCache::remove_static_script(const String &p_fqcn) {
	singleton->static_gdscript_cache.erase(p_fqcn);
}

Ref<PackedScene> GDScriptCache::get_packed_scene(const String &p_path, Error &r_error, const String &p_owner) {
	MutexLock lock(singleton->mutex);

	String path = p_path;
	if (path.begins_with("uid://")) {
		path = ResourceUID::get_singleton()->get_id_path(ResourceUID::get_singleton()->text_to_id(path));
	}

	if (singleton->packed_scene_cache.has(path)) {
		singleton->packed_scene_dependencies[path].insert(p_owner);
		return singleton->packed_scene_cache[path];
	}

	Ref<PackedScene> scene = ResourceCache::get_ref(path);
	if (scene.is_valid()) {
		singleton->packed_scene_cache[path] = scene;
		singleton->packed_scene_dependencies[path].insert(p_owner);
		return scene;
	}
	scene.instantiate();

	r_error = OK;
	if (path.is_empty()) {
		r_error = ERR_FILE_BAD_PATH;
		return scene;
	}

	scene->set_path(path);
	singleton->packed_scene_cache[path] = scene;
	singleton->packed_scene_dependencies[path].insert(p_owner);

	scene->reload_from_file();
	return scene;
}

void GDScriptCache::clear_unreferenced_packed_scenes() {
	if (singleton == nullptr) {
		return;
	}

	MutexLock lock(singleton->mutex);

	if (singleton->cleared) {
		return;
	}

	for (KeyValue<String, HashSet<String>> &E : singleton->packed_scene_dependencies) {
		if (E.value.size() > 0 || !ResourceLoader::is_imported(E.key)) {
			continue;
		}

		singleton->packed_scene_dependencies.erase(E.key);
		singleton->packed_scene_cache.erase(E.key);
	}
}

void GDScriptCache::clear() {
	if (singleton == nullptr) {
		return;
	}

	MutexLock lock(singleton->mutex);

	if (singleton->cleared) {
		return;
	}
	singleton->cleared = true;

	RBSet<Ref<GDScriptParserRef>> parser_map_refs;
	for (KeyValue<String, GDScriptParserRef *> &E : singleton->parser_map) {
		parser_map_refs.insert(E.value);
	}

	for (Ref<GDScriptParserRef> &E : parser_map_refs) {
		if (E.is_valid())
			E->clear();
	}

	singleton->packed_scene_dependencies.clear();
	singleton->packed_scene_cache.clear();

	parser_map_refs.clear();
	singleton->parser_map.clear();
	singleton->shallow_gdscript_cache.clear();
	singleton->full_gdscript_cache.clear();

	singleton->packed_scene_cache.clear();
	singleton->packed_scene_dependencies.clear();
}

GDScriptCache::GDScriptCache() {
	singleton = this;
}

GDScriptCache::~GDScriptCache() {
	if (!cleared) {
		clear();
	}
	singleton = nullptr;
}
