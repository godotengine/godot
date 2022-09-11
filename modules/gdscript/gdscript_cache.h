/*************************************************************************/
/*  gdscript_cache.h                                                     */
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

#ifndef GDSCRIPT_CACHE_H
#define GDSCRIPT_CACHE_H

#include "core/object/ref_counted.h"
#include "core/object/script_language.h"
#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "gdscript.h"
#include "scene/resources/packed_scene.h"

class GDScriptAnalyzer;
class GDScriptParser;

class GDScriptParserData : public RefCounted {
public:
	enum Status {
		EMPTY,
		PARSED,
		INHERITANCE_SOLVED,
		INTERFACE_SOLVED,
		FULLY_SOLVED,
	};

private:
	GDScriptParser *parser = nullptr;
	GDScriptAnalyzer *analyzer = nullptr;
	Status status = EMPTY;
	Error result = OK;
	String path;

	friend class GDScriptCache;

public:
	bool is_valid() const;
	Status get_status() const;
	GDScriptParser *get_parser() const;
	GDScriptAnalyzer *get_analyzer() const;
	Error raise_status(Status p_new_status);

	GDScriptParserData() {}
	~GDScriptParserData();
};

class GDScriptParserDataRef : public WeakRef {
public:
	Ref<GDScriptParserData> get_ref() const {
		return WeakRef::get_ref();
	}
	void set_ref(const Ref<GDScriptParserData> &p_ref) {
		WeakRef::set_ref(p_ref);
	}
};

class PackedSceneRef : public WeakRef {
public:
	Variant get(const StringName &p_name, bool *r_valid) const {
		if (get_ref().is_null()) {
			return Variant();
		}

		return get_ref()->get(p_name, r_valid);
	}

	void set(const StringName &p_name, const Variant &p_value, bool *r_valid) {
		if (get_ref().is_null()) {
			*r_valid = false;
			return;
		}

		get_ref()->set(p_name, p_value, r_valid);
	}

	Ref<PackedScene> get_ref() const {
		return WeakRef::get_ref();
	}

	void set_ref(const Ref<PackedScene> &p_ref) {
		WeakRef::set_ref(p_ref);
	}
};

class GDScriptCache {
	// String key is full path.
	HashMap<String, Ref<GDScriptParserData>> parser_map;
	HashMap<String, Ref<GDScript>> shallow_gdscript_cache;
	HashMap<String, Ref<GDScript>> full_gdscript_cache;
	HashMap<String, RBSet<String>> dependencies;
	HashMap<String, Ref<PackedScene>> packed_scene_cache;

	bool destructing;
	RBSet<String> removed_dependencies;

	friend class GDScript;
	friend class GDScriptParserData;

	static GDScriptCache *singleton;

	Mutex lock;

	static void remove_dependencies(const String &p_path);

public:
	static Ref<GDScriptParserDataRef> get_parser(const String &p_path, GDScriptParserData::Status status, Error &r_error, const String &p_owner = String());
	static String get_source_code(const String &p_path);
	static Ref<GDScriptRef> get_shallow_script(const String &p_path, const String &p_owner = String());
	static Ref<GDScriptRef> get_full_script(const String &p_path, Error &r_error, const String &p_owner = String(), bool p_update_from_disk = false);
	static Error finish_compiling(const String &p_owner);
	static void reload_script(const String &p_path);
	static void remove_script(const String &p_path);
	static Ref<PackedSceneRef> load_scene(const String &p_path, const String &p_owner);

	GDScriptCache();
	~GDScriptCache();
};

#endif // GDSCRIPT_CACHE_H
