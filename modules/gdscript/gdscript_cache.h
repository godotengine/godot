/**************************************************************************/
/*  gdscript_cache.h                                                      */
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

#ifndef GDSCRIPT_CACHE_H
#define GDSCRIPT_CACHE_H

#include "gdscript.h"

#include "core/object/ref_counted.h"
#include "core/os/mutex.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"

class GDScriptAnalyzer;
class GDScriptParser;

class GDScriptParserRef : public RefCounted {
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
	bool cleared = false;

	friend class GDScriptCache;

public:
	bool is_valid() const;
	Status get_status() const;
	GDScriptParser *get_parser() const;
	GDScriptAnalyzer *get_analyzer();
	Error raise_status(Status p_new_status);
	void clear();

	GDScriptParserRef() {}
	~GDScriptParserRef();
};

class GDScriptCache {
	// String key is full path.
	HashMap<String, GDScriptParserRef *> parser_map;
	HashMap<String, Ref<GDScript>> shallow_gdscript_cache;
	HashMap<String, Ref<GDScript>> full_gdscript_cache;
	HashMap<String, Ref<GDScript>> static_gdscript_cache;
	HashMap<String, HashSet<String>> dependencies;

	friend class GDScript;
	friend class GDScriptParserRef;
	friend class GDScriptInstance;

	static GDScriptCache *singleton;

	bool cleared = false;

	Mutex mutex;

public:
	static void move_script(const String &p_from, const String &p_to);
	static void remove_script(const String &p_path);
	static Ref<GDScriptParserRef> get_parser(const String &p_path, GDScriptParserRef::Status status, Error &r_error, const String &p_owner = String());
	static String get_source_code(const String &p_path);
	static Vector<uint8_t> get_binary_tokens(const String &p_path);
	static Ref<GDScript> get_shallow_script(const String &p_path, Error &r_error, const String &p_owner = String());
	static Ref<GDScript> get_full_script(const String &p_path, Error &r_error, const String &p_owner = String(), bool p_update_from_disk = false);
	static Ref<GDScript> get_cached_script(const String &p_path);
	static Error finish_compiling(const String &p_owner);
	static void add_static_script(Ref<GDScript> p_script);
	static void remove_static_script(const String &p_fqcn);

	static void clear();

	GDScriptCache();
	~GDScriptCache();
};

#endif // GDSCRIPT_CACHE_H
