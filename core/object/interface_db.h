/**************************************************************************/
/*  interface_db.h                                                        */
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

#pragma once

#include "core/object/object.h"
#include "core/os/mutex.h"
#include "core/string/string_name.h"
#include "core/templates/hash_map.h"
#include "core/variant/variant.h"

struct InterfaceInfo {
	StringName name;
	Vector<MethodInfo> required_methods;
	Vector<PropertyInfo> required_properties;
	StringName source_language; // "GDScript", "C#", "GDExtension", or "" for native.
	String script_path; // Path to defining script (if script-defined).
};

class InterfaceDB {
	static HashMap<StringName, InterfaceInfo> interfaces;
	static Mutex mutex;
	static uint64_t generation;

	struct CacheKey {
		ObjectID object_id;
		StringName interface_name;

		bool operator==(const CacheKey &p_other) const {
			return object_id == p_other.object_id && interface_name == p_other.interface_name;
		}

		uint32_t hash() const {
			uint32_t h = hash_one_uint64(object_id);
			return hash_fmix32(h ^ interface_name.hash());
		}
	};

	struct CacheEntry {
		bool result = false;
		uint64_t generation = 0;
	};

	static HashMap<CacheKey, CacheEntry> cache;

	static bool _object_structurally_satisfies_internal(const Object *p_object, const InterfaceInfo &p_info);

public:
	// Registration.
	static void register_interface(const InterfaceInfo &p_info);
	static void unregister_interface(const StringName &p_name);

	// Query.
	static bool interface_exists(const StringName &p_name);
	static InterfaceInfo get_interface_info(const StringName &p_name);
	static void get_interface_list(List<StringName> *r_interfaces);

	// Validation — the core question: "does this object implement interface X?"
	static bool object_implements_interface(const Object *p_object, const StringName &p_interface);

	// Structural matching (used internally by object_implements_interface, but public for testing).
	static bool object_structurally_satisfies(const Object *p_object, const StringName &p_interface);

	// Cache management.
	static void invalidate_cache();
	static void invalidate_cache_for_object(ObjectID p_object_id);

	// Generation counter (incremented on any registration change).
	static uint64_t get_generation() { return generation; }

	// Cleanup.
	static void cleanup();
};
