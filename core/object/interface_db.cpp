/**************************************************************************/
/*  interface_db.cpp                                                      */
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

#include "interface_db.h"

#include "core/io/resource.h"
#include "core/object/script_language.h"

HashMap<StringName, InterfaceInfo> InterfaceDB::interfaces;
Mutex InterfaceDB::mutex;
uint64_t InterfaceDB::generation = 0;
HashMap<InterfaceDB::CacheKey, InterfaceDB::CacheEntry> InterfaceDB::cache;

void InterfaceDB::register_interface(const InterfaceInfo &p_info) {
	MutexLock lock(mutex);
	interfaces[p_info.name] = p_info;
	generation++;
	// Invalidate all cache entries since a new/changed interface may affect results.
	cache.clear();
}

void InterfaceDB::unregister_interface(const StringName &p_name) {
	MutexLock lock(mutex);
	if (interfaces.erase(p_name)) {
		generation++;
		cache.clear();
	}
}

bool InterfaceDB::interface_exists(const StringName &p_name) {
	MutexLock lock(mutex);
	return interfaces.has(p_name);
}

InterfaceInfo InterfaceDB::get_interface_info(const StringName &p_name) {
	MutexLock lock(mutex);
	const InterfaceInfo *info = interfaces.getptr(p_name);
	ERR_FAIL_NULL_V_MSG(info, InterfaceInfo(), vformat("Interface '%s' not found.", p_name));
	return *info;
}

void InterfaceDB::get_interface_list(List<StringName> *r_interfaces) {
	MutexLock lock(mutex);
	for (const KeyValue<StringName, InterfaceInfo> &E : interfaces) {
		r_interfaces->push_back(E.key);
	}
}

bool InterfaceDB::object_implements_interface(const Object *p_object, const StringName &p_interface) {
	ERR_FAIL_NULL_V(p_object, false);

	MutexLock lock(mutex);

	if (!interfaces.has(p_interface)) {
		return false;
	}

	// Check cache.
	CacheKey key;
	key.object_id = p_object->get_instance_id();
	key.interface_name = p_interface;

	CacheEntry *cached = cache.getptr(key);
	if (cached && cached->generation == generation) {
		return cached->result;
	}

	bool result = false;

	// 1. Check if the object's script explicitly declares implementation.
	Ref<Script> script = p_object->get_script();
	if (script.is_valid() && script->implements_interface(p_interface)) {
		result = true;
	}

	// 2. If not explicitly declared, try structural matching (duck-typing).
	if (!result) {
		const InterfaceInfo &info = interfaces[p_interface];
		result = _object_structurally_satisfies_internal(p_object, info);
	}

	// Cache the result.
	CacheEntry entry;
	entry.result = result;
	entry.generation = generation;
	cache[key] = entry;

	return result;
}

bool InterfaceDB::object_structurally_satisfies(const Object *p_object, const StringName &p_interface) {
	ERR_FAIL_NULL_V(p_object, false);

	MutexLock lock(mutex);

	const InterfaceInfo *info = interfaces.getptr(p_interface);
	ERR_FAIL_NULL_V(info, false);

	return _object_structurally_satisfies_internal(p_object, *info);
}

bool InterfaceDB::_object_structurally_satisfies_internal(const Object *p_object, const InterfaceInfo &p_info) {
	// Check required methods.
	Ref<Script> script = p_object->get_script();
	for (const MethodInfo &mi : p_info.required_methods) {
		if (!p_object->has_method(mi.name)) {
			return false;
		}

		// Verify parameter count if the object's script can provide that info.
		if (script.is_valid()) {
			bool is_valid = false;
			int arg_count = script->get_script_method_argument_count(mi.name, &is_valid);
			if (is_valid && arg_count != mi.arguments.size()) {
				return false;
			}
		}
	}

	// Check required properties.
	if (!p_info.required_properties.is_empty()) {
		List<PropertyInfo> props;
		p_object->get_property_list(&props);
		for (const PropertyInfo &pi : p_info.required_properties) {
			bool found = false;
			for (const PropertyInfo &obj_pi : props) {
				if (obj_pi.name == pi.name) {
					// Check type compatibility if both types are specified.
					if (pi.type != Variant::NIL && obj_pi.type != Variant::NIL && pi.type != obj_pi.type) {
						return false;
					}
					found = true;
					break;
				}
			}
			if (!found) {
				return false;
			}
		}
	}

	return true;
}

void InterfaceDB::invalidate_cache() {
	MutexLock lock(mutex);
	cache.clear();
}

void InterfaceDB::invalidate_cache_for_object(ObjectID p_object_id) {
	MutexLock lock(mutex);
	// Remove all cache entries for this object.
	Vector<CacheKey> to_remove;
	for (const KeyValue<CacheKey, CacheEntry> &E : cache) {
		if (E.key.object_id == p_object_id) {
			to_remove.push_back(E.key);
		}
	}
	for (const CacheKey &key : to_remove) {
		cache.erase(key);
	}
}

void InterfaceDB::cleanup() {
	MutexLock lock(mutex);
	interfaces.clear();
	cache.clear();
	generation = 0;
}
