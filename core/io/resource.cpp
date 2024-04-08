/**************************************************************************/
/*  resource.cpp                                                          */
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

#include "resource.h"

#include "core/core_string_names.h"
#include "core/io/file_access.h"
#include "core/io/resource_loader.h"
#include "core/math/math_funcs.h"
#include "core/object/script_language.h"
#include "core/os/os.h"
#include "scene/main/node.h" //only so casting works

#include <stdio.h>

void Resource::emit_changed() {
	emit_signal(CoreStringNames::get_singleton()->changed);
}

void Resource::_resource_path_changed() {
}

void Resource::set_path(const String &p_path, bool p_take_over) {
	if (path_cache == p_path) {
		return;
	}

	if (p_path.is_empty()) {
		p_take_over = false; // Can't take over an empty path
	}

	ResourceCache::lock.lock();

	if (!path_cache.is_empty()) {
		ResourceCache::resources.erase(path_cache);
	}

	path_cache = "";

	Ref<Resource> existing = ResourceCache::get_ref(p_path);

	if (existing.is_valid()) {
		if (p_take_over) {
			existing->path_cache = String();
			ResourceCache::resources.erase(p_path);
		} else {
			ResourceCache::lock.unlock();
			ERR_FAIL_MSG("Another resource is loaded from path '" + p_path + "' (possible cyclic resource inclusion).");
		}
	}

	path_cache = p_path;

	if (!path_cache.is_empty()) {
		ResourceCache::resources[path_cache] = this;
	}
	ResourceCache::lock.unlock();

	_resource_path_changed();
}

String Resource::get_path() const {
	return path_cache;
}

void Resource::set_path_cache(const String &p_path) {
	path_cache = p_path;
}

String Resource::generate_scene_unique_id() {
	// Generate a unique enough hash, but still user-readable.
	// If it's not unique it does not matter because the saver will try again.
	OS::DateTime dt = OS::get_singleton()->get_datetime();
	uint32_t hash = hash_murmur3_one_32(OS::get_singleton()->get_ticks_usec());
	hash = hash_murmur3_one_32(dt.year, hash);
	hash = hash_murmur3_one_32(dt.month, hash);
	hash = hash_murmur3_one_32(dt.day, hash);
	hash = hash_murmur3_one_32(dt.hour, hash);
	hash = hash_murmur3_one_32(dt.minute, hash);
	hash = hash_murmur3_one_32(dt.second, hash);
	hash = hash_murmur3_one_32(Math::rand(), hash);

	static constexpr uint32_t characters = 5;
	static constexpr uint32_t char_count = ('z' - 'a');
	static constexpr uint32_t base = char_count + ('9' - '0');
	String id;
	for (uint32_t i = 0; i < characters; i++) {
		uint32_t c = hash % base;
		if (c < char_count) {
			id += String::chr('a' + c);
		} else {
			id += String::chr('0' + (c - char_count));
		}
		hash /= base;
	}

	return id;
}

void Resource::set_scene_unique_id(const String &p_id) {
	scene_unique_id = p_id;
}

String Resource::get_scene_unique_id() const {
	return scene_unique_id;
}

void Resource::set_name(const String &p_name) {
	name = p_name;
	emit_changed();
}

String Resource::get_name() const {
	return name;
}

void Resource::update_configuration_warning() {
	if (_update_configuration_warning) {
		_update_configuration_warning();
	}
}

bool Resource::editor_can_reload_from_file() {
	return true; //by default yes
}

void Resource::connect_changed(const Callable &p_callable, uint32_t p_flags) {
	if (!is_connected(CoreStringNames::get_singleton()->changed, p_callable) || p_flags & CONNECT_REFERENCE_COUNTED) {
		connect(CoreStringNames::get_singleton()->changed, p_callable, p_flags);
	}
}

void Resource::disconnect_changed(const Callable &p_callable) {
	if (is_connected(CoreStringNames::get_singleton()->changed, p_callable)) {
		disconnect(CoreStringNames::get_singleton()->changed, p_callable);
	}
}

void Resource::reset_state() {
}

Error Resource::copy_from(const Ref<Resource> &p_resource) {
	ERR_FAIL_COND_V(p_resource.is_null(), ERR_INVALID_PARAMETER);
	if (get_class() != p_resource->get_class()) {
		return ERR_INVALID_PARAMETER;
	}

	reset_state(); // May want to reset state.

	List<PropertyInfo> pi;
	p_resource->get_property_list(&pi);

	for (const PropertyInfo &E : pi) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
			continue;
		}
		if (E.name == "resource_path") {
			continue; //do not change path
		}

		set(E.name, p_resource->get(E.name));
	}
	return OK;
}
void Resource::reload_from_file() {
	String path = get_path();
	if (!path.is_resource_file()) {
		return;
	}

	Ref<Resource> s = ResourceLoader::load(ResourceLoader::path_remap(path), get_class(), ResourceFormatLoader::CACHE_MODE_IGNORE);

	if (!s.is_valid()) {
		return;
	}

	copy_from(s);
}

Ref<Resource> Resource::duplicate_for_local_scene(Node *p_for_scene, HashMap<Ref<Resource>, Ref<Resource>> &remap_cache) {
	List<PropertyInfo> plist;
	get_property_list(&plist);

	Ref<Resource> r = Object::cast_to<Resource>(ClassDB::instantiate(get_class()));
	ERR_FAIL_COND_V(r.is_null(), Ref<Resource>());

	r->local_scene = p_for_scene;

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
			continue;
		}
		Variant p = get(E.name);
		if (p.get_type() == Variant::OBJECT) {
			Ref<Resource> sr = p;
			if (sr.is_valid()) {
				if (sr->is_local_to_scene()) {
					if (remap_cache.has(sr)) {
						p = remap_cache[sr];
					} else {
						Ref<Resource> dupe = sr->duplicate_for_local_scene(p_for_scene, remap_cache);
						p = dupe;
						remap_cache[sr] = dupe;
					}
				}
			}
		}

		r->set(E.name, p);
	}

	return r;
}

void Resource::configure_for_local_scene(Node *p_for_scene, HashMap<Ref<Resource>, Ref<Resource>> &remap_cache) {
	List<PropertyInfo> plist;
	get_property_list(&plist);

	reset_local_to_scene();
	local_scene = p_for_scene;

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
			continue;
		}
		Variant p = get(E.name);
		if (p.get_type() == Variant::OBJECT) {
			Ref<Resource> sr = p;
			if (sr.is_valid()) {
				if (sr->is_local_to_scene()) {
					if (!remap_cache.has(sr)) {
						sr->configure_for_local_scene(p_for_scene, remap_cache);
						remap_cache[sr] = sr;
					}
				}
			}
		}
	}
}

Ref<Resource> Resource::duplicate(bool p_subresources) const {
	List<PropertyInfo> plist;
	get_property_list(&plist);

	Ref<Resource> r = static_cast<Resource *>(ClassDB::instantiate(get_class()));
	ERR_FAIL_COND_V(r.is_null(), Ref<Resource>());

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
			continue;
		}
		Variant p = get(E.name);

		switch (p.get_type()) {
			case Variant::Type::DICTIONARY:
			case Variant::Type::ARRAY:
			case Variant::Type::PACKED_BYTE_ARRAY:
			case Variant::Type::PACKED_COLOR_ARRAY:
			case Variant::Type::PACKED_INT32_ARRAY:
			case Variant::Type::PACKED_INT64_ARRAY:
			case Variant::Type::PACKED_FLOAT32_ARRAY:
			case Variant::Type::PACKED_FLOAT64_ARRAY:
			case Variant::Type::PACKED_STRING_ARRAY:
			case Variant::Type::PACKED_VECTOR2_ARRAY:
			case Variant::Type::PACKED_VECTOR3_ARRAY: {
				r->set(E.name, p.duplicate(p_subresources));
			} break;

			case Variant::Type::OBJECT: {
				if (!(E.usage & PROPERTY_USAGE_NEVER_DUPLICATE) && (p_subresources || (E.usage & PROPERTY_USAGE_ALWAYS_DUPLICATE))) {
					Ref<Resource> sr = p;
					if (sr.is_valid()) {
						r->set(E.name, sr->duplicate(p_subresources));
					}
				} else {
					r->set(E.name, p);
				}
			} break;

			default: {
				r->set(E.name, p);
			}
		}
	}

	return r;
}

void Resource::_set_path(const String &p_path) {
	set_path(p_path, false);
}

void Resource::_take_over_path(const String &p_path) {
	set_path(p_path, true);
}

RID Resource::get_rid() const {
	if (get_script_instance()) {
		Callable::CallError ce;
		RID ret = get_script_instance()->callp(SNAME("_get_rid"), nullptr, 0, ce);
		if (ce.error == Callable::CallError::CALL_OK && ret.is_valid()) {
			return ret;
		}
	}
	if (_get_extension() && _get_extension()->get_rid) {
		RID ret = RID::from_uint64(_get_extension()->get_rid(_get_extension_instance()));
		if (ret.is_valid()) {
			return ret;
		}
	}

	return RID();
}

#ifdef TOOLS_ENABLED

uint32_t Resource::hash_edited_version() const {
	uint32_t hash = hash_murmur3_one_32(get_edited_version());

	List<PropertyInfo> plist;
	get_property_list(&plist);

	for (const PropertyInfo &E : plist) {
		if (E.usage & PROPERTY_USAGE_STORAGE && E.type == Variant::OBJECT && E.hint == PROPERTY_HINT_RESOURCE_TYPE) {
			Ref<Resource> res = get(E.name);
			if (res.is_valid()) {
				hash = hash_murmur3_one_32(res->hash_edited_version(), hash);
			}
		}
	}

	return hash;
}

#endif

void Resource::set_local_to_scene(bool p_enable) {
	local_to_scene = p_enable;
}

bool Resource::is_local_to_scene() const {
	return local_to_scene;
}

Node *Resource::get_local_scene() const {
	if (local_scene) {
		return local_scene;
	}

	if (_get_local_scene_func) {
		return _get_local_scene_func();
	}

	return nullptr;
}

void Resource::setup_local_to_scene() {
	emit_signal(SNAME("setup_local_to_scene_requested"));
	GDVIRTUAL_CALL(_setup_local_to_scene);
}

void Resource::reset_local_to_scene() {
	// Restores the state as if setup_local_to_scene() hadn't been called.
}

Node *(*Resource::_get_local_scene_func)() = nullptr;
void (*Resource::_update_configuration_warning)() = nullptr;

void Resource::set_as_translation_remapped(bool p_remapped) {
	if (remapped_list.in_list() == p_remapped) {
		return;
	}

	ResourceCache::lock.lock();

	if (p_remapped) {
		ResourceLoader::remapped_list.add(&remapped_list);
	} else {
		ResourceLoader::remapped_list.remove(&remapped_list);
	}

	ResourceCache::lock.unlock();
}

#ifdef TOOLS_ENABLED
//helps keep IDs same number when loading/saving scenes. -1 clears ID and it Returns -1 when no id stored
void Resource::set_id_for_path(const String &p_path, const String &p_id) {
	if (p_id.is_empty()) {
		ResourceCache::path_cache_lock.write_lock();
		ResourceCache::resource_path_cache[p_path].erase(get_path());
		ResourceCache::path_cache_lock.write_unlock();
	} else {
		ResourceCache::path_cache_lock.write_lock();
		ResourceCache::resource_path_cache[p_path][get_path()] = p_id;
		ResourceCache::path_cache_lock.write_unlock();
	}
}

String Resource::get_id_for_path(const String &p_path) const {
	ResourceCache::path_cache_lock.read_lock();
	if (ResourceCache::resource_path_cache[p_path].has(get_path())) {
		String result = ResourceCache::resource_path_cache[p_path][get_path()];
		ResourceCache::path_cache_lock.read_unlock();
		return result;
	} else {
		ResourceCache::path_cache_lock.read_unlock();
		return "";
	}
}
#endif

void Resource::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_path", "path"), &Resource::_set_path);
	ClassDB::bind_method(D_METHOD("take_over_path", "path"), &Resource::_take_over_path);
	ClassDB::bind_method(D_METHOD("get_path"), &Resource::get_path);
	ClassDB::bind_method(D_METHOD("set_name", "name"), &Resource::set_name);
	ClassDB::bind_method(D_METHOD("get_name"), &Resource::get_name);
	ClassDB::bind_method(D_METHOD("get_rid"), &Resource::get_rid);
	ClassDB::bind_method(D_METHOD("set_local_to_scene", "enable"), &Resource::set_local_to_scene);
	ClassDB::bind_method(D_METHOD("is_local_to_scene"), &Resource::is_local_to_scene);
	ClassDB::bind_method(D_METHOD("get_local_scene"), &Resource::get_local_scene);
	ClassDB::bind_method(D_METHOD("setup_local_to_scene"), &Resource::setup_local_to_scene);

	ClassDB::bind_method(D_METHOD("emit_changed"), &Resource::emit_changed);

	ClassDB::bind_method(D_METHOD("duplicate", "subresources"), &Resource::duplicate, DEFVAL(false));
	ADD_SIGNAL(MethodInfo("changed"));
	ADD_SIGNAL(MethodInfo("setup_local_to_scene_requested"));

	ADD_GROUP("Resource", "resource_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "resource_local_to_scene"), "set_local_to_scene", "is_local_to_scene");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "resource_path", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_path", "get_path");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "resource_name"), "set_name", "get_name");

	MethodInfo get_rid_bind("_get_rid");
	get_rid_bind.return_val.type = Variant::RID;

	::ClassDB::add_virtual_method(get_class_static(), get_rid_bind, true, Vector<String>(), true);
	GDVIRTUAL_BIND(_setup_local_to_scene);
}

Resource::Resource() :
		remapped_list(this) {}

Resource::~Resource() {
	if (!path_cache.is_empty()) {
		ResourceCache::lock.lock();
		ResourceCache::resources.erase(path_cache);
		ResourceCache::lock.unlock();
	}
}

HashMap<String, Resource *> ResourceCache::resources;
#ifdef TOOLS_ENABLED
HashMap<String, HashMap<String, String>> ResourceCache::resource_path_cache;
#endif

Mutex ResourceCache::lock;
#ifdef TOOLS_ENABLED
RWLock ResourceCache::path_cache_lock;
#endif

void ResourceCache::clear() {
	if (resources.size()) {
		ERR_PRINT("Resources still in use at exit (run with --verbose for details).");
		if (OS::get_singleton()->is_stdout_verbose()) {
			for (const KeyValue<String, Resource *> &E : resources) {
				print_line(vformat("Resource still in use: %s (%s)", E.key, E.value->get_class()));
			}
		}
	}

	resources.clear();
}

bool ResourceCache::has(const String &p_path) {
	lock.lock();

	Resource **res = resources.getptr(p_path);

	if (res && (*res)->get_reference_count() == 0) {
		// This resource is in the process of being deleted, ignore its existence.
		(*res)->path_cache = String();
		resources.erase(p_path);
		res = nullptr;
	}

	lock.unlock();

	if (!res) {
		return false;
	}

	return true;
}

Ref<Resource> ResourceCache::get_ref(const String &p_path) {
	Ref<Resource> ref;
	lock.lock();

	Resource **res = resources.getptr(p_path);

	if (res) {
		ref = Ref<Resource>(*res);
	}

	if (res && !ref.is_valid()) {
		// This resource is in the process of being deleted, ignore its existence
		(*res)->path_cache = String();
		resources.erase(p_path);
		res = nullptr;
	}

	lock.unlock();

	return ref;
}

void ResourceCache::get_cached_resources(List<Ref<Resource>> *p_resources) {
	lock.lock();

	LocalVector<String> to_remove;

	for (KeyValue<String, Resource *> &E : resources) {
		Ref<Resource> ref = Ref<Resource>(E.value);

		if (!ref.is_valid()) {
			// This resource is in the process of being deleted, ignore its existence
			E.value->path_cache = String();
			to_remove.push_back(E.key);
			continue;
		}

		p_resources->push_back(ref);
	}

	for (const String &E : to_remove) {
		resources.erase(E);
	}

	lock.unlock();
}

int ResourceCache::get_cached_resource_count() {
	lock.lock();
	int rc = resources.size();
	lock.unlock();

	return rc;
}
