/*************************************************************************/
/*  resource.cpp                                                         */
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

	if (!path_cache.is_empty()) {
		ResourceCache::lock.write_lock();
		ResourceCache::resources.erase(path_cache);
		ResourceCache::lock.write_unlock();
	}

	path_cache = "";

	ResourceCache::lock.read_lock();
	bool has_path = ResourceCache::resources.has(p_path);
	ResourceCache::lock.read_unlock();

	if (has_path) {
		if (p_take_over) {
			ResourceCache::lock.write_lock();
			Resource **res = ResourceCache::resources.getptr(p_path);
			if (res) {
				(*res)->set_name("");
			}
			ResourceCache::lock.write_unlock();
		} else {
			ResourceCache::lock.read_lock();
			bool exists = ResourceCache::resources.has(p_path);
			ResourceCache::lock.read_unlock();

			ERR_FAIL_COND_MSG(exists, "Another resource is loaded from path '" + p_path + "' (possible cyclic resource inclusion).");
		}
	}
	path_cache = p_path;

	if (!path_cache.is_empty()) {
		ResourceCache::lock.write_lock();
		ResourceCache::resources[path_cache] = this;
		ResourceCache::lock.write_unlock();
	}

	_resource_path_changed();
}

String Resource::get_path() const {
	return path_cache;
}

String Resource::generate_scene_unique_id() {
	// Generate a unique enough hash, but still user-readable.
	// If it's not unique it does not matter because the saver will try again.
	OS::Date date = OS::get_singleton()->get_date();
	OS::Time time = OS::get_singleton()->get_time();
	uint32_t hash = hash_djb2_one_32(OS::get_singleton()->get_ticks_usec());
	hash = hash_djb2_one_32(date.year, hash);
	hash = hash_djb2_one_32(date.month, hash);
	hash = hash_djb2_one_32(date.day, hash);
	hash = hash_djb2_one_32(time.hour, hash);
	hash = hash_djb2_one_32(time.minute, hash);
	hash = hash_djb2_one_32(time.second, hash);
	hash = hash_djb2_one_32(Math::rand(), hash);

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

void Resource::reset_state() {
}
Error Resource::copy_from(const Ref<Resource> &p_resource) {
	ERR_FAIL_COND_V(p_resource.is_null(), ERR_INVALID_PARAMETER);
	if (get_class() != p_resource->get_class()) {
		return ERR_INVALID_PARAMETER;
	}

	reset_state(); //may want to reset state

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

Ref<Resource> Resource::duplicate_for_local_scene(Node *p_for_scene, Map<Ref<Resource>, Ref<Resource>> &remap_cache) {
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
			RES sr = p;
			if (sr.is_valid()) {
				if (sr->is_local_to_scene()) {
					if (remap_cache.has(sr)) {
						p = remap_cache[sr];
					} else {
						RES dupe = sr->duplicate_for_local_scene(p_for_scene, remap_cache);
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

void Resource::configure_for_local_scene(Node *p_for_scene, Map<Ref<Resource>, Ref<Resource>> &remap_cache) {
	List<PropertyInfo> plist;
	get_property_list(&plist);

	local_scene = p_for_scene;

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
			continue;
		}
		Variant p = get(E.name);
		if (p.get_type() == Variant::OBJECT) {
			RES sr = p;
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

	Ref<Resource> r = (Resource *)ClassDB::instantiate(get_class());
	ERR_FAIL_COND_V(r.is_null(), Ref<Resource>());

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
			continue;
		}
		Variant p = get(E.name);

		if ((p.get_type() == Variant::DICTIONARY || p.get_type() == Variant::ARRAY)) {
			r->set(E.name, p.duplicate(p_subresources));
		} else if (p.get_type() == Variant::OBJECT && (p_subresources || (E.usage & PROPERTY_USAGE_DO_NOT_SHARE_ON_DUPLICATE))) {
			RES sr = p;
			if (sr.is_valid()) {
				r->set(E.name, sr->duplicate(p_subresources));
			}
		} else {
			r->set(E.name, p);
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
	return RID();
}

void Resource::register_owner(Object *p_owner) {
	owners.insert(p_owner->get_instance_id());
}

void Resource::unregister_owner(Object *p_owner) {
	owners.erase(p_owner->get_instance_id());
}

void Resource::notify_change_to_owners() {
	for (Set<ObjectID>::Element *E = owners.front(); E; E = E->next()) {
		Object *obj = ObjectDB::get_instance(E->get());
		ERR_CONTINUE_MSG(!obj, "Object was deleted, while still owning a resource."); //wtf
		//TODO store string
		obj->call("resource_changed", RES(this));
	}
}

#ifdef TOOLS_ENABLED

uint32_t Resource::hash_edited_version() const {
	uint32_t hash = hash_djb2_one_32(get_edited_version());

	List<PropertyInfo> plist;
	get_property_list(&plist);

	for (const PropertyInfo &E : plist) {
		if (E.usage & PROPERTY_USAGE_STORAGE && E.type == Variant::OBJECT && E.hint == PROPERTY_HINT_RESOURCE_TYPE) {
			RES res = get(E.name);
			if (res.is_valid()) {
				hash = hash_djb2_one_32(res->hash_edited_version(), hash);
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
	// Can't use GDVIRTUAL in Resource, so this will have to be done with a signal
	emit_signal(SNAME("setup_local_to_scene_requested"));
}

Node *(*Resource::_get_local_scene_func)() = nullptr;
void (*Resource::_update_configuration_warning)() = nullptr;

void Resource::set_as_translation_remapped(bool p_remapped) {
	if (remapped_list.in_list() == p_remapped) {
		return;
	}

	ResourceCache::lock.write_lock();

	if (p_remapped) {
		ResourceLoader::remapped_list.add(&remapped_list);
	} else {
		ResourceLoader::remapped_list.remove(&remapped_list);
	}

	ResourceCache::lock.write_unlock();
}

bool Resource::is_translation_remapped() const {
	return remapped_list.in_list();
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
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "resource_name"), "set_name", "get_name");
}

Resource::Resource() :
		remapped_list(this) {}

Resource::~Resource() {
	if (!path_cache.is_empty()) {
		ResourceCache::lock.write_lock();
		ResourceCache::resources.erase(path_cache);
		ResourceCache::lock.write_unlock();
	}
	if (owners.size()) {
		WARN_PRINT("Resource is still owned.");
	}
}

HashMap<String, Resource *> ResourceCache::resources;
#ifdef TOOLS_ENABLED
HashMap<String, HashMap<String, String>> ResourceCache::resource_path_cache;
#endif

RWLock ResourceCache::lock;
#ifdef TOOLS_ENABLED
RWLock ResourceCache::path_cache_lock;
#endif

void ResourceCache::clear() {
	if (resources.size()) {
		ERR_PRINT("Resources still in use at exit (run with --verbose for details).");
		if (OS::get_singleton()->is_stdout_verbose()) {
			const String *K = nullptr;
			while ((K = resources.next(K))) {
				Resource *r = resources[*K];
				print_line(vformat("Resource still in use: %s (%s)", *K, r->get_class()));
			}
		}
	}

	resources.clear();
}

void ResourceCache::reload_externals() {
}

bool ResourceCache::has(const String &p_path) {
	lock.read_lock();
	bool b = resources.has(p_path);
	lock.read_unlock();

	return b;
}

Resource *ResourceCache::get(const String &p_path) {
	lock.read_lock();

	Resource **res = resources.getptr(p_path);

	lock.read_unlock();

	if (!res) {
		return nullptr;
	}

	return *res;
}

void ResourceCache::get_cached_resources(List<Ref<Resource>> *p_resources) {
	lock.read_lock();
	const String *K = nullptr;
	while ((K = resources.next(K))) {
		Resource *r = resources[*K];
		p_resources->push_back(Ref<Resource>(r));
	}
	lock.read_unlock();
}

int ResourceCache::get_cached_resource_count() {
	lock.read_lock();
	int rc = resources.size();
	lock.read_unlock();

	return rc;
}

void ResourceCache::dump(const char *p_file, bool p_short) {
#ifdef DEBUG_ENABLED
	lock.read_lock();

	Map<String, int> type_count;

	FileAccess *f = nullptr;
	if (p_file) {
		f = FileAccess::open(p_file, FileAccess::WRITE);
		ERR_FAIL_COND_MSG(!f, "Cannot create file at path '" + String(p_file) + "'.");
	}

	const String *K = nullptr;
	while ((K = resources.next(K))) {
		Resource *r = resources[*K];

		if (!type_count.has(r->get_class())) {
			type_count[r->get_class()] = 0;
		}

		type_count[r->get_class()]++;

		if (!p_short) {
			if (f) {
				f->store_line(r->get_class() + ": " + r->get_path());
			}
		}
	}

	for (const KeyValue<String, int> &E : type_count) {
		if (f) {
			f->store_line(E.key + " count: " + itos(E.value));
		}
	}
	if (f) {
		f->close();
		memdelete(f);
	}

	lock.read_unlock();
#else
	WARN_PRINT("ResourceCache::dump only with in debug builds.");
#endif
}
