/**************************************************************************/
/*  class_db.cpp                                                          */
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

#include "class_db.h"

#include "core/config/engine.h"
#include "core/io/resource_loader.h"
#include "core/object/script_language.h"
#include "core/os/mutex.h"
#include "core/version.h"

#define OBJTYPE_RLOCK RWLockRead _rw_lockr_(lock);
#define OBJTYPE_WLOCK RWLockWrite _rw_lockw_(lock);

#ifdef DEBUG_METHODS_ENABLED

MethodDefinition D_METHODP(const char *p_name, const char *const **p_args, uint32_t p_argcount) {
	MethodDefinition md;
	md.name = StaticCString::create(p_name);
	md.args.resize(p_argcount);
	for (uint32_t i = 0; i < p_argcount; i++) {
		md.args.write[i] = StaticCString::create(*p_args[i]);
	}
	return md;
}

#endif

ClassDB::APIType ClassDB::current_api = API_CORE;
HashMap<ClassDB::APIType, uint32_t> ClassDB::api_hashes_cache;

void ClassDB::set_current_api(APIType p_api) {
	DEV_ASSERT(!api_hashes_cache.has(p_api)); // This API type may not be suitable for caching of hash if it can change later.
	current_api = p_api;
}

ClassDB::APIType ClassDB::get_current_api() {
	return current_api;
}

HashMap<StringName, ClassDB::ClassInfo> ClassDB::classes;
HashMap<StringName, StringName> ClassDB::resource_base_extensions;
HashMap<StringName, StringName> ClassDB::compat_classes;

bool ClassDB::_is_parent_class(const StringName &p_class, const StringName &p_inherits) {
	if (!classes.has(p_class)) {
		return false;
	}

	StringName inherits = p_class;
	while (inherits.operator String().length()) {
		if (inherits == p_inherits) {
			return true;
		}
		inherits = _get_parent_class(inherits);
	}

	return false;
}

bool ClassDB::is_parent_class(const StringName &p_class, const StringName &p_inherits) {
	OBJTYPE_RLOCK;

	return _is_parent_class(p_class, p_inherits);
}

void ClassDB::get_class_list(List<StringName> *p_classes) {
	OBJTYPE_RLOCK;

	for (const KeyValue<StringName, ClassInfo> &E : classes) {
		p_classes->push_back(E.key);
	}

	p_classes->sort_custom<StringName::AlphCompare>();
}

#ifdef TOOLS_ENABLED
void ClassDB::get_extensions_class_list(List<StringName> *p_classes) {
	OBJTYPE_RLOCK;

	for (const KeyValue<StringName, ClassInfo> &E : classes) {
		if (E.value.api != API_EXTENSION && E.value.api != API_EDITOR_EXTENSION) {
			continue;
		}
		p_classes->push_back(E.key);
	}

	p_classes->sort_custom<StringName::AlphCompare>();
}
#endif

void ClassDB::get_inheriters_from_class(const StringName &p_class, List<StringName> *p_classes) {
	OBJTYPE_RLOCK;

	for (const KeyValue<StringName, ClassInfo> &E : classes) {
		if (E.key != p_class && _is_parent_class(E.key, p_class)) {
			p_classes->push_back(E.key);
		}
	}
}

void ClassDB::get_direct_inheriters_from_class(const StringName &p_class, List<StringName> *p_classes) {
	OBJTYPE_RLOCK;

	for (const KeyValue<StringName, ClassInfo> &E : classes) {
		if (E.key != p_class && _get_parent_class(E.key) == p_class) {
			p_classes->push_back(E.key);
		}
	}
}

StringName ClassDB::get_parent_class_nocheck(const StringName &p_class) {
	OBJTYPE_RLOCK;

	ClassInfo *ti = classes.getptr(p_class);
	if (!ti) {
		return StringName();
	}
	return ti->inherits;
}

StringName ClassDB::get_compatibility_remapped_class(const StringName &p_class) {
	if (classes.has(p_class)) {
		return p_class;
	}

	if (compat_classes.has(p_class)) {
		return compat_classes[p_class];
	}

	return p_class;
}

StringName ClassDB::_get_parent_class(const StringName &p_class) {
	ClassInfo *ti = classes.getptr(p_class);
	ERR_FAIL_NULL_V_MSG(ti, StringName(), "Cannot get class '" + String(p_class) + "'.");
	return ti->inherits;
}

StringName ClassDB::get_parent_class(const StringName &p_class) {
	OBJTYPE_RLOCK;

	return _get_parent_class(p_class);
}

ClassDB::APIType ClassDB::get_api_type(const StringName &p_class) {
	OBJTYPE_RLOCK;

	ClassInfo *ti = classes.getptr(p_class);

	ERR_FAIL_NULL_V_MSG(ti, API_NONE, "Cannot get class '" + String(p_class) + "'.");
	return ti->api;
}

uint32_t ClassDB::get_api_hash(APIType p_api) {
#ifdef DEBUG_METHODS_ENABLED
	OBJTYPE_WLOCK;

	if (api_hashes_cache.has(p_api)) {
		return api_hashes_cache[p_api];
	}

	uint64_t hash = hash_murmur3_one_64(HashMapHasherDefault::hash(VERSION_FULL_CONFIG));

	List<StringName> class_list;
	for (const KeyValue<StringName, ClassInfo> &E : classes) {
		class_list.push_back(E.key);
	}
	// Must be alphabetically sorted for hash to compute.
	class_list.sort_custom<StringName::AlphCompare>();

	for (const StringName &E : class_list) {
		ClassInfo *t = classes.getptr(E);
		ERR_FAIL_NULL_V_MSG(t, 0, "Cannot get class '" + String(E) + "'.");
		if (t->api != p_api || !t->exposed) {
			continue;
		}
		hash = hash_murmur3_one_64(t->name.hash(), hash);
		hash = hash_murmur3_one_64(t->inherits.hash(), hash);

		{ //methods

			List<StringName> snames;

			for (const KeyValue<StringName, MethodBind *> &F : t->method_map) {
				String name = F.key.operator String();

				ERR_CONTINUE(name.is_empty());

				if (name[0] == '_') {
					continue; // Ignore non-virtual methods that start with an underscore
				}

				snames.push_back(F.key);
			}

			snames.sort_custom<StringName::AlphCompare>();

			for (const StringName &F : snames) {
				MethodBind *mb = t->method_map[F];
				hash = hash_murmur3_one_64(mb->get_name().hash(), hash);
				hash = hash_murmur3_one_64(mb->get_argument_count(), hash);
				hash = hash_murmur3_one_64(mb->get_argument_type(-1), hash); //return

				for (int i = 0; i < mb->get_argument_count(); i++) {
					const PropertyInfo info = mb->get_argument_info(i);
					hash = hash_murmur3_one_64(info.type, hash);
					hash = hash_murmur3_one_64(info.name.hash(), hash);
					hash = hash_murmur3_one_64(info.hint, hash);
					hash = hash_murmur3_one_64(info.hint_string.hash(), hash);
				}

				hash = hash_murmur3_one_64(mb->get_default_argument_count(), hash);

				for (int i = 0; i < mb->get_argument_count(); i++) {
					if (mb->has_default_argument(i)) {
						Variant da = mb->get_default_argument(i);
						hash = hash_murmur3_one_64(da.hash(), hash);
					}
				}

				hash = hash_murmur3_one_64(mb->get_hint_flags(), hash);
			}
		}

		{ //constants

			List<StringName> snames;

			for (const KeyValue<StringName, int64_t> &F : t->constant_map) {
				snames.push_back(F.key);
			}

			snames.sort_custom<StringName::AlphCompare>();

			for (const StringName &F : snames) {
				hash = hash_murmur3_one_64(F.hash(), hash);
				hash = hash_murmur3_one_64(t->constant_map[F], hash);
			}
		}

		{ //signals

			List<StringName> snames;

			for (const KeyValue<StringName, MethodInfo> &F : t->signal_map) {
				snames.push_back(F.key);
			}

			snames.sort_custom<StringName::AlphCompare>();

			for (const StringName &F : snames) {
				MethodInfo &mi = t->signal_map[F];
				hash = hash_murmur3_one_64(F.hash(), hash);
				for (int i = 0; i < mi.arguments.size(); i++) {
					hash = hash_murmur3_one_64(mi.arguments[i].type, hash);
				}
			}
		}

		{ //properties

			List<StringName> snames;

			for (const KeyValue<StringName, PropertySetGet> &F : t->property_setget) {
				snames.push_back(F.key);
			}

			snames.sort_custom<StringName::AlphCompare>();

			for (const StringName &F : snames) {
				PropertySetGet *psg = t->property_setget.getptr(F);
				ERR_FAIL_NULL_V(psg, 0);

				hash = hash_murmur3_one_64(F.hash(), hash);
				hash = hash_murmur3_one_64(psg->setter.hash(), hash);
				hash = hash_murmur3_one_64(psg->getter.hash(), hash);
			}
		}

		//property list
		for (const PropertyInfo &F : t->property_list) {
			hash = hash_murmur3_one_64(F.name.hash(), hash);
			hash = hash_murmur3_one_64(F.type, hash);
			hash = hash_murmur3_one_64(F.hint, hash);
			hash = hash_murmur3_one_64(F.hint_string.hash(), hash);
			hash = hash_murmur3_one_64(F.usage, hash);
		}
	}

	hash = hash_fmix32(hash);

	// Extension API changes at runtime; let's just not cache them by now.
	if (p_api != API_EXTENSION && p_api != API_EDITOR_EXTENSION) {
		api_hashes_cache[p_api] = hash;
	}

	return hash;
#else
	return 0;
#endif
}

bool ClassDB::class_exists(const StringName &p_class) {
	OBJTYPE_RLOCK;
	return classes.has(p_class);
}

void ClassDB::add_compatibility_class(const StringName &p_class, const StringName &p_fallback) {
	OBJTYPE_WLOCK;
	compat_classes[p_class] = p_fallback;
}

StringName ClassDB::get_compatibility_class(const StringName &p_class) {
	if (compat_classes.has(p_class)) {
		return compat_classes[p_class];
	}
	return StringName();
}

Object *ClassDB::instantiate(const StringName &p_class) {
	ClassInfo *ti;
	{
		OBJTYPE_RLOCK;
		ti = classes.getptr(p_class);
		if (!ti || ti->disabled || !ti->creation_func || (ti->gdextension && !ti->gdextension->create_instance)) {
			if (compat_classes.has(p_class)) {
				ti = classes.getptr(compat_classes[p_class]);
			}
		}
		ERR_FAIL_NULL_V_MSG(ti, nullptr, "Cannot get class '" + String(p_class) + "'.");
		ERR_FAIL_COND_V_MSG(ti->disabled, nullptr, "Class '" + String(p_class) + "' is disabled.");
		ERR_FAIL_NULL_V_MSG(ti->creation_func, nullptr, "Class '" + String(p_class) + "' or its base class cannot be instantiated.");
	}
#ifdef TOOLS_ENABLED
	if (ti->api == API_EDITOR && !Engine::get_singleton()->is_editor_hint()) {
		ERR_PRINT("Class '" + String(p_class) + "' can only be instantiated by editor.");
		return nullptr;
	}
#endif
	if (ti->gdextension && ti->gdextension->create_instance) {
		Object *obj = (Object *)ti->gdextension->create_instance(ti->gdextension->class_userdata);
#ifdef TOOLS_ENABLED
		if (ti->gdextension->track_instance) {
			ti->gdextension->track_instance(ti->gdextension->tracking_userdata, obj);
		}
#endif
		return obj;
	} else {
		return ti->creation_func();
	}
}

void ClassDB::set_object_extension_instance(Object *p_object, const StringName &p_class, GDExtensionClassInstancePtr p_instance) {
	ERR_FAIL_NULL(p_object);
	ClassInfo *ti;
	{
		OBJTYPE_RLOCK;
		ti = classes.getptr(p_class);
		if (!ti || ti->disabled || !ti->creation_func || (ti->gdextension && !ti->gdextension->create_instance)) {
			if (compat_classes.has(p_class)) {
				ti = classes.getptr(compat_classes[p_class]);
			}
		}
		ERR_FAIL_NULL_MSG(ti, "Cannot get class '" + String(p_class) + "'.");
		ERR_FAIL_COND_MSG(ti->disabled, "Class '" + String(p_class) + "' is disabled.");
		ERR_FAIL_NULL_MSG(ti->gdextension, "Class '" + String(p_class) + "' has no native extension.");
	}

	p_object->_extension = ti->gdextension;
	p_object->_extension_instance = p_instance;
}

bool ClassDB::can_instantiate(const StringName &p_class) {
	OBJTYPE_RLOCK;

	ClassInfo *ti = classes.getptr(p_class);
	if (!ti) {
		if (!ScriptServer::is_global_class(p_class)) {
			ERR_FAIL_V_MSG(false, "Cannot get class '" + String(p_class) + "'.");
		}
		String path = ScriptServer::get_global_class_path(p_class);
		Ref<Script> scr = ResourceLoader::load(path);
		return scr.is_valid() && scr->is_valid() && !scr->is_abstract();
	}
#ifdef TOOLS_ENABLED
	if (ti->api == API_EDITOR && !Engine::get_singleton()->is_editor_hint()) {
		return false;
	}
#endif
	return (!ti->disabled && ti->creation_func != nullptr && !(ti->gdextension && !ti->gdextension->create_instance));
}

bool ClassDB::is_virtual(const StringName &p_class) {
	OBJTYPE_RLOCK;

	ClassInfo *ti = classes.getptr(p_class);
	if (!ti) {
		if (!ScriptServer::is_global_class(p_class)) {
			ERR_FAIL_V_MSG(false, "Cannot get class '" + String(p_class) + "'.");
		}
		String path = ScriptServer::get_global_class_path(p_class);
		Ref<Script> scr = ResourceLoader::load(path);
		return scr.is_valid() && scr->is_valid() && scr->is_abstract();
	}
#ifdef TOOLS_ENABLED
	if (ti->api == API_EDITOR && !Engine::get_singleton()->is_editor_hint()) {
		return false;
	}
#endif
	return (!ti->disabled && ti->creation_func != nullptr && !(ti->gdextension && !ti->gdextension->create_instance) && ti->is_virtual);
}

void ClassDB::_add_class2(const StringName &p_class, const StringName &p_inherits) {
	OBJTYPE_WLOCK;

	const StringName &name = p_class;

	ERR_FAIL_COND_MSG(classes.has(name), "Class '" + String(p_class) + "' already exists.");

	classes[name] = ClassInfo();
	ClassInfo &ti = classes[name];
	ti.name = name;
	ti.inherits = p_inherits;
	ti.api = current_api;

	if (ti.inherits) {
		ERR_FAIL_COND(!classes.has(ti.inherits)); //it MUST be registered.
		ti.inherits_ptr = &classes[ti.inherits];

	} else {
		ti.inherits_ptr = nullptr;
	}
}

static MethodInfo info_from_bind(MethodBind *p_method) {
	MethodInfo minfo;
	minfo.name = p_method->get_name();
	minfo.id = p_method->get_method_id();

	for (int i = 0; i < p_method->get_argument_count(); i++) {
		minfo.arguments.push_back(p_method->get_argument_info(i));
	}

	minfo.return_val = p_method->get_return_info();
	minfo.flags = p_method->get_hint_flags();

	for (int i = 0; i < p_method->get_argument_count(); i++) {
		if (p_method->has_default_argument(i)) {
			minfo.default_arguments.push_back(p_method->get_default_argument(i));
		}
	}

	return minfo;
}

void ClassDB::get_method_list(const StringName &p_class, List<MethodInfo> *p_methods, bool p_no_inheritance, bool p_exclude_from_properties) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		if (type->disabled) {
			if (p_no_inheritance) {
				break;
			}

			type = type->inherits_ptr;
			continue;
		}

#ifdef DEBUG_METHODS_ENABLED
		for (const MethodInfo &E : type->virtual_methods) {
			p_methods->push_back(E);
		}

		for (const StringName &E : type->method_order) {
			if (p_exclude_from_properties && type->methods_in_properties.has(E)) {
				continue;
			}

			MethodBind *method = type->method_map.get(E);
			MethodInfo minfo = info_from_bind(method);

			p_methods->push_back(minfo);
		}
#else
		for (KeyValue<StringName, MethodBind *> &E : type->method_map) {
			MethodBind *m = E.value;
			MethodInfo minfo = info_from_bind(m);
			p_methods->push_back(minfo);
		}
#endif

		if (p_no_inheritance) {
			break;
		}

		type = type->inherits_ptr;
	}
}

void ClassDB::get_method_list_with_compatibility(const StringName &p_class, List<Pair<MethodInfo, uint32_t>> *p_methods, bool p_no_inheritance, bool p_exclude_from_properties) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		if (type->disabled) {
			if (p_no_inheritance) {
				break;
			}

			type = type->inherits_ptr;
			continue;
		}

#ifdef DEBUG_METHODS_ENABLED
		for (const MethodInfo &E : type->virtual_methods) {
			Pair<MethodInfo, uint32_t> pair(E, 0);
			p_methods->push_back(pair);
		}

		for (const StringName &E : type->method_order) {
			if (p_exclude_from_properties && type->methods_in_properties.has(E)) {
				continue;
			}

			MethodBind *method = type->method_map.get(E);
			MethodInfo minfo = info_from_bind(method);

			Pair<MethodInfo, uint32_t> pair(minfo, method->get_hash());
			p_methods->push_back(pair);
		}
#else
		for (KeyValue<StringName, MethodBind *> &E : type->method_map) {
			MethodBind *method = E.value;
			MethodInfo minfo = info_from_bind(method);

			Pair<MethodInfo, uint32_t> pair(minfo, method->get_hash());
			p_methods->push_back(pair);
		}
#endif

		for (const KeyValue<StringName, LocalVector<MethodBind *, unsigned int, false, false>> &E : type->method_map_compatibility) {
			LocalVector<MethodBind *> compat = E.value;
			for (MethodBind *method : compat) {
				MethodInfo minfo = info_from_bind(method);

				Pair<MethodInfo, uint32_t> pair(minfo, method->get_hash());
				p_methods->push_back(pair);
			}
		}

		if (p_no_inheritance) {
			break;
		}

		type = type->inherits_ptr;
	}
}

bool ClassDB::get_method_info(const StringName &p_class, const StringName &p_method, MethodInfo *r_info, bool p_no_inheritance, bool p_exclude_from_properties) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		if (type->disabled) {
			if (p_no_inheritance) {
				break;
			}

			type = type->inherits_ptr;
			continue;
		}

#ifdef DEBUG_METHODS_ENABLED
		MethodBind **method = type->method_map.getptr(p_method);
		if (method && *method) {
			if (r_info != nullptr) {
				MethodInfo minfo = info_from_bind(*method);
				*r_info = minfo;
			}
			return true;
		} else if (type->virtual_methods_map.has(p_method)) {
			if (r_info) {
				*r_info = type->virtual_methods_map[p_method];
			}
			return true;
		}
#else
		if (type->method_map.has(p_method)) {
			if (r_info) {
				MethodBind *m = type->method_map[p_method];
				MethodInfo minfo = info_from_bind(m);
				*r_info = minfo;
			}
			return true;
		}
#endif

		if (p_no_inheritance) {
			break;
		}

		type = type->inherits_ptr;
	}

	return false;
}

MethodBind *ClassDB::get_method(const StringName &p_class, const StringName &p_name) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		MethodBind **method = type->method_map.getptr(p_name);
		if (method && *method) {
			return *method;
		}
		type = type->inherits_ptr;
	}
	return nullptr;
}

Vector<uint32_t> ClassDB::get_method_compatibility_hashes(const StringName &p_class, const StringName &p_name) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		if (type->method_map_compatibility.has(p_name)) {
			LocalVector<MethodBind *> *c = type->method_map_compatibility.getptr(p_name);
			Vector<uint32_t> ret;
			for (uint32_t i = 0; i < c->size(); i++) {
				ret.push_back((*c)[i]->get_hash());
			}
			return ret;
		}
		type = type->inherits_ptr;
	}
	return Vector<uint32_t>();
}

MethodBind *ClassDB::get_method_with_compatibility(const StringName &p_class, const StringName &p_name, uint64_t p_hash, bool *r_method_exists, bool *r_is_deprecated) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		MethodBind **method = type->method_map.getptr(p_name);
		if (method && *method) {
			if (r_method_exists) {
				*r_method_exists = true;
			}
			if ((*method)->get_hash() == p_hash) {
				return *method;
			}
		}

		LocalVector<MethodBind *> *compat = type->method_map_compatibility.getptr(p_name);
		if (compat) {
			if (r_method_exists) {
				*r_method_exists = true;
			}
			for (uint32_t i = 0; i < compat->size(); i++) {
				if ((*compat)[i]->get_hash() == p_hash) {
					if (r_is_deprecated) {
						*r_is_deprecated = true;
					}
					return (*compat)[i];
				}
			}
		}
		type = type->inherits_ptr;
	}
	return nullptr;
}

void ClassDB::bind_integer_constant(const StringName &p_class, const StringName &p_enum, const StringName &p_name, int64_t p_constant, bool p_is_bitfield) {
	OBJTYPE_WLOCK;

	ClassInfo *type = classes.getptr(p_class);

	ERR_FAIL_NULL(type);

	if (type->constant_map.has(p_name)) {
		ERR_FAIL();
	}

	type->constant_map[p_name] = p_constant;

	String enum_name = p_enum;
	if (!enum_name.is_empty()) {
		if (enum_name.contains(".")) {
			enum_name = enum_name.get_slicec('.', 1);
		}

		ClassInfo::EnumInfo *constants_list = type->enum_map.getptr(enum_name);

		if (constants_list) {
			constants_list->constants.push_back(p_name);
			constants_list->is_bitfield = p_is_bitfield;
		} else {
			ClassInfo::EnumInfo new_list;
			new_list.is_bitfield = p_is_bitfield;
			new_list.constants.push_back(p_name);
			type->enum_map[enum_name] = new_list;
		}
	}

#ifdef DEBUG_METHODS_ENABLED
	type->constant_order.push_back(p_name);
#endif
}

void ClassDB::get_integer_constant_list(const StringName &p_class, List<String> *p_constants, bool p_no_inheritance) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
#ifdef DEBUG_METHODS_ENABLED
		for (const StringName &E : type->constant_order) {
			p_constants->push_back(E);
		}
#else

		for (const KeyValue<StringName, int64_t> &E : type->constant_map) {
			p_constants->push_back(E.key);
		}

#endif
		if (p_no_inheritance) {
			break;
		}

		type = type->inherits_ptr;
	}
}

int64_t ClassDB::get_integer_constant(const StringName &p_class, const StringName &p_name, bool *p_success) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		int64_t *constant = type->constant_map.getptr(p_name);
		if (constant) {
			if (p_success) {
				*p_success = true;
			}
			return *constant;
		}

		type = type->inherits_ptr;
	}

	if (p_success) {
		*p_success = false;
	}

	return 0;
}

bool ClassDB::has_integer_constant(const StringName &p_class, const StringName &p_name, bool p_no_inheritance) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		if (type->constant_map.has(p_name)) {
			return true;
		}
		if (p_no_inheritance) {
			return false;
		}

		type = type->inherits_ptr;
	}

	return false;
}

StringName ClassDB::get_integer_constant_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		for (KeyValue<StringName, ClassInfo::EnumInfo> &E : type->enum_map) {
			List<StringName> &constants_list = E.value.constants;
			const List<StringName>::Element *found = constants_list.find(p_name);
			if (found) {
				return E.key;
			}
		}

		if (p_no_inheritance) {
			break;
		}

		type = type->inherits_ptr;
	}

	return StringName();
}

void ClassDB::get_enum_list(const StringName &p_class, List<StringName> *p_enums, bool p_no_inheritance) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		for (KeyValue<StringName, ClassInfo::EnumInfo> &E : type->enum_map) {
			p_enums->push_back(E.key);
		}

		if (p_no_inheritance) {
			break;
		}

		type = type->inherits_ptr;
	}
}

void ClassDB::get_enum_constants(const StringName &p_class, const StringName &p_enum, List<StringName> *p_constants, bool p_no_inheritance) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		const ClassInfo::EnumInfo *constants = type->enum_map.getptr(p_enum);

		if (constants) {
			for (const List<StringName>::Element *E = constants->constants.front(); E; E = E->next()) {
				p_constants->push_back(E->get());
			}
		}

		if (p_no_inheritance) {
			break;
		}

		type = type->inherits_ptr;
	}
}

void ClassDB::set_method_error_return_values(const StringName &p_class, const StringName &p_method, const Vector<Error> &p_values) {
#ifdef DEBUG_METHODS_ENABLED
	OBJTYPE_WLOCK;
	ClassInfo *type = classes.getptr(p_class);

	ERR_FAIL_NULL(type);

	type->method_error_values[p_method] = p_values;
#endif
}

Vector<Error> ClassDB::get_method_error_return_values(const StringName &p_class, const StringName &p_method) {
#ifdef DEBUG_METHODS_ENABLED
	OBJTYPE_RLOCK;
	ClassInfo *type = classes.getptr(p_class);

	ERR_FAIL_NULL_V(type, Vector<Error>());

	if (!type->method_error_values.has(p_method)) {
		return Vector<Error>();
	}
	return type->method_error_values[p_method];
#else
	return Vector<Error>();
#endif
}

bool ClassDB::has_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		if (type->enum_map.has(p_name)) {
			return true;
		}
		if (p_no_inheritance) {
			return false;
		}

		type = type->inherits_ptr;
	}

	return false;
}

bool ClassDB::is_enum_bitfield(const StringName &p_class, const StringName &p_name, bool p_no_inheritance) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);

	while (type) {
		if (type->enum_map.has(p_name) && type->enum_map[p_name].is_bitfield) {
			return true;
		}
		if (p_no_inheritance) {
			return false;
		}

		type = type->inherits_ptr;
	}

	return false;
}

void ClassDB::add_signal(const StringName &p_class, const MethodInfo &p_signal) {
	OBJTYPE_WLOCK;

	ClassInfo *type = classes.getptr(p_class);
	ERR_FAIL_NULL(type);

	StringName sname = p_signal.name;

#ifdef DEBUG_METHODS_ENABLED
	ClassInfo *check = type;
	while (check) {
		ERR_FAIL_COND_MSG(check->signal_map.has(sname), "Class '" + String(p_class) + "' already has signal '" + String(sname) + "'.");
		check = check->inherits_ptr;
	}
#endif

	type->signal_map[sname] = p_signal;
}

void ClassDB::get_signal_list(const StringName &p_class, List<MethodInfo> *p_signals, bool p_no_inheritance) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);
	ERR_FAIL_NULL(type);

	ClassInfo *check = type;

	while (check) {
		for (KeyValue<StringName, MethodInfo> &E : check->signal_map) {
			p_signals->push_back(E.value);
		}

		if (p_no_inheritance) {
			return;
		}

		check = check->inherits_ptr;
	}
}

bool ClassDB::has_signal(const StringName &p_class, const StringName &p_signal, bool p_no_inheritance) {
	OBJTYPE_RLOCK;
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		if (check->signal_map.has(p_signal)) {
			return true;
		}
		if (p_no_inheritance) {
			return false;
		}
		check = check->inherits_ptr;
	}

	return false;
}

bool ClassDB::get_signal(const StringName &p_class, const StringName &p_signal, MethodInfo *r_signal) {
	OBJTYPE_RLOCK;
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		if (check->signal_map.has(p_signal)) {
			if (r_signal) {
				*r_signal = check->signal_map[p_signal];
			}
			return true;
		}
		check = check->inherits_ptr;
	}

	return false;
}

void ClassDB::add_property_group(const StringName &p_class, const String &p_name, const String &p_prefix, int p_indent_depth) {
	OBJTYPE_WLOCK;
	ClassInfo *type = classes.getptr(p_class);
	ERR_FAIL_NULL(type);

	String prefix = p_prefix;
	if (p_indent_depth > 0) {
		prefix = vformat("%s,%d", p_prefix, p_indent_depth);
	}

	type->property_list.push_back(PropertyInfo(Variant::NIL, p_name, PROPERTY_HINT_NONE, prefix, PROPERTY_USAGE_GROUP));
}

void ClassDB::add_property_subgroup(const StringName &p_class, const String &p_name, const String &p_prefix, int p_indent_depth) {
	OBJTYPE_WLOCK;
	ClassInfo *type = classes.getptr(p_class);
	ERR_FAIL_NULL(type);

	String prefix = p_prefix;
	if (p_indent_depth > 0) {
		prefix = vformat("%s,%d", p_prefix, p_indent_depth);
	}

	type->property_list.push_back(PropertyInfo(Variant::NIL, p_name, PROPERTY_HINT_NONE, prefix, PROPERTY_USAGE_SUBGROUP));
}

void ClassDB::add_property_array_count(const StringName &p_class, const String &p_label, const StringName &p_count_property, const StringName &p_count_setter, const StringName &p_count_getter, const String &p_array_element_prefix, uint32_t p_count_usage) {
	add_property(p_class, PropertyInfo(Variant::INT, p_count_property, PROPERTY_HINT_NONE, "", p_count_usage | PROPERTY_USAGE_ARRAY, vformat("%s,%s", p_label, p_array_element_prefix)), p_count_setter, p_count_getter);
}

void ClassDB::add_property_array(const StringName &p_class, const StringName &p_path, const String &p_array_element_prefix) {
	OBJTYPE_WLOCK;
	ClassInfo *type = classes.getptr(p_class);
	ERR_FAIL_NULL(type);

	type->property_list.push_back(PropertyInfo(Variant::NIL, p_path, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_ARRAY, p_array_element_prefix));
}

// NOTE: For implementation simplicity reasons, this method doesn't allow setters to have optional arguments at the end.
void ClassDB::add_property(const StringName &p_class, const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter, int p_index) {
	lock.read_lock();
	ClassInfo *type = classes.getptr(p_class);
	lock.read_unlock();

	ERR_FAIL_NULL(type);

	MethodBind *mb_set = nullptr;
	if (p_setter) {
		mb_set = get_method(p_class, p_setter);
#ifdef DEBUG_METHODS_ENABLED

		ERR_FAIL_NULL_MSG(mb_set, "Invalid setter '" + p_class + "::" + p_setter + "' for property '" + p_pinfo.name + "'.");

		int exp_args = 1 + (p_index >= 0 ? 1 : 0);
		ERR_FAIL_COND_MSG(mb_set->get_argument_count() != exp_args, "Invalid function for setter '" + p_class + "::" + p_setter + " for property '" + p_pinfo.name + "'.");
#endif
	}

	MethodBind *mb_get = nullptr;
	if (p_getter) {
		mb_get = get_method(p_class, p_getter);
#ifdef DEBUG_METHODS_ENABLED

		ERR_FAIL_NULL_MSG(mb_get, "Invalid getter '" + p_class + "::" + p_getter + "' for property '" + p_pinfo.name + "'.");

		int exp_args = 0 + (p_index >= 0 ? 1 : 0);
		ERR_FAIL_COND_MSG(mb_get->get_argument_count() != exp_args, "Invalid function for getter '" + p_class + "::" + p_getter + "' for property: '" + p_pinfo.name + "'.");
#endif
	}

#ifdef DEBUG_METHODS_ENABLED
	ERR_FAIL_COND_MSG(type->property_setget.has(p_pinfo.name), "Object '" + p_class + "' already has property '" + p_pinfo.name + "'.");
#endif

	OBJTYPE_WLOCK

	type->property_list.push_back(p_pinfo);
	type->property_map[p_pinfo.name] = p_pinfo;
#ifdef DEBUG_METHODS_ENABLED
	if (mb_get) {
		type->methods_in_properties.insert(p_getter);
	}
	if (mb_set) {
		type->methods_in_properties.insert(p_setter);
	}
#endif
	PropertySetGet psg;
	psg.setter = p_setter;
	psg.getter = p_getter;
	psg._setptr = mb_set;
	psg._getptr = mb_get;
	psg.index = p_index;
	psg.type = p_pinfo.type;

	type->property_setget[p_pinfo.name] = psg;
}

void ClassDB::set_property_default_value(const StringName &p_class, const StringName &p_name, const Variant &p_default) {
	if (!default_values.has(p_class)) {
		default_values[p_class] = HashMap<StringName, Variant>();
	}
	default_values[p_class][p_name] = p_default;
}

void ClassDB::add_linked_property(const StringName &p_class, const String &p_property, const String &p_linked_property) {
#ifdef TOOLS_ENABLED
	OBJTYPE_WLOCK;
	ClassInfo *type = classes.getptr(p_class);
	ERR_FAIL_NULL(type);

	ERR_FAIL_COND(!type->property_map.has(p_property));
	ERR_FAIL_COND(!type->property_map.has(p_linked_property));

	if (!type->linked_properties.has(p_property)) {
		type->linked_properties.insert(p_property, List<StringName>());
	}
	type->linked_properties[p_property].push_back(p_linked_property);

#endif
}

void ClassDB::get_property_list(const StringName &p_class, List<PropertyInfo> *p_list, bool p_no_inheritance, const Object *p_validator) {
	OBJTYPE_RLOCK;

	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		for (const PropertyInfo &pi : check->property_list) {
			if (p_validator) {
				// Making a copy as we may modify it.
				PropertyInfo pi_mut = pi;
				p_validator->validate_property(pi_mut);
				p_list->push_back(pi_mut);
			} else {
				p_list->push_back(pi);
			}
		}

		if (p_no_inheritance) {
			return;
		}
		check = check->inherits_ptr;
	}
}

void ClassDB::get_linked_properties_info(const StringName &p_class, const StringName &p_property, List<StringName> *r_properties, bool p_no_inheritance) {
#ifdef TOOLS_ENABLED
	ClassInfo *check = classes.getptr(p_class);
	while (check) {
		if (!check->linked_properties.has(p_property)) {
			return;
		}
		for (const StringName &E : check->linked_properties[p_property]) {
			r_properties->push_back(E);
		}

		if (p_no_inheritance) {
			break;
		}
		check = check->inherits_ptr;
	}
#endif
}

bool ClassDB::get_property_info(const StringName &p_class, const StringName &p_property, PropertyInfo *r_info, bool p_no_inheritance, const Object *p_validator) {
	OBJTYPE_RLOCK;

	ClassInfo *check = classes.getptr(p_class);
	while (check) {
		if (check->property_map.has(p_property)) {
			PropertyInfo pinfo = check->property_map[p_property];
			if (p_validator) {
				p_validator->validate_property(pinfo);
			}
			if (r_info) {
				*r_info = pinfo;
			}
			return true;
		}
		if (p_no_inheritance) {
			break;
		}
		check = check->inherits_ptr;
	}

	return false;
}

bool ClassDB::set_property(Object *p_object, const StringName &p_property, const Variant &p_value, bool *r_valid) {
	ERR_FAIL_NULL_V(p_object, false);

	ClassInfo *type = classes.getptr(p_object->get_class_name());
	ClassInfo *check = type;
	while (check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {
			if (!psg->setter) {
				if (r_valid) {
					*r_valid = false;
				}
				return true; //return true but do nothing
			}

			Callable::CallError ce;

			if (psg->index >= 0) {
				Variant index = psg->index;
				const Variant *arg[2] = { &index, &p_value };
				//p_object->call(psg->setter,arg,2,ce);
				if (psg->_setptr) {
					psg->_setptr->call(p_object, arg, 2, ce);
				} else {
					p_object->callp(psg->setter, arg, 2, ce);
				}

			} else {
				const Variant *arg[1] = { &p_value };
				if (psg->_setptr) {
					psg->_setptr->call(p_object, arg, 1, ce);
				} else {
					p_object->callp(psg->setter, arg, 1, ce);
				}
			}

			if (r_valid) {
				*r_valid = ce.error == Callable::CallError::CALL_OK;
			}

			return true;
		}

		check = check->inherits_ptr;
	}

	return false;
}

bool ClassDB::get_property(Object *p_object, const StringName &p_property, Variant &r_value) {
	ERR_FAIL_NULL_V(p_object, false);

	ClassInfo *type = classes.getptr(p_object->get_class_name());
	ClassInfo *check = type;
	while (check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {
			if (!psg->getter) {
				return true; //return true but do nothing
			}

			if (psg->index >= 0) {
				Variant index = psg->index;
				const Variant *arg[1] = { &index };
				Callable::CallError ce;
				r_value = p_object->callp(psg->getter, arg, 1, ce);

			} else {
				Callable::CallError ce;
				if (psg->_getptr) {
					r_value = psg->_getptr->call(p_object, nullptr, 0, ce);
				} else {
					r_value = p_object->callp(psg->getter, nullptr, 0, ce);
				}
			}
			return true;
		}

		const int64_t *c = check->constant_map.getptr(p_property); //constants count
		if (c) {
			r_value = *c;
			return true;
		}

		if (check->method_map.has(p_property)) { //methods count
			r_value = Callable(p_object, p_property);
			return true;
		}

		if (check->signal_map.has(p_property)) { //signals count
			r_value = Signal(p_object, p_property);
			return true;
		}

		check = check->inherits_ptr;
	}

	return false;
}

int ClassDB::get_property_index(const StringName &p_class, const StringName &p_property, bool *r_is_valid) {
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {
			if (r_is_valid) {
				*r_is_valid = true;
			}

			return psg->index;
		}

		check = check->inherits_ptr;
	}
	if (r_is_valid) {
		*r_is_valid = false;
	}

	return -1;
}

Variant::Type ClassDB::get_property_type(const StringName &p_class, const StringName &p_property, bool *r_is_valid) {
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {
			if (r_is_valid) {
				*r_is_valid = true;
			}

			return psg->type;
		}

		check = check->inherits_ptr;
	}
	if (r_is_valid) {
		*r_is_valid = false;
	}

	return Variant::NIL;
}

StringName ClassDB::get_property_setter(const StringName &p_class, const StringName &p_property) {
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {
			return psg->setter;
		}

		check = check->inherits_ptr;
	}

	return StringName();
}

StringName ClassDB::get_property_getter(const StringName &p_class, const StringName &p_property) {
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		const PropertySetGet *psg = check->property_setget.getptr(p_property);
		if (psg) {
			return psg->getter;
		}

		check = check->inherits_ptr;
	}

	return StringName();
}

bool ClassDB::has_property(const StringName &p_class, const StringName &p_property, bool p_no_inheritance) {
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		if (check->property_setget.has(p_property)) {
			return true;
		}

		if (p_no_inheritance) {
			break;
		}
		check = check->inherits_ptr;
	}

	return false;
}

void ClassDB::set_method_flags(const StringName &p_class, const StringName &p_method, int p_flags) {
	OBJTYPE_WLOCK;
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	ERR_FAIL_NULL(check);
	ERR_FAIL_COND(!check->method_map.has(p_method));
	check->method_map[p_method]->set_hint_flags(p_flags);
}

bool ClassDB::has_method(const StringName &p_class, const StringName &p_method, bool p_no_inheritance) {
	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		if (check->method_map.has(p_method)) {
			return true;
		}
		if (p_no_inheritance) {
			return false;
		}
		check = check->inherits_ptr;
	}

	return false;
}

void ClassDB::bind_method_custom(const StringName &p_class, MethodBind *p_method) {
	_bind_method_custom(p_class, p_method, false);
}
void ClassDB::bind_compatibility_method_custom(const StringName &p_class, MethodBind *p_method) {
	_bind_method_custom(p_class, p_method, true);
}

void ClassDB::_bind_compatibility(ClassInfo *type, MethodBind *p_method) {
	if (!type->method_map_compatibility.has(p_method->get_name())) {
		type->method_map_compatibility.insert(p_method->get_name(), LocalVector<MethodBind *>());
	}
	type->method_map_compatibility[p_method->get_name()].push_back(p_method);
}

void ClassDB::_bind_method_custom(const StringName &p_class, MethodBind *p_method, bool p_compatibility) {
	OBJTYPE_WLOCK;

	ClassInfo *type = classes.getptr(p_class);
	if (!type) {
		ERR_FAIL_MSG("Couldn't bind custom method '" + p_method->get_name() + "' for instance '" + p_class + "'.");
	}

	if (p_compatibility) {
		_bind_compatibility(type, p_method);
		return;
	}

	if (type->method_map.has(p_method->get_name())) {
		// overloading not supported
		ERR_FAIL_MSG("Method already bound '" + p_class + "::" + p_method->get_name() + "'.");
	}

#ifdef DEBUG_METHODS_ENABLED
	type->method_order.push_back(p_method->get_name());
#endif

	type->method_map[p_method->get_name()] = p_method;
}

MethodBind *ClassDB::_bind_vararg_method(MethodBind *p_bind, const StringName &p_name, const Vector<Variant> &p_default_args, bool p_compatibility) {
	MethodBind *bind = p_bind;
	bind->set_name(p_name);
	bind->set_default_arguments(p_default_args);

	String instance_type = bind->get_instance_class();

	ClassInfo *type = classes.getptr(instance_type);
	if (!type) {
		memdelete(bind);
		ERR_FAIL_NULL_V(type, nullptr);
	}

	if (p_compatibility) {
		_bind_compatibility(type, bind);
		return bind;
	}

	if (type->method_map.has(p_name)) {
		memdelete(bind);
		// Overloading not supported
		ERR_FAIL_V_MSG(nullptr, "Method already bound: " + instance_type + "::" + p_name + ".");
	}
	type->method_map[p_name] = bind;
#ifdef DEBUG_METHODS_ENABLED
	// FIXME: <reduz> set_return_type is no longer in MethodBind, so I guess it should be moved to vararg method bind
	//bind->set_return_type("Variant");
	type->method_order.push_back(p_name);
#endif

	return bind;
}

#ifdef DEBUG_METHODS_ENABLED
MethodBind *ClassDB::bind_methodfi(uint32_t p_flags, MethodBind *p_bind, bool p_compatibility, const MethodDefinition &method_name, const Variant **p_defs, int p_defcount) {
	StringName mdname = method_name.name;
#else
MethodBind *ClassDB::bind_methodfi(uint32_t p_flags, MethodBind *p_bind, bool p_compatibility, const char *method_name, const Variant **p_defs, int p_defcount) {
	StringName mdname = StaticCString::create(method_name);
#endif

	OBJTYPE_WLOCK;
	ERR_FAIL_NULL_V(p_bind, nullptr);
	p_bind->set_name(mdname);

	String instance_type = p_bind->get_instance_class();

#ifdef DEBUG_ENABLED

	ERR_FAIL_COND_V_MSG(!p_compatibility && has_method(instance_type, mdname), nullptr, "Class " + String(instance_type) + " already has a method " + String(mdname) + ".");
#endif

	ClassInfo *type = classes.getptr(instance_type);
	if (!type) {
		memdelete(p_bind);
		ERR_FAIL_V_MSG(nullptr, "Couldn't bind method '" + mdname + "' for instance '" + instance_type + "'.");
	}

	if (!p_compatibility && type->method_map.has(mdname)) {
		memdelete(p_bind);
		// overloading not supported
		ERR_FAIL_V_MSG(nullptr, "Method already bound '" + instance_type + "::" + mdname + "'.");
	}

#ifdef DEBUG_METHODS_ENABLED

	if (method_name.args.size() > p_bind->get_argument_count()) {
		memdelete(p_bind);
		ERR_FAIL_V_MSG(nullptr, "Method definition provides more arguments than the method actually has '" + instance_type + "::" + mdname + "'.");
	}

	p_bind->set_argument_names(method_name.args);

	if (!p_compatibility) {
		type->method_order.push_back(mdname);
	}
#endif

	if (p_compatibility) {
		_bind_compatibility(type, p_bind);
	} else {
		type->method_map[mdname] = p_bind;
	}

	Vector<Variant> defvals;

	defvals.resize(p_defcount);
	for (int i = 0; i < p_defcount; i++) {
		defvals.write[i] = *p_defs[i];
	}

	p_bind->set_default_arguments(defvals);
	p_bind->set_hint_flags(p_flags);
	return p_bind;
}

void ClassDB::add_virtual_method(const StringName &p_class, const MethodInfo &p_method, bool p_virtual, const Vector<String> &p_arg_names, bool p_object_core) {
	ERR_FAIL_COND_MSG(!classes.has(p_class), "Request for nonexistent class '" + p_class + "'.");

	OBJTYPE_WLOCK;

#ifdef DEBUG_METHODS_ENABLED
	MethodInfo mi = p_method;
	if (p_virtual) {
		mi.flags |= METHOD_FLAG_VIRTUAL;
	}
	if (p_object_core) {
		mi.flags |= METHOD_FLAG_OBJECT_CORE;
	}

	if (!p_object_core) {
		if (p_arg_names.size() != mi.arguments.size()) {
			WARN_PRINT("Mismatch argument name count for virtual method: " + String(p_class) + "::" + p_method.name);
		} else {
			for (int i = 0; i < p_arg_names.size(); i++) {
				mi.arguments[i].name = p_arg_names[i];
			}
		}
	}

	if (classes[p_class].virtual_methods_map.has(p_method.name)) {
		// overloading not supported
		ERR_FAIL_MSG("Virtual method already bound '" + String(p_class) + "::" + p_method.name + "'.");
	}
	classes[p_class].virtual_methods.push_back(mi);
	classes[p_class].virtual_methods_map[p_method.name] = mi;

#endif
}

void ClassDB::get_virtual_methods(const StringName &p_class, List<MethodInfo> *p_methods, bool p_no_inheritance) {
	ERR_FAIL_COND_MSG(!classes.has(p_class), "Request for nonexistent class '" + p_class + "'.");

#ifdef DEBUG_METHODS_ENABLED

	ClassInfo *type = classes.getptr(p_class);
	ClassInfo *check = type;
	while (check) {
		for (const MethodInfo &E : check->virtual_methods) {
			p_methods->push_back(E);
		}

		if (p_no_inheritance) {
			return;
		}
		check = check->inherits_ptr;
	}

#endif
}

void ClassDB::set_class_enabled(const StringName &p_class, bool p_enable) {
	OBJTYPE_WLOCK;

	ERR_FAIL_COND_MSG(!classes.has(p_class), "Request for nonexistent class '" + p_class + "'.");
	classes[p_class].disabled = !p_enable;
}

bool ClassDB::is_class_enabled(const StringName &p_class) {
	OBJTYPE_RLOCK;

	ClassInfo *ti = classes.getptr(p_class);
	if (!ti || !ti->creation_func) {
		if (compat_classes.has(p_class)) {
			ti = classes.getptr(compat_classes[p_class]);
		}
	}

	ERR_FAIL_NULL_V_MSG(ti, false, "Cannot get class '" + String(p_class) + "'.");
	return !ti->disabled;
}

bool ClassDB::is_class_exposed(const StringName &p_class) {
	OBJTYPE_RLOCK;

	ClassInfo *ti = classes.getptr(p_class);
	ERR_FAIL_NULL_V_MSG(ti, false, "Cannot get class '" + String(p_class) + "'.");
	return ti->exposed;
}

bool ClassDB::is_class_reloadable(const StringName &p_class) {
	OBJTYPE_RLOCK;

	ClassInfo *ti = classes.getptr(p_class);
	ERR_FAIL_NULL_V_MSG(ti, false, "Cannot get class '" + String(p_class) + "'.");
	return ti->reloadable;
}

void ClassDB::add_resource_base_extension(const StringName &p_extension, const StringName &p_class) {
	if (resource_base_extensions.has(p_extension)) {
		return;
	}

	resource_base_extensions[p_extension] = p_class;
}

void ClassDB::get_resource_base_extensions(List<String> *p_extensions) {
	for (const KeyValue<StringName, StringName> &E : resource_base_extensions) {
		p_extensions->push_back(E.key);
	}
}

bool ClassDB::is_resource_extension(const StringName &p_extension) {
	return resource_base_extensions.has(p_extension);
}

void ClassDB::get_extensions_for_type(const StringName &p_class, List<String> *p_extensions) {
	for (const KeyValue<StringName, StringName> &E : resource_base_extensions) {
		if (is_parent_class(p_class, E.value) || is_parent_class(E.value, p_class)) {
			p_extensions->push_back(E.key);
		}
	}
}

HashMap<StringName, HashMap<StringName, Variant>> ClassDB::default_values;
HashSet<StringName> ClassDB::default_values_cached;

Variant ClassDB::class_get_default_property_value(const StringName &p_class, const StringName &p_property, bool *r_valid) {
	if (!default_values_cached.has(p_class)) {
		if (!default_values.has(p_class)) {
			default_values[p_class] = HashMap<StringName, Variant>();
		}

		Object *c = nullptr;
		bool cleanup_c = false;

		if (Engine::get_singleton()->has_singleton(p_class)) {
			c = Engine::get_singleton()->get_singleton_object(p_class);
			cleanup_c = false;
		} else if (ClassDB::can_instantiate(p_class) && !ClassDB::is_virtual(p_class)) { // Keep this condition in sync with doc_tools.cpp get_documentation_default_value.
			c = ClassDB::instantiate(p_class);
			cleanup_c = true;
		}

		if (c) {
			List<PropertyInfo> plist;
			c->get_property_list(&plist);
			for (const PropertyInfo &E : plist) {
				if (E.usage & (PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_EDITOR)) {
					if (!default_values[p_class].has(E.name)) {
						Variant v = c->get(E.name);
						default_values[p_class][E.name] = v;
					}
				}
			}

			if (cleanup_c) {
				memdelete(c);
			}
		}

		default_values_cached.insert(p_class);
	}

	if (!default_values.has(p_class)) {
		if (r_valid != nullptr) {
			*r_valid = false;
		}
		return Variant();
	}

	if (!default_values[p_class].has(p_property)) {
		if (r_valid != nullptr) {
			*r_valid = false;
		}
		return Variant();
	}

	if (r_valid != nullptr) {
		*r_valid = true;
	}

	Variant var = default_values[p_class][p_property];

#ifdef DEBUG_ENABLED
	// Some properties may have an instantiated Object as default value,
	// (like Path2D's `curve` used to have), but that's not a good practice.
	// Instead, those properties should use PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT
	// to be auto-instantiated when created in the editor with the following method:
	// EditorNode::get_editor_data().instantiate_object_properties(obj);
	if (var.get_type() == Variant::OBJECT) {
		Object *obj = var.get_validated_object();
		if (obj) {
			WARN_PRINT(vformat("Instantiated %s used as default value for %s's \"%s\" property.", obj->get_class(), p_class, p_property));
		}
	}
#endif

	return var;
}

void ClassDB::register_extension_class(ObjectGDExtension *p_extension) {
	GLOBAL_LOCK_FUNCTION;

	ERR_FAIL_COND_MSG(classes.has(p_extension->class_name), "Class already registered: " + String(p_extension->class_name));
	ERR_FAIL_COND_MSG(!classes.has(p_extension->parent_class_name), "Parent class name for extension class not found: " + String(p_extension->parent_class_name));

	ClassInfo *parent = classes.getptr(p_extension->parent_class_name);

	ClassInfo c;
	c.api = p_extension->editor_class ? API_EDITOR_EXTENSION : API_EXTENSION;
	c.gdextension = p_extension;
	c.name = p_extension->class_name;
	c.is_virtual = p_extension->is_virtual;
	if (!p_extension->is_abstract) {
		// Find the closest ancestor which is either non-abstract or native (or both).
		ClassInfo *concrete_ancestor = parent;
		while (concrete_ancestor->creation_func == nullptr &&
				concrete_ancestor->inherits_ptr != nullptr &&
				concrete_ancestor->gdextension != nullptr) {
			concrete_ancestor = concrete_ancestor->inherits_ptr;
		}
		ERR_FAIL_NULL_MSG(concrete_ancestor->creation_func, "Extension class " + String(p_extension->class_name) + " cannot extend native abstract class " + String(concrete_ancestor->name));
		c.creation_func = concrete_ancestor->creation_func;
	}
	c.inherits = parent->name;
	c.class_ptr = parent->class_ptr;
	c.inherits_ptr = parent;
	c.exposed = p_extension->is_exposed;
	if (c.exposed) {
		// The parent classes should be exposed if it has an exposed child class.
		while (parent && !parent->exposed) {
			parent->exposed = true;
			parent = classes.getptr(parent->name);
		}
	}
	c.reloadable = p_extension->reloadable;

	classes[p_extension->class_name] = c;
}

void ClassDB::unregister_extension_class(const StringName &p_class, bool p_free_method_binds) {
	ClassInfo *c = classes.getptr(p_class);
	ERR_FAIL_NULL_MSG(c, "Class '" + String(p_class) + "' does not exist.");
	if (p_free_method_binds) {
		for (KeyValue<StringName, MethodBind *> &F : c->method_map) {
			memdelete(F.value);
		}
	}
	classes.erase(p_class);
}

HashMap<StringName, ClassDB::NativeStruct> ClassDB::native_structs;
void ClassDB::register_native_struct(const StringName &p_name, const String &p_code, uint64_t p_current_size) {
	NativeStruct ns;
	ns.ccode = p_code;
	ns.struct_size = p_current_size;
	native_structs[p_name] = ns;
}

void ClassDB::get_native_struct_list(List<StringName> *r_names) {
	for (const KeyValue<StringName, NativeStruct> &E : native_structs) {
		r_names->push_back(E.key);
	}
}

String ClassDB::get_native_struct_code(const StringName &p_name) {
	ERR_FAIL_COND_V(!native_structs.has(p_name), String());
	return native_structs[p_name].ccode;
}

uint64_t ClassDB::get_native_struct_size(const StringName &p_name) {
	ERR_FAIL_COND_V(!native_structs.has(p_name), 0);
	return native_structs[p_name].struct_size;
}

RWLock ClassDB::lock;

void ClassDB::cleanup_defaults() {
	default_values.clear();
	default_values_cached.clear();
}

void ClassDB::cleanup() {
	//OBJTYPE_LOCK; hah not here

	for (KeyValue<StringName, ClassInfo> &E : classes) {
		ClassInfo &ti = E.value;

		for (KeyValue<StringName, MethodBind *> &F : ti.method_map) {
			memdelete(F.value);
		}
		for (KeyValue<StringName, LocalVector<MethodBind *>> &F : ti.method_map_compatibility) {
			for (uint32_t i = 0; i < F.value.size(); i++) {
				memdelete(F.value[i]);
			}
		}
	}

	classes.clear();
	resource_base_extensions.clear();
	compat_classes.clear();
	native_structs.clear();
}

//
