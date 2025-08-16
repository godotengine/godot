/**************************************************************************/
/*  object.hpp                                                            */
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

#include <godot_cpp/core/defs.hpp>

#include <godot_cpp/core/object_id.hpp>

#include <godot_cpp/core/property_info.hpp>

#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/classes/object.hpp>

#include <godot_cpp/godot.hpp>

#include <gdextension_interface.h>

#include <vector>

#define ADD_SIGNAL(m_signal) ::godot::ClassDB::add_signal(get_class_static(), m_signal)
#define ADD_GROUP(m_name, m_prefix) ::godot::ClassDB::add_property_group(get_class_static(), m_name, m_prefix)
#define ADD_SUBGROUP(m_name, m_prefix) ::godot::ClassDB::add_property_subgroup(get_class_static(), m_name, m_prefix)
#define ADD_PROPERTY(m_property, m_setter, m_getter) ::godot::ClassDB::add_property(get_class_static(), m_property, m_setter, m_getter)
#define ADD_PROPERTYI(m_property, m_setter, m_getter, m_index) ::godot::ClassDB::add_property(get_class_static(), m_property, m_setter, m_getter, m_index)

namespace godot {

namespace internal {

Object *get_object_instance_binding(GodotObject *);

} // namespace internal

struct MethodInfo {
	StringName name;
	PropertyInfo return_val;
	uint32_t flags;
	int id = 0;
	std::vector<PropertyInfo> arguments;
	std::vector<Variant> default_arguments;
	GDExtensionClassMethodArgumentMetadata return_val_metadata;
	std::vector<GDExtensionClassMethodArgumentMetadata> arguments_metadata;

	inline bool operator==(const MethodInfo &p_method) const { return id == p_method.id; }
	inline bool operator<(const MethodInfo &p_method) const { return id == p_method.id ? (name < p_method.name) : (id < p_method.id); }

	operator Dictionary() const;

	static MethodInfo from_dict(const Dictionary &p_dict);

	MethodInfo();
	MethodInfo(StringName p_name);
	template <typename... Args>
	MethodInfo(StringName p_name, const Args &...args);
	MethodInfo(Variant::Type ret);
	MethodInfo(Variant::Type ret, StringName p_name);
	template <typename... Args>
	MethodInfo(Variant::Type ret, StringName p_name, const Args &...args);
	MethodInfo(const PropertyInfo &p_ret, StringName p_name);
	template <typename... Args>
	MethodInfo(const PropertyInfo &p_ret, StringName p_name, const Args &...);
};

template <typename... Args>
MethodInfo::MethodInfo(StringName p_name, const Args &...args) :
		name(p_name), flags(GDEXTENSION_METHOD_FLAG_NORMAL) {
	arguments = { args... };
}

template <typename... Args>
MethodInfo::MethodInfo(Variant::Type ret, StringName p_name, const Args &...args) :
		name(p_name), flags(GDEXTENSION_METHOD_FLAG_NORMAL) {
	return_val.type = ret;
	arguments = { args... };
}

template <typename... Args>
MethodInfo::MethodInfo(const PropertyInfo &p_ret, StringName p_name, const Args &...args) :
		name(p_name), return_val(p_ret), flags(GDEXTENSION_METHOD_FLAG_NORMAL) {
	arguments = { args... };
}

class ObjectDB {
public:
	static Object *get_instance(uint64_t p_object_id) {
		GDExtensionObjectPtr obj = internal::gdextension_interface_object_get_instance_from_id(p_object_id);
		if (obj == nullptr) {
			return nullptr;
		}
		return internal::get_object_instance_binding(obj);
	}
};

template <typename T>
T *Object::cast_to(Object *p_object) {
	if (p_object == nullptr) {
		return nullptr;
	}
	StringName class_name = T::get_class_static();
	GDExtensionObjectPtr casted = internal::gdextension_interface_object_cast_to(p_object->_owner, internal::gdextension_interface_classdb_get_class_tag(class_name._native_ptr()));
	if (casted == nullptr) {
		return nullptr;
	}
	return dynamic_cast<T *>(internal::get_object_instance_binding(casted));
}

template <typename T>
const T *Object::cast_to(const Object *p_object) {
	if (p_object == nullptr) {
		return nullptr;
	}
	StringName class_name = T::get_class_static();
	GDExtensionObjectPtr casted = internal::gdextension_interface_object_cast_to(p_object->_owner, internal::gdextension_interface_classdb_get_class_tag(class_name._native_ptr()));
	if (casted == nullptr) {
		return nullptr;
	}
	return dynamic_cast<const T *>(internal::get_object_instance_binding(casted));
}

} // namespace godot
