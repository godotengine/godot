/**************************************************************************/
/*  gdtype.cpp                                                            */
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

#include "gdtype.h"

#include "core/object/method_bind.h"
#include "core/os/memory.h"
#include "core/os/thread.h"

GDType::GDType(const GDType *p_super_type, StringName p_name) :
		super_type(p_super_type), name(std::move(p_name)) {
	name_hierarchy.push_back(name);

	if (super_type) {
		for (const StringName &ancestor_name : super_type->name_hierarchy) {
			name_hierarchy.push_back(ancestor_name);
		}
	}
}

GDType::~GDType() {
	for (const KeyValue<StringName, Property> &kv : self_property_map) {
		if (kv.value.type == Property::Type::ENUM) {
			memdelete(const_cast<EnumInfo *>(kv.value.payload.enum_info));
		} else if (kv.value.type == Property::Type::SIGNAL) {
			memdelete(const_cast<MethodInfo *>(kv.value.payload.signal));
		}
	}
	for (MethodBind *bind : owned_method_map) {
		memdelete(bind);
	}
	for (const KeyValue<StringName, LocalVector<MethodBind *>> &kv : self_compatibility_method_map) {
		for (MethodBind *bind : kv.value) {
			memdelete(bind);
		}
	}
	for (const PropertyInfo *property : ordered_self_properties) {
		memdelete(const_cast<PropertyInfo *>(property));
	}
}

void GDType::initialize() {
	ERR_FAIL_COND(init_state != InitState::UNINITIALIZED);

	if (super_type) {
		// Now that a subtype is registered, the supertype cannot change anymore.
		// Otherwise, our caches would become invalid.
		// This shouldn't be a problem, since classes should register all their
		// parts in _bind_methods, which is called on registration.
		super_type->init_state = InitState::FINALIZED;

		property_map = super_type->property_map;
	}

	init_state = InitState::MUTABLE;
}

void GDType::bind_integer_constant(const StringName &p_enum, const StringName &p_name, int64_t p_constant, bool p_is_bitfield) {
	ERR_FAIL_COND(!Thread::is_main_thread());
	ERR_FAIL_COND(init_state != InitState::MUTABLE);
	ERR_FAIL_COND_MSG(property_map.has(p_name), vformat("Object '%s' already has property '%s'.", get_name(), p_name));

	EnumInfo *enum_info = nullptr;

	String enum_name = p_enum;
	if (!enum_name.is_empty()) {
		if (enum_name.contains_char('.')) {
			enum_name = enum_name.get_slicec('.', 1);
		}

		const Property *enum_property = self_property_map.getptr(enum_name);
		ERR_FAIL_COND_MSG(!enum_property && property_map.has(enum_name), vformat("Cannot bind integer constant '%s' to enum '%s' from class '%s' because the enum belongs to a parent class.", p_name, enum_name, get_name()));
		ERR_FAIL_COND_MSG(enum_property && enum_property->type != Property::Type::ENUM, vformat("Object '%s' already has property '%s'.", get_name(), enum_name));

		if (enum_property) {
			enum_info = const_cast<EnumInfo *>(enum_property->payload.enum_info);
			enum_info->values.insert(p_name, p_constant);
			enum_info->is_bitfield = p_is_bitfield;
		} else {
			enum_info = memnew(EnumInfo);
			enum_info->name = enum_name;
			enum_info->is_bitfield = p_is_bitfield;
			enum_info->values.insert(p_name, p_constant);
			self_property_map.insert(enum_name, Property::create_enum(enum_info));
			property_map.insert(enum_name, Property::create_enum(enum_info));
		}
	}

	Property::IntegerConstant entry{ p_constant, enum_info };
	property_map.insert(p_name, Property::create_integer_constant(entry));
	self_property_map.insert(p_name, Property::create_integer_constant(entry));
}

const GDType::EnumInfo *GDType::get_integer_constant_enum(const StringName &p_name, bool p_no_inheritance) const {
	const Property *property = get_property_map(p_no_inheritance).getptr(p_name);
	if (!property || property->type != Property::Type::INTEGER_CONSTANT) {
		return nullptr;
	}
	return property->payload.integer_constant.enum_info;
}

void GDType::add_signal(MethodInfo p_signal) {
	ERR_FAIL_COND(!Thread::is_main_thread());
	ERR_FAIL_COND(init_state != InitState::MUTABLE);

	const StringName signal_name(p_signal.name);
	ERR_FAIL_COND_MSG(property_map.has(signal_name), vformat("Object '%s' already has property '%s'.", get_name(), signal_name));

	const MethodInfo *ptr = memnew(MethodInfo(std::move(p_signal)));

	property_map.insert(ptr->name, Property::create_signal(ptr));
	self_property_map.insert(ptr->name, Property::create_signal(ptr));
}

bool GDType::bind_method(MethodBind *p_method, bool p_take_ownership) {
	ERR_FAIL_COND_V(!Thread::is_main_thread(), false);
	ERR_FAIL_COND_V(init_state != InitState::MUTABLE, false);

	if (property_map.has(p_method->get_name())) {
		if (p_take_ownership) {
			memdelete(p_method);
		}
		ERR_FAIL_V_MSG(false, vformat("Object '%s' already has property '%s'.", get_name(), p_method->get_name()));
	}

	if (p_take_ownership) {
		owned_method_map.push_back(p_method);
	}
	property_map.insert(p_method->get_name(), Property::create_method(p_method));
	self_property_map.insert(p_method->get_name(), Property::create_method(p_method));

	return true;
}

void GDType::set_method_flags(const StringName &p_method, int p_flags) {
	ERR_FAIL_COND(!Thread::is_main_thread());
	ERR_FAIL_COND(init_state != InitState::MUTABLE);

	const Property *property = self_property_map.getptr(p_method);
	ERR_FAIL_NULL(property);
	ERR_FAIL_COND(property->type != Property::Type::METHOD);

	const_cast<MethodBind *>(property->payload.method)->set_hint_flags(p_flags);
}

bool GDType::bind_compatibility_method(MethodBind *p_method) {
	ERR_FAIL_COND_V(!Thread::is_main_thread(), false);
	ERR_FAIL_COND_V(init_state != InitState::MUTABLE, false);

	if (!self_compatibility_method_map.has(p_method->get_name())) {
		self_compatibility_method_map.insert(p_method->get_name(), LocalVector<MethodBind *>());
	}
	self_compatibility_method_map[p_method->get_name()].push_back(p_method);
	return true;
}

void GDType::add_property(const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter,
		int p_index) {
	ERR_FAIL_COND(!Thread::is_main_thread());
	ERR_FAIL_COND(init_state != InitState::MUTABLE);

	ERR_FAIL_COND_MSG(property_map.has(p_pinfo.name), vformat("Object '%s' already has property '%s'.", get_name(), p_pinfo.name));

	const MethodBind *mb_set = nullptr;
	if (p_setter) {
		const Property *set_prop = property_map.getptr(p_setter);

		ERR_FAIL_COND_MSG(!set_prop || set_prop->type != Property::Type::METHOD, vformat("Invalid setter '%s::%s' for property '%s'.", get_name(), p_setter, p_pinfo.name));

		mb_set = set_prop->payload.method;

		int exp_args = 1 + (p_index >= 0 ? 1 : 0);
		ERR_FAIL_COND_MSG(mb_set->get_argument_count() != exp_args, vformat("Invalid function for setter '%s::%s' for property '%s'.", get_name(), p_setter, p_pinfo.name));
	}

	const MethodBind *mb_get = nullptr;
	if (p_getter) {
		const Property *get_prop = property_map.getptr(p_getter);

		ERR_FAIL_COND_MSG(!get_prop || get_prop->type != Property::Type::METHOD, vformat("Invalid getter '%s::%s' for property '%s'.", get_name(), p_getter, p_pinfo.name));

		mb_get = get_prop->payload.method;

		int exp_args = 0 + (p_index >= 0 ? 1 : 0);
		ERR_FAIL_COND_MSG(mb_get->get_argument_count() != exp_args, vformat("Invalid function for getter '%s::%s' for property '%s'.", get_name(), p_getter, p_pinfo.name));
	}

	PropertyInfo *info = memnew(PropertyInfo(p_pinfo));

	Property::SetGet psg;
	psg.property_info = info;
	psg.setter = mb_set;
	psg.getter = mb_get;
	psg.index = p_index;

	property_map.insert(p_pinfo.name, Property::create_setget(psg));
	self_property_map.insert(p_pinfo.name, Property::create_setget(psg));
	ordered_self_properties.push_back(info);
}

void GDType::add_to_ordered_properties(const PropertyInfo &p_pinfo) {
	ERR_FAIL_COND(!Thread::is_main_thread());
	ERR_FAIL_COND(init_state != InitState::MUTABLE);

	PropertyInfo *info = memnew(PropertyInfo(p_pinfo));
	ordered_self_properties.push_back(info);
}
