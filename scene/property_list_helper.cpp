/**************************************************************************/
/*  property_list_helper.cpp                                              */
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

#include "property_list_helper.h"

Vector<PropertyListHelper *> PropertyListHelper::base_helpers; // static

void PropertyListHelper::clear_base_helpers() { // static
	for (PropertyListHelper *helper : base_helpers) {
		helper->clear();
	}
	base_helpers.clear();
}

void PropertyListHelper::register_base_helper(PropertyListHelper *p_helper) { // static
	base_helpers.push_back(p_helper);
}

const PropertyListHelper::Property *PropertyListHelper::_get_property(const String &p_property, int *r_index, bool p_allow_oob) const {
	const Vector<String> components = p_property.rsplit("/", true, 1);
	if (components.size() < 2 || !components[0].begins_with(prefix)) {
		return nullptr;
	}

	const String index_string = components[0].trim_prefix(prefix);
	if (!index_string.is_valid_int()) {
		return nullptr;
	}

	int index = index_string.to_int();
	if (index < 0 || (!p_allow_oob && index >= _call_array_length_getter())) {
		return nullptr;
	}

	*r_index = index;
	return property_list.getptr(components[1]);
}

void PropertyListHelper::_call_setter(const MethodBind *p_setter, int p_index, const Variant &p_value) const {
	DEV_ASSERT(p_setter);
	Variant args[] = { p_index, p_value };
	const Variant *argptrs[] = { &args[0], &args[1] };
	Callable::CallError ce;
	p_setter->call(object, argptrs, 2, ce);
}

Variant PropertyListHelper::_call_getter(const Property *p_property, int p_index) const {
	if (!p_property->getter) {
		return object->get(prefix + itos(p_index) + "/" + p_property->info.name);
	}

	Callable::CallError ce;
	Variant args[] = { p_index };
	const Variant *argptrs[] = { &args[0] };
	return p_property->getter->call(object, argptrs, 1, ce);
}

int PropertyListHelper::_call_array_length_getter() const {
	Callable::CallError ce;
	return array_length_getter->call(object, nullptr, 0, ce);
}

void PropertyListHelper::set_prefix(const String &p_prefix) {
	prefix = p_prefix;
}

void PropertyListHelper::register_property(const PropertyInfo &p_info, const Variant &p_default) {
	Property property;
	property.info = p_info;
	property.default_value = p_default;

	property_list[p_info.name] = property;
}

bool PropertyListHelper::is_initialized() const {
	return !property_list.is_empty();
}

void PropertyListHelper::setup_for_instance(const PropertyListHelper &p_base, Object *p_object) {
	DEV_ASSERT(!p_base.prefix.is_empty());
	DEV_ASSERT(p_base.array_length_getter != nullptr);
	DEV_ASSERT(!p_base.property_list.is_empty());
	DEV_ASSERT(p_object != nullptr);

	prefix = p_base.prefix;
	array_length_getter = p_base.array_length_getter;
	property_list = p_base.property_list;
	object = p_object;
}

bool PropertyListHelper::is_property_valid(const String &p_property, int *r_index) const {
	const Vector<String> components = p_property.rsplit("/", true, 1);
	if (components.size() < 2 || !components[0].begins_with(prefix)) {
		return false;
	}

	{
		const String index_string = components[0].trim_prefix(prefix);
		if (!index_string.is_valid_int()) {
			return false;
		}

		if (r_index) {
			*r_index = index_string.to_int();
		}
	}

	return property_list.has(components[1]);
}

void PropertyListHelper::get_property_list(List<PropertyInfo> *p_list) const {
	const int property_count = _call_array_length_getter();
	for (int i = 0; i < property_count; i++) {
		for (const KeyValue<String, Property> &E : property_list) {
			const Property &property = E.value;

			PropertyInfo info = property.info;
			if (!(info.usage & PROPERTY_USAGE_STORE_IF_NULL) && _call_getter(&property, i) == property.default_value) {
				info.usage &= (~PROPERTY_USAGE_STORAGE);
			}

			info.name = vformat("%s%d/%s", prefix, i, info.name);
			p_list->push_back(info);
		}
	}
}

bool PropertyListHelper::property_get_value(const String &p_property, Variant &r_ret) const {
	int index;
	const Property *property = _get_property(p_property, &index);

	if (property) {
		r_ret = _call_getter(property, index);
		return true;
	}
	return false;
}

bool PropertyListHelper::property_set_value(const String &p_property, const Variant &p_value) const {
	int index;
	const Property *property = _get_property(p_property, &index, allow_oob_assign);

	if (property) {
		_call_setter(property->setter, index, p_value);
		return true;
	}
	return false;
}

bool PropertyListHelper::property_can_revert(const String &p_property) const {
	return is_property_valid(p_property);
}

bool PropertyListHelper::property_get_revert(const String &p_property, Variant &r_value) const {
	int index;
	const Property *property = _get_property(p_property, &index);

	if (property) {
		r_value = property->default_value;
		return true;
	}
	return false;
}

void PropertyListHelper::clear() {
	if (is_initialized()) {
		memdelete(array_length_getter);

		for (const KeyValue<String, Property> &E : property_list) {
			if (E.value.setter) {
				memdelete(E.value.setter);
				memdelete(E.value.getter);
			}
		}
		property_list.clear();
	}
}
