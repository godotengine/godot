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

const PropertyListHelper::Property *PropertyListHelper::_get_property(const String &p_property, int *r_index) const {
	const Vector<String> components = p_property.split("/", true, 2);
	if (components.size() < 2 || !components[0].begins_with(prefix)) {
		return nullptr;
	}

	{
		const String index_string = components[0].trim_prefix(prefix);
		if (!index_string.is_valid_int()) {
			return nullptr;
		}
		*r_index = index_string.to_int();
	}

	return property_list.getptr(components[1]);
}

void PropertyListHelper::_bind_property(const Property &p_property, const Object *p_object) {
	Property property = p_property;
	property.info = p_property.info;
	property.default_value = p_property.default_value;
	property.setter = Callable(p_object, p_property.setter_name);
	property.getter = Callable(p_object, p_property.getter_name);

	property_list[property.info.name] = property;
}

void PropertyListHelper::set_prefix(const String &p_prefix) {
	prefix = p_prefix;
}

void PropertyListHelper::register_property(const PropertyInfo &p_info, const Variant &p_default, const StringName &p_setter, const StringName &p_getter) {
	Property property;
	property.info = p_info;
	property.default_value = p_default;
	property.setter_name = p_setter;
	property.getter_name = p_getter;

	property_list[p_info.name] = property;
}

void PropertyListHelper::setup_for_instance(const PropertyListHelper &p_base, const Object *p_object) {
	prefix = p_base.prefix;
	for (const KeyValue<String, Property> &E : p_base.property_list) {
		_bind_property(E.value, p_object);
	}
}

void PropertyListHelper::get_property_list(List<PropertyInfo> *p_list, int p_count) const {
	for (int i = 0; i < p_count; i++) {
		for (const KeyValue<String, Property> &E : property_list) {
			const Property &property = E.value;

			PropertyInfo info = property.info;
			if (property.getter.call(i) == property.default_value) {
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
		r_ret = property->getter.call(index);
		return true;
	}
	return false;
}

bool PropertyListHelper::property_set_value(const String &p_property, const Variant &p_value) const {
	int index;
	const Property *property = _get_property(p_property, &index);

	if (property) {
		property->setter.call(index, p_value);
		return true;
	}
	return false;
}

bool PropertyListHelper::property_can_revert(const String &p_property) const {
	int index;
	const Property *property = _get_property(p_property, &index);

	if (property) {
		return property->getter.call(index) != property->default_value;
	}
	return false;
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
