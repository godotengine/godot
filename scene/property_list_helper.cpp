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

void PropertyListHelper::_call_setter(const MethodBind *p_setter, int p_index, const Variant &p_value) const {
	Variant args[] = { p_index, p_value };
	const Variant *argptrs[] = { &args[0], &args[1] };
	Callable::CallError ce;
	p_setter->call(object, argptrs, 2, ce);
}

Variant PropertyListHelper::_call_getter(const MethodBind *p_getter, int p_index) const {
	Callable::CallError ce;
	Variant args[] = { p_index };
	const Variant *argptrs[] = { &args[0] };
	return p_getter->call(object, argptrs, 1, ce);
}

void PropertyListHelper::set_prefix(const String &p_prefix) {
	prefix = p_prefix;
}

void PropertyListHelper::setup_for_instance(const PropertyListHelper &p_base, Object *p_object) {
	prefix = p_base.prefix;
	property_list = p_base.property_list;
	object = p_object;
}

void PropertyListHelper::get_property_list(List<PropertyInfo> *p_list, int p_count) const {
	for (int i = 0; i < p_count; i++) {
		for (const KeyValue<String, Property> &E : property_list) {
			const Property &property = E.value;

			PropertyInfo info = property.info;
			if (_call_getter(property.getter, i) == property.default_value) {
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
		r_ret = _call_getter(property->getter, index);
		return true;
	}
	return false;
}

bool PropertyListHelper::property_set_value(const String &p_property, const Variant &p_value) const {
	int index;
	const Property *property = _get_property(p_property, &index);

	if (property) {
		_call_setter(property->setter, index, p_value);
		return true;
	}
	return false;
}

bool PropertyListHelper::property_can_revert(const String &p_property) const {
	int index;
	return _get_property(p_property, &index) != nullptr;
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

PropertyListHelper::~PropertyListHelper() {
	// No object = it's the main helper. Do a cleanup.
	if (!object) {
		for (const KeyValue<String, Property> &E : property_list) {
			memdelete(E.value.setter);
			memdelete(E.value.getter);
		}
	}
}
