/**************************************************************************/
/*  actor.cpp                                                             */
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


#include "actor.h"

#include "core/object/class_db.h"
#include "core/object/script_language.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"


bool Actor::has_component(StringName component_class) const {
	return _component_resources.has(component_class);
}


Ref<Component> Actor::get_component(StringName component_class) const {
	if (!has_component(component_class)) {
		return nullptr;
	}

	Ref<Component> result;

	result = _component_resources.get(component_class);

	return result;
}


void Actor::set_component(Ref<Component> value) {
	ERR_FAIL_COND_MSG(!value.is_valid(), vformat("Can't add a null component."));

	_component_resources[value->get_component_class()] = value;
	print_line("Success setting component: ", value->get_component_class());

	notify_property_list_changed();
}


void Actor::remove_component(StringName component_class) {
	if (_component_resources.has(component_class)) {
		_component_resources.erase(component_class);
		notify_property_list_changed();
	}
}


void Actor::get_component_list(List<Ref<Component>> *out) const {
	for (const KeyValue<StringName, Ref<Component>> &K : _component_resources) {
		out->push_back(K.value);
	}
}


void Actor::get_component_class_list(List<StringName> *out) const {
	for (const KeyValue<StringName, Ref<Component>> &K : _component_resources) {
		out->push_back(K.key);
	}
}


void Actor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("has_component", "component_class"), &Actor::has_component);
	ClassDB::bind_method(D_METHOD("get_component", "component_class"), &Actor::get_component);
	ClassDB::bind_method(D_METHOD("set_component", "value"), &Actor::set_component);
	ClassDB::bind_method(D_METHOD("remove_component", "component_class"), &Actor::remove_component);
}


void Actor::_get_property_list(List<PropertyInfo> *out) const {
	for (const KeyValue<StringName, Ref<Component>> &k_v: _component_resources) {
		PropertyInfo property_info = PropertyInfo(Variant::OBJECT, "components/" + k_v.key.operator String(), PROPERTY_HINT_RESOURCE_TYPE, "Component", PropertyUsageFlags::PROPERTY_USAGE_DEFAULT | PropertyUsageFlags::PROPERTY_USAGE_READ_ONLY);
		out->push_back(property_info);
	}
}


bool Actor::_get(const StringName &p_property, Variant &r_value) const {
	bool result = false;

	const String COMPONENTS = "components/";
	String p_string = p_property;
	if (p_string.begins_with(COMPONENTS)) {
		String key = p_string.trim_prefix(COMPONENTS);

		result = has_component(key);

		if (result) {
			r_value = get_component(key);
		}
	}

	return result;
}


bool Actor::_set(const StringName &p_property, const Variant &p_value) {
	bool result = false;

	const String COMPONENTS = "components/";
	String p_string = p_property;
	if (p_string.begins_with(COMPONENTS)) {
		set_component(p_value);
		result = true;
	}

	return result;
}
