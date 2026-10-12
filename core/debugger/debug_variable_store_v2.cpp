/**************************************************************************/
/*  debug_variable_store_v2.cpp                                           */
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

#include "debug_variable_store_v2.h"

#include "core/object/object.h"

uint64_t DebugVariableStoreV2::_store_value(const Variant &p_value) {
	HandleEntry entry;
	entry.kind = HANDLE_VALUE;
	entry.stop_id = active_stop_id;
	entry.value = p_value;
	const uint64_t handle = next_handle++;
	handles[handle] = entry;
	return handle;
}

int DebugVariableStoreV2::_child_count(const Variant &p_value) const {
	switch (p_value.get_type()) {
		case Variant::ARRAY:
			return Array(p_value).size();
		case Variant::DICTIONARY:
			return Dictionary(p_value).size();
		case Variant::STRING:
			return ((String)p_value).length() > preview_length ? (((String)p_value).length() + 1023) / 1024 : 0;
		case Variant::OBJECT: {
			Object *object = p_value;
			if (!object) {
				return 0;
			}
			List<PropertyInfo> properties;
			object->get_property_list(&properties, true);
			int count = 0;
			for (List<PropertyInfo>::Element *element = properties.front(); element; element = element->next()) {
				if (element->get().usage & PROPERTY_USAGE_EDITOR) {
					count++;
				}
			}
			return count;
		}
		default:
			return 0;
	}
}

String DebugVariableStoreV2::_preview(const Variant &p_value) const {
	switch (p_value.get_type()) {
		case Variant::NIL:
			return "null";
		case Variant::BOOL:
			return (bool)p_value ? "true" : "false";
		case Variant::INT:
			return String::num_int64((int64_t)p_value);
		case Variant::FLOAT:
			return String::num((double)p_value);
		case Variant::STRING: {
			const String text = p_value;
			return text.length() > preview_length ? text.substr(0, preview_length) + "…" : text;
		}
		case Variant::ARRAY:
			return "Array[" + String::num_int64(Array(p_value).size()) + "]";
		case Variant::DICTIONARY:
			return "Dictionary[" + String::num_int64(Dictionary(p_value).size()) + "]";
		case Variant::OBJECT: {
			Object *object = p_value;
			return object ? "<" + object->get_class() + "#" + String::num_int64(object->get_instance_id()) + ">" : "<freed object>";
		}
		default:
			return String(p_value);
	}
}

Array DebugVariableStoreV2::_describe(const String &p_name, const Variant &p_value) {
	const int children = _child_count(p_value);
	uint64_t handle = 0;
	if (children > 0) {
		handle = _store_value(p_value);
	}

	Array descriptor;
	descriptor.push_back(p_name);
	descriptor.push_back(Variant::get_type_name(p_value.get_type()));
	descriptor.push_back(_preview(p_value));
	descriptor.push_back((int64_t)handle);
	descriptor.push_back(p_value.get_type() == Variant::ARRAY || p_value.get_type() == Variant::STRING ? children : 0);
	descriptor.push_back(p_value.get_type() == Variant::DICTIONARY || p_value.get_type() == Variant::OBJECT ? children : 0);
	return descriptor;
}

void DebugVariableStoreV2::start_stop(uint64_t p_stop_id, int p_page_size) {
	handles.clear();
	active_stop_id = p_stop_id;
	page_size = MAX(1, p_page_size);
	next_handle = (p_stop_id << 24) | 1;
	if (next_handle == 0 || next_handle > INT64_MAX) {
		next_handle = 1;
	}
}

void DebugVariableStoreV2::invalidate() {
	handles.clear();
	active_stop_id = 0;
}

uint64_t DebugVariableStoreV2::create_scope_handle(int p_frame, ScopeKind p_scope) {
	HandleEntry entry;
	entry.kind = HANDLE_SCOPE;
	entry.stop_id = active_stop_id;
	entry.frame = p_frame;
	entry.scope = p_scope;
	const uint64_t handle = next_handle++;
	handles[handle] = entry;
	return handle;
}

bool DebugVariableStoreV2::get_scope(uint64_t p_handle, int &r_frame, ScopeKind &r_scope) const {
	const RBMap<uint64_t, HandleEntry>::Element *element = handles.find(p_handle);
	if (!element || element->get().stop_id != active_stop_id || element->get().kind != HANDLE_SCOPE) {
		return false;
	}
	r_frame = element->get().frame;
	r_scope = element->get().scope;
	return true;
}

Error DebugVariableStoreV2::describe_named(const List<String> &p_names, const List<Variant> &p_values, int p_start, int p_count, Array &r_items, int &r_total, bool &r_has_more) {
	r_total = MIN(p_names.size(), p_values.size());
	const int start = CLAMP(p_start, 0, r_total);
	const int count = CLAMP(p_count <= 0 ? page_size : p_count, 1, page_size);
	const int end = MIN(start + count, r_total);

	const List<String>::Element *name = p_names.front();
	const List<Variant>::Element *value = p_values.front();
	for (int index = 0; name && value; index++, name = name->next(), value = value->next()) {
		if (index >= start && index < end) {
			r_items.push_back(_describe(name->get(), value->get()));
		}
		if (index >= end) {
			break;
		}
	}
	r_has_more = end < r_total;
	return OK;
}

Error DebugVariableStoreV2::describe_handle(uint64_t p_handle, int p_start, int p_count, Array &r_items, int &r_total, bool &r_has_more) {
	RBMap<uint64_t, HandleEntry>::Element *element = handles.find(p_handle);
	if (!element || element->get().stop_id != active_stop_id || element->get().kind != HANDLE_VALUE) {
		return ERR_DOES_NOT_EXIST;
	}

	const Variant value = element->get().value;
	r_total = _child_count(value);
	const int start = CLAMP(p_start, 0, r_total);
	const int count = CLAMP(p_count <= 0 ? page_size : p_count, 1, page_size);
	const int end = MIN(start + count, r_total);

	if (value.get_type() == Variant::ARRAY) {
		const Array array = value;
		for (int index = start; index < end; index++) {
			r_items.push_back(_describe("[" + String::num_int64(index) + "]", array[index]));
		}
	} else if (value.get_type() == Variant::DICTIONARY) {
		const Dictionary dictionary = value;
		const LocalVector<Variant> keys = dictionary.get_key_list();
		for (int index = 0; index < (int)keys.size(); index++) {
			if (index >= start && index < end) {
				r_items.push_back(_describe(_preview(keys[index]), dictionary[keys[index]]));
			}
			if (index >= end) {
				break;
			}
		}
	} else if (value.get_type() == Variant::STRING) {
		const String text = value;
		for (int index = start; index < end; index++) {
			r_items.push_back(_describe("[" + String::num_int64(index) + "]", text.substr(index * 1024, 1024)));
		}
	} else if (value.get_type() == Variant::OBJECT) {
		Object *object = value;
		if (!object) {
			return ERR_DOES_NOT_EXIST;
		}
		List<PropertyInfo> properties;
		object->get_property_list(&properties, true);
		int index = 0;
		for (List<PropertyInfo>::Element *property = properties.front(); property; property = property->next()) {
			if (!(property->get().usage & PROPERTY_USAGE_EDITOR)) {
				continue;
			}
			if (index >= start && index < end) {
				r_items.push_back(_describe(property->get().name, object->get(property->get().name)));
			}
			index++;
			if (index >= end) {
				break;
			}
		}
	} else {
		return ERR_INVALID_PARAMETER;
	}

	r_has_more = end < r_total;
	return OK;
}
