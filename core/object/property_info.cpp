/**************************************************************************/
/*  property_info.cpp                                                     */
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

#include "property_info.h"

#include "core/variant/dictionary.h"
#include "core/variant/typed_array.h"

PropertyInfo::operator Dictionary() const {
	Dictionary d;
	d["name"] = name;
	d["class_name"] = class_name;
	d["type"] = type;
	d["hint"] = hint;
	d["hint_string"] = hint_string;
	d["usage"] = usage;
	return d;
}

PropertyInfo PropertyInfo::from_dict(const Dictionary &p_dict) {
	PropertyInfo pi;

	if (p_dict.has("type")) {
		pi.type = Variant::Type(int(p_dict["type"]));
	}

	if (p_dict.has("name")) {
		pi.name = p_dict["name"];
	}

	if (p_dict.has("class_name")) {
		pi.class_name = p_dict["class_name"];
	}

	if (p_dict.has("hint")) {
		pi.hint = PropertyHint(int(p_dict["hint"]));
	}

	if (p_dict.has("hint_string")) {
		pi.hint_string = p_dict["hint_string"];
	}

	if (p_dict.has("usage")) {
		pi.usage = p_dict["usage"];
	}

	return pi;
}

PropertyInfo::PropertyInfo(const Variant::Type p_type, const String &p_name, const PropertyHint p_hint,
		const String &p_hint_string, const uint32_t p_usage, const StringName &p_class_name) :
		type(p_type),
		name(p_name),
		hint(p_hint),
		hint_string(p_hint_string),
		usage(p_usage) {
	if (hint == PROPERTY_HINT_RESOURCE_TYPE) {
		class_name = hint_string;
	} else {
		class_name = p_class_name;
	}
}

PropertyInfo::PropertyInfo(const Variant::Type p_type, const char *p_name, const PropertyHint p_hint,
		const char *p_hint_string, const uint32_t p_usage, const char *p_class_name) :
		type(p_type),
		name(p_name),
		hint(p_hint),
		hint_string(p_hint_string),
		usage(p_usage) {
	if (hint == PROPERTY_HINT_RESOURCE_TYPE) {
		class_name = hint_string;
	} else if (p_class_name) {
		class_name = StringName(p_class_name);
	}
}

TypedArray<Dictionary> convert_property_list(const List<PropertyInfo> *p_list) {
	TypedArray<Dictionary> va;
	for (const List<PropertyInfo>::Element *E = p_list->front(); E; E = E->next()) {
		va.push_back(Dictionary(E->get()));
	}

	return va;
}

TypedArray<Dictionary> convert_property_list(const Vector<PropertyInfo> &p_vector) {
	TypedArray<Dictionary> va;
	for (const PropertyInfo &E : p_vector) {
		va.push_back(Dictionary(E));
	}

	return va;
}
