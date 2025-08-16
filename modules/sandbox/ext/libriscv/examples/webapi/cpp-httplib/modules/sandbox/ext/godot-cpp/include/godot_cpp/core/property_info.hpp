/**************************************************************************/
/*  property_info.hpp                                                     */
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

#include <godot_cpp/classes/global_constants.hpp>

#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/godot.hpp>

#include <gdextension_interface.h>

namespace godot {

struct PropertyInfo {
	Variant::Type type = Variant::NIL;
	StringName name;
	StringName class_name;
	uint32_t hint = PROPERTY_HINT_NONE;
	String hint_string;
	uint32_t usage = PROPERTY_USAGE_DEFAULT;

	PropertyInfo() = default;

	PropertyInfo(Variant::Type p_type, const StringName &p_name, PropertyHint p_hint = PROPERTY_HINT_NONE, const String &p_hint_string = "", uint32_t p_usage = PROPERTY_USAGE_DEFAULT, const StringName &p_class_name = "") :
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

	PropertyInfo(GDExtensionVariantType p_type, const StringName &p_name, PropertyHint p_hint = PROPERTY_HINT_NONE, const String &p_hint_string = "", uint32_t p_usage = PROPERTY_USAGE_DEFAULT, const StringName &p_class_name = "") :
			PropertyInfo((Variant::Type)p_type, p_name, p_hint, p_hint_string, p_usage, p_class_name) {}

	PropertyInfo(const GDExtensionPropertyInfo *p_info) :
			PropertyInfo(p_info->type, *reinterpret_cast<StringName *>(p_info->name), (PropertyHint)p_info->hint, *reinterpret_cast<String *>(p_info->hint_string), p_info->usage, *reinterpret_cast<StringName *>(p_info->class_name)) {}

	operator Dictionary() const {
		Dictionary dict;
		dict["name"] = name;
		dict["class_name"] = class_name;
		dict["type"] = type;
		dict["hint"] = hint;
		dict["hint_string"] = hint_string;
		dict["usage"] = usage;
		return dict;
	}

	static PropertyInfo from_dict(const Dictionary &p_dict) {
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

	void _update(GDExtensionPropertyInfo *p_info) {
		p_info->type = (GDExtensionVariantType)type;
		*(reinterpret_cast<StringName *>(p_info->name)) = name;
		p_info->hint = hint;
		*(reinterpret_cast<String *>(p_info->hint_string)) = hint_string;
		p_info->usage = usage;
		*(reinterpret_cast<StringName *>(p_info->class_name)) = class_name;
	}

	GDExtensionPropertyInfo _to_gdextension() const {
		return {
			(GDExtensionVariantType)type,
			name._native_ptr(),
			class_name._native_ptr(),
			hint,
			hint_string._native_ptr(),
			usage,
		};
	}
};

} // namespace godot
