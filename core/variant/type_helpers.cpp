/**************************************************************************/
/*  type_helpers.cpp                                                      */
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

#include "type_helpers.h"

#include "core/variant/container_type_validate.h"

String VariantTypeHelpers::get_display_name(const Script &p_script) {
	const StringName &global_name = p_script.get_global_name();
	if (global_name.is_empty()) {
		return p_script.get_path();
	}
	return global_name;
}

String VariantTypeHelpers::get_display_name(const ContainerType &p_type) {
	if (p_type.variant_type != Variant::OBJECT) {
		return Variant::get_type_name(p_type.variant_type);
	}
	if (p_type.script.is_valid()) {
		return get_display_name(**p_type.script);
	}
	return p_type.class_name;
}

String VariantTypeHelpers::get_type_display_name(const Variant &p_value) {
	if (p_value.get_type() == Variant::ARRAY) {
		const Array &arr = p_value.to<Array>();
		if (arr.is_typed()) {
			return vformat("Array[%s]", get_display_name(arr.get_element_type()));
		}
	}
	if (p_value.get_type() == Variant::DICTIONARY) {
		const Dictionary &dict = p_value.to<Dictionary>();
		if (dict.is_typed()) {
			return vformat("Dictionary[%s, %s]", get_display_name(dict.get_value_type()), get_display_name(dict.get_key_type()));
		}
	}
	if (p_value.get_type() != Variant::OBJECT) {
		return Variant::get_type_name(p_value.get_type());
	}
	const Object &obj = *p_value.to<Object *>();
	const ScriptInstance *si = obj.get_script_instance();
	if (si != nullptr) {
		const Script *scr = si->get_script();
		if (scr != nullptr) {
			return get_display_name(*scr);
		}
	}
	return obj.get_class();
}
