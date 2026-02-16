/**************************************************************************/
/*  doc_data.cpp                                                          */
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

#include "doc_data.h"

String DocData::get_default_value_string(const Variant &p_value) {
	const Variant::Type type = p_value.get_type();
	if (type == Variant::ARRAY) {
		return Variant(Array(p_value, 0, StringName(), Variant())).get_construct_string().replace_char('\n', ' ');
	} else if (type == Variant::DICTIONARY) {
		return Variant(Dictionary(p_value, 0, StringName(), Variant(), 0, StringName(), Variant())).get_construct_string().replace_char('\n', ' ');
	} else if (type == Variant::INT) {
		return itos(p_value);
	} else if (type == Variant::FLOAT) {
		// Since some values are 32-bit internally, use 32-bit for all
		// documentation values to avoid garbage digits at the end.
		const String s = String::num_scientific((float)p_value);
		// Use float literals for floats in the documentation for clarity.
		if (s != "inf" && s != "-inf" && s != "nan") {
			if (!s.contains_char('.') && !s.contains_char('e')) {
				return s + ".0";
			}
		}
		return s;
	} else {
		return p_value.get_construct_string().replace_char('\n', ' ');
	}
}

void DocData::doctype_from_propinfo(const PropertyInfo &p_info, String &r_type, String &r_enum, bool &r_is_bitfield, bool p_is_return) {
	r_type = String();
	r_enum = String();
	r_is_bitfield = false;

	if (p_info.type == Variant::INT) {
		if (p_info.hint == PROPERTY_HINT_INT_IS_POINTER) {
			r_type = p_info.hint_string.is_empty() ? "void*" : p_info.hint_string + "*";
		} else {
			r_type = "int";
			if (p_info.usage & (PROPERTY_USAGE_CLASS_IS_ENUM | PROPERTY_USAGE_CLASS_IS_BITFIELD)) {
				r_enum = p_info.class_name;
				if (r_enum.begins_with("_")) { // Proxy class.
					r_enum = r_enum.substr(1);
				}
				r_is_bitfield = p_info.usage & PROPERTY_USAGE_CLASS_IS_BITFIELD;
			}
		}
	} else if (p_info.type == Variant::ARRAY) {
		if (p_info.hint == PROPERTY_HINT_ARRAY_TYPE && !p_info.hint_string.is_empty()) {
			r_type = p_info.hint_string + "[]";
		} else {
			r_type = "Array";
		}
	} else if (p_info.type == Variant::DICTIONARY) {
		if (p_info.hint == PROPERTY_HINT_DICTIONARY_TYPE && !p_info.hint_string.is_empty()) {
			r_type = "Dictionary[" + p_info.hint_string.replace(";", ", ") + "]";
		} else {
			r_type = "Dictionary";
		}
	} else if (p_info.type == Variant::NIL) {
		if (p_info.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
			r_type = "Variant";
		} else {
			//r_type = p_is_return ? "void" : "null"; // `Variant` constructors do not support `usage`.
			r_type = p_is_return ? "void" : "Variant";
		}
	} else if (p_info.class_name != StringName()) {
		r_type = p_info.class_name;
	} else {
		r_type = Variant::get_type_name(p_info.type);
	}
}

void DocData::method_doc_from_methodinfo(DocData::MethodDoc &p_method, const MethodInfo &p_methodinfo, const String &p_desc) {
	p_method.name = p_methodinfo.name;
	p_method.description = p_desc;

	if (p_methodinfo.flags & METHOD_FLAG_VIRTUAL) {
		p_method.qualifiers = "virtual";
	}

	if (p_methodinfo.flags & METHOD_FLAG_VIRTUAL_REQUIRED) {
		if (!p_method.qualifiers.is_empty()) {
			p_method.qualifiers += " ";
		}
		p_method.qualifiers += "required";
	}

	if (p_methodinfo.flags & METHOD_FLAG_CONST) {
		if (!p_method.qualifiers.is_empty()) {
			p_method.qualifiers += " ";
		}
		p_method.qualifiers += "const";
	}

	if (p_methodinfo.flags & METHOD_FLAG_VARARG) {
		if (!p_method.qualifiers.is_empty()) {
			p_method.qualifiers += " ";
		}
		p_method.qualifiers += "vararg";
	}

	if (p_methodinfo.flags & METHOD_FLAG_STATIC) {
		if (!p_method.qualifiers.is_empty()) {
			p_method.qualifiers += " ";
		}
		p_method.qualifiers += "static";
	}

	return_doc_from_retinfo(p_method, p_methodinfo.return_val);

	for (int64_t i = 0; i < p_methodinfo.arguments.size(); ++i) {
		DocData::ArgumentDoc argument;
		argument_doc_from_arginfo(argument, p_methodinfo.arguments[i]);
		int64_t default_arg_index = i - (p_methodinfo.arguments.size() - p_methodinfo.default_arguments.size());
		if (default_arg_index >= 0) {
			Variant default_arg = p_methodinfo.default_arguments[default_arg_index];
			argument.default_value = get_default_value_string(default_arg);
		}
		p_method.arguments.push_back(argument);
	}
}
