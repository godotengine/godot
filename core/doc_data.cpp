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

#include "core/core_constants.h"

String DocData::get_default_value_string(const Variant &p_value, const PropertyInfo &p_info) {
	const Variant::Type type = p_value.get_type();
	if (type == Variant::ARRAY) {
		return Variant(Array(p_value, 0, StringName(), Variant())).get_construct_string().replace_char('\n', ' ');
	} else if (type == Variant::DICTIONARY) {
		return Variant(Dictionary(p_value, 0, StringName(), Variant(), 0, StringName(), Variant())).get_construct_string().replace_char('\n', ' ');
	} else if (type == Variant::INT) {
		String val = itos(p_value);
		if (p_info.usage & (PROPERTY_USAGE_CLASS_IS_ENUM | PROPERTY_USAGE_CLASS_IS_BITFIELD)) {
			String enum_name = p_info.class_name;
			if (enum_name.begins_with("_")) {
				enum_name = enum_name.substr(1);
			}
			String class_name = enum_name.get_slicec('.', 0);
			enum_name = enum_name.get_slicec('.', 1);
			bool is_bitfield = p_info.usage & PROPERTY_USAGE_CLASS_IS_BITFIELD;
			int64_t int_val = p_value;
			HashMap<StringName, int64_t> enum_constants;
			if (ClassDB::has_enum(class_name, enum_name)) {
				List<StringName> constant_list;
				ClassDB::get_enum_constants(class_name, enum_name, &constant_list);
				for (const StringName &constant : constant_list) {
					enum_constants[constant] = ClassDB::get_integer_constant(class_name, constant);
				}
			}
			struct ConstantSort {
				constexpr bool operator()(const KeyValue<StringName, int64_t> &p_lhs, const KeyValue<StringName, int64_t> &p_rhs) const {
					return p_lhs.value >= p_rhs.value;
				}
			};
			enum_constants.sort_custom<ConstantSort>();
			if (enum_name.is_empty() && CoreConstants::is_global_enum(class_name)) {
				CoreConstants::get_enum_values(class_name, &enum_constants);
			}
			if (!enum_constants.is_empty()) {
				if (is_bitfield) {
					val = String();
					int64_t rem_val = int_val;
					for (const KeyValue<StringName, int64_t> &constant : enum_constants) {
						if (constant.value != 0 && (rem_val & constant.value) == constant.value) {
							if (!val.is_empty()) {
								val += " | ";
							}
							val += vformat("%s", constant.key);
							rem_val &= ~constant.value;
						}
					}
					if (rem_val != 0) {
						if (!val.is_empty()) {
							val += " | ";
						}
						val += itos(rem_val);
						WARN_PRINT(vformat("Default value of '%s.%s' is not a valid combination of enum constants (%d).", p_info.class_name, p_info.name, rem_val));
					}
					if (val.is_empty()) {
						val = itos(p_value);
					}
				} else {
					for (const KeyValue<StringName, int64_t> &constant : enum_constants) {
						if (int_val == constant.value) {
							val = vformat("%s", constant.key);
							break;
						}
					}
				}
			}
		}
		return val;
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

void DocData::return_doc_from_retinfo(DocData::MethodDoc &p_method, const PropertyInfo &p_retinfo) {
	if (p_retinfo.type == Variant::INT && p_retinfo.hint == PROPERTY_HINT_INT_IS_POINTER) {
		p_method.return_type = p_retinfo.hint_string;
		if (p_method.return_type.is_empty()) {
			p_method.return_type = "void*";
		} else {
			p_method.return_type += "*";
		}
	} else if (p_retinfo.type == Variant::INT && p_retinfo.usage & (PROPERTY_USAGE_CLASS_IS_ENUM | PROPERTY_USAGE_CLASS_IS_BITFIELD)) {
		p_method.return_enum = p_retinfo.class_name;
		if (p_method.return_enum.begins_with("_")) { //proxy class
			p_method.return_enum = p_method.return_enum.substr(1);
		}
		p_method.return_is_bitfield = p_retinfo.usage & PROPERTY_USAGE_CLASS_IS_BITFIELD;
		p_method.return_type = "int";
	} else if (p_retinfo.class_name != StringName()) {
		p_method.return_type = p_retinfo.class_name;
	} else if (p_retinfo.type == Variant::ARRAY && p_retinfo.hint == PROPERTY_HINT_ARRAY_TYPE) {
		p_method.return_type = p_retinfo.hint_string + "[]";
	} else if (p_retinfo.type == Variant::DICTIONARY && p_retinfo.hint == PROPERTY_HINT_DICTIONARY_TYPE) {
		p_method.return_type = "Dictionary[" + p_retinfo.hint_string.replace(";", ", ") + "]";
	} else if (p_retinfo.hint == PROPERTY_HINT_RESOURCE_TYPE) {
		p_method.return_type = p_retinfo.hint_string;
	} else if (p_retinfo.type == Variant::NIL && p_retinfo.usage & PROPERTY_USAGE_NIL_IS_VARIANT) {
		p_method.return_type = "Variant";
	} else if (p_retinfo.type == Variant::NIL) {
		p_method.return_type = "void";
	} else {
		p_method.return_type = Variant::get_type_name(p_retinfo.type);
	}
}

void DocData::argument_doc_from_arginfo(DocData::ArgumentDoc &p_argument, const PropertyInfo &p_arginfo) {
	p_argument.name = p_arginfo.name;

	if (p_arginfo.type == Variant::INT && p_arginfo.hint == PROPERTY_HINT_INT_IS_POINTER) {
		p_argument.type = p_arginfo.hint_string;
		if (p_argument.type.is_empty()) {
			p_argument.type = "void*";
		} else {
			p_argument.type += "*";
		}
	} else if (p_arginfo.type == Variant::INT && p_arginfo.usage & (PROPERTY_USAGE_CLASS_IS_ENUM | PROPERTY_USAGE_CLASS_IS_BITFIELD)) {
		p_argument.enumeration = p_arginfo.class_name;
		if (p_argument.enumeration.begins_with("_")) { //proxy class
			p_argument.enumeration = p_argument.enumeration.substr(1);
		}
		p_argument.is_bitfield = p_arginfo.usage & PROPERTY_USAGE_CLASS_IS_BITFIELD;
		p_argument.type = "int";
	} else if (p_arginfo.class_name != StringName()) {
		p_argument.type = p_arginfo.class_name;
	} else if (p_arginfo.type == Variant::ARRAY && p_arginfo.hint == PROPERTY_HINT_ARRAY_TYPE) {
		p_argument.type = p_arginfo.hint_string + "[]";
	} else if (p_arginfo.type == Variant::DICTIONARY && p_arginfo.hint == PROPERTY_HINT_DICTIONARY_TYPE) {
		p_argument.type = "Dictionary[" + p_arginfo.hint_string.replace(";", ", ") + "]";
	} else if (p_arginfo.hint == PROPERTY_HINT_RESOURCE_TYPE) {
		p_argument.type = p_arginfo.hint_string;
	} else if (p_arginfo.type == Variant::NIL) {
		// Parameters cannot be void, so PROPERTY_USAGE_NIL_IS_VARIANT is not necessary
		p_argument.type = "Variant";
	} else {
		p_argument.type = Variant::get_type_name(p_arginfo.type);
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
			argument.default_value = get_default_value_string(default_arg, p_methodinfo.arguments[i]);
		}
		p_method.arguments.push_back(argument);
	}
}
