/*************************************************************************/
/*  doc_data.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef DOC_DATA_H
#define DOC_DATA_H

#include "core/io/xml_parser.h"
#include "core/templates/map.h"
#include "core/variant/variant.h"

struct ScriptMemberInfo {
	PropertyInfo propinfo;
	String doc_string;
	StringName setter;
	StringName getter;

	bool has_default_value = false;
	Variant default_value;
};

class DocData {
public:
	struct ArgumentDoc {
		String name;
		String type;
		String enumeration;
		String default_value;
		bool operator<(const ArgumentDoc &p_arg) const {
			if (name == p_arg.name) {
				return type < p_arg.type;
			}
			return name < p_arg.name;
		}
	};

	struct MethodDoc {
		String name;
		String return_type;
		String return_enum;
		String qualifiers;
		String description;
		Vector<ArgumentDoc> arguments;
		Vector<int> errors_returned;
		bool operator<(const MethodDoc &p_method) const {
			if (name == p_method.name) {
				// Must be a constructor since there is no overloading.
				// We want this arbitrary order for a class "Foo":
				// - 1. Default constructor: Foo()
				// - 2. Copy constructor: Foo(Foo)
				// - 3+. Other constructors Foo(Bar, ...) based on first argument's name
				if (arguments.size() == 0 || p_method.arguments.size() == 0) { // 1.
					return arguments.size() < p_method.arguments.size();
				}
				if (arguments[0].type == return_type || p_method.arguments[0].type == p_method.return_type) { // 2.
					return (arguments[0].type == return_type) || (p_method.arguments[0].type != p_method.return_type);
				}
				return arguments[0] < p_method.arguments[0];
			}
			return name < p_method.name;
		}
	};

	struct ConstantDoc {
		String name;
		String value;
		bool is_value_valid = false;
		String enumeration;
		String description;
		bool operator<(const ConstantDoc &p_const) const {
			return name < p_const.name;
		}
	};

	struct EnumDoc {
		String name = "@unnamed_enum";
		String description;
		Vector<DocData::ConstantDoc> values;
	};

	struct PropertyDoc {
		String name;
		String type;
		String enumeration;
		String description;
		String setter, getter;
		String default_value;
		bool overridden = false;
		bool operator<(const PropertyDoc &p_prop) const {
			return name < p_prop.name;
		}
	};

	struct ThemeItemDoc {
		String name;
		String type;
		String data_type;
		String description;
		String default_value;
		bool operator<(const ThemeItemDoc &p_theme_item) const {
			return name < p_theme_item.name;
		}
	};

	struct TutorialDoc {
		String link;
		String title;
	};

	struct ClassDoc {
		String name;
		String inherits;
		String category;
		String brief_description;
		String description;
		Vector<TutorialDoc> tutorials;
		Vector<MethodDoc> constructors;
		Vector<MethodDoc> methods;
		Vector<MethodDoc> operators;
		Vector<MethodDoc> signals;
		Vector<ConstantDoc> constants;
		Map<String, String> enums;
		Vector<PropertyDoc> properties;
		Vector<ThemeItemDoc> theme_properties;
		bool is_script_doc = false;
		String script_path;
		bool operator<(const ClassDoc &p_class) const {
			return name < p_class.name;
		}
	};

	static void return_doc_from_retinfo(DocData::MethodDoc &p_method, const PropertyInfo &p_retinfo);
	static void argument_doc_from_arginfo(DocData::ArgumentDoc &p_argument, const PropertyInfo &p_arginfo);
	static void property_doc_from_scriptmemberinfo(DocData::PropertyDoc &p_property, const ScriptMemberInfo &p_memberinfo);
	static void method_doc_from_methodinfo(DocData::MethodDoc &p_method, const MethodInfo &p_methodinfo, const String &p_desc);
	static void constant_doc_from_variant(DocData::ConstantDoc &p_const, const StringName &p_name, const Variant &p_value, const String &p_desc);
	static void signal_doc_from_methodinfo(DocData::MethodDoc &p_signal, const MethodInfo &p_methodinfo, const String &p_desc);
};

#endif // DOC_DATA_H
