/**************************************************************************/
/*  doc_data.h                                                            */
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

#ifndef DOC_DATA_H
#define DOC_DATA_H

#include "core/io/xml_parser.h"
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
		static ArgumentDoc from_dict(const Dictionary &p_dict) {
			ArgumentDoc doc;

			if (p_dict.has("name")) {
				doc.name = p_dict["name"];
			}

			if (p_dict.has("type")) {
				doc.type = p_dict["type"];
			}

			if (p_dict.has("enumeration")) {
				doc.enumeration = p_dict["enumeration"];
			}

			if (p_dict.has("default_value")) {
				doc.default_value = p_dict["default_value"];
			}

			return doc;
		}
	};

	struct MethodDoc {
		String name;
		String return_type;
		String return_enum;
		String qualifiers;
		String description;
		bool is_deprecated = false;
		bool is_experimental = false;
		Vector<ArgumentDoc> arguments;
		Vector<int> errors_returned;
		bool operator<(const MethodDoc &p_method) const {
			if (name == p_method.name) {
				// Must be an operator or a constructor since there is no other overloading
				if (name.left(8) == "operator") {
					if (arguments.size() == p_method.arguments.size()) {
						if (arguments.size() == 0) {
							return false;
						}
						return arguments[0].type < p_method.arguments[0].type;
					}
					return arguments.size() < p_method.arguments.size();
				} else {
					// Must be a constructor
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
			}
			return name < p_method.name;
		}
		static MethodDoc from_dict(const Dictionary &p_dict) {
			MethodDoc doc;

			if (p_dict.has("name")) {
				doc.name = p_dict["name"];
			}

			if (p_dict.has("return_type")) {
				doc.return_type = p_dict["return_type"];
			}

			if (p_dict.has("return_enum")) {
				doc.return_enum = p_dict["return_enum"];
			}

			if (p_dict.has("qualifiers")) {
				doc.qualifiers = p_dict["qualifiers"];
			}

			if (p_dict.has("description")) {
				doc.description = p_dict["description"];
			}

			if (p_dict.has("is_deprecated")) {
				doc.is_deprecated = p_dict["is_deprecated"];
			}

			if (p_dict.has("is_experimental")) {
				doc.is_experimental = p_dict["is_experimental"];
			}

			Array arguments;
			if (p_dict.has("arguments")) {
				arguments = p_dict["arguments"];
			}
			for (int i = 0; i < arguments.size(); i++) {
				doc.arguments.push_back(ArgumentDoc::from_dict(arguments[i]));
			}

			Array errors_returned;
			if (p_dict.has("errors_returned")) {
				errors_returned = p_dict["errors_returned"];
			}
			for (int i = 0; i < errors_returned.size(); i++) {
				doc.errors_returned.push_back(errors_returned[i]);
			}

			return doc;
		}
	};

	struct ConstantDoc {
		String name;
		String value;
		bool is_value_valid = false;
		String enumeration;
		bool is_bitfield = false;
		String description;
		bool is_deprecated = false;
		bool is_experimental = false;
		bool operator<(const ConstantDoc &p_const) const {
			return name < p_const.name;
		}
		static ConstantDoc from_dict(const Dictionary &p_dict) {
			ConstantDoc doc;

			if (p_dict.has("name")) {
				doc.name = p_dict["name"];
			}

			if (p_dict.has("value")) {
				doc.value = p_dict["value"];
			}

			if (p_dict.has("is_value_valid")) {
				doc.is_value_valid = p_dict["is_value_valid"];
			}

			if (p_dict.has("enumeration")) {
				doc.enumeration = p_dict["enumeration"];
			}

			if (p_dict.has("is_bitfield")) {
				doc.is_bitfield = p_dict["is_bitfield"];
			}

			if (p_dict.has("description")) {
				doc.description = p_dict["description"];
			}

			if (p_dict.has("is_deprecated")) {
				doc.is_deprecated = p_dict["is_deprecated"];
			}

			if (p_dict.has("is_experimental")) {
				doc.is_experimental = p_dict["is_experimental"];
			}

			return doc;
		}
	};

	struct EnumDoc {
		String name = "@unnamed_enum";
		bool is_bitfield = false;
		String description;
		Vector<DocData::ConstantDoc> values;
		static EnumDoc from_dict(const Dictionary &p_dict) {
			EnumDoc doc;

			if (p_dict.has("name")) {
				doc.name = p_dict["name"];
			}

			if (p_dict.has("is_bitfield")) {
				doc.is_bitfield = p_dict["is_bitfield"];
			}

			if (p_dict.has("description")) {
				doc.description = p_dict["description"];
			}

			Array values;
			if (p_dict.has("values")) {
				values = p_dict["values"];
			}
			for (int i = 0; i < values.size(); i++) {
				doc.values.push_back(ConstantDoc::from_dict(values[i]));
			}

			return doc;
		}
	};

	struct PropertyDoc {
		String name;
		String type;
		String enumeration;
		String description;
		String setter, getter;
		String default_value;
		bool overridden = false;
		String overrides;
		bool is_deprecated = false;
		bool is_experimental = false;
		bool operator<(const PropertyDoc &p_prop) const {
			return name < p_prop.name;
		}
		static PropertyDoc from_dict(const Dictionary &p_dict) {
			PropertyDoc doc;

			if (p_dict.has("name")) {
				doc.name = p_dict["name"];
			}

			if (p_dict.has("type")) {
				doc.type = p_dict["type"];
			}

			if (p_dict.has("enumeration")) {
				doc.enumeration = p_dict["enumeration"];
			}

			if (p_dict.has("description")) {
				doc.description = p_dict["description"];
			}

			if (p_dict.has("setter")) {
				doc.setter = p_dict["setter"];
			}

			if (p_dict.has("getter")) {
				doc.getter = p_dict["getter"];
			}

			if (p_dict.has("default_value")) {
				doc.default_value = p_dict["default_value"];
			}

			if (p_dict.has("overridden")) {
				doc.overridden = p_dict["overridden"];
			}

			if (p_dict.has("overrides")) {
				doc.overrides = p_dict["overrides"];
			}

			if (p_dict.has("is_deprecated")) {
				doc.is_deprecated = p_dict["is_deprecated"];
			}

			if (p_dict.has("is_experimental")) {
				doc.is_experimental = p_dict["is_experimental"];
			}

			return doc;
		}
	};

	struct ThemeItemDoc {
		String name;
		String type;
		String data_type;
		String description;
		String default_value;
		bool operator<(const ThemeItemDoc &p_theme_item) const {
			// First sort by the data type, then by name.
			if (data_type == p_theme_item.data_type) {
				return name < p_theme_item.name;
			}
			return data_type < p_theme_item.data_type;
		}
		static ThemeItemDoc from_dict(const Dictionary &p_dict) {
			ThemeItemDoc doc;

			if (p_dict.has("name")) {
				doc.name = p_dict["name"];
			}

			if (p_dict.has("type")) {
				doc.type = p_dict["type"];
			}

			if (p_dict.has("data_type")) {
				doc.data_type = p_dict["data_type"];
			}

			if (p_dict.has("description")) {
				doc.description = p_dict["description"];
			}

			if (p_dict.has("default_value")) {
				doc.default_value = p_dict["default_value"];
			}

			return doc;
		}
	};

	struct TutorialDoc {
		String link;
		String title;
		static TutorialDoc from_dict(const Dictionary &p_dict) {
			TutorialDoc doc;

			if (p_dict.has("link")) {
				doc.link = p_dict["link"];
			}

			if (p_dict.has("title")) {
				doc.title = p_dict["title"];
			}

			return doc;
		}
	};

	struct ClassDoc {
		String name;
		String inherits;
		String brief_description;
		String description;
		Vector<TutorialDoc> tutorials;
		Vector<MethodDoc> constructors;
		Vector<MethodDoc> methods;
		Vector<MethodDoc> operators;
		Vector<MethodDoc> signals;
		Vector<ConstantDoc> constants;
		HashMap<String, String> enums;
		Vector<PropertyDoc> properties;
		Vector<MethodDoc> annotations;
		Vector<ThemeItemDoc> theme_properties;
		bool is_deprecated = false;
		bool is_experimental = false;
		bool is_script_doc = false;
		String script_path;
		bool operator<(const ClassDoc &p_class) const {
			return name < p_class.name;
		}
		static ClassDoc from_dict(const Dictionary &p_dict) {
			ClassDoc doc;

			if (p_dict.has("name")) {
				doc.name = p_dict["name"];
			}

			if (p_dict.has("inherits")) {
				doc.inherits = p_dict["inherits"];
			}

			if (p_dict.has("brief_description")) {
				doc.brief_description = p_dict["brief_description"];
			}

			if (p_dict.has("description")) {
				doc.description = p_dict["description"];
			}

			Array tutorials;
			if (p_dict.has("tutorials")) {
				tutorials = p_dict["tutorials"];
			}
			for (int i = 0; i < tutorials.size(); i++) {
				doc.tutorials.push_back(TutorialDoc::from_dict(tutorials[i]));
			}

			Array constructors;
			if (p_dict.has("constructors")) {
				constructors = p_dict["constructors"];
			}
			for (int i = 0; i < constructors.size(); i++) {
				doc.constructors.push_back(MethodDoc::from_dict(constructors[i]));
			}

			Array methods;
			if (p_dict.has("methods")) {
				methods = p_dict["methods"];
			}
			for (int i = 0; i < methods.size(); i++) {
				doc.methods.push_back(MethodDoc::from_dict(methods[i]));
			}

			Array operators;
			if (p_dict.has("operators")) {
				operators = p_dict["operators"];
			}
			for (int i = 0; i < operators.size(); i++) {
				doc.operators.push_back(MethodDoc::from_dict(operators[i]));
			}

			Array signals;
			if (p_dict.has("signals")) {
				signals = p_dict["signals"];
			}
			for (int i = 0; i < signals.size(); i++) {
				doc.signals.push_back(MethodDoc::from_dict(signals[i]));
			}

			Array constants;
			if (p_dict.has("constants")) {
				constants = p_dict["constants"];
			}
			for (int i = 0; i < constants.size(); i++) {
				doc.constants.push_back(ConstantDoc::from_dict(constants[i]));
			}

			Dictionary enums;
			if (p_dict.has("enums")) {
				enums = p_dict["enums"];
			}
			for (int i = 0; i < enums.size(); i++) {
				doc.enums[enums.get_key_at_index(i)] = enums.get_value_at_index(i);
			}

			Array properties;
			if (p_dict.has("properties")) {
				properties = p_dict["properties"];
			}
			for (int i = 0; i < properties.size(); i++) {
				doc.properties.push_back(PropertyDoc::from_dict(properties[i]));
			}

			Array annotations;
			if (p_dict.has("annotations")) {
				annotations = p_dict["annotations"];
			}
			for (int i = 0; i < annotations.size(); i++) {
				doc.annotations.push_back(MethodDoc::from_dict(annotations[i]));
			}

			Array theme_properties;
			if (p_dict.has("theme_properties")) {
				theme_properties = p_dict["theme_properties"];
			}
			for (int i = 0; i < theme_properties.size(); i++) {
				doc.theme_properties.push_back(ThemeItemDoc::from_dict(theme_properties[i]));
			}

			if (p_dict.has("is_deprecated")) {
				doc.is_deprecated = p_dict["is_deprecated"];
			}

			if (p_dict.has("is_experimental")) {
				doc.is_experimental = p_dict["is_experimental"];
			}

			if (p_dict.has("is_script_doc")) {
				doc.is_script_doc = p_dict["is_script_doc"];
			}

			if (p_dict.has("script_path")) {
				doc.script_path = p_dict["script_path"];
			}

			return doc;
		}
	};

	static String get_default_value_string(const Variant &p_value);

	static void return_doc_from_retinfo(DocData::MethodDoc &p_method, const PropertyInfo &p_retinfo);
	static void argument_doc_from_arginfo(DocData::ArgumentDoc &p_argument, const PropertyInfo &p_arginfo);
	static void property_doc_from_scriptmemberinfo(DocData::PropertyDoc &p_property, const ScriptMemberInfo &p_memberinfo);
	static void method_doc_from_methodinfo(DocData::MethodDoc &p_method, const MethodInfo &p_methodinfo, const String &p_desc);
	static void constant_doc_from_variant(DocData::ConstantDoc &p_const, const StringName &p_name, const Variant &p_value, const String &p_desc);
	static void signal_doc_from_methodinfo(DocData::MethodDoc &p_signal, const MethodInfo &p_methodinfo, const String &p_desc);
};

#endif // DOC_DATA_H
