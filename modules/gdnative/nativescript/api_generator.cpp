/*************************************************************************/
/*  api_generator.cpp                                                    */
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

#include "api_generator.h"

#ifdef TOOLS_ENABLED

#include "core/config/engine.h"
#include "core/core_constants.h"
#include "core/object/class_db.h"
#include "core/os/file_access.h"
#include "core/string/string_builder.h"
#include "core/templates/pair.h"

// helper stuff

static Error save_file(const String &p_path, const List<String> &p_content) {
	FileAccessRef file = FileAccess::open(p_path, FileAccess::WRITE);

	ERR_FAIL_COND_V(!file, ERR_FILE_CANT_WRITE);

	for (const List<String>::Element *e = p_content.front(); e != nullptr; e = e->next()) {
		file->store_string(e->get());
	}

	file->close();

	return OK;
}

// helper stuff end

struct MethodAPI {
	String method_name;
	String return_type;

	List<String> argument_types;
	List<String> argument_names;

	Map<int, Variant> default_arguments;

	int argument_count = 0;
	bool has_varargs = false;
	bool is_editor = false;
	bool is_noscript = false;
	bool is_const = false;
	bool is_static = false; // For builtin types.
	bool is_reverse = false;
	bool is_virtual = false;
	bool is_from_script = false;
};

struct PropertyAPI {
	String name;
	String getter;
	String setter;
	String type;
	int index = 0;
};

struct ConstantAPI {
	String constant_name;
	int constant_value = 0;
	Variant builtin_constant_value; // For builtin types;
	String builtin_constant_type; // For builtin types;
};

struct SignalAPI {
	String name;
	List<String> argument_types;
	List<String> argument_names;
	Map<int, Variant> default_arguments;
};

struct EnumAPI {
	String name;
	List<Pair<int, String>> values;
};

struct OperatorAPI { // For builtin types;
	String name;
	int oper = Variant::OP_MAX;
	String other_type;
	String return_type;
};

struct ClassAPI {
	String class_name;
	String super_class_name;

	ClassDB::APIType api_type = ClassDB::API_NONE;

	bool is_singleton = false;
	String singleton_name;
	bool is_instantiable = false;
	// @Unclear
	bool is_reference = false;
	bool has_indexing = false; // For builtin types.
	String indexed_type; // For builtin types.
	bool is_keyed = false; // For builtin types.

	List<MethodAPI> methods;
	List<MethodAPI> constructors; // For builtin types.
	List<PropertyAPI> properties;
	List<ConstantAPI> constants;
	List<SignalAPI> signals_;
	List<EnumAPI> enums;
	List<OperatorAPI> operators; // For builtin types.
};

static String get_type_name(const PropertyInfo &info) {
	if (info.type == Variant::INT && (info.usage & PROPERTY_USAGE_CLASS_IS_ENUM)) {
		return String("enum.") + String(info.class_name).replace(".", "::");
	}
	if (info.class_name != StringName()) {
		return info.class_name;
	}
	if (info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
		return info.class_name;
	}
	if (info.type == Variant::NIL && (info.usage & PROPERTY_USAGE_NIL_IS_VARIANT)) {
		return "Variant";
	}
	if (info.type == Variant::NIL) {
		return "void";
	}
	return Variant::get_type_name(info.type);
}

/*
 * Some comparison helper functions we need
 */

struct MethodInfoComparator {
	StringName::AlphCompare compare;
	bool operator()(const MethodInfo &p_a, const MethodInfo &p_b) const {
		return compare(p_a.name, p_b.name);
	}
};

struct PropertyInfoComparator {
	StringName::AlphCompare compare;
	bool operator()(const PropertyInfo &p_a, const PropertyInfo &p_b) const {
		return compare(p_a.name, p_b.name);
	}
};

struct ConstantAPIComparator {
	NoCaseComparator compare;
	bool operator()(const ConstantAPI &p_a, const ConstantAPI &p_b) const {
		return compare(p_a.constant_name, p_b.constant_name);
	}
};

/*
 * Reads the entire Godot API to a list
 */
List<ClassAPI> generate_c_api_classes() {
	List<ClassAPI> api;

	List<StringName> classes;
	ClassDB::get_class_list(&classes);
	classes.sort_custom<StringName::AlphCompare>();

	// Register global constants as a fake CoreConstants singleton class
	{
		ClassAPI global_constants_api;
		global_constants_api.class_name = "CoreConstants";
		global_constants_api.api_type = ClassDB::API_CORE;
		global_constants_api.is_singleton = true;
		global_constants_api.singleton_name = "CoreConstants";
		global_constants_api.is_instantiable = false;
		const int constants_count = CoreConstants::get_global_constant_count();

		Map<StringName, EnumAPI> enum_api_map;
		for (int i = 0; i < constants_count; ++i) {
			StringName enum_name = CoreConstants::get_global_constant_enum(i);
			String name = String(CoreConstants::get_global_constant_name(i));
			int value = CoreConstants::get_global_constant_value(i);

			if (enum_name == StringName()) {
				ConstantAPI constant_api;
				constant_api.constant_name = name;
				constant_api.constant_value = value;
				global_constants_api.constants.push_back(constant_api);
			} else {
				EnumAPI enum_api;
				if (enum_api_map.has(enum_name)) {
					enum_api = enum_api_map[enum_name];
				} else {
					enum_api.name = String(enum_name);
				}
				enum_api.values.push_back(Pair(value, name));

				enum_api_map[enum_name] = enum_api;
			}
		}
		for (const Map<StringName, EnumAPI>::Element *E = enum_api_map.front(); E; E = E->next()) {
			global_constants_api.enums.push_back(E->get());
		}
		global_constants_api.constants.sort_custom<ConstantAPIComparator>();
		api.push_back(global_constants_api);
	}

	for (List<StringName>::Element *e = classes.front(); e != nullptr; e = e->next()) {
		StringName class_name = e->get();

		if (!ClassDB::is_class_exposed(class_name)) {
			continue;
		}

		ClassAPI class_api;
		class_api.api_type = ClassDB::get_api_type(e->get());
		class_api.class_name = class_name;
		class_api.super_class_name = ClassDB::get_parent_class(class_name);
		{
			String name = class_name;
			if (name.begins_with("_")) {
				name.remove(0);
			}
			class_api.is_singleton = Engine::get_singleton()->has_singleton(name);
			if (class_api.is_singleton) {
				class_api.singleton_name = name;
			}
		}
		class_api.is_instantiable = !class_api.is_singleton && ClassDB::can_instance(class_name);

		{
			List<StringName> inheriters;
			ClassDB::get_inheriters_from_class("Reference", &inheriters);
			bool is_reference = !!inheriters.find(class_name) || class_name == "Reference";
			// @Unclear
			class_api.is_reference = !class_api.is_singleton && is_reference;
		}

		// constants
		{
			List<String> constant;
			ClassDB::get_integer_constant_list(class_name, &constant, true);
			constant.sort_custom<NoCaseComparator>();
			for (List<String>::Element *c = constant.front(); c != nullptr; c = c->next()) {
				ConstantAPI constant_api;
				constant_api.constant_name = c->get();
				constant_api.constant_value = ClassDB::get_integer_constant(class_name, c->get());

				class_api.constants.push_back(constant_api);
			}
		}

		// signals
		{
			List<MethodInfo> signals_;
			ClassDB::get_signal_list(class_name, &signals_, true);
			signals_.sort_custom<MethodInfoComparator>();

			for (int i = 0; i < signals_.size(); i++) {
				SignalAPI signal;

				MethodInfo method_info = signals_[i];
				signal.name = method_info.name;

				for (int j = 0; j < method_info.arguments.size(); j++) {
					PropertyInfo argument = method_info.arguments[j];
					String type;
					String name = argument.name;

					if (argument.name.find(":") != -1) {
						type = argument.name.get_slice(":", 1);
						name = argument.name.get_slice(":", 0);
					} else {
						type = get_type_name(argument);
					}

					signal.argument_names.push_back(name);
					signal.argument_types.push_back(type);
				}

				Vector<Variant> default_arguments = method_info.default_arguments;

				int default_start = signal.argument_names.size() - default_arguments.size();

				for (int j = 0; j < default_arguments.size(); j++) {
					signal.default_arguments[default_start + j] = default_arguments[j];
				}

				class_api.signals_.push_back(signal);
			}
		}

		//properties
		{
			List<PropertyInfo> properties;
			ClassDB::get_property_list(class_name, &properties, true);
			properties.sort_custom<PropertyInfoComparator>();

			for (List<PropertyInfo>::Element *p = properties.front(); p != nullptr; p = p->next()) {
				PropertyAPI property_api;

				property_api.name = p->get().name;
				property_api.getter = ClassDB::get_property_getter(class_name, p->get().name);
				property_api.setter = ClassDB::get_property_setter(class_name, p->get().name);

				if (p->get().name.find(":") != -1) {
					property_api.type = p->get().name.get_slice(":", 1);
					property_api.name = p->get().name.get_slice(":", 0);
				} else {
					MethodInfo minfo;
					ClassDB::get_method_info(class_name, property_api.getter, &minfo, true, false);
					property_api.type = get_type_name(minfo.return_val);
				}

				property_api.index = ClassDB::get_property_index(class_name, p->get().name);

				if (!property_api.setter.is_empty() || !property_api.getter.is_empty()) {
					class_api.properties.push_back(property_api);
				}
			}
		}

		//methods
		{
			List<MethodInfo> methods;
			ClassDB::get_method_list(class_name, &methods, true);
			methods.sort_custom<MethodInfoComparator>();

			for (List<MethodInfo>::Element *m = methods.front(); m != nullptr; m = m->next()) {
				MethodAPI method_api;
				MethodBind *method_bind = ClassDB::get_method(class_name, m->get().name);
				MethodInfo &method_info = m->get();

				//method name
				method_api.method_name = method_info.name;
				//method return type
				if (method_api.method_name.find(":") != -1) {
					method_api.return_type = method_api.method_name.get_slice(":", 1);
					method_api.method_name = method_api.method_name.get_slice(":", 0);
				} else {
					method_api.return_type = get_type_name(m->get().return_val);
				}

				method_api.argument_count = method_info.arguments.size();
				method_api.has_varargs = method_bind && method_bind->is_vararg();

				// Method flags
				method_api.is_virtual = false;
				if (method_info.flags) {
					const uint32_t flags = method_info.flags;
					method_api.is_editor = flags & METHOD_FLAG_EDITOR;
					method_api.is_noscript = flags & METHOD_FLAG_NOSCRIPT;
					method_api.is_const = flags & METHOD_FLAG_CONST;
					method_api.is_reverse = flags & METHOD_FLAG_REVERSE;
					method_api.is_virtual = flags & METHOD_FLAG_VIRTUAL;
					method_api.is_from_script = flags & METHOD_FLAG_FROM_SCRIPT;
				}

				method_api.is_virtual = method_api.is_virtual || method_api.method_name[0] == '_';

				// method argument name and type

				for (int i = 0; i < method_api.argument_count; i++) {
					String arg_name;
					String arg_type;
					PropertyInfo arg_info = method_info.arguments[i];

					arg_name = arg_info.name;

					if (arg_info.name.find(":") != -1) {
						arg_type = arg_info.name.get_slice(":", 1);
						arg_name = arg_info.name.get_slice(":", 0);
					} else if (arg_info.hint == PROPERTY_HINT_RESOURCE_TYPE) {
						arg_type = arg_info.class_name;
					} else if (arg_info.type == Variant::NIL) {
						arg_type = "Variant";
					} else if (arg_info.type == Variant::OBJECT) {
						arg_type = arg_info.class_name;
						if (arg_type == "") {
							arg_type = Variant::get_type_name(arg_info.type);
						}
					} else {
						arg_type = Variant::get_type_name(arg_info.type);
					}

					method_api.argument_names.push_back(arg_name);
					method_api.argument_types.push_back(arg_type);

					if (method_bind && method_bind->has_default_argument(i)) {
						method_api.default_arguments[i] = method_bind->get_default_argument(i);
					}
				}

				class_api.methods.push_back(method_api);
			}
		}

		// enums
		{
			List<EnumAPI> enums;
			List<StringName> enum_names;
			ClassDB::get_enum_list(class_name, &enum_names, true);
			for (List<StringName>::Element *E = enum_names.front(); E; E = E->next()) {
				List<StringName> value_names;
				EnumAPI enum_api;
				enum_api.name = E->get();
				ClassDB::get_enum_constants(class_name, E->get(), &value_names, true);
				for (List<StringName>::Element *val_e = value_names.front(); val_e; val_e = val_e->next()) {
					int int_val = ClassDB::get_integer_constant(class_name, val_e->get(), nullptr);
					enum_api.values.push_back(Pair<int, String>(int_val, val_e->get()));
				}
				enum_api.values.sort_custom<PairSort<int, String>>();
				class_api.enums.push_back(enum_api);
			}
		}

		api.push_back(class_api);
	}

	return api;
}

/*
 * Reads the builtin Variant API to a list
 */
List<ClassAPI> generate_c_builtin_api_types() {
	List<ClassAPI> api;

	// Special class for the utility methods.
	{
		ClassAPI utility_api;
		utility_api.class_name = "Utilities";
		utility_api.is_instantiable = false;

		List<StringName> utility_functions;
		Variant::get_utility_function_list(&utility_functions);
		for (const List<StringName>::Element *E = utility_functions.front(); E; E = E->next()) {
			const StringName &function_name = E->get();

			MethodAPI function_api;
			function_api.method_name = function_name;
			function_api.has_varargs = Variant::is_utility_function_vararg(function_name);
			function_api.argument_count = function_api.has_varargs ? 0 : Variant::get_utility_function_argument_count(function_name);
			function_api.is_const = Variant::get_utility_function_type(function_name) == Variant::UTILITY_FUNC_TYPE_MATH;

			for (int i = 0; i < function_api.argument_count; i++) {
				function_api.argument_names.push_back(Variant::get_utility_function_argument_name(function_name, i));
				Variant::Type arg_type = Variant::get_utility_function_argument_type(function_name, i);
				function_api.argument_types.push_back(arg_type == Variant::NIL ? "Variant" : Variant::get_type_name(arg_type));
			}

			if (Variant::has_utility_function_return_value(function_name)) {
				Variant::Type ret_type = Variant::get_utility_function_return_type(function_name);
				function_api.return_type = ret_type == Variant::NIL ? "Variant" : Variant::get_type_name(ret_type);
			} else {
				function_api.return_type = "void";
			}

			utility_api.methods.push_back(function_api);
		}

		api.push_back(utility_api);
	}

	for (int t = 0; t < Variant::VARIANT_MAX; t++) {
		Variant::Type type = (Variant::Type)t;

		ClassAPI class_api;
		class_api.class_name = Variant::get_type_name(type);
		class_api.is_instantiable = true;
		class_api.has_indexing = Variant::has_indexing(type);
		class_api.indexed_type = Variant::get_type_name(Variant::get_indexed_element_type(type));
		class_api.is_keyed = Variant::is_keyed(type);
		// Types that are passed by reference.
		switch (type) {
			case Variant::OBJECT:
			case Variant::DICTIONARY:
			case Variant::ARRAY:
			case Variant::PACKED_BYTE_ARRAY:
			case Variant::PACKED_INT32_ARRAY:
			case Variant::PACKED_INT64_ARRAY:
			case Variant::PACKED_FLOAT32_ARRAY:
			case Variant::PACKED_FLOAT64_ARRAY:
			case Variant::PACKED_STRING_ARRAY:
			case Variant::PACKED_VECTOR2_ARRAY:
			case Variant::PACKED_VECTOR3_ARRAY:
			case Variant::PACKED_COLOR_ARRAY:
				class_api.is_reference = true;
				break;
			default:
				class_api.is_reference = false;
				break;
		}

		// Methods.

		List<StringName> methods;
		Variant::get_builtin_method_list(type, &methods);
		for (const List<StringName>::Element *E = methods.front(); E; E = E->next()) {
			const StringName &method_name = E->get();

			MethodAPI method_api;

			method_api.method_name = method_name;
			method_api.argument_count = Variant::get_builtin_method_argument_count(type, method_name);
			method_api.has_varargs = Variant::is_builtin_method_vararg(type, method_name);
			method_api.is_const = Variant::is_builtin_method_const(type, method_name);
			method_api.is_static = Variant::is_builtin_method_static(type, method_name);

			for (int i = 0; i < method_api.argument_count; i++) {
				method_api.argument_names.push_back(Variant::get_builtin_method_argument_name(type, method_name, i));
				Variant::Type arg_type = Variant::get_builtin_method_argument_type(type, method_name, i);
				method_api.argument_types.push_back(arg_type == Variant::NIL ? "Variant" : Variant::get_type_name(arg_type));
			}

			Vector<Variant> default_arguments = Variant::get_builtin_method_default_arguments(type, method_name);

			int default_start = method_api.argument_names.size() - default_arguments.size();

			for (int i = 0; i < default_arguments.size(); i++) {
				method_api.default_arguments[default_start + i] = default_arguments[i];
			}

			if (Variant::has_builtin_method_return_value(type, method_name)) {
				Variant::Type ret_type = Variant::get_builtin_method_return_type(type, method_name);
				method_api.return_type = ret_type == Variant::NIL ? "Variant" : Variant::get_type_name(ret_type);
			} else {
				method_api.return_type = "void";
			}

			class_api.methods.push_back(method_api);
		}

		// Constructors.

		for (int c = 0; c < Variant::get_constructor_count(type); c++) {
			MethodAPI constructor_api;

			constructor_api.method_name = Variant::get_type_name(type);
			constructor_api.argument_count = Variant::get_constructor_argument_count(type, c);
			constructor_api.return_type = Variant::get_type_name(type);

			for (int i = 0; i < constructor_api.argument_count; i++) {
				constructor_api.argument_names.push_back(Variant::get_constructor_argument_name(type, c, i));
				Variant::Type arg_type = Variant::get_constructor_argument_type(type, c, i);
				constructor_api.argument_types.push_back(arg_type == Variant::NIL ? "Variant" : Variant::get_type_name(arg_type));
			}

			class_api.constructors.push_back(constructor_api);
		}

		// Constants.

		List<StringName> constants;
		Variant::get_constants_for_type(type, &constants);
		for (const List<StringName>::Element *E = constants.front(); E; E = E->next()) {
			const StringName &constant_name = E->get();
			ConstantAPI constant_api;

			constant_api.constant_name = constant_name;
			constant_api.builtin_constant_value = Variant::get_constant_value(type, constant_name);
			constant_api.builtin_constant_type = Variant::get_type_name(constant_api.builtin_constant_value.get_type());

			class_api.constants.push_back(constant_api);
		}

		// Members.

		List<StringName> members;
		Variant::get_member_list(type, &members);
		for (const List<StringName>::Element *E = members.front(); E; E = E->next()) {
			const StringName &member_name = E->get();

			PropertyAPI member_api;
			member_api.name = member_name;
			Variant::Type member_type = Variant::get_member_type(type, member_name);
			member_api.type = member_type == Variant::NIL ? "Variant" : Variant::get_type_name(member_type);

			class_api.properties.push_back(member_api);
		}

		// Operators.

		for (int op = 0; op < Variant::OP_MAX; op++) {
			Variant::Operator oper = (Variant::Operator)op;

			for (int ot = 0; ot < Variant::VARIANT_MAX; ot++) {
				Variant::Type other_type = (Variant::Type)ot;

				if (!Variant::get_validated_operator_evaluator(oper, type, other_type)) {
					continue;
				}

				OperatorAPI oper_api;
				oper_api.name = Variant::get_operator_name(oper);
				oper_api.oper = oper;
				oper_api.other_type = Variant::get_type_name(other_type);
				oper_api.return_type = Variant::get_type_name(Variant::get_operator_return_type(oper, type, other_type));

				class_api.operators.push_back(oper_api);
			}
		}

		api.push_back(class_api);
	}

	return api;
}

/*
 * Generates the JSON source from the API in p_api
 */
static List<String> generate_c_api_json(const List<ClassAPI> &p_api) {
	// I'm sorry for the \t mess

	List<String> source;

	source.push_back("[\n");

	for (const List<ClassAPI>::Element *c = p_api.front(); c != nullptr; c = c->next()) {
		ClassAPI api = c->get();

		source.push_back("\t{\n");

		source.push_back("\t\t\"name\": \"" + api.class_name + "\",\n");
		source.push_back("\t\t\"base_class\": \"" + api.super_class_name + "\",\n");
		source.push_back(String("\t\t\"api_type\": \"") + (api.api_type == ClassDB::API_CORE ? "core" : (api.api_type == ClassDB::API_EDITOR ? "tools" : "none")) + "\",\n");
		source.push_back(String("\t\t\"singleton\": ") + (api.is_singleton ? "true" : "false") + ",\n");
		source.push_back("\t\t\"singleton_name\": \"" + api.singleton_name + "\",\n");
		source.push_back(String("\t\t\"instantiable\": ") + (api.is_instantiable ? "true" : "false") + ",\n");
		source.push_back(String("\t\t\"is_reference\": ") + (api.is_reference ? "true" : "false") + ",\n");

		source.push_back("\t\t\"constants\": {\n");
		for (List<ConstantAPI>::Element *e = api.constants.front(); e; e = e->next()) {
			source.push_back("\t\t\t\"" + e->get().constant_name + "\": " + String::num_int64(e->get().constant_value) + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t},\n");

		source.push_back("\t\t\"properties\": [\n");
		for (List<PropertyAPI>::Element *e = api.properties.front(); e; e = e->next()) {
			source.push_back("\t\t\t{\n");
			source.push_back("\t\t\t\t\"name\": \"" + e->get().name + "\",\n");
			source.push_back("\t\t\t\t\"type\": \"" + e->get().type + "\",\n");
			source.push_back("\t\t\t\t\"getter\": \"" + e->get().getter + "\",\n");
			source.push_back("\t\t\t\t\"setter\": \"" + e->get().setter + "\",\n");
			source.push_back(String("\t\t\t\t\"index\": ") + itos(e->get().index) + "\n");
			source.push_back(String("\t\t\t}") + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t],\n");

		source.push_back("\t\t\"signals\": [\n");
		for (List<SignalAPI>::Element *e = api.signals_.front(); e; e = e->next()) {
			source.push_back("\t\t\t{\n");
			source.push_back("\t\t\t\t\"name\": \"" + e->get().name + "\",\n");
			source.push_back("\t\t\t\t\"arguments\": [\n");
			for (int i = 0; i < e->get().argument_names.size(); i++) {
				source.push_back("\t\t\t\t\t{\n");
				source.push_back("\t\t\t\t\t\t\"name\": \"" + e->get().argument_names[i] + "\",\n");
				source.push_back("\t\t\t\t\t\t\"type\": \"" + e->get().argument_types[i] + "\",\n");
				source.push_back(String("\t\t\t\t\t\t\"has_default_value\": ") + (e->get().default_arguments.has(i) ? "true" : "false") + ",\n");
				source.push_back("\t\t\t\t\t\t\"default_value\": \"" + (e->get().default_arguments.has(i) ? (String)e->get().default_arguments[i] : "") + "\"\n");
				source.push_back(String("\t\t\t\t\t}") + ((i < e->get().argument_names.size() - 1) ? "," : "") + "\n");
			}
			source.push_back("\t\t\t\t]\n");
			source.push_back(String("\t\t\t}") + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t],\n");

		source.push_back("\t\t\"methods\": [\n");
		for (List<MethodAPI>::Element *e = api.methods.front(); e; e = e->next()) {
			source.push_back("\t\t\t{\n");
			source.push_back("\t\t\t\t\"name\": \"" + e->get().method_name + "\",\n");
			source.push_back("\t\t\t\t\"return_type\": \"" + e->get().return_type + "\",\n");
			source.push_back(String("\t\t\t\t\"is_editor\": ") + (e->get().is_editor ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_noscript\": ") + (e->get().is_noscript ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_const\": ") + (e->get().is_const ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_reverse\": ") + (e->get().is_reverse ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_virtual\": ") + (e->get().is_virtual ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"has_varargs\": ") + (e->get().has_varargs ? "true" : "false") + ",\n");
			source.push_back(String("\t\t\t\t\"is_from_script\": ") + (e->get().is_from_script ? "true" : "false") + ",\n");
			source.push_back("\t\t\t\t\"arguments\": [\n");
			for (int i = 0; i < e->get().argument_names.size(); i++) {
				source.push_back("\t\t\t\t\t{\n");
				source.push_back("\t\t\t\t\t\t\"name\": \"" + e->get().argument_names[i] + "\",\n");
				source.push_back("\t\t\t\t\t\t\"type\": \"" + e->get().argument_types[i] + "\",\n");
				source.push_back(String("\t\t\t\t\t\t\"has_default_value\": ") + (e->get().default_arguments.has(i) ? "true" : "false") + ",\n");
				source.push_back("\t\t\t\t\t\t\"default_value\": \"" + (e->get().default_arguments.has(i) ? (String)e->get().default_arguments[i] : "") + "\"\n");
				source.push_back(String("\t\t\t\t\t}") + ((i < e->get().argument_names.size() - 1) ? "," : "") + "\n");
			}
			source.push_back("\t\t\t\t]\n");
			source.push_back(String("\t\t\t}") + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t],\n");

		source.push_back("\t\t\"enums\": [\n");
		for (List<EnumAPI>::Element *e = api.enums.front(); e; e = e->next()) {
			source.push_back("\t\t\t{\n");
			source.push_back("\t\t\t\t\"name\": \"" + e->get().name + "\",\n");
			source.push_back("\t\t\t\t\"values\": {\n");
			for (List<Pair<int, String>>::Element *val_e = e->get().values.front(); val_e; val_e = val_e->next()) {
				source.push_back("\t\t\t\t\t\"" + val_e->get().second + "\": " + itos(val_e->get().first));
				source.push_back(String((val_e->next() ? "," : "")) + "\n");
			}
			source.push_back("\t\t\t\t}\n");
			source.push_back(String("\t\t\t}") + (e->next() ? "," : "") + "\n");
		}
		source.push_back("\t\t]\n");

		source.push_back(String("\t}") + (c->next() ? "," : "") + "\n");
	}
	source.push_back("]");

	return source;
}

static int indent_level = 0;

static void append_indented(StringBuilder &p_source, const String &p_text) {
	for (int i = 0; i < indent_level; i++) {
		p_source.append("\t");
	}
	p_source.append(p_text);
	p_source.append("\n");
}

static void append_indented(StringBuilder &p_source, const char *p_text) {
	for (int i = 0; i < indent_level; i++) {
		p_source.append("\t");
	}
	p_source.append(p_text);
	p_source.append("\n");
}

static void write_builtin_method(StringBuilder &p_source, const MethodAPI &p_method) {
	append_indented(p_source, vformat(R"("name": "%s",)", p_method.method_name));
	append_indented(p_source, vformat(R"("return_type": "%s",)", p_method.return_type));
	append_indented(p_source, vformat(R"("is_const": %s,)", p_method.is_const ? "true" : "false"));
	append_indented(p_source, vformat(R"("is_static": %s,)", p_method.is_static ? "true" : "false"));
	append_indented(p_source, vformat(R"("has_varargs": %s,)", p_method.has_varargs ? "true" : "false"));

	append_indented(p_source, R"("arguments": [)");
	indent_level++;
	for (int i = 0; i < p_method.argument_count; i++) {
		append_indented(p_source, "{");
		indent_level++;

		append_indented(p_source, vformat(R"("name": "%s",)", p_method.argument_names[i]));
		append_indented(p_source, vformat(R"("type": "%s",)", p_method.argument_types[i]));
		append_indented(p_source, vformat(R"("has_default_value": %s,)", p_method.default_arguments.has(i) ? "true" : "false"));
		append_indented(p_source, vformat(R"("default_value": "%s")", p_method.default_arguments.has(i) ? p_method.default_arguments[i].operator String() : ""));

		indent_level--;
		append_indented(p_source, i < p_method.argument_count - 1 ? "}," : "}");
	}
	indent_level--;
	append_indented(p_source, "]");
}

static List<String> generate_c_builtin_api_json(const List<ClassAPI> &p_api) {
	StringBuilder source;

	source.append("[\n");

	indent_level = 1;

	for (const List<ClassAPI>::Element *C = p_api.front(); C; C = C->next()) {
		const ClassAPI &class_api = C->get();
		append_indented(source, "{");
		indent_level++;

		append_indented(source, vformat(R"("name": "%s",)", class_api.class_name));
		append_indented(source, vformat(R"("is_instantiable": %s,)", class_api.is_instantiable ? "true" : "false"));
		append_indented(source, vformat(R"("is_reference": %s,)", class_api.is_reference ? "true" : "false"));
		append_indented(source, vformat(R"("has_indexing": %s,)", class_api.has_indexing ? "true" : "false"));
		append_indented(source, vformat(R"("indexed_type": "%s",)", class_api.has_indexing && class_api.indexed_type == "Nil" ? "Variant" : class_api.indexed_type));
		append_indented(source, vformat(R"("is_keyed": %s,)", class_api.is_keyed ? "true" : "false"));

		// Constructors.
		append_indented(source, R"("constructors": [)");
		indent_level++;
		for (const List<MethodAPI>::Element *E = class_api.constructors.front(); E; E = E->next()) {
			const MethodAPI &constructor = E->get();
			append_indented(source, "{");
			indent_level++;

			write_builtin_method(source, constructor);

			indent_level--;
			append_indented(source, E->next() ? "}," : "}");
		}
		indent_level--;
		append_indented(source, "],");

		// Constants.
		append_indented(source, R"("constants": [)");
		indent_level++;
		for (const List<ConstantAPI>::Element *E = class_api.constants.front(); E; E = E->next()) {
			const ConstantAPI &constant = E->get();
			append_indented(source, "{");
			indent_level++;

			append_indented(source, vformat(R"("name": "%s",)", constant.constant_name));
			append_indented(source, vformat(R"("type": "%s",)", constant.builtin_constant_type));
			append_indented(source, vformat(R"("value": "%s")", constant.builtin_constant_value.operator String()));

			indent_level--;
			append_indented(source, E->next() ? "}," : "}");
		}
		indent_level--;
		append_indented(source, "],");

		// Methods.
		append_indented(source, R"("methods": [)");
		indent_level++;
		for (const List<MethodAPI>::Element *E = class_api.methods.front(); E; E = E->next()) {
			const MethodAPI &method = E->get();
			append_indented(source, "{");
			indent_level++;

			write_builtin_method(source, method);

			indent_level--;
			append_indented(source, E->next() ? "}," : "}");
		}
		indent_level--;
		append_indented(source, "],");

		// Members.
		append_indented(source, R"("members": [)");
		indent_level++;
		for (const List<PropertyAPI>::Element *E = class_api.properties.front(); E; E = E->next()) {
			const PropertyAPI &member = E->get();
			append_indented(source, "{");
			indent_level++;

			append_indented(source, vformat(R"("name": "%s",)", member.name));
			append_indented(source, vformat(R"("type": "%s")", member.type));

			indent_level--;
			append_indented(source, E->next() ? "}," : "}");
		}
		indent_level--;
		append_indented(source, "],");

		// Operators.
		append_indented(source, R"("operators": [)");
		indent_level++;
		for (const List<OperatorAPI>::Element *E = class_api.operators.front(); E; E = E->next()) {
			const OperatorAPI &oper = E->get();
			append_indented(source, "{");
			indent_level++;

			append_indented(source, vformat(R"("name": "%s",)", oper.name));
			append_indented(source, vformat(R"("operator": %d,)", oper.oper));
			append_indented(source, vformat(R"("other_type": "%s",)", oper.other_type));
			append_indented(source, vformat(R"("return_type": "%s")", oper.return_type));

			indent_level--;
			append_indented(source, E->next() ? "}," : "}");
		}
		indent_level--;
		append_indented(source, "]");

		indent_level--;
		append_indented(source, C->next() ? "}," : "}");
	}

	indent_level--;
	source.append("]\n");

	List<String> result;
	result.push_back(source.as_string());
	return result;
}

#endif

/*
 * Saves the whole Godot API to a JSON file located at
 *  p_path
 */
Error generate_c_api(const String &p_path) {
#ifndef TOOLS_ENABLED
	return ERR_BUG;
#else

	List<ClassAPI> api = generate_c_api_classes();

	List<String> json_source = generate_c_api_json(api);

	return save_file(p_path, json_source);
#endif
}
/*
 * Saves the builtin Godot API to a JSON file located at
 *  p_path
 */
Error generate_c_builtin_api(const String &p_path) {
#ifndef TOOLS_ENABLED
	return ERR_BUG;
#else

	List<ClassAPI> api = generate_c_builtin_api_types();

	List<String> json_source = generate_c_builtin_api_json(api);

	return save_file(p_path, json_source);
#endif
}
