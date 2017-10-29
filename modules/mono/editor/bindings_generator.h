/*************************************************************************/
/*  bindings_generator.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef BINDINGS_GENERATOR_H
#define BINDINGS_GENERATOR_H

#include "class_db.h"
#include "editor/doc/doc_data.h"
#include "editor/editor_help.h"

#ifdef DEBUG_METHODS_ENABLED

#include "ustring.h"

class BindingsGenerator {
	struct ArgumentInterface {
		enum DefaultParamMode {
			CONSTANT,
			NULLABLE_VAL,
			NULLABLE_REF
		};

		String type;
		String name;
		String default_argument;
		DefaultParamMode def_param_mode;

		ArgumentInterface() {
			def_param_mode = CONSTANT;
		}
	};

	struct MethodInterface {
		String name;

		/**
		 * Name of the C# method
		 */
		String proxy_name;

		/**
		 * [TypeInterface::name] of the return type
		 */
		String return_type;

		/**
		 * Determines if the method has a variable number of arguments (VarArg)
		 */
		bool is_vararg;

		/**
		 * Virtual methods ("virtual" as defined by the Godot API) are methods that by default do nothing,
		 * but can be overridden by the user to add custom functionality.
		 * e.g.: _ready, _process, etc.
		 */
		bool is_virtual;

		/**
		 * Determines if the call should fallback to Godot's object.Call(string, params) in C#.
		 */
		bool requires_object_call;

		/**
		 * Determines if the method visibility is `internal` (visible only to files in the same assembly).
		 * Currently, we only use this for methods that are not meant to be exposed,
		 * but are required by properties as getters or setters.
		 * Methods that are not meant to be exposed are those that begin with underscore and are not virtual.
		 */
		bool is_internal;

		List<ArgumentInterface> arguments;

		const DocData::MethodDoc *method_doc;

		void add_argument(const ArgumentInterface &argument) {
			arguments.push_back(argument);
		}

		MethodInterface() {
			return_type = "void";
			is_vararg = false;
			is_virtual = false;
			requires_object_call = false;
			is_internal = false;
			method_doc = NULL;
		}
	};

	struct TypeInterface {
		/**
		 * Identifier name for this type.
		 * Also used to format [c_out].
		 */
		String name;

		/**
		 * Identifier name of the base class.
		 */
		String base_name;

		/**
		 * Name of the C# class
		 */
		String proxy_name;

		ClassDB::APIType api_type;

		bool is_object_type;
		bool is_singleton;
		bool is_reference;

		/**
		 * Used only by Object-derived types.
		 * Determines if this type is not virtual (incomplete).
		 * e.g.: CanvasItem cannot be instantiated.
		 */
		bool is_instantiable;

		/**
		 * Used only by Object-derived types.
		 * Determines if the C# class owns the native handle and must free it somehow when disposed.
		 * e.g.: Reference types must notify when the C# instance is disposed, for proper refcounting.
		 */
		bool memory_own;

		/**
		 * Determines if the file must have a using directive for System.Collections.Generic
		 * e.g.: When the generated class makes use of Dictionary
		 */
		bool requires_collections;

		// !! The comments of the following fields make reference to other fields via square brackets, e.g.: [field_name]
		// !! When renaming those fields, make sure to rename their references in the comments

		// --- C INTERFACE ---

		static const char *DEFAULT_VARARG_C_IN;

		/**
		 * One or more statements that manipulate the parameter before being passed as argument of a ptrcall.
		 * If the statement adds a local that must be passed as the argument instead of the parameter,
		 * the name of that local must be specified with [c_arg_in].
		 * For variadic methods, this field is required and, if empty, [DEFAULT_VARARG_C_IN] is used instead.
		 * Formatting elements:
		 * %0: [c_type] of the parameter
		 * %1: name of the parameter
		 */
		String c_in;

		/**
		 * Determines the name of the variable that will be passed as argument to a ptrcall.
		 * By default the value equals the name of the parameter,
		 * this varies for types that require special manipulation via [c_in].
		 * Formatting elements:
		 * %0 or %s: name of the parameter
		 */
		String c_arg_in;

		/**
		 * One or more statements that determine how a variable of this type is returned from a function.
		 * It must contain the return statement(s).
		 * Formatting elements:
		 * %0: [c_type_out] of the return type
		 * %1: name of the variable to be returned
		 * %2: [name] of the return type
		 */
		String c_out;

		/**
		 * The actual expected type, as seen (in most cases) in Variant copy constructors
		 * Used for the type of the return variable and to format [c_in].
		 * The value must be the following depending of the type:
		 * Object-derived types: Object*
		 * Other types: [name]
		 * -- Exceptions --
		 * VarArg (fictitious type to represent variable arguments): Array
		 * float: double (because ptrcall only supports double)
		 * int: int64_t (because ptrcall only supports int64_t and uint64_t)
		 * Reference types override this for the type of the return variable: Ref<Reference>
		 */
		String c_type;

		/**
		 * Determines the type used for parameters in function signatures.
		 */
		String c_type_in;

		/**
		 * Determines the return type used for function signatures.
		 * Also used to construct a default value to return in case of errors,
		 * and to format [c_out].
		 */
		String c_type_out;

		// --- C# INTERFACE ---

		/**
		 * An expression that overrides the way the parameter is passed to the internal call.
		 * If empty, the parameter is passed as is.
		 * Formatting elements:
		 * %0 or %s: name of the parameter
		 */
		String cs_in;

		/**
		 * One or more statements that determine how a variable of this type is returned from a method.
		 * It must contain the return statement(s).
		 * Formatting elements:
		 * %0 or %s: name of the variable to be returned
		 */
		String cs_out;

		/**
		 * Type used for method signatures, both for parameters and the return type.
		 * Same as [proxy_name] except for variable arguments (VarArg).
		 */
		String cs_type;

		/**
		 * Type used for parameters of internal call methods.
		 */
		String im_type_in;

		/**
		 * Type used for the return type of internal call methods.
		 * If [cs_out] is not empty and the method return type is not void,
		 * it is also used for the type of the return variable.
		 */
		String im_type_out;

		const DocData::ClassDoc *class_doc;

		List<MethodInterface> methods;

		const MethodInterface *find_method_by_name(const String &p_name) const {

			for (const List<MethodInterface>::Element *E = methods.front(); E; E = E->next()) {
				if (E->get().name == p_name)
					return &E->get();
			}

			return NULL;
		}

		static TypeInterface create_value_type(const String &p_name) {
			TypeInterface itype;

			itype.name = p_name;
			itype.proxy_name = p_name;

			itype.c_type = itype.name;
			itype.c_type_in = "void*";
			itype.c_type_out = "MonoObject*";
			itype.cs_type = itype.proxy_name;
			itype.im_type_in = "ref " + itype.proxy_name;
			itype.im_type_out = itype.proxy_name;
			itype.class_doc = &EditorHelp::get_doc_data()->class_list[itype.proxy_name];

			return itype;
		}

		static TypeInterface create_object_type(const String &p_name, ClassDB::APIType p_api_type) {
			TypeInterface itype;

			itype.name = p_name;
			itype.proxy_name = p_name.begins_with("_") ? p_name.substr(1, p_name.length()) : p_name;
			itype.api_type = p_api_type;
			itype.is_object_type = true;
			itype.class_doc = &EditorHelp::get_doc_data()->class_list[itype.proxy_name];

			return itype;
		}

		static void create_placeholder_type(TypeInterface &r_itype, const String &p_name) {
			r_itype.name = p_name;
			r_itype.proxy_name = p_name;

			r_itype.c_type = r_itype.name;
			r_itype.c_type_in = "MonoObject*";
			r_itype.c_type_out = "MonoObject*";
			r_itype.cs_type = r_itype.proxy_name;
			r_itype.im_type_in = r_itype.proxy_name;
			r_itype.im_type_out = r_itype.proxy_name;
		}

		TypeInterface() {

			api_type = ClassDB::API_NONE;

			is_object_type = false;
			is_singleton = false;
			is_reference = false;
			is_instantiable = false;

			memory_own = false;
			requires_collections = false;

			c_arg_in = "%s";

			class_doc = NULL;
		}
	};

	struct InternalCall {
		String name;
		String im_type_out; // Return type for the C# method declaration. Also used as companion of [unique_siq]
		String im_sig; // Signature for the C# method declaration
		String unique_sig; // Unique signature to avoid duplicates in containers
		bool editor_only;

		InternalCall() {}

		InternalCall(const String &p_name, const String &p_im_type_out, const String &p_im_sig = String(), const String &p_unique_sig = String()) {
			name = p_name;
			im_type_out = p_im_type_out;
			im_sig = p_im_sig;
			unique_sig = p_unique_sig;
			editor_only = false;
		}

		InternalCall(ClassDB::APIType api_type, const String &p_name, const String &p_im_type_out, const String &p_im_sig = String(), const String &p_unique_sig = String()) {
			name = p_name;
			im_type_out = p_im_type_out;
			im_sig = p_im_sig;
			unique_sig = p_unique_sig;
			editor_only = api_type == ClassDB::API_EDITOR;
		}

		inline bool operator==(const InternalCall &p_a) const {
			return p_a.unique_sig == unique_sig;
		}
	};

	static bool verbose_output;

	Map<String, TypeInterface> placeholder_types;
	Map<String, TypeInterface> builtin_types;
	Map<String, TypeInterface> obj_types;

	Map<String, String> extra_members;

	List<InternalCall> method_icalls;
	Map<const MethodInterface *, const InternalCall *> method_icalls_map;

	List<const InternalCall *> generated_icall_funcs;

	List<InternalCall> core_custom_icalls;
	List<InternalCall> editor_custom_icalls;

	const List<InternalCall>::Element *find_icall_by_name(const String &p_name, const List<InternalCall> &p_list) {

		const List<InternalCall>::Element *it = p_list.front();
		while (it) {
			if (it->get().name == p_name) return it;
			it = it->next();
		}
		return NULL;
	}

	inline String get_unique_sig(const TypeInterface &p_type) {
		if (p_type.is_reference)
			return "Ref";
		else if (p_type.is_object_type)
			return "Obj";

		return p_type.name;
	}

	void _generate_header_icalls();
	void _generate_method_icalls(const TypeInterface &p_itype);

	const TypeInterface *_get_type_by_name_or_null(const String &p_name);
	const TypeInterface *_get_type_by_name_or_placeholder(const String &p_name);

	void _default_argument_from_variant(const Variant &p_var, ArgumentInterface &r_iarg);
	void _populate_builtin_type(TypeInterface &r_type, Variant::Type vtype);

	void _populate_object_type_interfaces();
	void _populate_builtin_type_interfaces();

	Error _generate_cs_type(const TypeInterface &itype, const String &p_output_file);

	Error _generate_cs_property(const TypeInterface &p_itype, const DocData::PropertyDoc &p_prop_doc, List<String> &p_output);
	Error _generate_cs_method(const TypeInterface &p_itype, const MethodInterface &p_imethod, int &p_method_bind_count, List<String> &p_output);

	Error _generate_glue_method(const TypeInterface &p_itype, const MethodInterface &p_imethod, List<String> &p_output);

	Error _save_file(const String &path, const List<String> &content);

	BindingsGenerator();

	BindingsGenerator(const BindingsGenerator &);
	BindingsGenerator &operator=(const BindingsGenerator &);

public:
	Error generate_cs_core_project(const String &p_output_dir, bool p_verbose_output = true);
	Error generate_cs_editor_project(const String &p_output_dir, const String &p_core_dll_path, bool p_verbose_output = true);
	Error generate_glue(const String &p_output_dir);

	static BindingsGenerator &get_singleton() {
		static BindingsGenerator singleton;
		return singleton;
	}

	static void handle_cmdline_args(const List<String> &p_cmdline_args);
};

#endif

#endif // BINDINGS_GENERATOR_H
