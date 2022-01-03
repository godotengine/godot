/*************************************************************************/
/*  bindings_generator.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/doc_data.h"
#include "core/object/class_db.h"
#include "core/string/string_builder.h"
#include "editor/doc_tools.h"
#include "editor/editor_help.h"

#if defined(DEBUG_METHODS_ENABLED) && defined(TOOLS_ENABLED)

#include "core/string/ustring.h"

class BindingsGenerator {
	struct ConstantInterface {
		String name;
		String proxy_name;
		int value = 0;
		const DocData::ConstantDoc *const_doc;

		ConstantInterface() {}

		ConstantInterface(const String &p_name, const String &p_proxy_name, int p_value) {
			name = p_name;
			proxy_name = p_proxy_name;
			value = p_value;
		}
	};

	struct EnumInterface {
		StringName cname;
		List<ConstantInterface> constants;

		_FORCE_INLINE_ bool operator==(const EnumInterface &p_ienum) const {
			return p_ienum.cname == cname;
		}

		EnumInterface() {}

		EnumInterface(const StringName &p_cname) {
			cname = p_cname;
		}
	};

	struct PropertyInterface {
		StringName cname;
		String proxy_name;
		int index = 0;

		StringName setter;
		StringName getter;

		const DocData::PropertyDoc *prop_doc;
	};

	struct TypeReference {
		StringName cname;
		bool is_enum = false;

		TypeReference() {}

		TypeReference(const StringName &p_cname) :
				cname(p_cname) {}
	};

	struct ArgumentInterface {
		enum DefaultParamMode {
			CONSTANT,
			NULLABLE_VAL,
			NULLABLE_REF
		};

		TypeReference type;

		String name;

		Variant def_param_value;
		DefaultParamMode def_param_mode = CONSTANT;

		/**
		 * Determines the expression for the parameter default value.
		 * Formatting elements:
		 * %0 or %s: [cs_type] of the argument type
		 */
		String default_argument;

		ArgumentInterface() {}
	};

	struct MethodInterface {
		String name;
		StringName cname;

		/**
		 * Name of the C# method
		 */
		String proxy_name;

		/**
		 * [TypeInterface::name] of the return type
		 */
		TypeReference return_type;

		/**
		 * Determines if the method has a variable number of arguments (VarArg)
		 */
		bool is_vararg = false;

		/**
		 * Virtual methods ("virtual" as defined by the Godot API) are methods that by default do nothing,
		 * but can be overridden by the user to add custom functionality.
		 * e.g.: _ready, _process, etc.
		 */
		bool is_virtual = false;

		/**
		 * Determines if the call should fallback to Godot's object.Call(string, params) in C#.
		 */
		bool requires_object_call = false;

		/**
		 * Determines if the method visibility is 'internal' (visible only to files in the same assembly).
		 * Currently, we only use this for methods that are not meant to be exposed,
		 * but are required by properties as getters or setters.
		 * Methods that are not meant to be exposed are those that begin with underscore and are not virtual.
		 */
		bool is_internal = false;

		List<ArgumentInterface> arguments;

		const DocData::MethodDoc *method_doc = nullptr;

		bool is_deprecated = false;
		String deprecation_message;

		void add_argument(const ArgumentInterface &argument) {
			arguments.push_back(argument);
		}

		MethodInterface() {}
	};

	struct SignalInterface {
		String name;
		StringName cname;

		/**
		 * Name of the C# method
		 */
		String proxy_name;

		List<ArgumentInterface> arguments;

		const DocData::MethodDoc *method_doc = nullptr;

		bool is_deprecated = false;
		String deprecation_message;

		void add_argument(const ArgumentInterface &argument) {
			arguments.push_back(argument);
		}

		SignalInterface() {}
	};

	struct TypeInterface {
		/**
		 * Identifier name for this type.
		 * Also used to format [c_out].
		 */
		String name;
		StringName cname;

		/**
		 * Identifier name of the base class.
		 */
		StringName base_name;

		/**
		 * Name of the C# class
		 */
		String proxy_name;

		ClassDB::APIType api_type = ClassDB::API_NONE;

		bool is_enum = false;
		bool is_object_type = false;
		bool is_singleton = false;
		bool is_ref_counted = false;

		/**
		 * Used only by Object-derived types.
		 * Determines if this type is not abstract (incomplete).
		 * e.g.: CanvasItem cannot be instantiated.
		 */
		bool is_instantiable = false;

		/**
		 * Used only by Object-derived types.
		 * Determines if the C# class owns the native handle and must free it somehow when disposed.
		 * e.g.: RefCounted types must notify when the C# instance is disposed, for proper refcounting.
		 */
		bool memory_own = false;

		/**
		 * This must be set to true for any struct bigger than 32-bits. Those cannot be passed/returned by value
		 * with internal calls, so we must use pointers instead. Returns must be replace with out parameters.
		 * In this case, [c_out] and [cs_out] must have a different format, explained below.
		 * The Mono IL interpreter icall trampolines don't support passing structs bigger than 32-bits by value (at least not on WASM).
		 */
		bool ret_as_byref_arg = false;

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
		 * Determines the expression that will be passed as argument to ptrcall.
		 * By default the value equals the name of the parameter,
		 * this varies for types that require special manipulation via [c_in].
		 * Formatting elements:
		 * %0 or %s: name of the parameter
		 */
		String c_arg_in = "%s";

		/**
		 * One or more statements that determine how a variable of this type is returned from a function.
		 * It must contain the return statement(s).
		 * Formatting elements:
		 * %0: [c_type_out] of the return type
		 * %1: name of the variable to be returned
		 * %2: [name] of the return type
		 * ---------------------------------------
		 * If [ret_as_byref_arg] is true, the format is different. Instead of using a return statement,
		 * the value must be assigned to a parameter. This type of this parameter is a pointer to [c_type_out].
		 * Formatting elements:
		 * %0: [c_type_out] of the return type
		 * %1: name of the variable to be returned
		 * %2: [name] of the return type
		 * %3: name of the parameter that must be assigned the return value
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
		 * RefCounted types override this for the type of the return variable: Ref<RefCounted>
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
		 * %0: internal method name
		 * %1: internal method call arguments without surrounding parenthesis
		 * %2: [cs_type] of the return type
		 * %3: [im_type_out] of the return type
		 */
		String cs_out;

		/**
		 * Type used for method signatures, both for parameters and the return type.
		 * Same as [proxy_name] except for variable arguments (VarArg) and collections (which include the namespace).
		 */
		String cs_type;

		/**
		 * Type used for parameters of internal call methods.
		 */
		String im_type_in;

		/**
		 * Type used for the return type of internal call methods.
		 */
		String im_type_out;

		const DocData::ClassDoc *class_doc = nullptr;

		List<ConstantInterface> constants;
		List<EnumInterface> enums;
		List<PropertyInterface> properties;
		List<MethodInterface> methods;
		List<SignalInterface> signals_;

		const MethodInterface *find_method_by_name(const StringName &p_cname) const {
			for (const MethodInterface &E : methods) {
				if (E.cname == p_cname) {
					return &E;
				}
			}

			return nullptr;
		}

		const PropertyInterface *find_property_by_name(const StringName &p_cname) const {
			for (const PropertyInterface &E : properties) {
				if (E.cname == p_cname) {
					return &E;
				}
			}

			return nullptr;
		}

		const PropertyInterface *find_property_by_proxy_name(const String &p_proxy_name) const {
			for (const PropertyInterface &E : properties) {
				if (E.proxy_name == p_proxy_name) {
					return &E;
				}
			}

			return nullptr;
		}

		const MethodInterface *find_method_by_proxy_name(const String &p_proxy_name) const {
			for (const MethodInterface &E : methods) {
				if (E.proxy_name == p_proxy_name) {
					return &E;
				}
			}

			return nullptr;
		}

	private:
		static void _init_value_type(TypeInterface &itype) {
			itype.proxy_name = itype.name;

			itype.c_type = itype.name;
			itype.cs_type = itype.proxy_name;
			itype.im_type_in = "ref " + itype.proxy_name;
			itype.im_type_out = itype.proxy_name;
			itype.class_doc = &EditorHelp::get_doc_data()->class_list[itype.proxy_name];
		}

	public:
		static TypeInterface create_value_type(const String &p_name) {
			TypeInterface itype;
			itype.name = p_name;
			itype.cname = StringName(p_name);
			_init_value_type(itype);
			return itype;
		}

		static TypeInterface create_value_type(const StringName &p_name) {
			TypeInterface itype;
			itype.name = p_name.operator String();
			itype.cname = p_name;
			_init_value_type(itype);
			return itype;
		}

		static TypeInterface create_object_type(const StringName &p_cname, ClassDB::APIType p_api_type) {
			TypeInterface itype;

			itype.name = p_cname;
			itype.cname = p_cname;
			itype.proxy_name = itype.name.begins_with("_") ? itype.name.substr(1, itype.name.length()) : itype.name;
			itype.api_type = p_api_type;
			itype.is_object_type = true;
			itype.class_doc = &EditorHelp::get_doc_data()->class_list[itype.proxy_name];

			return itype;
		}

		static void create_placeholder_type(TypeInterface &r_itype, const StringName &p_cname) {
			r_itype.name = p_cname;
			r_itype.cname = p_cname;
			r_itype.proxy_name = r_itype.name;

			r_itype.c_type = r_itype.name;
			r_itype.c_type_in = "MonoObject*";
			r_itype.c_type_out = "MonoObject*";
			r_itype.cs_type = r_itype.proxy_name;
			r_itype.im_type_in = r_itype.proxy_name;
			r_itype.im_type_out = r_itype.proxy_name;
		}

		static void postsetup_enum_type(TypeInterface &r_enum_itype) {
			// C interface for enums is the same as that of 'uint32_t'. Remember to apply
			// any of the changes done here to the 'uint32_t' type interface as well.

			r_enum_itype.c_arg_in = "&%s_in";
			{
				// The expected types for parameters and return value in ptrcall are 'int64_t' or 'uint64_t'.
				r_enum_itype.c_in = "\t%0 %1_in = (%0)%1;\n";
				r_enum_itype.c_out = "\treturn (%0)%1;\n";
				r_enum_itype.c_type = "int64_t";
			}
			r_enum_itype.c_type_in = "int32_t";
			r_enum_itype.c_type_out = r_enum_itype.c_type_in;

			r_enum_itype.cs_type = r_enum_itype.proxy_name;
			r_enum_itype.cs_in = "(int)%s";
			r_enum_itype.cs_out = "return (%2)%0(%1);";
			r_enum_itype.im_type_in = "int";
			r_enum_itype.im_type_out = "int";
			r_enum_itype.class_doc = &EditorHelp::get_doc_data()->class_list[r_enum_itype.proxy_name];
		}

		TypeInterface() {}
	};

	struct InternalCall {
		String name;
		String im_type_out; // Return type for the C# method declaration. Also used as companion of [unique_siq]
		String im_sig; // Signature for the C# method declaration
		String unique_sig; // Unique signature to avoid duplicates in containers
		bool editor_only = false;

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

	bool log_print_enabled = true;
	bool initialized = false;

	OrderedHashMap<StringName, TypeInterface> obj_types;

	Map<StringName, TypeInterface> placeholder_types;
	Map<StringName, TypeInterface> builtin_types;
	Map<StringName, TypeInterface> enum_types;

	List<EnumInterface> global_enums;
	List<ConstantInterface> global_constants;

	List<InternalCall> method_icalls;
	Map<const MethodInterface *, const InternalCall *> method_icalls_map;

	List<const InternalCall *> generated_icall_funcs;

	List<InternalCall> core_custom_icalls;
	List<InternalCall> editor_custom_icalls;

	Map<StringName, List<StringName>> blacklisted_methods;

	void _initialize_blacklisted_methods();

	struct NameCache {
		StringName type_void = StaticCString::create("void");
		StringName type_Variant = StaticCString::create("Variant");
		StringName type_VarArg = StaticCString::create("VarArg");
		StringName type_Object = StaticCString::create("Object");
		StringName type_RefCounted = StaticCString::create("RefCounted");
		StringName type_RID = StaticCString::create("RID");
		StringName type_Callable = StaticCString::create("Callable");
		StringName type_Signal = StaticCString::create("Signal");
		StringName type_String = StaticCString::create("String");
		StringName type_StringName = StaticCString::create("StringName");
		StringName type_NodePath = StaticCString::create("NodePath");
		StringName type_at_GlobalScope = StaticCString::create("@GlobalScope");
		StringName enum_Error = StaticCString::create("Error");

		StringName type_sbyte = StaticCString::create("sbyte");
		StringName type_short = StaticCString::create("short");
		StringName type_int = StaticCString::create("int");
		StringName type_byte = StaticCString::create("byte");
		StringName type_ushort = StaticCString::create("ushort");
		StringName type_uint = StaticCString::create("uint");
		StringName type_long = StaticCString::create("long");
		StringName type_ulong = StaticCString::create("ulong");

		StringName type_bool = StaticCString::create("bool");
		StringName type_float = StaticCString::create("float");
		StringName type_double = StaticCString::create("double");

		StringName type_Vector2 = StaticCString::create("Vector2");
		StringName type_Rect2 = StaticCString::create("Rect2");
		StringName type_Vector3 = StaticCString::create("Vector3");

		// Object not included as it must be checked for all derived classes
		static constexpr int nullable_types_count = 17;
		StringName nullable_types[nullable_types_count] = {
			type_String,
			type_StringName,
			type_NodePath,

			StaticCString::create(_STR(Array)),
			StaticCString::create(_STR(Dictionary)),
			StaticCString::create(_STR(Callable)),
			StaticCString::create(_STR(Signal)),

			StaticCString::create(_STR(PackedByteArray)),
			StaticCString::create(_STR(PackedInt32Array)),
			StaticCString::create(_STR(PackedInt64Array)),
			StaticCString::create(_STR(PackedFloat32Array)),
			StaticCString::create(_STR(PackedFloat64Array)),
			StaticCString::create(_STR(PackedStringArray)),
			StaticCString::create(_STR(PackedVector2Array)),
			StaticCString::create(_STR(PackedVector3Array)),
			StaticCString::create(_STR(PackedColorArray)),
		};

		bool is_nullable_type(const StringName &p_type) const {
			for (int i = 0; i < nullable_types_count; i++) {
				if (p_type == nullable_types[i]) {
					return true;
				}
			}

			return false;
		}

		NameCache() {}

	private:
		NameCache(const NameCache &);
		void operator=(const NameCache &);
	};

	NameCache name_cache;

	const List<InternalCall>::Element *find_icall_by_name(const String &p_name, const List<InternalCall> &p_list) {
		const List<InternalCall>::Element *it = p_list.front();
		while (it) {
			if (it->get().name == p_name) {
				return it;
			}
			it = it->next();
		}
		return nullptr;
	}

	const ConstantInterface *find_constant_by_name(const String &p_name, const List<ConstantInterface> &p_constants) const {
		for (const ConstantInterface &E : p_constants) {
			if (E.name == p_name) {
				return &E;
			}
		}

		return nullptr;
	}

	inline String get_unique_sig(const TypeInterface &p_type) {
		if (p_type.is_ref_counted) {
			return "Ref";
		} else if (p_type.is_object_type) {
			return "Obj";
		} else if (p_type.is_enum) {
			return "int";
		}

		return p_type.name;
	}

	String bbcode_to_xml(const String &p_bbcode, const TypeInterface *p_itype);

	int _determine_enum_prefix(const EnumInterface &p_ienum);
	void _apply_prefix_to_enum_constants(EnumInterface &p_ienum, int p_prefix_length);

	void _generate_method_icalls(const TypeInterface &p_itype);

	const TypeInterface *_get_type_or_null(const TypeReference &p_typeref);
	const TypeInterface *_get_type_or_placeholder(const TypeReference &p_typeref);

	StringName _get_int_type_name_from_meta(GodotTypeInfo::Metadata p_meta);
	StringName _get_float_type_name_from_meta(GodotTypeInfo::Metadata p_meta);

	bool _arg_default_value_from_variant(const Variant &p_val, ArgumentInterface &r_iarg);
	bool _arg_default_value_is_assignable_to_type(const Variant &p_val, const TypeInterface &p_arg_type);

	bool _populate_object_type_interfaces();
	void _populate_builtin_type_interfaces();

	void _populate_global_constants();

	Error _generate_cs_type(const TypeInterface &itype, const String &p_output_file);

	Error _generate_cs_property(const TypeInterface &p_itype, const PropertyInterface &p_iprop, StringBuilder &p_output);
	Error _generate_cs_method(const TypeInterface &p_itype, const MethodInterface &p_imethod, int &p_method_bind_count, StringBuilder &p_output);
	Error _generate_cs_signal(const BindingsGenerator::TypeInterface &p_itype, const BindingsGenerator::SignalInterface &p_isignal, StringBuilder &p_output);

	void _generate_array_extensions(StringBuilder &p_output);
	void _generate_global_constants(StringBuilder &p_output);

	Error _generate_glue_method(const TypeInterface &p_itype, const MethodInterface &p_imethod, StringBuilder &p_output);

	Error _save_file(const String &p_path, const StringBuilder &p_content);

	void _log(const char *p_format, ...) _PRINTF_FORMAT_ATTRIBUTE_2_3;

	void _initialize();

public:
	Error generate_cs_core_project(const String &p_proj_dir);
	Error generate_cs_editor_project(const String &p_proj_dir);
	Error generate_cs_api(const String &p_output_dir);
	Error generate_glue(const String &p_output_dir);

	_FORCE_INLINE_ bool is_log_print_enabled() { return log_print_enabled; }
	_FORCE_INLINE_ void set_log_print_enabled(bool p_enabled) { log_print_enabled = p_enabled; }

	_FORCE_INLINE_ bool is_initialized() { return initialized; }

	static uint32_t get_version();

	static void handle_cmdline_args(const List<String> &p_cmdline_args);

	BindingsGenerator() {
		_initialize();
	}
};

#endif

#endif // BINDINGS_GENERATOR_H
