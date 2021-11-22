/*************************************************************************/
/*  class_db.h                                                           */
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

#ifndef CLASS_DB_H
#define CLASS_DB_H

#include "core/object/method_bind.h"
#include "core/object/object.h"
#include "core/string/print_string.h"

/** To bind more then 6 parameters include this:
 *
 */

// Makes callable_mp readily available in all classes connecting signals.
// Needs to come after method_bind and object have been included.
#include "core/object/callable_method_pointer.h"

#define DEFVAL(m_defval) (m_defval)

struct MethodDefinition {
	StringName name;
	Vector<StringName> args;
	MethodDefinition() {}
	MethodDefinition(const char *p_name) :
			name(p_name) {}
	MethodDefinition(const StringName &p_name) :
			name(p_name) {}
};

MethodDefinition D_METHOD(const char *p_name);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5, const char *p_arg6);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5, const char *p_arg6, const char *p_arg7);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5, const char *p_arg6, const char *p_arg7, const char *p_arg8);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5, const char *p_arg6, const char *p_arg7, const char *p_arg8, const char *p_arg9);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5, const char *p_arg6, const char *p_arg7, const char *p_arg8, const char *p_arg9, const char *p_arg10);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5, const char *p_arg6, const char *p_arg7, const char *p_arg8, const char *p_arg9, const char *p_arg10, const char *p_arg11);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5, const char *p_arg6, const char *p_arg7, const char *p_arg8, const char *p_arg9, const char *p_arg10, const char *p_arg11, const char *p_arg12);
MethodDefinition D_METHOD(const char *p_name, const char *p_arg1, const char *p_arg2, const char *p_arg3, const char *p_arg4, const char *p_arg5, const char *p_arg6, const char *p_arg7, const char *p_arg8, const char *p_arg9, const char *p_arg10, const char *p_arg11, const char *p_arg12, const char *p_arg13);

class ClassDB {
public:
	enum APIType {
		API_CORE,
		API_EDITOR,
		API_EXTENSION,
		API_EDITOR_EXTENSION,
		API_NONE
	};

public:
	struct PropertySetGet {
		int index;
		StringName setter;
		StringName getter;
		MethodBind *_setptr;
		MethodBind *_getptr;
		Variant::Type type;
	};

	struct ClassInfo {
		APIType api = API_NONE;
		ClassInfo *inherits_ptr = nullptr;
		void *class_ptr = nullptr;

		ObjectNativeExtension *native_extension = nullptr;

		HashMap<StringName, MethodBind *> method_map;
		HashMap<StringName, int> constant_map;
		HashMap<StringName, List<StringName>> enum_map;
		HashMap<StringName, MethodInfo> signal_map;
		List<PropertyInfo> property_list;
		HashMap<StringName, PropertyInfo> property_map;
#ifdef DEBUG_METHODS_ENABLED
		List<StringName> constant_order;
		List<StringName> method_order;
		Set<StringName> methods_in_properties;
		List<MethodInfo> virtual_methods;
		Map<StringName, MethodInfo> virtual_methods_map;
		StringName category;
		Map<StringName, Vector<Error>> method_error_values;
#endif
		HashMap<StringName, PropertySetGet> property_setget;

		StringName inherits;
		StringName name;
		bool disabled = false;
		bool exposed = false;
		Object *(*creation_func)() = nullptr;

		ClassInfo() {}
		~ClassInfo() {}
	};

	template <class T>
	static Object *creator() {
		return memnew(T);
	}

	static RWLock lock;
	static HashMap<StringName, ClassInfo> classes;
	static HashMap<StringName, StringName> resource_base_extensions;
	static HashMap<StringName, StringName> compat_classes;

	static MethodBind *bind_methodfi(uint32_t p_flags, MethodBind *p_bind, const MethodDefinition &method_name, const Variant **p_defs, int p_defcount);

	static APIType current_api;

	static void _add_class2(const StringName &p_class, const StringName &p_inherits);

	static HashMap<StringName, HashMap<StringName, Variant>> default_values;
	static Set<StringName> default_values_cached;

private:
	// Non-locking variants of get_parent_class and is_parent_class.
	static StringName _get_parent_class(const StringName &p_class);
	static bool _is_parent_class(const StringName &p_class, const StringName &p_inherits);

public:
	// DO NOT USE THIS!!!!!! NEEDS TO BE PUBLIC BUT DO NOT USE NO MATTER WHAT!!!
	template <class T>
	static void _add_class() {
		_add_class2(T::get_class_static(), T::get_parent_class_static());
	}

	template <class T>
	static void register_class() {
		GLOBAL_LOCK_FUNCTION;
		T::initialize_class();
		ClassInfo *t = classes.getptr(T::get_class_static());
		ERR_FAIL_COND(!t);
		t->creation_func = &creator<T>;
		t->exposed = true;
		t->class_ptr = T::get_class_ptr_static();
		t->api = current_api;
		T::register_custom_data_to_otdb();
	}

	template <class T>
	static void register_virtual_class() {
		GLOBAL_LOCK_FUNCTION;
		T::initialize_class();
		ClassInfo *t = classes.getptr(T::get_class_static());
		ERR_FAIL_COND(!t);
		t->exposed = true;
		t->class_ptr = T::get_class_ptr_static();
		t->api = current_api;
		//nothing
	}

	static void register_extension_class(ObjectNativeExtension *p_extension);
	static void unregister_extension_class(const StringName &p_class);

	template <class T>
	static Object *_create_ptr_func() {
		return T::create();
	}

	template <class T>
	static void register_custom_instance_class() {
		GLOBAL_LOCK_FUNCTION;
		T::initialize_class();
		ClassInfo *t = classes.getptr(T::get_class_static());
		ERR_FAIL_COND(!t);
		t->creation_func = &_create_ptr_func<T>;
		t->exposed = true;
		t->class_ptr = T::get_class_ptr_static();
		t->api = current_api;
		T::register_custom_data_to_otdb();
	}

	static void get_class_list(List<StringName> *p_classes);
	static void get_inheriters_from_class(const StringName &p_class, List<StringName> *p_classes);
	static void get_direct_inheriters_from_class(const StringName &p_class, List<StringName> *p_classes);
	static StringName get_parent_class_nocheck(const StringName &p_class);
	static StringName get_parent_class(const StringName &p_class);
	static StringName get_compatibility_remapped_class(const StringName &p_class);
	static bool class_exists(const StringName &p_class);
	static bool is_parent_class(const StringName &p_class, const StringName &p_inherits);
	static bool can_instantiate(const StringName &p_class);
	static Object *instantiate(const StringName &p_class);
	static Object *construct_object(Object *(*p_create_func)(), ObjectNativeExtension *p_extension);
	static void instance_get_native_extension_data(ObjectNativeExtension **r_extension, GDExtensionClassInstancePtr *r_extension_instance, Object *p_base);

	static APIType get_api_type(const StringName &p_class);

	static uint64_t get_api_hash(APIType p_api);

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method) {
		MethodBind *bind = create_method_bind(p_method);

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, nullptr, 0); //use static function, much smaller binary usage
	}

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method, const Variant &p_def1) {
		MethodBind *bind = create_method_bind(p_method);
		const Variant *ptr[1] = { &p_def1 };

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, ptr, 1);
	}

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method, const Variant &p_def1, const Variant &p_def2) {
		MethodBind *bind = create_method_bind(p_method);
		const Variant *ptr[2] = { &p_def1, &p_def2 };

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, ptr, 2);
	}

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method, const Variant &p_def1, const Variant &p_def2, const Variant &p_def3) {
		MethodBind *bind = create_method_bind(p_method);
		const Variant *ptr[3] = { &p_def1, &p_def2, &p_def3 };

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, ptr, 3);
	}

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method, const Variant &p_def1, const Variant &p_def2, const Variant &p_def3, const Variant &p_def4) {
		MethodBind *bind = create_method_bind(p_method);
		const Variant *ptr[4] = { &p_def1, &p_def2, &p_def3, &p_def4 };

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, ptr, 4);
	}

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method, const Variant &p_def1, const Variant &p_def2, const Variant &p_def3, const Variant &p_def4, const Variant &p_def5) {
		MethodBind *bind = create_method_bind(p_method);
		const Variant *ptr[5] = { &p_def1, &p_def2, &p_def3, &p_def4, &p_def5 };

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, ptr, 5);
	}

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method, const Variant &p_def1, const Variant &p_def2, const Variant &p_def3, const Variant &p_def4, const Variant &p_def5, const Variant &p_def6) {
		MethodBind *bind = create_method_bind(p_method);
		const Variant *ptr[6] = { &p_def1, &p_def2, &p_def3, &p_def4, &p_def5, &p_def6 };

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, ptr, 6);
	}

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method, const Variant &p_def1, const Variant &p_def2, const Variant &p_def3, const Variant &p_def4, const Variant &p_def5, const Variant &p_def6, const Variant &p_def7) {
		MethodBind *bind = create_method_bind(p_method);
		const Variant *ptr[7] = { &p_def1, &p_def2, &p_def3, &p_def4, &p_def5, &p_def6, &p_def7 };

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, ptr, 7);
	}

	template <class N, class M>
	static MethodBind *bind_method(N p_method_name, M p_method, const Variant &p_def1, const Variant &p_def2, const Variant &p_def3, const Variant &p_def4, const Variant &p_def5, const Variant &p_def6, const Variant &p_def7, const Variant &p_def8) {
		MethodBind *bind = create_method_bind(p_method);
		const Variant *ptr[8] = { &p_def1, &p_def2, &p_def3, &p_def4, &p_def5, &p_def6, &p_def7, &p_def8 };

		return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, ptr, 8);
	}

	template <class M>
	static MethodBind *bind_vararg_method(uint32_t p_flags, const StringName &p_name, M p_method, const MethodInfo &p_info = MethodInfo(), const Vector<Variant> &p_default_args = Vector<Variant>(), bool p_return_nil_is_variant = true) {
		GLOBAL_LOCK_FUNCTION;

		MethodBind *bind = create_vararg_method_bind(p_method, p_info, p_return_nil_is_variant);
		ERR_FAIL_COND_V(!bind, nullptr);

		bind->set_name(p_name);
		bind->set_default_arguments(p_default_args);

		String instance_type = bind->get_instance_class();

		ClassInfo *type = classes.getptr(instance_type);
		if (!type) {
			memdelete(bind);
			ERR_FAIL_COND_V(!type, nullptr);
		}

		if (type->method_map.has(p_name)) {
			memdelete(bind);
			// Overloading not supported
			ERR_FAIL_V_MSG(nullptr, "Method already bound: " + instance_type + "::" + p_name + ".");
		}
		type->method_map[p_name] = bind;
#ifdef DEBUG_METHODS_ENABLED
		// FIXME: <reduz> set_return_type is no longer in MethodBind, so I guess it should be moved to vararg method bind
		//bind->set_return_type("Variant");
		type->method_order.push_back(p_name);
#endif

		return bind;
	}

	static void bind_method_custom(const StringName &p_class, MethodBind *p_method);

	static void add_signal(const StringName &p_class, const MethodInfo &p_signal);
	static bool has_signal(const StringName &p_class, const StringName &p_signal, bool p_no_inheritance = false);
	static bool get_signal(const StringName &p_class, const StringName &p_signal, MethodInfo *r_signal);
	static void get_signal_list(const StringName &p_class, List<MethodInfo> *p_signals, bool p_no_inheritance = false);

	static void add_property_group(const StringName &p_class, const String &p_name, const String &p_prefix = "");
	static void add_property_subgroup(const StringName &p_class, const String &p_name, const String &p_prefix = "");
	static void add_property_array_count(const StringName &p_class, const String &p_label, const StringName &p_count_property, const StringName &p_count_setter, const StringName &p_count_getter, const String &p_array_element_prefix, uint32_t p_count_usage = PROPERTY_USAGE_DEFAULT);
	static void add_property_array(const StringName &p_class, const StringName &p_path, const String &p_array_element_prefix);
	static void add_property(const StringName &p_class, const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter, int p_index = -1);
	static void set_property_default_value(const StringName &p_class, const StringName &p_name, const Variant &p_default);
	static void add_linked_property(const StringName &p_class, const String &p_property, const String &p_linked_property);
	static void get_property_list(const StringName &p_class, List<PropertyInfo> *p_list, bool p_no_inheritance = false, const Object *p_validator = nullptr);
	static bool get_property_info(const StringName &p_class, const StringName &p_property, PropertyInfo *r_info, bool p_no_inheritance = false, const Object *p_validator = nullptr);
	static bool set_property(Object *p_object, const StringName &p_property, const Variant &p_value, bool *r_valid = nullptr);
	static bool get_property(Object *p_object, const StringName &p_property, Variant &r_value);
	static bool has_property(const StringName &p_class, const StringName &p_property, bool p_no_inheritance = false);
	static int get_property_index(const StringName &p_class, const StringName &p_property, bool *r_is_valid = nullptr);
	static Variant::Type get_property_type(const StringName &p_class, const StringName &p_property, bool *r_is_valid = nullptr);
	static StringName get_property_setter(const StringName &p_class, const StringName &p_property);
	static StringName get_property_getter(const StringName &p_class, const StringName &p_property);

	static bool has_method(const StringName &p_class, const StringName &p_method, bool p_no_inheritance = false);
	static void set_method_flags(const StringName &p_class, const StringName &p_method, int p_flags);

	static void get_method_list(const StringName &p_class, List<MethodInfo> *p_methods, bool p_no_inheritance = false, bool p_exclude_from_properties = false);
	static bool get_method_info(const StringName &p_class, const StringName &p_method, MethodInfo *r_info, bool p_no_inheritance = false, bool p_exclude_from_properties = false);
	static MethodBind *get_method(const StringName &p_class, const StringName &p_name);

	static void add_virtual_method(const StringName &p_class, const MethodInfo &p_method, bool p_virtual = true, const Vector<String> &p_arg_names = Vector<String>(), bool p_object_core = false);
	static void get_virtual_methods(const StringName &p_class, List<MethodInfo> *p_methods, bool p_no_inheritance = false);

	static void bind_integer_constant(const StringName &p_class, const StringName &p_enum, const StringName &p_name, int p_constant);
	static void get_integer_constant_list(const StringName &p_class, List<String> *p_constants, bool p_no_inheritance = false);
	static int get_integer_constant(const StringName &p_class, const StringName &p_name, bool *p_success = nullptr);
	static bool has_integer_constant(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false);

	static StringName get_integer_constant_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false);
	static void get_enum_list(const StringName &p_class, List<StringName> *p_enums, bool p_no_inheritance = false);
	static void get_enum_constants(const StringName &p_class, const StringName &p_enum, List<StringName> *p_constants, bool p_no_inheritance = false);
	static bool has_enum(const StringName &p_class, const StringName &p_name, bool p_no_inheritance = false);

	static void set_method_error_return_values(const StringName &p_class, const StringName &p_method, const Vector<Error> &p_values);
	static Vector<Error> get_method_error_return_values(const StringName &p_class, const StringName &p_method);
	static Variant class_get_default_property_value(const StringName &p_class, const StringName &p_property, bool *r_valid = nullptr);

	static StringName get_category(const StringName &p_node);

	static void set_class_enabled(const StringName &p_class, bool p_enable);
	static bool is_class_enabled(const StringName &p_class);

	static bool is_class_exposed(const StringName &p_class);

	static void add_resource_base_extension(const StringName &p_extension, const StringName &p_class);
	static void get_resource_base_extensions(List<String> *p_extensions);
	static void get_extensions_for_type(const StringName &p_class, List<String> *p_extensions);
	static bool is_resource_extension(const StringName &p_extension);

	static void add_compatibility_class(const StringName &p_class, const StringName &p_fallback);

	static void set_current_api(APIType p_api);
	static APIType get_current_api();
	static void cleanup_defaults();
	static void cleanup();
};

#ifdef DEBUG_METHODS_ENABLED

#define BIND_CONSTANT(m_constant) \
	::ClassDB::bind_integer_constant(get_class_static(), StringName(), #m_constant, m_constant);

#define BIND_ENUM_CONSTANT(m_constant) \
	::ClassDB::bind_integer_constant(get_class_static(), __constant_get_enum_name(m_constant, #m_constant), #m_constant, m_constant);

_FORCE_INLINE_ void errarray_add_str(Vector<Error> &arr) {
}

_FORCE_INLINE_ void errarray_add_str(Vector<Error> &arr, const Error &p_err) {
	arr.push_back(p_err);
}

template <class... P>
_FORCE_INLINE_ void errarray_add_str(Vector<Error> &arr, const Error &p_err, P... p_args) {
	arr.push_back(p_err);
	errarray_add_str(arr, p_args...);
}

template <class... P>
_FORCE_INLINE_ Vector<Error> errarray(P... p_args) {
	Vector<Error> arr;
	errarray_add_str(arr, p_args...);
	return arr;
}

#define BIND_METHOD_ERR_RETURN_DOC(m_method, ...) \
	::ClassDB::set_method_error_return_values(get_class_static(), m_method, errarray(__VA_ARGS__));

#else

#define BIND_CONSTANT(m_constant) \
	::ClassDB::bind_integer_constant(get_class_static(), StringName(), #m_constant, m_constant);

#define BIND_ENUM_CONSTANT(m_constant) \
	::ClassDB::bind_integer_constant(get_class_static(), StringName(), #m_constant, m_constant);

#define BIND_METHOD_ERR_RETURN_DOC(m_method, ...)

#endif

#define GDREGISTER_CLASS(m_class)                    \
	if (!GD_IS_DEFINED(ClassDB_Disable_##m_class)) { \
		::ClassDB::register_class<m_class>();        \
	}
#define GDREGISTER_VIRTUAL_CLASS(m_class)             \
	if (!GD_IS_DEFINED(ClassDB_Disable_##m_class)) {  \
		::ClassDB::register_virtual_class<m_class>(); \
	}

#include "core/disabled_classes.gen.h"

#endif // CLASS_DB_H
