/**************************************************************************/
/*  class_db.hpp                                                          */
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

#include <gdextension_interface.h>

#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/method_bind.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/core/print_string.hpp>

#include <godot_cpp/classes/class_db_singleton.hpp>

// Makes callable_mp readily available in all classes connecting signals.
// Needs to come after method_bind and object have been included.
#include <godot_cpp/variant/callable_method_pointer.hpp>

#include <list>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Needed to use StringName as key in `std::unordered_map`
template <>
struct std::hash<godot::StringName> {
	std::size_t operator()(godot::StringName const &s) const noexcept {
		return s.hash();
	}
};

namespace godot {

#define DEFVAL(m_defval) (m_defval)

struct MethodDefinition {
	StringName name;
	std::list<StringName> args;
	MethodDefinition() {}
	MethodDefinition(StringName p_name) :
			name(p_name) {}
};

MethodDefinition D_METHOD(StringName p_name);
MethodDefinition D_METHOD(StringName p_name, StringName p_arg1);
template <typename... Args>
MethodDefinition D_METHOD(StringName p_name, StringName p_arg1, Args... args) {
	MethodDefinition md = D_METHOD(p_name, args...);
	md.args.push_front(p_arg1);
	return md;
}

class ClassDB {
	static GDExtensionInitializationLevel current_level;

	friend class godot::GDExtensionBinding;

public:
	struct ClassInfo {
		struct VirtualMethod {
			GDExtensionClassCallVirtual func;
			uint32_t hash;
		};

		StringName name;
		StringName parent_name;
		GDExtensionInitializationLevel level = GDEXTENSION_INITIALIZATION_SCENE;
		std::unordered_map<StringName, MethodBind *> method_map;
		std::set<StringName> signal_names;
		std::unordered_map<StringName, VirtualMethod> virtual_methods;
		std::set<StringName> property_names;
		std::set<StringName> constant_names;
		// Pointer to the parent custom class, if any. Will be null if the parent class is a Godot class.
		ClassInfo *parent_ptr = nullptr;
	};

private:
	// This may only contain custom classes, not Godot classes
	static std::unordered_map<StringName, ClassInfo> classes;
	static std::unordered_map<StringName, const GDExtensionInstanceBindingCallbacks *> instance_binding_callbacks;
	// Used to remember the custom class registration order.
	static std::vector<StringName> class_register_order;
	static std::unordered_map<StringName, Object *> engine_singletons;
	static std::mutex engine_singletons_mutex;

	static MethodBind *bind_methodfi(uint32_t p_flags, MethodBind *p_bind, const MethodDefinition &method_name, const void **p_defs, int p_defcount);
	static void initialize_class(const ClassInfo &cl);
	static void bind_method_godot(const StringName &p_class_name, MethodBind *p_method);

	template <typename T, bool is_abstract>
	static void _register_class(bool p_virtual = false, bool p_exposed = true, bool p_runtime = false);

	template <typename T>
	static GDExtensionObjectPtr _create_instance_func(void *data, GDExtensionBool p_notify_postinitialize) {
		if constexpr (!std::is_abstract_v<T>) {
			Wrapped::_set_construct_info<T>();
			T *new_object = new ("", "") T;
			if (p_notify_postinitialize) {
				new_object->_postinitialize();
			}
			return new_object->_owner;
		} else {
			return nullptr;
		}
	}

	template <typename T>
	static GDExtensionClassInstancePtr _recreate_instance_func(void *data, GDExtensionObjectPtr obj) {
		if constexpr (!std::is_abstract_v<T>) {
#ifdef HOT_RELOAD_ENABLED
#ifdef _GODOT_CPP_AVOID_THREAD_LOCAL
			std::lock_guard<std::recursive_mutex> lk(Wrapped::_constructing_mutex);
#endif
			Wrapped::_constructing_recreate_owner = obj;
			T *new_instance = (T *)memalloc(sizeof(T));
			memnew_placement(new_instance, T);
			return new_instance;
#else
			return nullptr;
#endif
		} else {
			return nullptr;
		}
	}

public:
	template <typename T>
	static void register_class(bool p_virtual = false);
	template <typename T>
	static void register_abstract_class();
	template <typename T>
	static void register_internal_class();
	template <typename T>
	static void register_runtime_class();

	_FORCE_INLINE_ static void _register_engine_class(const StringName &p_name, const GDExtensionInstanceBindingCallbacks *p_callbacks) {
		instance_binding_callbacks[p_name] = p_callbacks;
	}

	static void _editor_get_classes_used_callback(GDExtensionTypePtr p_packed_string_array);

	static void _register_engine_singleton(const StringName &p_class_name, Object *p_singleton) {
		std::lock_guard<std::mutex> lock(engine_singletons_mutex);
		std::unordered_map<StringName, Object *>::const_iterator i = engine_singletons.find(p_class_name);
		if (i != engine_singletons.end()) {
			ERR_FAIL_COND((*i).second != p_singleton);
			return;
		}
		engine_singletons[p_class_name] = p_singleton;
	}

	static void _unregister_engine_singleton(const StringName &p_class_name) {
		std::lock_guard<std::mutex> lock(engine_singletons_mutex);
		engine_singletons.erase(p_class_name);
	}

	template <typename N, typename M, typename... VarArgs>
	static MethodBind *bind_method(N p_method_name, M p_method, VarArgs... p_args);

	template <typename N, typename M, typename... VarArgs>
	static MethodBind *bind_static_method(StringName p_class, N p_method_name, M p_method, VarArgs... p_args);

	template <typename M>
	static MethodBind *bind_vararg_method(uint32_t p_flags, StringName p_name, M p_method, const MethodInfo &p_info = MethodInfo(), const std::vector<Variant> &p_default_args = std::vector<Variant>{}, bool p_return_nil_is_variant = true);

	static void add_property_group(const StringName &p_class, const String &p_name, const String &p_prefix);
	static void add_property_subgroup(const StringName &p_class, const String &p_name, const String &p_prefix);
	static void add_property(const StringName &p_class, const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter, int p_index = -1);
	static void add_signal(const StringName &p_class, const MethodInfo &p_signal);
	static void bind_integer_constant(const StringName &p_class_name, const StringName &p_enum_name, const StringName &p_constant_name, GDExtensionInt p_constant_value, bool p_is_bitfield = false);
	// Binds an implementation of a virtual method defined in Godot.
	static void bind_virtual_method(const StringName &p_class, const StringName &p_method, GDExtensionClassCallVirtual p_call, uint32_t p_hash);
	// Add a new virtual method that can be implemented by scripts.
	static void add_virtual_method(const StringName &p_class, const MethodInfo &p_method, const Vector<StringName> &p_arg_names = Vector<StringName>());

	static MethodBind *get_method(const StringName &p_class, const StringName &p_method);

	static GDExtensionClassCallVirtual get_virtual_func(void *p_userdata, GDExtensionConstStringNamePtr p_name, uint32_t p_hash);
	static const GDExtensionInstanceBindingCallbacks *get_instance_binding_callbacks(const StringName &p_class);

	static void initialize(GDExtensionInitializationLevel p_level);
	static void deinitialize(GDExtensionInitializationLevel p_level);

	CLASSDB_SINGLETON_FORWARD_METHODS;
};

#define BIND_CONSTANT(m_constant) \
	::godot::ClassDB::bind_integer_constant(get_class_static(), "", #m_constant, m_constant);

#define BIND_ENUM_CONSTANT(m_constant) \
	::godot::ClassDB::bind_integer_constant(get_class_static(), ::godot::_gde_constant_get_enum_name(m_constant, #m_constant), #m_constant, m_constant);

#define BIND_BITFIELD_FLAG(m_constant) \
	::godot::ClassDB::bind_integer_constant(get_class_static(), ::godot::_gde_constant_get_bitfield_name(m_constant, #m_constant), #m_constant, m_constant, true);

#define BIND_VIRTUAL_METHOD(m_class, m_method, m_hash)                                                                                        \
	{                                                                                                                                         \
		auto _call##m_method = [](GDExtensionObjectPtr p_instance, const GDExtensionConstTypePtr *p_args, GDExtensionTypePtr p_ret) -> void { \
			call_with_ptr_args(reinterpret_cast<m_class *>(p_instance), &m_class::m_method, p_args, p_ret);                                   \
		};                                                                                                                                    \
		::godot::ClassDB::bind_virtual_method(m_class::get_class_static(), #m_method, _call##m_method, m_hash);                               \
	}

template <typename T, bool is_abstract>
void ClassDB::_register_class(bool p_virtual, bool p_exposed, bool p_runtime) {
	static_assert(TypesAreSame<typename T::self_type, T>::value, "Class not declared properly, please use GDCLASS.");
	static_assert(!FunctionsAreSame<T::self_type::_bind_methods, T::parent_type::_bind_methods>::value, "Class must declare 'static void _bind_methods'.");
	static_assert(!std::is_abstract_v<T> || is_abstract, "Class is abstract, please use GDREGISTER_ABSTRACT_CLASS.");
	instance_binding_callbacks[T::get_class_static()] = &T::_gde_binding_callbacks;

	// Register this class within our plugin
	ClassInfo cl;
	cl.name = T::get_class_static();
	cl.parent_name = T::get_parent_class_static();
	cl.level = current_level;
	std::unordered_map<StringName, ClassInfo>::iterator parent_it = classes.find(cl.parent_name);
	if (parent_it != classes.end()) {
		// Assign parent if it is also a custom class
		cl.parent_ptr = &parent_it->second;
	}
	classes[cl.name] = cl;
	class_register_order.push_back(cl.name);

	// Register this class with Godot
	GDExtensionClassCreationInfo5 class_info = {
		p_virtual, // GDExtensionBool is_virtual;
		is_abstract, // GDExtensionBool is_abstract;
		p_exposed, // GDExtensionBool is_exposed;
		p_runtime, // GDExtensionBool is_runtime;
		nullptr, // GDExtensionConstStringPtr icon_path;
		T::set_bind, // GDExtensionClassSet set_func;
		T::get_bind, // GDExtensionClassGet get_func;
		T::has_get_property_list() ? T::get_property_list_bind : nullptr, // GDExtensionClassGetPropertyList get_property_list_func;
		T::free_property_list_bind, // GDExtensionClassFreePropertyList2 free_property_list_func;
		T::property_can_revert_bind, // GDExtensionClassPropertyCanRevert property_can_revert_func;
		T::property_get_revert_bind, // GDExtensionClassPropertyGetRevert property_get_revert_func;
		T::validate_property_bind, // GDExtensionClassValidateProperty validate_property_func;
		T::notification_bind, // GDExtensionClassNotification2 notification_func;
		T::to_string_bind, // GDExtensionClassToString to_string_func;
		nullptr, // GDExtensionClassReference reference_func;
		nullptr, // GDExtensionClassUnreference unreference_func;
		&_create_instance_func<T>, // GDExtensionClassCreateInstance create_instance_func; /* this one is mandatory */
		T::free, // GDExtensionClassFreeInstance free_instance_func; /* this one is mandatory */
		&_recreate_instance_func<T>, // GDExtensionClassRecreateInstance recreate_instance_func;
		&ClassDB::get_virtual_func, // GDExtensionClassGetVirtual get_virtual_func;
		nullptr, // GDExtensionClassGetVirtualCallData get_virtual_call_data_func;
		nullptr, // GDExtensionClassCallVirtualWithData call_virtual_func;
		(void *)&T::get_class_static(), // void *class_userdata;
	};

	internal::gdextension_interface_classdb_register_extension_class5(internal::library, cl.name._native_ptr(), cl.parent_name._native_ptr(), &class_info);

	// call bind_methods etc. to register all members of the class
	T::initialize_class();

	// now register our class within ClassDB within Godot
	initialize_class(classes[cl.name]);
}

template <typename T>
void ClassDB::register_class(bool p_virtual) {
	ClassDB::_register_class<T, false>(p_virtual);
}

template <typename T>
void ClassDB::register_abstract_class() {
	ClassDB::_register_class<T, true>();
}

template <typename T>
void ClassDB::register_internal_class() {
	ClassDB::_register_class<T, false>(false, false);
}

template <typename T>
void ClassDB::register_runtime_class() {
	ClassDB::_register_class<T, false>(false, true, true);
}

template <typename N, typename M, typename... VarArgs>
MethodBind *ClassDB::bind_method(N p_method_name, M p_method, VarArgs... p_args) {
	Variant args[sizeof...(p_args) + 1] = { p_args..., Variant() }; // +1 makes sure zero sized arrays are also supported.
	const Variant *argptrs[sizeof...(p_args) + 1];
	for (uint32_t i = 0; i < sizeof...(p_args); i++) {
		argptrs[i] = &args[i];
	}
	MethodBind *bind = create_method_bind(p_method);
	return bind_methodfi(METHOD_FLAGS_DEFAULT, bind, p_method_name, sizeof...(p_args) == 0 ? nullptr : (const void **)argptrs, sizeof...(p_args));
}

template <typename N, typename M, typename... VarArgs>
MethodBind *ClassDB::bind_static_method(StringName p_class, N p_method_name, M p_method, VarArgs... p_args) {
	Variant args[sizeof...(p_args) + 1] = { p_args..., Variant() }; // +1 makes sure zero sized arrays are also supported.
	const Variant *argptrs[sizeof...(p_args) + 1];
	for (uint32_t i = 0; i < sizeof...(p_args); i++) {
		argptrs[i] = &args[i];
	}
	MethodBind *bind = create_static_method_bind(p_method);
	bind->set_instance_class(p_class);
	return bind_methodfi(0, bind, p_method_name, sizeof...(p_args) == 0 ? nullptr : (const void **)argptrs, sizeof...(p_args));
}

template <typename M>
MethodBind *ClassDB::bind_vararg_method(uint32_t p_flags, StringName p_name, M p_method, const MethodInfo &p_info, const std::vector<Variant> &p_default_args, bool p_return_nil_is_variant) {
	MethodBind *bind = create_vararg_method_bind(p_method, p_info, p_return_nil_is_variant);
	ERR_FAIL_NULL_V(bind, nullptr);

	bind->set_name(p_name);
	bind->set_default_arguments(p_default_args);

	StringName instance_type = bind->get_instance_class();

	std::unordered_map<StringName, ClassInfo>::iterator type_it = classes.find(instance_type);
	if (type_it == classes.end()) {
		memdelete(bind);
		ERR_FAIL_V_MSG(nullptr, String("Class '{0}' doesn't exist.").format(Array::make(instance_type)));
	}

	ClassInfo &type = type_it->second;

	if (type.method_map.find(p_name) != type.method_map.end()) {
		memdelete(bind);
		ERR_FAIL_V_MSG(nullptr, String("Binding duplicate method: {0}::{1}.").format(Array::make(instance_type, p_method)));
	}

	// register our method bind within our plugin
	type.method_map[p_name] = bind;

	// and register with godot
	bind_method_godot(type.name, bind);

	return bind;
}

#define GDREGISTER_CLASS(m_class) ::godot::ClassDB::register_class<m_class>();
#define GDREGISTER_VIRTUAL_CLASS(m_class) ::godot::ClassDB::register_class<m_class>(true);
#define GDREGISTER_ABSTRACT_CLASS(m_class) ::godot::ClassDB::register_abstract_class<m_class>();
#define GDREGISTER_INTERNAL_CLASS(m_class) ::godot::ClassDB::register_internal_class<m_class>();
#define GDREGISTER_RUNTIME_CLASS(m_class) ::godot::ClassDB::register_runtime_class<m_class>();

} // namespace godot

CLASSDB_SINGLETON_VARIANT_CAST;
