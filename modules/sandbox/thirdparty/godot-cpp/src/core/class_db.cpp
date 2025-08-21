/**************************************************************************/
/*  class_db.cpp                                                          */
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

#include <godot_cpp/core/class_db.hpp>

#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/templates/vector.hpp>

#include <godot_cpp/core/memory.hpp>

#include <algorithm>

namespace godot {

std::unordered_map<StringName, ClassDB::ClassInfo> ClassDB::classes;
std::unordered_map<StringName, const GDExtensionInstanceBindingCallbacks *> ClassDB::instance_binding_callbacks;
std::vector<StringName> ClassDB::class_register_order;
std::unordered_map<StringName, Object *> ClassDB::engine_singletons;
std::mutex ClassDB::engine_singletons_mutex;
GDExtensionInitializationLevel ClassDB::current_level = GDEXTENSION_INITIALIZATION_CORE;

MethodDefinition D_METHOD(StringName p_name) {
	return MethodDefinition(p_name);
}

MethodDefinition D_METHOD(StringName p_name, StringName p_arg1) {
	MethodDefinition method(p_name);
	method.args.push_front(p_arg1);
	return method;
}

void ClassDB::add_property_group(const StringName &p_class, const String &p_name, const String &p_prefix) {
	ERR_FAIL_COND_MSG(classes.find(p_class) == classes.end(), String("Trying to add property '{0}{1}' to non-existing class '{2}'.").format(Array::make(p_prefix, p_name, p_class)));

	internal::gdextension_interface_classdb_register_extension_class_property_group(internal::library, p_class._native_ptr(), p_name._native_ptr(), p_prefix._native_ptr());
}

void ClassDB::add_property_subgroup(const StringName &p_class, const String &p_name, const String &p_prefix) {
	ERR_FAIL_COND_MSG(classes.find(p_class) == classes.end(), String("Trying to add property '{0}{1}' to non-existing class '{2}'.").format(Array::make(p_prefix, p_name, p_class)));

	internal::gdextension_interface_classdb_register_extension_class_property_subgroup(internal::library, p_class._native_ptr(), p_name._native_ptr(), p_prefix._native_ptr());
}

void ClassDB::add_property(const StringName &p_class, const PropertyInfo &p_pinfo, const StringName &p_setter, const StringName &p_getter, int p_index) {
	ERR_FAIL_COND_MSG(classes.find(p_class) == classes.end(), String("Trying to add property '{0}' to non-existing class '{1}'.").format(Array::make(p_pinfo.name, p_class)));

	ClassInfo &info = classes[p_class];

	ERR_FAIL_COND_MSG(info.property_names.find(p_pinfo.name) != info.property_names.end(), String("Property '{0}' already exists in class '{1}'.").format(Array::make(p_pinfo.name, p_class)));

	MethodBind *setter = nullptr;
	if (p_setter != String("")) {
		setter = get_method(p_class, p_setter);

		ERR_FAIL_NULL_MSG(setter, String("Setter method '{0}::{1}()' not found for property '{2}::{3}'.").format(Array::make(p_class, p_setter, p_class, p_pinfo.name)));

		size_t exp_args = 1 + (p_index >= 0 ? 1 : 0);
		ERR_FAIL_COND_MSG((int)exp_args != setter->get_argument_count(), String("Setter method '{0}::{1}()' must take a single argument.").format(Array::make(p_class, p_setter)));
	}

	ERR_FAIL_COND_MSG(p_getter == String(""), String("Getter method must be specified for '{0}::{1}'.").format(Array::make(p_class, p_pinfo.name)));

	MethodBind *getter = get_method(p_class, p_getter);
	ERR_FAIL_NULL_MSG(getter, String("Getter method '{0}::{1}()' not found for property '{2}::{3}'.").format(Array::make(p_class, p_getter, p_class, p_pinfo.name)));
	{
		size_t exp_args = 0 + (p_index >= 0 ? 1 : 0);
		ERR_FAIL_COND_MSG((int)exp_args != getter->get_argument_count(), String("Getter method '{0}::{1}()' must not take any argument.").format(Array::make(p_class, p_getter)));
	}

	// register property with plugin
	info.property_names.insert(p_pinfo.name);

	// register with Godot
	GDExtensionPropertyInfo prop_info = {
		static_cast<GDExtensionVariantType>(p_pinfo.type), // GDExtensionVariantType type;
		p_pinfo.name._native_ptr(), // GDExtensionStringNamePtr name;
		p_pinfo.class_name._native_ptr(), // GDExtensionStringNamePtr class_name;
		p_pinfo.hint, // NONE //uint32_t hint;
		p_pinfo.hint_string._native_ptr(), // GDExtensionStringPtr hint_string;
		p_pinfo.usage, // DEFAULT //uint32_t usage;
	};

	internal::gdextension_interface_classdb_register_extension_class_property_indexed(internal::library, info.name._native_ptr(), &prop_info, p_setter._native_ptr(), p_getter._native_ptr(), p_index);
}

MethodBind *ClassDB::get_method(const StringName &p_class, const StringName &p_method) {
	ERR_FAIL_COND_V_MSG(classes.find(p_class) == classes.end(), nullptr, String("Class '{0}' not found.").format(Array::make(p_class)));

	ClassInfo *type = &classes[p_class];
	while (type) {
		std::unordered_map<StringName, MethodBind *>::iterator method = type->method_map.find(p_method);
		if (method != type->method_map.end()) {
			return method->second;
		}
		type = type->parent_ptr;
		continue;
	}

	return nullptr;
}

MethodBind *ClassDB::bind_methodfi(uint32_t p_flags, MethodBind *p_bind, const MethodDefinition &method_name, const void **p_defs, int p_defcount) {
	StringName instance_type = p_bind->get_instance_class();

	std::unordered_map<StringName, ClassInfo>::iterator type_it = classes.find(instance_type);
	if (type_it == classes.end()) {
		memdelete(p_bind);
		ERR_FAIL_V_MSG(nullptr, String("Class '{0}' doesn't exist.").format(Array::make(instance_type)));
	}

	ClassInfo &type = type_it->second;

	if (type.method_map.find(method_name.name) != type.method_map.end()) {
		memdelete(p_bind);
		ERR_FAIL_V_MSG(nullptr, String("Binding duplicate method: {0}::{1}().").format(Array::make(instance_type, method_name.name)));
	}

	if (type.virtual_methods.find(method_name.name) != type.virtual_methods.end()) {
		memdelete(p_bind);
		ERR_FAIL_V_MSG(nullptr, String("Method '{0}::{1}()' already bound as virtual.").format(Array::make(instance_type, method_name.name)));
	}

	p_bind->set_name(method_name.name);

	if ((int)method_name.args.size() > p_bind->get_argument_count()) {
		memdelete(p_bind);
		ERR_FAIL_V_MSG(nullptr, String("Method '{0}::{1}()' definition has more arguments than the actual method.").format(Array::make(instance_type, method_name.name)));
	}

	p_bind->set_hint_flags(p_flags);

	std::vector<StringName> args;
	args.resize(method_name.args.size());
	size_t arg_index = 0;
	for (StringName arg : method_name.args) {
		args[arg_index++] = arg;
	}

	p_bind->set_argument_names(args);

	std::vector<Variant> defvals;

	defvals.resize(p_defcount);
	for (int i = 0; i < p_defcount; i++) {
		defvals[i] = *static_cast<const Variant *>(p_defs[i]);
	}

	p_bind->set_default_arguments(defvals);
	p_bind->set_hint_flags(p_flags);

	// register our method bind within our plugin
	type.method_map[method_name.name] = p_bind;

	// and register with godot
	bind_method_godot(type.name, p_bind);

	return p_bind;
}

void ClassDB::bind_method_godot(const StringName &p_class_name, MethodBind *p_method) {
	std::vector<GDExtensionVariantPtr> def_args;
	const std::vector<Variant> &def_args_val = p_method->get_default_arguments();
	def_args.resize(def_args_val.size());
	for (size_t i = 0; i < def_args_val.size(); i++) {
		def_args[i] = (GDExtensionVariantPtr)&def_args_val[i];
	}

	std::vector<PropertyInfo> return_value_and_arguments_info = p_method->get_arguments_info_list();
	std::vector<GDExtensionClassMethodArgumentMetadata> return_value_and_arguments_metadata = p_method->get_arguments_metadata_list();

	std::vector<GDExtensionPropertyInfo> return_value_and_arguments_gdextension_info;
	return_value_and_arguments_gdextension_info.reserve(return_value_and_arguments_info.size());
	for (std::vector<PropertyInfo>::iterator it = return_value_and_arguments_info.begin(); it != return_value_and_arguments_info.end(); it++) {
		return_value_and_arguments_gdextension_info.push_back(
				GDExtensionPropertyInfo{
						static_cast<GDExtensionVariantType>(it->type), // GDExtensionVariantType type;
						it->name._native_ptr(), // GDExtensionStringNamePtr name;
						it->class_name._native_ptr(), // GDExtensionStringNamePtr class_name;
						it->hint, // uint32_t hint;
						it->hint_string._native_ptr(), // GDExtensionStringPtr hint_string;
						it->usage, // uint32_t usage;
				});
	}

	GDExtensionPropertyInfo *return_value_info = return_value_and_arguments_gdextension_info.data();
	GDExtensionClassMethodArgumentMetadata *return_value_metadata = return_value_and_arguments_metadata.data();
	GDExtensionPropertyInfo *arguments_info = return_value_and_arguments_gdextension_info.data() + 1;
	GDExtensionClassMethodArgumentMetadata *arguments_metadata = return_value_and_arguments_metadata.data() + 1;

	StringName name = p_method->get_name();
	GDExtensionClassMethodInfo method_info = {
		name._native_ptr(), // GDExtensionStringNamePtr;
		p_method, // void *method_userdata;
		MethodBind::bind_call, // GDExtensionClassMethodCall call_func;
		MethodBind::bind_ptrcall, // GDExtensionClassMethodPtrCall ptrcall_func;
		p_method->get_hint_flags(), // uint32_t method_flags; /* GDExtensionClassMethodFlags */
		(GDExtensionBool)p_method->has_return(), // GDExtensionBool has_return_value;
		return_value_info, // GDExtensionPropertyInfo *
		*return_value_metadata, // GDExtensionClassMethodArgumentMetadata *
		(uint32_t)p_method->get_argument_count(), // uint32_t argument_count;
		arguments_info, // GDExtensionPropertyInfo *
		arguments_metadata, // GDExtensionClassMethodArgumentMetadata *
		(uint32_t)p_method->get_default_argument_count(), // uint32_t default_argument_count;
		def_args.data(), // GDExtensionVariantPtr *default_arguments;
	};
	internal::gdextension_interface_classdb_register_extension_class_method(internal::library, p_class_name._native_ptr(), &method_info);
}

void ClassDB::add_signal(const StringName &p_class, const MethodInfo &p_signal) {
	std::unordered_map<StringName, ClassInfo>::iterator type_it = classes.find(p_class);

	ERR_FAIL_COND_MSG(type_it == classes.end(), String("Class '{0}' doesn't exist.").format(Array::make(p_class)));

	ClassInfo &cl = type_it->second;

	// Check if this signal is already register
	ClassInfo *check = &cl;
	while (check) {
		ERR_FAIL_COND_MSG(check->signal_names.find(p_signal.name) != check->signal_names.end(), String("Class '{0}' already has signal '{1}'.").format(Array::make(p_class, p_signal.name)));
		check = check->parent_ptr;
	}

	// register our signal in our plugin
	cl.signal_names.insert(p_signal.name);

	// register our signal in godot
	std::vector<GDExtensionPropertyInfo> parameters;
	parameters.reserve(p_signal.arguments.size());

	for (const PropertyInfo &par : p_signal.arguments) {
		parameters.push_back(GDExtensionPropertyInfo{
				static_cast<GDExtensionVariantType>(par.type), // GDExtensionVariantType type;
				par.name._native_ptr(), // GDExtensionStringNamePtr name;
				par.class_name._native_ptr(), // GDExtensionStringNamePtr class_name;
				par.hint, // NONE //uint32_t hint;
				par.hint_string._native_ptr(), // GDExtensionStringPtr hint_string;
				par.usage, // DEFAULT //uint32_t usage;
		});
	}

	internal::gdextension_interface_classdb_register_extension_class_signal(internal::library, cl.name._native_ptr(), p_signal.name._native_ptr(), parameters.data(), parameters.size());
}

void ClassDB::bind_integer_constant(const StringName &p_class_name, const StringName &p_enum_name, const StringName &p_constant_name, GDExtensionInt p_constant_value, bool p_is_bitfield) {
	std::unordered_map<StringName, ClassInfo>::iterator type_it = classes.find(p_class_name);

	ERR_FAIL_COND_MSG(type_it == classes.end(), String("Class '{0}' doesn't exist.").format(Array::make(p_class_name)));

	ClassInfo &type = type_it->second;

	// check if it already exists
	ERR_FAIL_COND_MSG(type.constant_names.find(p_constant_name) != type.constant_names.end(), String("Constant '{0}::{1}' already registered.").format(Array::make(p_class_name, p_constant_name)));

	// register it with our plugin (purely to check for duplicates)
	type.constant_names.insert(p_constant_name);

	// Register it with Godot
	internal::gdextension_interface_classdb_register_extension_class_integer_constant(internal::library, p_class_name._native_ptr(), p_enum_name._native_ptr(), p_constant_name._native_ptr(), p_constant_value, p_is_bitfield);
}
GDExtensionClassCallVirtual ClassDB::get_virtual_func(void *p_userdata, GDExtensionConstStringNamePtr p_name, uint32_t p_hash) {
	// This is called by Godot the first time it calls a virtual function, and it caches the result, per object instance.
	// Because of this, it can happen from different threads at once.
	// It should be ok not using any mutex as long as we only READ data.
	const StringName *class_name = reinterpret_cast<const StringName *>(p_userdata);
	const StringName *name = reinterpret_cast<const StringName *>(p_name);

	std::unordered_map<StringName, ClassInfo>::iterator type_it = classes.find(*class_name);
	ERR_FAIL_COND_V_MSG(type_it == classes.end(), nullptr, String("Class '{0}' doesn't exist.").format(Array::make(*class_name)));

	const ClassInfo *type = &type_it->second;

	// Find method in current class, or any of its parent classes (Godot classes not included)
	while (type != nullptr) {
		std::unordered_map<StringName, ClassInfo::VirtualMethod>::const_iterator method_it = type->virtual_methods.find(*name);

		if (method_it != type->virtual_methods.end() && method_it->second.hash == p_hash) {
			return method_it->second.func;
		}

		type = type->parent_ptr;
	}

	return nullptr;
}

const GDExtensionInstanceBindingCallbacks *ClassDB::get_instance_binding_callbacks(const StringName &p_class) {
	std::unordered_map<StringName, const GDExtensionInstanceBindingCallbacks *>::iterator callbacks_it = instance_binding_callbacks.find(p_class);
	if (likely(callbacks_it != instance_binding_callbacks.end())) {
		return callbacks_it->second;
	}

	// If we don't have an instance binding callback for the given class, find the closest parent where we do.
	StringName class_name = p_class;
	do {
		class_name = get_parent_class(class_name);
		ERR_FAIL_COND_V_MSG(class_name == StringName(), nullptr, String("Cannot find instance binding callbacks for class '{0}'.").format(Array::make(p_class)));
		callbacks_it = instance_binding_callbacks.find(class_name);
	} while (callbacks_it == instance_binding_callbacks.end());

	return callbacks_it->second;
}

void ClassDB::bind_virtual_method(const StringName &p_class, const StringName &p_method, GDExtensionClassCallVirtual p_call, uint32_t p_hash) {
	std::unordered_map<StringName, ClassInfo>::iterator type_it = classes.find(p_class);
	ERR_FAIL_COND_MSG(type_it == classes.end(), String("Class '{0}' doesn't exist.").format(Array::make(p_class)));

	ClassInfo &type = type_it->second;

	ERR_FAIL_COND_MSG(type.method_map.find(p_method) != type.method_map.end(), String("Method '{0}::{1}()' already registered as non-virtual.").format(Array::make(p_class, p_method)));
	ERR_FAIL_COND_MSG(type.virtual_methods.find(p_method) != type.virtual_methods.end(), String("Virtual '{0}::{1}()' method already registered.").format(Array::make(p_class, p_method)));

	type.virtual_methods[p_method] = ClassInfo::VirtualMethod{
		p_call,
		p_hash,
	};
}

void ClassDB::add_virtual_method(const StringName &p_class, const MethodInfo &p_method, const Vector<StringName> &p_arg_names) {
	std::unordered_map<StringName, ClassInfo>::iterator type_it = classes.find(p_class);
	ERR_FAIL_COND_MSG(type_it == classes.end(), String("Class '{0}' doesn't exist.").format(Array::make(p_class)));

	GDExtensionClassVirtualMethodInfo mi;
	mi.name = (GDExtensionStringNamePtr)&p_method.name;
	mi.method_flags = p_method.flags;
	mi.return_value = p_method.return_val._to_gdextension();
	mi.return_value_metadata = p_method.return_val_metadata;
	mi.argument_count = p_method.arguments.size();
	if (mi.argument_count > 0) {
		mi.arguments = (GDExtensionPropertyInfo *)memalloc(sizeof(GDExtensionPropertyInfo) * mi.argument_count);
		mi.arguments_metadata = (GDExtensionClassMethodArgumentMetadata *)memalloc(sizeof(GDExtensionClassMethodArgumentMetadata) * mi.argument_count);
		if (mi.argument_count != p_method.arguments_metadata.size()) {
			WARN_PRINT("Mismatch argument metadata count for virtual method: " + String(p_class) + "::" + p_method.name);
		}
		for (uint32_t i = 0; i < mi.argument_count; i++) {
			mi.arguments[i] = p_method.arguments[i]._to_gdextension();
			if (i < p_method.arguments_metadata.size()) {
				mi.arguments_metadata[i] = p_method.arguments_metadata[i];
			}
		}
	} else {
		mi.arguments = nullptr;
		mi.arguments_metadata = nullptr;
	}

	if (p_arg_names.size() != mi.argument_count) {
		WARN_PRINT("Mismatch argument name count for virtual method: " + String(p_class) + "::" + p_method.name);
	} else {
		for (int i = 0; i < p_arg_names.size(); i++) {
			mi.arguments[i].name = (GDExtensionStringNamePtr)&p_arg_names[i];
		}
	}

	internal::gdextension_interface_classdb_register_extension_class_virtual_method(internal::library, &p_class, &mi);

	if (mi.arguments) {
		memfree(mi.arguments);
	}
	if (mi.arguments_metadata) {
		memfree(mi.arguments_metadata);
	}
}

void ClassDB::_editor_get_classes_used_callback(GDExtensionTypePtr p_packed_string_array) {
	PackedStringArray *arr = reinterpret_cast<PackedStringArray *>(p_packed_string_array);
	arr->resize(instance_binding_callbacks.size());
	int index = 0;
	for (const std::pair<const StringName, const GDExtensionInstanceBindingCallbacks *> &pair : instance_binding_callbacks) {
		(*arr)[index++] = pair.first;
	}
}

void ClassDB::initialize_class(const ClassInfo &p_cl) {
}

void ClassDB::initialize(GDExtensionInitializationLevel p_level) {
	for (const std::pair<const StringName, ClassInfo> &pair : classes) {
		const ClassInfo &cl = pair.second;
		if (cl.level != p_level) {
			continue;
		}

		// Nothing to do here for now...
	}
}

void ClassDB::deinitialize(GDExtensionInitializationLevel p_level) {
	std::set<StringName> to_erase;
	for (std::vector<StringName>::reverse_iterator i = class_register_order.rbegin(); i != class_register_order.rend(); ++i) {
		const StringName &name = *i;
		const ClassInfo &cl = classes[name];

		if (cl.level != p_level) {
			continue;
		}

		internal::gdextension_interface_classdb_unregister_extension_class(internal::library, name._native_ptr());

		for (const std::pair<const StringName, MethodBind *> &method : cl.method_map) {
			memdelete(method.second);
		}

		classes.erase(name);
		to_erase.insert(name);
	}

	{
		// The following is equivalent to c++20 `std::erase_if(class_register_order, [&](const StringName& name){ return to_erase.contains(name); });`
		std::vector<StringName>::iterator it = std::remove_if(class_register_order.begin(), class_register_order.end(), [&](const StringName &p_name) {
			return to_erase.count(p_name) > 0;
		});
		class_register_order.erase(it, class_register_order.end());
	}

	if (p_level == GDEXTENSION_INITIALIZATION_CORE) {
		// Make a new list of the singleton objects, since freeing the instance bindings will lead to
		// elements getting removed from engine_singletons.
		std::vector<Object *> singleton_objects;
		{
			std::lock_guard<std::mutex> lock(engine_singletons_mutex);
			singleton_objects.reserve(engine_singletons.size());
			for (const std::pair<const StringName, Object *> &pair : engine_singletons) {
				singleton_objects.push_back(pair.second);
			}
		}
		for (std::vector<Object *>::iterator i = singleton_objects.begin(); i != singleton_objects.end(); i++) {
			internal::gdextension_interface_object_free_instance_binding((*i)->_owner, internal::token);
		}
	}
}

} // namespace godot
