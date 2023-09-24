/**************************************************************************/
/*  gdextension.cpp                                                       */
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

#include "gdextension.h"
#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/object/class_db.h"
#include "core/object/method_bind.h"
#include "core/os/os.h"
#include "core/version.h"

extern void gdextension_setup_interface();
extern GDExtensionInterfaceFunctionPtr gdextension_get_proc_address(const char *p_name);

typedef GDExtensionBool (*GDExtensionLegacyInitializationFunction)(void *p_interface, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization);

String GDExtension::get_extension_list_config_file() {
	return ProjectSettings::get_singleton()->get_project_data_path().path_join("extension_list.cfg");
}

String GDExtension::find_extension_library(const String &p_path, Ref<ConfigFile> p_config, std::function<bool(String)> p_has_feature, PackedStringArray *r_tags) {
	// First, check the explicit libraries.
	if (p_config->has_section("libraries")) {
		List<String> libraries;
		p_config->get_section_keys("libraries", &libraries);

		// Iterate the libraries, finding the best matching tags.
		String best_library_path;
		Vector<String> best_library_tags;
		for (const String &E : libraries) {
			Vector<String> tags = E.split(".");
			bool all_tags_met = true;
			for (int i = 0; i < tags.size(); i++) {
				String tag = tags[i].strip_edges();
				if (!p_has_feature(tag)) {
					all_tags_met = false;
					break;
				}
			}

			if (all_tags_met && tags.size() > best_library_tags.size()) {
				best_library_path = p_config->get_value("libraries", E);
				best_library_tags = tags;
			}
		}

		if (!best_library_path.is_empty()) {
			if (best_library_path.is_relative_path()) {
				best_library_path = p_path.get_base_dir().path_join(best_library_path);
			}
			if (r_tags != nullptr) {
				r_tags->append_array(best_library_tags);
			}
			return best_library_path;
		}
	}

	// Second, try to autodetect
	String autodetect_library_prefix;
	if (p_config->has_section_key("configuration", "autodetect_library_prefix")) {
		autodetect_library_prefix = p_config->get_value("configuration", "autodetect_library_prefix");
	}
	if (!autodetect_library_prefix.is_empty()) {
		String autodetect_path = autodetect_library_prefix;
		if (autodetect_path.is_relative_path()) {
			autodetect_path = p_path.get_base_dir().path_join(autodetect_path);
		}

		// Find the folder and file parts of the prefix.
		String folder;
		String file_prefix;
		if (DirAccess::dir_exists_absolute(autodetect_path)) {
			folder = autodetect_path;
		} else if (DirAccess::dir_exists_absolute(autodetect_path.get_base_dir())) {
			folder = autodetect_path.get_base_dir();
			file_prefix = autodetect_path.get_file();
		} else {
			ERR_FAIL_V_MSG(String(), vformat("Error in extension: %s. Could not find folder for automatic detection of libraries files. autodetect_library_prefix=\"%s\"", p_path, autodetect_library_prefix));
		}

		// Open the folder.
		Ref<DirAccess> dir = DirAccess::open(folder);
		ERR_FAIL_COND_V_MSG(!dir.is_valid(), String(), vformat("Error in extension: %s. Could not open folder for automatic detection of libraries files. autodetect_library_prefix=\"%s\"", p_path, autodetect_library_prefix));

		// Iterate the files and check the prefixes, finding the best matching file.
		String best_file;
		Vector<String> best_file_tags;
		dir->list_dir_begin();
		String file_name = dir->_get_next();
		while (file_name != "") {
			if (!dir->current_is_dir() && file_name.begins_with(file_prefix)) {
				// Check if the files matches all requested feature tags.
				String tags_str = file_name.trim_prefix(file_prefix);
				tags_str = tags_str.trim_suffix(tags_str.get_extension());

				Vector<String> tags = tags_str.split(".", false);
				bool all_tags_met = true;
				for (int i = 0; i < tags.size(); i++) {
					String tag = tags[i].strip_edges();
					if (!p_has_feature(tag)) {
						all_tags_met = false;
						break;
					}
				}

				// If all tags are found in the feature list, and we found more tags than before, use this file.
				if (all_tags_met && tags.size() > best_file_tags.size()) {
					best_file_tags = tags;
					best_file = file_name;
				}
			}
			file_name = dir->_get_next();
		}

		if (!best_file.is_empty()) {
			String library_path = folder.path_join(best_file);
			if (r_tags != nullptr) {
				r_tags->append_array(best_file_tags);
			}
			return library_path;
		}
	}
	return String();
}

class GDExtensionMethodBind : public MethodBind {
	GDExtensionClassMethodCall call_func;
	GDExtensionClassMethodValidatedCall validated_call_func;
	GDExtensionClassMethodPtrCall ptrcall_func;
	void *method_userdata;
	bool vararg;
	uint32_t argument_count;
	PropertyInfo return_value_info;
	GodotTypeInfo::Metadata return_value_metadata;
	List<PropertyInfo> arguments_info;
	List<GodotTypeInfo::Metadata> arguments_metadata;

protected:
	virtual Variant::Type _gen_argument_type(int p_arg) const override {
		if (p_arg < 0) {
			return return_value_info.type;
		} else {
			return arguments_info[p_arg].type;
		}
	}
	virtual PropertyInfo _gen_argument_type_info(int p_arg) const override {
		if (p_arg < 0) {
			return return_value_info;
		} else {
			return arguments_info[p_arg];
		}
	}

public:
#ifdef DEBUG_METHODS_ENABLED
	virtual GodotTypeInfo::Metadata get_argument_meta(int p_arg) const override {
		if (p_arg < 0) {
			return return_value_metadata;
		} else {
			return arguments_metadata[p_arg];
		}
	}
#endif

	virtual Variant call(Object *p_object, const Variant **p_args, int p_arg_count, Callable::CallError &r_error) const override {
		Variant ret;
		GDExtensionClassInstancePtr extension_instance = is_static() ? nullptr : p_object->_get_extension_instance();
		GDExtensionCallError ce{ GDEXTENSION_CALL_OK, 0, 0 };
		call_func(method_userdata, extension_instance, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), p_arg_count, (GDExtensionVariantPtr)&ret, &ce);
		r_error.error = Callable::CallError::Error(ce.error);
		r_error.argument = ce.argument;
		r_error.expected = ce.expected;
		return ret;
	}
	virtual void validated_call(Object *p_object, const Variant **p_args, Variant *r_ret) const override {
		ERR_FAIL_COND_MSG(vararg, "Validated methods don't have ptrcall support. This is most likely an engine bug.");
		GDExtensionClassInstancePtr extension_instance = is_static() ? nullptr : p_object->_get_extension_instance();

		if (validated_call_func) {
			// This is added here, but it's unlikely to be provided by most extensions.
			validated_call_func(method_userdata, extension_instance, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), (GDExtensionVariantPtr)r_ret);
		} else {
#if 1
			// Slow code-path, but works for the time being.
			Callable::CallError ce;
			call(p_object, p_args, argument_count, ce);
#else
			// This is broken, because it needs more information to do the calling properly

			// If not provided, go via ptrcall, which is faster than resorting to regular call.
			const void **argptrs = (const void **)alloca(argument_count * sizeof(void *));
			for (uint32_t i = 0; i < argument_count; i++) {
				argptrs[i] = VariantInternal::get_opaque_pointer(p_args[i]);
			}

			bool returns = true;
			void *ret_opaque;
			if (returns) {
				ret_opaque = VariantInternal::get_opaque_pointer(r_ret);
			} else {
				ret_opaque = nullptr; // May be unnecessary as this is ignored, but just in case.
			}

			ptrcall(p_object, argptrs, ret_opaque);
#endif
		}
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
		ERR_FAIL_COND_MSG(vararg, "Vararg methods don't have ptrcall support. This is most likely an engine bug.");
		GDExtensionClassInstancePtr extension_instance = p_object->_get_extension_instance();
		ptrcall_func(method_userdata, extension_instance, reinterpret_cast<GDExtensionConstTypePtr *>(p_args), (GDExtensionTypePtr)r_ret);
	}

	virtual bool is_vararg() const override {
		return false;
	}

	explicit GDExtensionMethodBind(const GDExtensionClassMethodInfo *p_method_info) {
		method_userdata = p_method_info->method_userdata;
		call_func = p_method_info->call_func;
		validated_call_func = nullptr;
		ptrcall_func = p_method_info->ptrcall_func;
		set_name(*reinterpret_cast<StringName *>(p_method_info->name));

		if (p_method_info->has_return_value) {
			return_value_info = PropertyInfo(*p_method_info->return_value_info);
			return_value_metadata = GodotTypeInfo::Metadata(p_method_info->return_value_metadata);
		}

		for (uint32_t i = 0; i < p_method_info->argument_count; i++) {
			arguments_info.push_back(PropertyInfo(p_method_info->arguments_info[i]));
			arguments_metadata.push_back(GodotTypeInfo::Metadata(p_method_info->arguments_metadata[i]));
		}

		set_hint_flags(p_method_info->method_flags);
		argument_count = p_method_info->argument_count;
		vararg = p_method_info->method_flags & GDEXTENSION_METHOD_FLAG_VARARG;
		_set_returns(p_method_info->has_return_value);
		_set_const(p_method_info->method_flags & GDEXTENSION_METHOD_FLAG_CONST);
		_set_static(p_method_info->method_flags & GDEXTENSION_METHOD_FLAG_STATIC);
#ifdef DEBUG_METHODS_ENABLED
		_generate_argument_types(p_method_info->argument_count);
#endif
		set_argument_count(p_method_info->argument_count);

		Vector<Variant> defargs;
		defargs.resize(p_method_info->default_argument_count);
		for (uint32_t i = 0; i < p_method_info->default_argument_count; i++) {
			defargs.write[i] = *static_cast<Variant *>(p_method_info->default_arguments[i]);
		}

		set_default_arguments(defargs);
	}
};

#ifndef DISABLE_DEPRECATED
void GDExtension::_register_extension_class(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, const GDExtensionClassCreationInfo *p_extension_funcs) {
	const GDExtensionClassCreationInfo2 class_info2 = {
		p_extension_funcs->is_virtual, // GDExtensionBool is_virtual;
		p_extension_funcs->is_abstract, // GDExtensionBool is_abstract;
		true, // GDExtensionBool is_exposed;
		p_extension_funcs->set_func, // GDExtensionClassSet set_func;
		p_extension_funcs->get_func, // GDExtensionClassGet get_func;
		p_extension_funcs->get_property_list_func, // GDExtensionClassGetPropertyList get_property_list_func;
		p_extension_funcs->free_property_list_func, // GDExtensionClassFreePropertyList free_property_list_func;
		p_extension_funcs->property_can_revert_func, // GDExtensionClassPropertyCanRevert property_can_revert_func;
		p_extension_funcs->property_get_revert_func, // GDExtensionClassPropertyGetRevert property_get_revert_func;
		nullptr, // GDExtensionClassValidateProperty validate_property_func;
		nullptr, // GDExtensionClassNotification2 notification_func;
		p_extension_funcs->to_string_func, // GDExtensionClassToString to_string_func;
		p_extension_funcs->reference_func, // GDExtensionClassReference reference_func;
		p_extension_funcs->unreference_func, // GDExtensionClassUnreference unreference_func;
		p_extension_funcs->create_instance_func, // GDExtensionClassCreateInstance create_instance_func; /* this one is mandatory */
		p_extension_funcs->free_instance_func, // GDExtensionClassFreeInstance free_instance_func; /* this one is mandatory */
		p_extension_funcs->get_virtual_func, // GDExtensionClassGetVirtual get_virtual_func;
		nullptr, // GDExtensionClassGetVirtualCallData get_virtual_call_data_func;
		nullptr, // GDExtensionClassCallVirtualWithData call_virtual_func;
		p_extension_funcs->get_rid_func, // GDExtensionClassGetRID get_rid;
		p_extension_funcs->class_userdata, // void *class_userdata;
	};

	const ClassCreationDeprecatedInfo legacy = {
		p_extension_funcs->notification_func,
	};
	_register_extension_class_internal(p_library, p_class_name, p_parent_class_name, &class_info2, &legacy);
}
#endif // DISABLE_DEPRECATED

void GDExtension::_register_extension_class2(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, const GDExtensionClassCreationInfo2 *p_extension_funcs) {
	_register_extension_class_internal(p_library, p_class_name, p_parent_class_name, p_extension_funcs);
}

void GDExtension::_register_extension_class_internal(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, const GDExtensionClassCreationInfo2 *p_extension_funcs, const ClassCreationDeprecatedInfo *p_deprecated_funcs) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	StringName parent_class_name = *reinterpret_cast<const StringName *>(p_parent_class_name);
	ERR_FAIL_COND_MSG(!String(class_name).is_valid_identifier(), "Attempt to register extension class '" + class_name + "', which is not a valid class identifier.");
	ERR_FAIL_COND_MSG(ClassDB::class_exists(class_name), "Attempt to register extension class '" + class_name + "', which appears to be already registered.");

	Extension *parent_extension = nullptr;

	if (self->extension_classes.has(parent_class_name)) {
		parent_extension = &self->extension_classes[parent_class_name];
	} else if (ClassDB::class_exists(parent_class_name)) {
		if (ClassDB::get_api_type(parent_class_name) == ClassDB::API_EXTENSION || ClassDB::get_api_type(parent_class_name) == ClassDB::API_EDITOR_EXTENSION) {
			ERR_PRINT("Unimplemented yet");
			//inheriting from another extension
		} else {
			//inheriting from engine class
		}
	} else {
		ERR_FAIL_MSG("Attempt to register an extension class '" + String(class_name) + "' using non-existing parent class '" + String(parent_class_name) + "'");
	}

	self->extension_classes[class_name] = Extension();

	Extension *extension = &self->extension_classes[class_name];

	if (parent_extension) {
		extension->gdextension.parent = &parent_extension->gdextension;
		parent_extension->gdextension.children.push_back(&extension->gdextension);
	}

	extension->gdextension.library = self;
	extension->gdextension.parent_class_name = parent_class_name;
	extension->gdextension.class_name = class_name;
	extension->gdextension.editor_class = self->level_initialized == INITIALIZATION_LEVEL_EDITOR;
	extension->gdextension.is_virtual = p_extension_funcs->is_virtual;
	extension->gdextension.is_abstract = p_extension_funcs->is_abstract;
	extension->gdextension.is_exposed = p_extension_funcs->is_exposed;
	extension->gdextension.set = p_extension_funcs->set_func;
	extension->gdextension.get = p_extension_funcs->get_func;
	extension->gdextension.get_property_list = p_extension_funcs->get_property_list_func;
	extension->gdextension.free_property_list = p_extension_funcs->free_property_list_func;
	extension->gdextension.property_can_revert = p_extension_funcs->property_can_revert_func;
	extension->gdextension.property_get_revert = p_extension_funcs->property_get_revert_func;
	extension->gdextension.validate_property = p_extension_funcs->validate_property_func;
#ifndef DISABLE_DEPRECATED
	if (p_deprecated_funcs) {
		extension->gdextension.notification = p_deprecated_funcs->notification_func;
	}
#endif // DISABLE_DEPRECATED
	extension->gdextension.notification2 = p_extension_funcs->notification_func;
	extension->gdextension.to_string = p_extension_funcs->to_string_func;
	extension->gdextension.reference = p_extension_funcs->reference_func;
	extension->gdextension.unreference = p_extension_funcs->unreference_func;
	extension->gdextension.class_userdata = p_extension_funcs->class_userdata;
	extension->gdextension.create_instance = p_extension_funcs->create_instance_func;
	extension->gdextension.free_instance = p_extension_funcs->free_instance_func;
	extension->gdextension.get_virtual = p_extension_funcs->get_virtual_func;
	extension->gdextension.get_virtual_call_data = p_extension_funcs->get_virtual_call_data_func;
	extension->gdextension.call_virtual_with_data = p_extension_funcs->call_virtual_with_data_func;
	extension->gdextension.get_rid = p_extension_funcs->get_rid_func;

	ClassDB::register_extension_class(&extension->gdextension);
}

void GDExtension::_register_extension_class_method(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionClassMethodInfo *p_method_info) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	StringName method_name = *reinterpret_cast<const StringName *>(p_method_info->name);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension method '" + String(method_name) + "' for unexisting class '" + class_name + "'.");

	//Extension *extension = &self->extension_classes[class_name];

	GDExtensionMethodBind *method = memnew(GDExtensionMethodBind(p_method_info));
	method->set_instance_class(class_name);

	ClassDB::bind_method_custom(class_name, method);
}
void GDExtension::_register_extension_class_integer_constant(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_enum_name, GDExtensionConstStringNamePtr p_constant_name, GDExtensionInt p_constant_value, GDExtensionBool p_is_bitfield) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	StringName enum_name = *reinterpret_cast<const StringName *>(p_enum_name);
	StringName constant_name = *reinterpret_cast<const StringName *>(p_constant_name);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension constant '" + constant_name + "' for unexisting class '" + class_name + "'.");

	ClassDB::bind_integer_constant(class_name, enum_name, constant_name, p_constant_value, p_is_bitfield);
}

void GDExtension::_register_extension_class_property(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionPropertyInfo *p_info, GDExtensionConstStringNamePtr p_setter, GDExtensionConstStringNamePtr p_getter) {
	_register_extension_class_property_indexed(p_library, p_class_name, p_info, p_setter, p_getter, -1);
}

void GDExtension::_register_extension_class_property_indexed(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionPropertyInfo *p_info, GDExtensionConstStringNamePtr p_setter, GDExtensionConstStringNamePtr p_getter, GDExtensionInt p_index) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	StringName setter = *reinterpret_cast<const StringName *>(p_setter);
	StringName getter = *reinterpret_cast<const StringName *>(p_getter);
	String property_name = *reinterpret_cast<const StringName *>(p_info->name);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension class property '" + property_name + "' for unexisting class '" + class_name + "'.");

	PropertyInfo pinfo(*p_info);

	ClassDB::add_property(class_name, pinfo, setter, getter, p_index);
}

void GDExtension::_register_extension_class_property_group(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringPtr p_group_name, GDExtensionConstStringPtr p_prefix) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	String group_name = *reinterpret_cast<const String *>(p_group_name);
	String prefix = *reinterpret_cast<const String *>(p_prefix);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension class property group '" + group_name + "' for unexisting class '" + class_name + "'.");

	ClassDB::add_property_group(class_name, group_name, prefix);
}

void GDExtension::_register_extension_class_property_subgroup(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringPtr p_subgroup_name, GDExtensionConstStringPtr p_prefix) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	String subgroup_name = *reinterpret_cast<const String *>(p_subgroup_name);
	String prefix = *reinterpret_cast<const String *>(p_prefix);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension class property subgroup '" + subgroup_name + "' for unexisting class '" + class_name + "'.");

	ClassDB::add_property_subgroup(class_name, subgroup_name, prefix);
}

void GDExtension::_register_extension_class_signal(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_signal_name, const GDExtensionPropertyInfo *p_argument_info, GDExtensionInt p_argument_count) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	StringName signal_name = *reinterpret_cast<const StringName *>(p_signal_name);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension class signal '" + signal_name + "' for unexisting class '" + class_name + "'.");

	MethodInfo s;
	s.name = signal_name;
	for (int i = 0; i < p_argument_count; i++) {
		PropertyInfo arg(p_argument_info[i]);
		s.arguments.push_back(arg);
	}
	ClassDB::add_signal(class_name, s);
}

void GDExtension::_unregister_extension_class(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to unregister unexisting extension class '" + class_name + "'.");
	Extension *ext = &self->extension_classes[class_name];
	ERR_FAIL_COND_MSG(ext->gdextension.children.size(), "Attempt to unregister class '" + class_name + "' while other extension classes inherit from it.");

	ClassDB::unregister_extension_class(class_name);
	if (ext->gdextension.parent != nullptr) {
		ext->gdextension.parent->children.erase(&ext->gdextension);
	}
	self->extension_classes.erase(class_name);
}

void GDExtension::_get_library_path(GDExtensionClassLibraryPtr p_library, GDExtensionUninitializedStringPtr r_path) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	memnew_placement(r_path, String(self->library_path));
}

HashMap<StringName, GDExtensionInterfaceFunctionPtr> gdextension_interface_functions;

void GDExtension::register_interface_function(StringName p_function_name, GDExtensionInterfaceFunctionPtr p_function_pointer) {
	ERR_FAIL_COND_MSG(gdextension_interface_functions.has(p_function_name), "Attempt to register interface function '" + p_function_name + "', which appears to be already registered.");
	gdextension_interface_functions.insert(p_function_name, p_function_pointer);
}

GDExtensionInterfaceFunctionPtr GDExtension::get_interface_function(StringName p_function_name) {
	GDExtensionInterfaceFunctionPtr *function = gdextension_interface_functions.getptr(p_function_name);
	ERR_FAIL_NULL_V_MSG(function, nullptr, "Attempt to get non-existent interface function: " + String(p_function_name) + ".");
	return *function;
}

Error GDExtension::open_library(const String &p_path, const String &p_entry_symbol) {
	Error err = OS::get_singleton()->open_dynamic_library(p_path, library, true, &library_path);
	if (err != OK) {
		ERR_PRINT("GDExtension dynamic library not found: " + p_path);
		return err;
	}

	void *entry_funcptr = nullptr;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(library, p_entry_symbol, entry_funcptr, false);

	if (err != OK) {
		ERR_PRINT("GDExtension entry point '" + p_entry_symbol + "' not found in library " + p_path);
		OS::get_singleton()->close_dynamic_library(library);
		return err;
	}

	GDExtensionInitializationFunction initialization_function = (GDExtensionInitializationFunction)entry_funcptr;
	GDExtensionBool ret = initialization_function(&gdextension_get_proc_address, this, &initialization);

	if (ret) {
		level_initialized = -1;
		return OK;
	} else {
		ERR_PRINT("GDExtension initialization function '" + p_entry_symbol + "' returned an error.");
		return FAILED;
	}
}

void GDExtension::close_library() {
	ERR_FAIL_NULL(library);
	OS::get_singleton()->close_dynamic_library(library);

#if defined(TOOLS_ENABLED) && defined(WINDOWS_ENABLED)
	// Delete temporary copy of library if it exists.
	if (!temp_lib_path.is_empty() && Engine::get_singleton()->is_editor_hint()) {
		DirAccess::remove_absolute(temp_lib_path);
	}
#endif

	library = nullptr;
}

bool GDExtension::is_library_open() const {
	return library != nullptr;
}

GDExtension::InitializationLevel GDExtension::get_minimum_library_initialization_level() const {
	ERR_FAIL_NULL_V(library, INITIALIZATION_LEVEL_CORE);
	return InitializationLevel(initialization.minimum_initialization_level);
}

void GDExtension::initialize_library(InitializationLevel p_level) {
	ERR_FAIL_NULL(library);
	ERR_FAIL_COND_MSG(p_level <= int32_t(level_initialized), vformat("Level '%d' must be higher than the current level '%d'", p_level, level_initialized));

	level_initialized = int32_t(p_level);

	ERR_FAIL_COND(initialization.initialize == nullptr);

	initialization.initialize(initialization.userdata, GDExtensionInitializationLevel(p_level));
}
void GDExtension::deinitialize_library(InitializationLevel p_level) {
	ERR_FAIL_NULL(library);
	ERR_FAIL_COND(p_level > int32_t(level_initialized));

	level_initialized = int32_t(p_level) - 1;
	initialization.deinitialize(initialization.userdata, GDExtensionInitializationLevel(p_level));
}

void GDExtension::_bind_methods() {
	ClassDB::bind_method(D_METHOD("open_library", "path", "entry_symbol"), &GDExtension::open_library);
	ClassDB::bind_method(D_METHOD("close_library"), &GDExtension::close_library);
	ClassDB::bind_method(D_METHOD("is_library_open"), &GDExtension::is_library_open);

	ClassDB::bind_method(D_METHOD("get_minimum_library_initialization_level"), &GDExtension::get_minimum_library_initialization_level);
	ClassDB::bind_method(D_METHOD("initialize_library", "level"), &GDExtension::initialize_library);

	BIND_ENUM_CONSTANT(INITIALIZATION_LEVEL_CORE);
	BIND_ENUM_CONSTANT(INITIALIZATION_LEVEL_SERVERS);
	BIND_ENUM_CONSTANT(INITIALIZATION_LEVEL_SCENE);
	BIND_ENUM_CONSTANT(INITIALIZATION_LEVEL_EDITOR);
}

GDExtension::GDExtension() {
}

GDExtension::~GDExtension() {
	if (library != nullptr) {
		close_library();
	}
}

void GDExtension::initialize_gdextensions() {
	gdextension_setup_interface();

#ifndef DISABLE_DEPRECATED
	register_interface_function("classdb_register_extension_class", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class);
#endif // DISABLE_DEPRECATED
	register_interface_function("classdb_register_extension_class2", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class2);
	register_interface_function("classdb_register_extension_class_method", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class_method);
	register_interface_function("classdb_register_extension_class_integer_constant", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class_integer_constant);
	register_interface_function("classdb_register_extension_class_property", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class_property);
	register_interface_function("classdb_register_extension_class_property_indexed", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class_property_indexed);
	register_interface_function("classdb_register_extension_class_property_group", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class_property_group);
	register_interface_function("classdb_register_extension_class_property_subgroup", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class_property_subgroup);
	register_interface_function("classdb_register_extension_class_signal", (GDExtensionInterfaceFunctionPtr)&GDExtension::_register_extension_class_signal);
	register_interface_function("classdb_unregister_extension_class", (GDExtensionInterfaceFunctionPtr)&GDExtension::_unregister_extension_class);
	register_interface_function("get_library_path", (GDExtensionInterfaceFunctionPtr)&GDExtension::_get_library_path);
}

Ref<Resource> GDExtensionResourceLoader::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<ConfigFile> config;
	config.instantiate();

	Error err = config->load(p_path);

	if (r_error) {
		*r_error = err;
	}

	if (err != OK) {
		ERR_PRINT("Error loading GDExtension configuration file: " + p_path);
		return Ref<Resource>();
	}

	if (!config->has_section_key("configuration", "entry_symbol")) {
		if (r_error) {
			*r_error = ERR_INVALID_DATA;
		}
		ERR_PRINT("GDExtension configuration file must contain a \"configuration/entry_symbol\" key: " + p_path);
		return Ref<Resource>();
	}

	String entry_symbol = config->get_value("configuration", "entry_symbol");

	uint32_t compatibility_minimum[3] = { 0, 0, 0 };
	if (config->has_section_key("configuration", "compatibility_minimum")) {
		String compat_string = config->get_value("configuration", "compatibility_minimum");
		Vector<int> parts = compat_string.split_ints(".");
		for (int i = 0; i < parts.size(); i++) {
			if (i >= 3) {
				break;
			}
			if (parts[i] >= 0) {
				compatibility_minimum[i] = parts[i];
			}
		}
	} else {
		if (r_error) {
			*r_error = ERR_INVALID_DATA;
		}
		ERR_PRINT("GDExtension configuration file must contain a \"configuration/compatibility_minimum\" key: " + p_path);
		return Ref<Resource>();
	}

	if (compatibility_minimum[0] < 4 || (compatibility_minimum[0] == 4 && compatibility_minimum[1] == 0)) {
		if (r_error) {
			*r_error = ERR_INVALID_DATA;
		}
		ERR_PRINT(vformat("GDExtension's compatibility_minimum (%d.%d.%d) must be at least 4.1.0: %s", compatibility_minimum[0], compatibility_minimum[1], compatibility_minimum[2], p_path));
		return Ref<Resource>();
	}

	bool compatible = true;
	// Check version lexicographically.
	if (VERSION_MAJOR != compatibility_minimum[0]) {
		compatible = VERSION_MAJOR > compatibility_minimum[0];
	} else if (VERSION_MINOR != compatibility_minimum[1]) {
		compatible = VERSION_MINOR > compatibility_minimum[1];
	} else {
		compatible = VERSION_PATCH >= compatibility_minimum[2];
	}
	if (!compatible) {
		if (r_error) {
			*r_error = ERR_INVALID_DATA;
		}
		ERR_PRINT(vformat("GDExtension only compatible with Godot version %d.%d.%d or later: %s", compatibility_minimum[0], compatibility_minimum[1], compatibility_minimum[2], p_path));
		return Ref<Resource>();
	}

	String library_path = GDExtension::find_extension_library(p_path, config, [](String p_feature) { return OS::get_singleton()->has_feature(p_feature); });

	if (library_path.is_empty()) {
		if (r_error) {
			*r_error = ERR_FILE_NOT_FOUND;
		}
		const String os_arch = OS::get_singleton()->get_name().to_lower() + "." + Engine::get_singleton()->get_architecture_name();
		ERR_PRINT(vformat("No GDExtension library found for current OS and architecture (%s) in configuration file: %s", os_arch, p_path));
		return Ref<Resource>();
	}

	if (!library_path.is_resource_file() && !library_path.is_absolute_path()) {
		library_path = p_path.get_base_dir().path_join(library_path);
	}

	Ref<GDExtension> lib;
	lib.instantiate();
	String abs_path = ProjectSettings::get_singleton()->globalize_path(library_path);

#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	// If running on the editor on Windows, we copy the library and open the copy.
	// This is so the original file isn't locked and can be updated by a compiler.
	if (Engine::get_singleton()->is_editor_hint()) {
		if (!FileAccess::exists(abs_path)) {
			if (r_error) {
				*r_error = ERR_FILE_NOT_FOUND;
			}
			ERR_PRINT("GDExtension library not found: " + library_path);
			return Ref<Resource>();
		}

		// Copy the file to the same directory as the original with a prefix in the name.
		// This is so relative path to dependencies are satisfied.
		String copy_path = abs_path.get_base_dir().path_join("~" + abs_path.get_file());

		// If there's a left-over copy (possibly from a crash) then delete it first.
		if (FileAccess::exists(copy_path)) {
			DirAccess::remove_absolute(copy_path);
		}

		Error copy_err = DirAccess::copy_absolute(abs_path, copy_path);
		if (copy_err) {
			if (r_error) {
				*r_error = ERR_CANT_CREATE;
			}
			ERR_PRINT("Error copying GDExtension library: " + library_path);
			return Ref<Resource>();
		}
		FileAccess::set_hidden_attribute(copy_path, true);

		// Save the copied path so it can be deleted later.
		lib->set_temp_library_path(copy_path);

		// Use the copy to open the library.
		abs_path = copy_path;
	}
#endif
	err = lib->open_library(abs_path, entry_symbol);

	if (r_error) {
		*r_error = err;
	}

	if (err != OK) {
#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
		// If the DLL fails to load, make sure that temporary DLL copies are cleaned up.
		if (Engine::get_singleton()->is_editor_hint()) {
			DirAccess::remove_absolute(lib->get_temp_library_path());
		}
#endif
		// Errors already logged in open_library()
		return Ref<Resource>();
	}

	// Handle icons if any are specified.
	if (config->has_section("icons")) {
		List<String> keys;
		config->get_section_keys("icons", &keys);
		for (const String &key : keys) {
			lib->class_icon_paths[key] = config->get_value("icons", key);
		}
	}

	return lib;
}

void GDExtensionResourceLoader::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("gdextension");
}

bool GDExtensionResourceLoader::handles_type(const String &p_type) const {
	return p_type == "GDExtension";
}

String GDExtensionResourceLoader::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "gdextension") {
		return "GDExtension";
	}
	return "";
}

#ifdef TOOLS_ENABLED
Vector<StringName> GDExtensionEditorPlugins::extension_classes;
GDExtensionEditorPlugins::EditorPluginRegisterFunc GDExtensionEditorPlugins::editor_node_add_plugin = nullptr;
GDExtensionEditorPlugins::EditorPluginRegisterFunc GDExtensionEditorPlugins::editor_node_remove_plugin = nullptr;

void GDExtensionEditorPlugins::add_extension_class(const StringName &p_class_name) {
	if (editor_node_add_plugin) {
		editor_node_add_plugin(p_class_name);
	} else {
		extension_classes.push_back(p_class_name);
	}
}

void GDExtensionEditorPlugins::remove_extension_class(const StringName &p_class_name) {
	if (editor_node_remove_plugin) {
		editor_node_remove_plugin(p_class_name);
	} else {
		extension_classes.erase(p_class_name);
	}
}
#endif // TOOLS_ENABLED
