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
#include "gdextension_manager.h"

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

#ifdef TOOLS_ENABLED
	friend class GDExtension;

	StringName name;
	bool is_reloading = false;
	bool valid = true;
#endif

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
#ifdef TOOLS_ENABLED
	virtual bool is_valid() const override { return valid; }
#endif

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
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_V_MSG(!valid, Variant(), vformat("Cannot call invalid GDExtension method bind '%s'. It's probably cached - you may need to restart Godot.", name));
#endif
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
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(!valid, vformat("Cannot call invalid GDExtension method bind '%s'. It's probably cached - you may need to restart Godot.", name));
#endif
		ERR_FAIL_COND_MSG(vararg, "Vararg methods don't have validated call support. This is most likely an engine bug.");
		GDExtensionClassInstancePtr extension_instance = is_static() ? nullptr : p_object->_get_extension_instance();

		if (validated_call_func) {
			// This is added here, but it's unlikely to be provided by most extensions.
			validated_call_func(method_userdata, extension_instance, reinterpret_cast<GDExtensionConstVariantPtr *>(p_args), (GDExtensionVariantPtr)r_ret);
		} else {
			// If not provided, go via ptrcall, which is faster than resorting to regular call.
			const void **argptrs = (const void **)alloca(argument_count * sizeof(void *));
			for (uint32_t i = 0; i < argument_count; i++) {
				argptrs[i] = VariantInternal::get_opaque_pointer(p_args[i]);
			}

			void *ret_opaque = nullptr;
			if (r_ret) {
				VariantInternal::initialize(r_ret, return_value_info.type);
				ret_opaque = r_ret->get_type() == Variant::NIL ? r_ret : VariantInternal::get_opaque_pointer(r_ret);
			}

			ptrcall(p_object, argptrs, ret_opaque);

			if (r_ret && r_ret->get_type() == Variant::OBJECT) {
				VariantInternal::update_object_id(r_ret);
			}
		}
	}

	virtual void ptrcall(Object *p_object, const void **p_args, void *r_ret) const override {
#ifdef TOOLS_ENABLED
		ERR_FAIL_COND_MSG(!valid, vformat("Cannot call invalid GDExtension method bind '%s'. It's probably cached - you may need to restart Godot.", name));
#endif
		ERR_FAIL_COND_MSG(vararg, "Vararg methods don't have ptrcall support. This is most likely an engine bug.");
		GDExtensionClassInstancePtr extension_instance = p_object->_get_extension_instance();
		ptrcall_func(method_userdata, extension_instance, reinterpret_cast<GDExtensionConstTypePtr *>(p_args), (GDExtensionTypePtr)r_ret);
	}

	virtual bool is_vararg() const override {
		return false;
	}

#ifdef TOOLS_ENABLED
	bool try_update(const GDExtensionClassMethodInfo *p_method_info) {
		if (is_static() != (bool)(p_method_info->method_flags & GDEXTENSION_METHOD_FLAG_STATIC)) {
			return false;
		}

		if (vararg != (bool)(p_method_info->method_flags & GDEXTENSION_METHOD_FLAG_VARARG)) {
			return false;
		}

		if (has_return() != (bool)p_method_info->has_return_value) {
			return false;
		}

		if (has_return() && return_value_info.type != (Variant::Type)p_method_info->return_value_info->type) {
			return false;
		}

		if (argument_count != p_method_info->argument_count) {
			return false;
		}

		for (uint32_t i = 0; i < p_method_info->argument_count; i++) {
			if (arguments_info[i].type != (Variant::Type)p_method_info->arguments_info[i].type) {
				return false;
			}
		}

		update(p_method_info);
		return true;
	}
#endif

	void update(const GDExtensionClassMethodInfo *p_method_info) {
#ifdef TOOLS_ENABLED
		name = *reinterpret_cast<StringName *>(p_method_info->name);
#endif
		method_userdata = p_method_info->method_userdata;
		call_func = p_method_info->call_func;
		validated_call_func = nullptr;
		ptrcall_func = p_method_info->ptrcall_func;
		set_name(*reinterpret_cast<StringName *>(p_method_info->name));

		if (p_method_info->has_return_value) {
			return_value_info = PropertyInfo(*p_method_info->return_value_info);
			return_value_metadata = GodotTypeInfo::Metadata(p_method_info->return_value_metadata);
		}

		arguments_info.clear();
		arguments_metadata.clear();
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

	explicit GDExtensionMethodBind(const GDExtensionClassMethodInfo *p_method_info) {
		update(p_method_info);
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
		nullptr, // GDExtensionClassRecreateInstance recreate_instance_func;
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

#ifdef TOOLS_ENABLED
	Extension *extension = nullptr;
	if (self->is_reloading && self->extension_classes.has(class_name)) {
		extension = &self->extension_classes[class_name];
		if (!parent_extension && parent_class_name != extension->gdextension.parent_class_name) {
			ERR_FAIL_MSG(vformat("GDExtension class '%s' attempt to change parent type from '%s' to '%s' on hot reload. Restart Godot for this change to take effect.", class_name, extension->gdextension.parent_class_name, parent_class_name));
		}
		extension->is_reloading = false;
	} else {
		self->extension_classes[class_name] = Extension();
		extension = &self->extension_classes[class_name];
	}
#else
	self->extension_classes[class_name] = Extension();
	Extension *extension = &self->extension_classes[class_name];
#endif

	if (parent_extension) {
		extension->gdextension.parent = &parent_extension->gdextension;
		parent_extension->gdextension.children.push_back(&extension->gdextension);
	}

	if (self->reloadable && p_extension_funcs->recreate_instance_func == nullptr) {
		ERR_PRINT(vformat("Extension marked as reloadable, but attempted to register class '%s' which doesn't support reloading. Perhaps your language binding don't support it? Reloading disabled for this extension.", class_name));
		self->reloadable = false;
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
	extension->gdextension.recreate_instance = p_extension_funcs->recreate_instance_func;
	extension->gdextension.get_virtual = p_extension_funcs->get_virtual_func;
	extension->gdextension.get_virtual_call_data = p_extension_funcs->get_virtual_call_data_func;
	extension->gdextension.call_virtual_with_data = p_extension_funcs->call_virtual_with_data_func;
	extension->gdextension.get_rid = p_extension_funcs->get_rid_func;

	extension->gdextension.reloadable = self->reloadable;
#ifdef TOOLS_ENABLED
	if (extension->gdextension.reloadable) {
		extension->gdextension.tracking_userdata = extension;
		extension->gdextension.track_instance = &GDExtension::_track_instance;
		extension->gdextension.untrack_instance = &GDExtension::_untrack_instance;
	} else {
		extension->gdextension.tracking_userdata = nullptr;
		extension->gdextension.track_instance = nullptr;
		extension->gdextension.untrack_instance = nullptr;
	}
#endif

	ClassDB::register_extension_class(&extension->gdextension);
}

void GDExtension::_register_extension_class_method(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionClassMethodInfo *p_method_info) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	StringName method_name = *reinterpret_cast<const StringName *>(p_method_info->name);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension method '" + String(method_name) + "' for unexisting class '" + class_name + "'.");

#ifdef TOOLS_ENABLED
	Extension *extension = &self->extension_classes[class_name];
	GDExtensionMethodBind *method = nullptr;

	// If the extension is still marked as reloading, that means it failed to register again.
	if (extension->is_reloading) {
		return;
	}

	if (self->is_reloading && extension->methods.has(method_name)) {
		method = extension->methods[method_name];

		// Try to update the method bind. If it doesn't work (because it's incompatible) then
		// mark as invalid and create a new one.
		if (!method->is_reloading || !method->try_update(p_method_info)) {
			method->valid = false;
			self->invalid_methods.push_back(method);

			method = nullptr;
		}
	}

	if (method == nullptr) {
		method = memnew(GDExtensionMethodBind(p_method_info));
		method->set_instance_class(class_name);
		extension->methods[method_name] = method;
	} else {
		method->is_reloading = false;
	}
#else
	GDExtensionMethodBind *method = memnew(GDExtensionMethodBind(p_method_info));
	method->set_instance_class(class_name);
#endif

	ClassDB::bind_method_custom(class_name, method);
}
void GDExtension::_register_extension_class_integer_constant(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_enum_name, GDExtensionConstStringNamePtr p_constant_name, GDExtensionInt p_constant_value, GDExtensionBool p_is_bitfield) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	StringName enum_name = *reinterpret_cast<const StringName *>(p_enum_name);
	StringName constant_name = *reinterpret_cast<const StringName *>(p_constant_name);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension constant '" + constant_name + "' for unexisting class '" + class_name + "'.");

#ifdef TOOLS_ENABLED
	// If the extension is still marked as reloading, that means it failed to register again.
	Extension *extension = &self->extension_classes[class_name];
	if (extension->is_reloading) {
		return;
	}
#endif

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

#ifdef TOOLS_ENABLED
	// If the extension is still marked as reloading, that means it failed to register again.
	Extension *extension = &self->extension_classes[class_name];
	if (extension->is_reloading) {
		return;
	}
#endif

	PropertyInfo pinfo(*p_info);

	ClassDB::add_property(class_name, pinfo, setter, getter, p_index);
}

void GDExtension::_register_extension_class_property_group(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringPtr p_group_name, GDExtensionConstStringPtr p_prefix) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	String group_name = *reinterpret_cast<const String *>(p_group_name);
	String prefix = *reinterpret_cast<const String *>(p_prefix);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension class property group '" + group_name + "' for unexisting class '" + class_name + "'.");

#ifdef TOOLS_ENABLED
	// If the extension is still marked as reloading, that means it failed to register again.
	Extension *extension = &self->extension_classes[class_name];
	if (extension->is_reloading) {
		return;
	}
#endif

	ClassDB::add_property_group(class_name, group_name, prefix);
}

void GDExtension::_register_extension_class_property_subgroup(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringPtr p_subgroup_name, GDExtensionConstStringPtr p_prefix) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	String subgroup_name = *reinterpret_cast<const String *>(p_subgroup_name);
	String prefix = *reinterpret_cast<const String *>(p_prefix);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension class property subgroup '" + subgroup_name + "' for unexisting class '" + class_name + "'.");

#ifdef TOOLS_ENABLED
	// If the extension is still marked as reloading, that means it failed to register again.
	Extension *extension = &self->extension_classes[class_name];
	if (extension->is_reloading) {
		return;
	}
#endif

	ClassDB::add_property_subgroup(class_name, subgroup_name, prefix);
}

void GDExtension::_register_extension_class_signal(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_signal_name, const GDExtensionPropertyInfo *p_argument_info, GDExtensionInt p_argument_count) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	StringName class_name = *reinterpret_cast<const StringName *>(p_class_name);
	StringName signal_name = *reinterpret_cast<const StringName *>(p_signal_name);
	ERR_FAIL_COND_MSG(!self->extension_classes.has(class_name), "Attempt to register extension class signal '" + signal_name + "' for unexisting class '" + class_name + "'.");

#ifdef TOOLS_ENABLED
	// If the extension is still marked as reloading, that means it failed to register again.
	Extension *extension = &self->extension_classes[class_name];
	if (extension->is_reloading) {
		return;
	}
#endif

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
#ifdef TOOLS_ENABLED
	if (ext->is_reloading) {
		self->_clear_extension(ext);
	}
#endif
	ERR_FAIL_COND_MSG(ext->gdextension.children.size(), "Attempt to unregister class '" + class_name + "' while other extension classes inherit from it.");

#ifdef TOOLS_ENABLED
	ClassDB::unregister_extension_class(class_name, !ext->is_reloading);
#else
	ClassDB::unregister_extension_class(class_name);
#endif

	if (ext->gdextension.parent != nullptr) {
		ext->gdextension.parent->children.erase(&ext->gdextension);
	}

#ifdef TOOLS_ENABLED
	if (!ext->is_reloading) {
		self->extension_classes.erase(class_name);
	}
#else
	self->extension_classes.erase(class_name);
#endif
}

void GDExtension::_get_library_path(GDExtensionClassLibraryPtr p_library, GDExtensionUninitializedStringPtr r_path) {
	GDExtension *self = reinterpret_cast<GDExtension *>(p_library);

	memnew_placement(r_path, String(self->library_path));
}

HashMap<StringName, GDExtensionInterfaceFunctionPtr> GDExtension::gdextension_interface_functions;

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
	String abs_path = ProjectSettings::get_singleton()->globalize_path(p_path);
#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	// If running on the editor on Windows, we copy the library and open the copy.
	// This is so the original file isn't locked and can be updated by a compiler.
	bool library_copied = false;
	if (Engine::get_singleton()->is_editor_hint()) {
		if (!FileAccess::exists(abs_path)) {
			ERR_PRINT("GDExtension library not found: " + library_path);
			return ERR_FILE_NOT_FOUND;
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
			ERR_PRINT("Error copying GDExtension library: " + library_path);
			return ERR_CANT_CREATE;
		}
		FileAccess::set_hidden_attribute(copy_path, true);
		library_copied = true;

		// Save the copied path so it can be deleted later.
		temp_lib_path = copy_path;

		// Use the copy to open the library.
		abs_path = copy_path;
	}
#endif

	Error err = OS::get_singleton()->open_dynamic_library(abs_path, library, true, &library_path);
	if (err != OK) {
		ERR_PRINT("GDExtension dynamic library not found: " + abs_path);
		return err;
	}

#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	// If we copied the file, let's change the library path to point at the original,
	// because that's what we want to check to see if it's changed.
	if (library_copied) {
		library_path = library_path.get_base_dir() + "\\" + p_path.get_file();
	}
#endif

	void *entry_funcptr = nullptr;

	err = OS::get_singleton()->get_dynamic_library_symbol_handle(library, p_entry_symbol, entry_funcptr, false);

	if (err != OK) {
		ERR_PRINT("GDExtension entry point '" + p_entry_symbol + "' not found in library " + abs_path);
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
		OS::get_singleton()->close_dynamic_library(library);
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
	class_icon_paths.clear();

#ifdef TOOLS_ENABLED
	instance_bindings.clear();
#endif
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

	ERR_FAIL_NULL(initialization.initialize);

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
#ifdef TOOLS_ENABLED
	// If we have any invalid method binds still laying around, we can finally free them!
	for (GDExtensionMethodBind *E : invalid_methods) {
		memdelete(E);
	}
#endif
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

void GDExtension::finalize_gdextensions() {
	gdextension_interface_functions.clear();
}

Error GDExtensionResourceLoader::load_gdextension_resource(const String &p_path, Ref<GDExtension> &p_extension) {
	ERR_FAIL_COND_V_MSG(p_extension.is_valid() && p_extension->is_library_open(), ERR_ALREADY_IN_USE, "Cannot load GDExtension resource into already opened library.");

	Ref<ConfigFile> config;
	config.instantiate();

	Error err = config->load(p_path);

	if (err != OK) {
		ERR_PRINT("Error loading GDExtension configuration file: " + p_path);
		return err;
	}

	if (!config->has_section_key("configuration", "entry_symbol")) {
		ERR_PRINT("GDExtension configuration file must contain a \"configuration/entry_symbol\" key: " + p_path);
		return ERR_INVALID_DATA;
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
		ERR_PRINT("GDExtension configuration file must contain a \"configuration/compatibility_minimum\" key: " + p_path);
		return ERR_INVALID_DATA;
	}

	if (compatibility_minimum[0] < 4 || (compatibility_minimum[0] == 4 && compatibility_minimum[1] == 0)) {
		ERR_PRINT(vformat("GDExtension's compatibility_minimum (%d.%d.%d) must be at least 4.1.0: %s", compatibility_minimum[0], compatibility_minimum[1], compatibility_minimum[2], p_path));
		return ERR_INVALID_DATA;
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
		ERR_PRINT(vformat("GDExtension only compatible with Godot version %d.%d.%d or later: %s", compatibility_minimum[0], compatibility_minimum[1], compatibility_minimum[2], p_path));
		return ERR_INVALID_DATA;
	}

	String library_path = GDExtension::find_extension_library(p_path, config, [](String p_feature) { return OS::get_singleton()->has_feature(p_feature); });

	if (library_path.is_empty()) {
		const String os_arch = OS::get_singleton()->get_name().to_lower() + "." + Engine::get_singleton()->get_architecture_name();
		ERR_PRINT(vformat("No GDExtension library found for current OS and architecture (%s) in configuration file: %s", os_arch, p_path));
		return ERR_FILE_NOT_FOUND;
	}

	bool is_static_library = library_path.ends_with(".a") || library_path.ends_with(".xcframework");

	if (!library_path.is_resource_file() && !library_path.is_absolute_path()) {
		library_path = p_path.get_base_dir().path_join(library_path);
	}

	if (p_extension.is_null()) {
		p_extension.instantiate();
	}

#ifdef TOOLS_ENABLED
	p_extension->set_reloadable(config->get_value("configuration", "reloadable", false) && Engine::get_singleton()->is_extension_reloading_enabled());

	p_extension->update_last_modified_time(
			FileAccess::get_modified_time(p_path),
			FileAccess::get_modified_time(library_path));
#endif

	err = p_extension->open_library(is_static_library ? String() : library_path, entry_symbol);
	if (err != OK) {
#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
		// If the DLL fails to load, make sure that temporary DLL copies are cleaned up.
		if (Engine::get_singleton()->is_editor_hint()) {
			DirAccess::remove_absolute(p_extension->get_temp_library_path());
		}
#endif

		// Unreference the extension so that this loading can be considered a failure.
		p_extension.unref();

		// Errors already logged in open_library()
		return err;
	}

	// Handle icons if any are specified.
	if (config->has_section("icons")) {
		List<String> keys;
		config->get_section_keys("icons", &keys);
		for (const String &key : keys) {
			String icon_path = config->get_value("icons", key);
			if (icon_path.is_relative_path()) {
				icon_path = p_path.get_base_dir().path_join(icon_path);
			}

			p_extension->class_icon_paths[key] = icon_path;
		}
	}

	return OK;
}

Ref<Resource> GDExtensionResourceLoader::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	// We can't have two GDExtension resource object representing the same library, because
	// loading (or unloading) a GDExtension affects global data. So, we need reuse the same
	// object if one has already been loaded (even if caching is disabled at the resource
	// loader level).
	GDExtensionManager *manager = GDExtensionManager::get_singleton();
	if (manager->is_extension_loaded(p_path)) {
		return manager->get_extension(p_path);
	}

	Ref<GDExtension> lib;
	Error err = load_gdextension_resource(p_path, lib);
	if (err != OK && r_error) {
		// Errors already logged in load_gdextension_resource().
		*r_error = err;
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
bool GDExtension::has_library_changed() const {
	// Check only that the last modified time is different (rather than checking
	// that it's newer) since some OS's (namely Windows) will preserve the modified
	// time by default when copying files.
	if (FileAccess::get_modified_time(get_path()) != resource_last_modified_time) {
		return true;
	}
	if (FileAccess::get_modified_time(library_path) != library_last_modified_time) {
		return true;
	}
	return false;
}

void GDExtension::prepare_reload() {
	is_reloading = true;

	for (KeyValue<StringName, Extension> &E : extension_classes) {
		E.value.is_reloading = true;

		for (KeyValue<StringName, GDExtensionMethodBind *> &M : E.value.methods) {
			M.value->is_reloading = true;
		}

		for (const ObjectID &obj_id : E.value.instances) {
			Object *obj = ObjectDB::get_instance(obj_id);
			if (!obj) {
				continue;
			}

			// Store instance state so it can be restored after reload.
			List<Pair<String, Variant>> state;
			List<PropertyInfo> prop_list;
			obj->get_property_list(&prop_list);
			for (const PropertyInfo &P : prop_list) {
				if (!(P.usage & PROPERTY_USAGE_STORAGE)) {
					continue;
				}

				Variant value = obj->get(P.name);
				Variant default_value = ClassDB::class_get_default_property_value(obj->get_class_name(), P.name);

				if (default_value.get_type() != Variant::NIL && bool(Variant::evaluate(Variant::OP_EQUAL, value, default_value))) {
					continue;
				}

				if (P.type == Variant::OBJECT && value.is_zero() && !(P.usage & PROPERTY_USAGE_STORE_IF_NULL)) {
					continue;
				}

				state.push_back(Pair<String, Variant>(P.name, value));
			}
			E.value.instance_state[obj_id] = state;
		}
	}
}

void GDExtension::_clear_extension(Extension *p_extension) {
	// Clear out hierarchy information because it may change.
	p_extension->gdextension.parent = nullptr;
	p_extension->gdextension.children.clear();

	// Clear all objects of any GDExtension data. It will become its native parent class
	// until the reload can reset the object with the new GDExtension data.
	for (const ObjectID &obj_id : p_extension->instances) {
		Object *obj = ObjectDB::get_instance(obj_id);
		if (!obj) {
			continue;
		}

		obj->clear_internal_extension();
	}
}

void GDExtension::track_instance_binding(Object *p_object) {
	instance_bindings.push_back(p_object->get_instance_id());
}

void GDExtension::untrack_instance_binding(Object *p_object) {
	instance_bindings.erase(p_object->get_instance_id());
}

void GDExtension::clear_instance_bindings() {
	for (ObjectID obj_id : instance_bindings) {
		Object *obj = ObjectDB::get_instance(obj_id);
		if (!obj) {
			continue;
		}

		obj->free_instance_binding(this);
	}
	instance_bindings.clear();
}

void GDExtension::finish_reload() {
	is_reloading = false;

	// Clean up any classes or methods that didn't get re-added.
	Vector<StringName> classes_to_remove;
	for (KeyValue<StringName, Extension> &E : extension_classes) {
		if (E.value.is_reloading) {
			E.value.is_reloading = false;
			classes_to_remove.push_back(E.key);
		}

		Vector<StringName> methods_to_remove;
		for (KeyValue<StringName, GDExtensionMethodBind *> &M : E.value.methods) {
			if (M.value->is_reloading) {
				M.value->valid = false;
				invalid_methods.push_back(M.value);

				M.value->is_reloading = false;
				methods_to_remove.push_back(M.key);
			}
		}
		for (const StringName &method_name : methods_to_remove) {
			E.value.methods.erase(method_name);
		}
	}
	for (const StringName &class_name : classes_to_remove) {
		extension_classes.erase(class_name);
	}

	// Reset any the extension on instances made from the classes that remain.
	for (KeyValue<StringName, Extension> &E : extension_classes) {
		// Loop over 'instance_state' rather than 'instance' because new instances
		// may have been created when re-initializing the extension.
		for (const KeyValue<ObjectID, List<Pair<String, Variant>>> &S : E.value.instance_state) {
			Object *obj = ObjectDB::get_instance(S.key);
			if (!obj) {
				continue;
			}

			obj->reset_internal_extension(&E.value.gdextension);
		}
	}

	// Now that all the classes are back, restore the state.
	for (KeyValue<StringName, Extension> &E : extension_classes) {
		for (const KeyValue<ObjectID, List<Pair<String, Variant>>> &S : E.value.instance_state) {
			Object *obj = ObjectDB::get_instance(S.key);
			if (!obj) {
				continue;
			}

			for (const Pair<String, Variant> &state : S.value) {
				obj->set(state.first, state.second);
			}
		}
	}

	// Finally, let the objects know that we are done reloading them.
	for (KeyValue<StringName, Extension> &E : extension_classes) {
		for (const KeyValue<ObjectID, List<Pair<String, Variant>>> &S : E.value.instance_state) {
			Object *obj = ObjectDB::get_instance(S.key);
			if (!obj) {
				continue;
			}

			obj->notification(NOTIFICATION_EXTENSION_RELOADED);
		}

		// Clear the instance state, we're done looping.
		E.value.instance_state.clear();
	}
}

void GDExtension::_track_instance(void *p_user_data, void *p_instance) {
	Extension *extension = reinterpret_cast<Extension *>(p_user_data);
	Object *obj = reinterpret_cast<Object *>(p_instance);

	extension->instances.insert(obj->get_instance_id());
}

void GDExtension::_untrack_instance(void *p_user_data, void *p_instance) {
	Extension *extension = reinterpret_cast<Extension *>(p_user_data);
	Object *obj = reinterpret_cast<Object *>(p_instance);

	extension->instances.erase(obj->get_instance_id());
}

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
