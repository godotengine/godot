/**************************************************************************/
/*  gdextension.h                                                         */
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

#ifndef GDEXTENSION_H
#define GDEXTENSION_H

#include <functional>

#include "core/extension/gdextension_interface.h"
#include "core/io/config_file.h"
#include "core/io/resource_loader.h"
#include "core/object/ref_counted.h"

class GDExtensionMethodBind;

class GDExtension : public Resource {
	GDCLASS(GDExtension, Resource)

	friend class GDExtensionManager;

	void *library = nullptr; // pointer if valid,
	String library_path;
#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	String temp_lib_path;
#endif
	bool reloadable = false;

	struct Extension {
		ObjectGDExtension gdextension;

#ifdef TOOLS_ENABLED
		bool is_reloading = false;
		HashMap<StringName, GDExtensionMethodBind *> methods;
		HashSet<ObjectID> instances;
		HashMap<ObjectID, List<Pair<String, Variant>>> instance_state;
#endif
	};

	HashMap<StringName, Extension> extension_classes;

	struct ClassCreationDeprecatedInfo {
#ifndef DISABLE_DEPRECATED
		GDExtensionClassNotification notification_func = nullptr;
#endif // DISABLE_DEPRECATED
	};

#ifndef DISABLE_DEPRECATED
	static void _register_extension_class(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, const GDExtensionClassCreationInfo *p_extension_funcs);
#endif // DISABLE_DEPRECATED
	static void _register_extension_class2(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, const GDExtensionClassCreationInfo2 *p_extension_funcs);
	static void _register_extension_class_internal(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_parent_class_name, const GDExtensionClassCreationInfo2 *p_extension_funcs, const ClassCreationDeprecatedInfo *p_deprecated_funcs = nullptr);
	static void _register_extension_class_method(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionClassMethodInfo *p_method_info);
	static void _register_extension_class_integer_constant(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_enum_name, GDExtensionConstStringNamePtr p_constant_name, GDExtensionInt p_constant_value, GDExtensionBool p_is_bitfield);
	static void _register_extension_class_property(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionPropertyInfo *p_info, GDExtensionConstStringNamePtr p_setter, GDExtensionConstStringNamePtr p_getter);
	static void _register_extension_class_property_indexed(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, const GDExtensionPropertyInfo *p_info, GDExtensionConstStringNamePtr p_setter, GDExtensionConstStringNamePtr p_getter, GDExtensionInt p_index);
	static void _register_extension_class_property_group(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_group_name, GDExtensionConstStringNamePtr p_prefix);
	static void _register_extension_class_property_subgroup(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_subgroup_name, GDExtensionConstStringNamePtr p_prefix);
	static void _register_extension_class_signal(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name, GDExtensionConstStringNamePtr p_signal_name, const GDExtensionPropertyInfo *p_argument_info, GDExtensionInt p_argument_count);
	static void _unregister_extension_class(GDExtensionClassLibraryPtr p_library, GDExtensionConstStringNamePtr p_class_name);
	static void _get_library_path(GDExtensionClassLibraryPtr p_library, GDExtensionStringPtr r_path);

	GDExtensionInitialization initialization;
	int32_t level_initialized = -1;

#ifdef TOOLS_ENABLED
	uint64_t resource_last_modified_time = 0;
	uint64_t library_last_modified_time = 0;
	bool is_reloading = false;
	Vector<GDExtensionMethodBind *> invalid_methods;
	Vector<ObjectID> instance_bindings;

	static void _track_instance(void *p_user_data, void *p_instance);
	static void _untrack_instance(void *p_user_data, void *p_instance);

	void _clear_extension(Extension *p_extension);

	// Only called by GDExtensionManager during the reload process.
	void prepare_reload();
	void finish_reload();
	void clear_instance_bindings();
#endif

	static HashMap<StringName, GDExtensionInterfaceFunctionPtr> gdextension_interface_functions;

protected:
	static void _bind_methods();

public:
	HashMap<String, String> class_icon_paths;

	virtual bool editor_can_reload_from_file() override { return false; } // Reloading is handled in a special way.

	static String get_extension_list_config_file();
	static String find_extension_library(const String &p_path, Ref<ConfigFile> p_config, std::function<bool(String)> p_has_feature, PackedStringArray *r_tags = nullptr);

	Error open_library(const String &p_path, const String &p_entry_symbol);
	void close_library();

#if defined(WINDOWS_ENABLED) && defined(TOOLS_ENABLED)
	String get_temp_library_path() const { return temp_lib_path; }
#endif

	enum InitializationLevel {
		INITIALIZATION_LEVEL_CORE = GDEXTENSION_INITIALIZATION_CORE,
		INITIALIZATION_LEVEL_SERVERS = GDEXTENSION_INITIALIZATION_SERVERS,
		INITIALIZATION_LEVEL_SCENE = GDEXTENSION_INITIALIZATION_SCENE,
		INITIALIZATION_LEVEL_EDITOR = GDEXTENSION_INITIALIZATION_EDITOR
	};

	bool is_library_open() const;

#ifdef TOOLS_ENABLED
	bool is_reloadable() const { return reloadable; }
	void set_reloadable(bool p_reloadable) { reloadable = p_reloadable; }

	bool has_library_changed() const;
	void update_last_modified_time(uint64_t p_resource_last_modified_time, uint64_t p_library_last_modified_time) {
		resource_last_modified_time = p_resource_last_modified_time;
		library_last_modified_time = p_library_last_modified_time;
	}

	void track_instance_binding(Object *p_object);
	void untrack_instance_binding(Object *p_object);
#endif

	InitializationLevel get_minimum_library_initialization_level() const;
	void initialize_library(InitializationLevel p_level);
	void deinitialize_library(InitializationLevel p_level);

	static void register_interface_function(StringName p_function_name, GDExtensionInterfaceFunctionPtr p_function_pointer);
	static GDExtensionInterfaceFunctionPtr get_interface_function(StringName p_function_name);
	static void initialize_gdextensions();
	static void finalize_gdextensions();

	GDExtension();
	~GDExtension();
};

VARIANT_ENUM_CAST(GDExtension::InitializationLevel)

class GDExtensionResourceLoader : public ResourceFormatLoader {
public:
	static Error load_gdextension_resource(const String &p_path, Ref<GDExtension> &p_extension);

	virtual Ref<Resource> load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

#ifdef TOOLS_ENABLED
class GDExtensionEditorPlugins {
private:
	static Vector<StringName> extension_classes;

protected:
	friend class EditorNode;

	// Since this in core, we can't directly reference EditorNode, so it will
	// set these function pointers in its constructor.
	typedef void (*EditorPluginRegisterFunc)(const StringName &p_class_name);
	static EditorPluginRegisterFunc editor_node_add_plugin;
	static EditorPluginRegisterFunc editor_node_remove_plugin;

public:
	static void add_extension_class(const StringName &p_class_name);
	static void remove_extension_class(const StringName &p_class_name);

	static const Vector<StringName> &get_extension_classes() {
		return extension_classes;
	}
};
#endif // TOOLS_ENABLED

#endif // GDEXTENSION_H
