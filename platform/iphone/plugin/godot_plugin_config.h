/*************************************************************************/
/*  godot_plugin_config.h                                                */
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

#ifndef GODOT_PLUGIN_CONFIG_H
#define GODOT_PLUGIN_CONFIG_H

#include "core/error/error_list.h"
#include "core/io/config_file.h"
#include "core/string/ustring.h"

/*
 The `config` section and fields are required and defined as follow:
- **name**: name of the plugin
- **binary**: path to static `.a` library

The `dependencies` and fields are optional.
- **linked**: dependencies that should only be linked.
- **embedded**: dependencies that should be linked and embedded into application.
- **system**: system dependencies that should be linked.
- **capabilities**: capabilities that would be used for `UIRequiredDeviceCapabilities` options in Info.plist file.
- **files**: files that would be copied into application

The `plist` section are optional.
- **key**: key and value that would be added in Info.plist file.
 */

struct PluginConfigIOS {
	inline static const char *PLUGIN_CONFIG_EXT = ".gdip";

	inline static const char *CONFIG_SECTION = "config";
	inline static const char *CONFIG_NAME_KEY = "name";
	inline static const char *CONFIG_BINARY_KEY = "binary";
	inline static const char *CONFIG_INITIALIZE_KEY = "initialization";
	inline static const char *CONFIG_DEINITIALIZE_KEY = "deinitialization";

	inline static const char *DEPENDENCIES_SECTION = "dependencies";
	inline static const char *DEPENDENCIES_LINKED_KEY = "linked";
	inline static const char *DEPENDENCIES_EMBEDDED_KEY = "embedded";
	inline static const char *DEPENDENCIES_SYSTEM_KEY = "system";
	inline static const char *DEPENDENCIES_CAPABILITIES_KEY = "capabilities";
	inline static const char *DEPENDENCIES_FILES_KEY = "files";
	inline static const char *DEPENDENCIES_LINKER_FLAGS = "linker_flags";

	inline static const char *PLIST_SECTION = "plist";

	// Set to true when the config file is properly loaded.
	bool valid_config = false;
	bool supports_targets = false;
	// Unix timestamp of last change to this plugin.
	uint64_t last_updated = 0;

	// Required config section
	String name;
	String binary;
	String initialization_method;
	String deinitialization_method;

	// Optional dependencies section
	Vector<String> linked_dependencies;
	Vector<String> embedded_dependencies;
	Vector<String> system_dependencies;

	Vector<String> files_to_copy;
	Vector<String> capabilities;

	Vector<String> linker_flags;

	// Optional plist section
	// Supports only string types for now
	HashMap<String, String> plist;
};

static inline String resolve_local_dependency_path(String plugin_config_dir, String dependency_path) {
	String absolute_path;

	if (dependency_path.is_empty()) {
		return absolute_path;
	}

	if (dependency_path.is_abs_path()) {
		return dependency_path;
	}

	String res_path = ProjectSettings::get_singleton()->globalize_path("res://");
	absolute_path = plugin_config_dir.plus_file(dependency_path);

	return absolute_path.replace(res_path, "res://");
}

static inline String resolve_system_dependency_path(String dependency_path) {
	String absolute_path;

	if (dependency_path.is_empty()) {
		return absolute_path;
	}

	if (dependency_path.is_abs_path()) {
		return dependency_path;
	}

	String system_path = "/System/Library/Frameworks";

	return system_path.plus_file(dependency_path);
}

static inline Vector<String> resolve_local_dependencies(String plugin_config_dir, Vector<String> p_paths) {
	Vector<String> paths;

	for (int i = 0; i < p_paths.size(); i++) {
		String path = resolve_local_dependency_path(plugin_config_dir, p_paths[i]);

		if (path.is_empty()) {
			continue;
		}

		paths.push_back(path);
	}

	return paths;
}

static inline Vector<String> resolve_system_dependencies(Vector<String> p_paths) {
	Vector<String> paths;

	for (int i = 0; i < p_paths.size(); i++) {
		String path = resolve_system_dependency_path(p_paths[i]);

		if (path.is_empty()) {
			continue;
		}

		paths.push_back(path);
	}

	return paths;
}

static inline bool validate_plugin(PluginConfigIOS &plugin_config) {
	bool valid_name = !plugin_config.name.is_empty();
	bool valid_binary_name = !plugin_config.binary.is_empty();
	bool valid_initialize = !plugin_config.initialization_method.is_empty();
	bool valid_deinitialize = !plugin_config.deinitialization_method.is_empty();

	bool fields_value = valid_name && valid_binary_name && valid_initialize && valid_deinitialize;

	if (!fields_value) {
		return false;
	}

	String plugin_extension = plugin_config.binary.get_extension().to_lower();

	if ((plugin_extension == "a" && FileAccess::exists(plugin_config.binary)) ||
			(plugin_extension == "xcframework" && DirAccess::exists(plugin_config.binary))) {
		plugin_config.valid_config = true;
		plugin_config.supports_targets = false;
	} else {
		String file_path = plugin_config.binary.get_base_dir();
		String file_name = plugin_config.binary.get_basename().get_file();
		String file_extension = plugin_config.binary.get_extension();
		String release_file_name = file_path.plus_file(file_name + ".release." + file_extension);
		String debug_file_name = file_path.plus_file(file_name + ".debug." + file_extension);

		if ((plugin_extension == "a" && FileAccess::exists(release_file_name) && FileAccess::exists(debug_file_name)) ||
				(plugin_extension == "xcframework" && DirAccess::exists(release_file_name) && DirAccess::exists(debug_file_name))) {
			plugin_config.valid_config = true;
			plugin_config.supports_targets = true;
		}
	}

	return plugin_config.valid_config;
}

static inline String get_plugin_main_binary(PluginConfigIOS &plugin_config, bool p_debug) {
	if (!plugin_config.supports_targets) {
		return plugin_config.binary;
	}

	String plugin_binary_dir = plugin_config.binary.get_base_dir();
	String plugin_name_prefix = plugin_config.binary.get_basename().get_file();
	String plugin_extension = plugin_config.binary.get_extension();
	String plugin_file = plugin_name_prefix + "." + (p_debug ? "debug" : "release") + "." + plugin_extension;

	return plugin_binary_dir.plus_file(plugin_file);
}

static inline uint64_t get_plugin_modification_time(const PluginConfigIOS &plugin_config, const String &config_path) {
	uint64_t last_updated = FileAccess::get_modified_time(config_path);

	if (!plugin_config.supports_targets) {
		last_updated = MAX(last_updated, FileAccess::get_modified_time(plugin_config.binary));
	} else {
		String file_path = plugin_config.binary.get_base_dir();
		String file_name = plugin_config.binary.get_basename().get_file();
		String plugin_extension = plugin_config.binary.get_extension();
		String release_file_name = file_path.plus_file(file_name + ".release." + plugin_extension);
		String debug_file_name = file_path.plus_file(file_name + ".debug." + plugin_extension);

		last_updated = MAX(last_updated, FileAccess::get_modified_time(release_file_name));
		last_updated = MAX(last_updated, FileAccess::get_modified_time(debug_file_name));
	}

	return last_updated;
}

static inline PluginConfigIOS load_plugin_config(Ref<ConfigFile> config_file, const String &path) {
	PluginConfigIOS plugin_config = {};

	if (!config_file.is_valid()) {
		return plugin_config;
	}

	Error err = config_file->load(path);

	if (err != OK) {
		return plugin_config;
	}

	String config_base_dir = path.get_base_dir();

	plugin_config.name = config_file->get_value(PluginConfigIOS::CONFIG_SECTION, PluginConfigIOS::CONFIG_NAME_KEY, String());
	plugin_config.initialization_method = config_file->get_value(PluginConfigIOS::CONFIG_SECTION, PluginConfigIOS::CONFIG_INITIALIZE_KEY, String());
	plugin_config.deinitialization_method = config_file->get_value(PluginConfigIOS::CONFIG_SECTION, PluginConfigIOS::CONFIG_DEINITIALIZE_KEY, String());

	String binary_path = config_file->get_value(PluginConfigIOS::CONFIG_SECTION, PluginConfigIOS::CONFIG_BINARY_KEY, String());
	plugin_config.binary = resolve_local_dependency_path(config_base_dir, binary_path);

	if (config_file->has_section(PluginConfigIOS::DEPENDENCIES_SECTION)) {
		Vector<String> linked_dependencies = config_file->get_value(PluginConfigIOS::DEPENDENCIES_SECTION, PluginConfigIOS::DEPENDENCIES_LINKED_KEY, Vector<String>());
		Vector<String> embedded_dependencies = config_file->get_value(PluginConfigIOS::DEPENDENCIES_SECTION, PluginConfigIOS::DEPENDENCIES_EMBEDDED_KEY, Vector<String>());
		Vector<String> system_dependencies = config_file->get_value(PluginConfigIOS::DEPENDENCIES_SECTION, PluginConfigIOS::DEPENDENCIES_SYSTEM_KEY, Vector<String>());
		Vector<String> files = config_file->get_value(PluginConfigIOS::DEPENDENCIES_SECTION, PluginConfigIOS::DEPENDENCIES_FILES_KEY, Vector<String>());

		plugin_config.linked_dependencies = resolve_local_dependencies(config_base_dir, linked_dependencies);
		plugin_config.embedded_dependencies = resolve_local_dependencies(config_base_dir, embedded_dependencies);
		plugin_config.system_dependencies = resolve_system_dependencies(system_dependencies);

		plugin_config.files_to_copy = resolve_local_dependencies(config_base_dir, files);

		plugin_config.capabilities = config_file->get_value(PluginConfigIOS::DEPENDENCIES_SECTION, PluginConfigIOS::DEPENDENCIES_CAPABILITIES_KEY, Vector<String>());

		plugin_config.linker_flags = config_file->get_value(PluginConfigIOS::DEPENDENCIES_SECTION, PluginConfigIOS::DEPENDENCIES_LINKER_FLAGS, Vector<String>());
	}

	if (config_file->has_section(PluginConfigIOS::PLIST_SECTION)) {
		List<String> keys;
		config_file->get_section_keys(PluginConfigIOS::PLIST_SECTION, &keys);

		for (int i = 0; i < keys.size(); i++) {
			String value = config_file->get_value(PluginConfigIOS::PLIST_SECTION, keys[i], String());

			if (value.is_empty()) {
				continue;
			}

			plugin_config.plist[keys[i]] = value;
		}
	}

	if (validate_plugin(plugin_config)) {
		plugin_config.last_updated = get_plugin_modification_time(plugin_config, path);
	}

	return plugin_config;
}

#endif // GODOT_PLUGIN_CONFIG_H
