/*************************************************************************/
/*  godot_plugin_config.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/error_list.h"
#include "core/io/config_file.h"
#include "core/ustring.h"

static const char *PLUGIN_CONFIG_EXT = ".gdap";

static const char *CONFIG_SECTION = "config";
static const char *CONFIG_NAME_KEY = "name";
static const char *CONFIG_BINARY_TYPE_KEY = "binary_type";
static const char *CONFIG_BINARY_KEY = "binary";

static const char *DEPENDENCIES_SECTION = "dependencies";
static const char *DEPENDENCIES_LOCAL_KEY = "local";
static const char *DEPENDENCIES_REMOTE_KEY = "remote";
static const char *DEPENDENCIES_CUSTOM_MAVEN_REPOS_KEY = "custom_maven_repos";

static const char *BINARY_TYPE_LOCAL = "local";
static const char *BINARY_TYPE_REMOTE = "remote";

static const char *PLUGIN_VALUE_SEPARATOR = "|";

/*
 The `config` section and fields are required and defined as follow:
- **name**: name of the plugin
- **binary_type**: can be either `local` or `remote`.  The type affects the **binary** field
- **binary**:
  - if **binary_type** is `local`, then this should be the filename of the plugin `aar` file in the `res://android/plugins` directory (e.g: `MyPlugin.aar`).
  - if **binary_type** is `remote`, then this should be a declaration for a remote gradle binary (e.g: "org.godot.example:my-plugin:0.0.0").

The `dependencies` section and fields are optional and defined as follow:
- **local**: contains a list of local `.aar` binary files the plugin depends on. The local binary dependencies must also be located in the `res://android/plugins` directory.
- **remote**: contains a list of remote binary gradle dependencies for the plugin.
- **custom_maven_repos**: contains a list of urls specifying custom maven repos required for the plugin's dependencies.

 See https://github.com/godotengine/godot/issues/38157#issuecomment-618773871
 */
struct PluginConfig {
	// Set to true when the config file is properly loaded.
	bool valid_config = false;
	// Unix timestamp of last change to this plugin.
	uint64_t last_updated = 0;

	// Required config section
	String name;
	String binary_type;
	String binary;

	// Optional dependencies section
	Vector<String> local_dependencies;
	Vector<String> remote_dependencies;
	Vector<String> custom_maven_repos;
};

/*
 * Set of prebuilt plugins.
 * Currently unused, this is just for future reference:
 */
// static const PluginConfig MY_PREBUILT_PLUGIN = {
//	/*.valid_config =*/true,
//	/*.last_updated =*/0,
//	/*.name =*/"GodotPayment",
//	/*.binary_type =*/"local",
//	/*.binary =*/"res://android/build/libs/plugins/GodotPayment.release.aar",
//	/*.local_dependencies =*/{},
// 	/*.remote_dependencies =*/String("com.android.billingclient:billing:2.2.1").split("|"),
// 	/*.custom_maven_repos =*/{}
// };

static inline String resolve_local_dependency_path(String plugin_config_dir, String dependency_path) {
	String absolute_path;
	if (!dependency_path.empty()) {
		if (dependency_path.is_abs_path()) {
			absolute_path = ProjectSettings::get_singleton()->globalize_path(dependency_path);
		} else {
			absolute_path = plugin_config_dir.plus_file(dependency_path);
		}
	}

	return absolute_path;
}

static inline PluginConfig resolve_prebuilt_plugin(PluginConfig prebuilt_plugin, String plugin_config_dir) {
	PluginConfig resolved = prebuilt_plugin;
	resolved.binary = resolved.binary_type == BINARY_TYPE_LOCAL ? resolve_local_dependency_path(plugin_config_dir, prebuilt_plugin.binary) : prebuilt_plugin.binary;
	if (!prebuilt_plugin.local_dependencies.empty()) {
		resolved.local_dependencies.clear();
		for (int i = 0; i < prebuilt_plugin.local_dependencies.size(); i++) {
			resolved.local_dependencies.push_back(resolve_local_dependency_path(plugin_config_dir, prebuilt_plugin.local_dependencies[i]));
		}
	}
	return resolved;
}

static inline Vector<PluginConfig> get_prebuilt_plugins(String plugins_base_dir) {
	Vector<PluginConfig> prebuilt_plugins;
	// prebuilt_plugins.push_back(resolve_prebuilt_plugin(MY_PREBUILT_PLUGIN, plugins_base_dir));
	return prebuilt_plugins;
}

static inline bool is_plugin_config_valid(PluginConfig plugin_config) {
	bool valid_name = !plugin_config.name.empty();
	bool valid_binary_type = plugin_config.binary_type == BINARY_TYPE_LOCAL ||
							 plugin_config.binary_type == BINARY_TYPE_REMOTE;

	bool valid_binary = false;
	if (valid_binary_type) {
		valid_binary = !plugin_config.binary.empty() &&
					   (plugin_config.binary_type == BINARY_TYPE_REMOTE ||
							   FileAccess::exists(plugin_config.binary));
	}

	bool valid_local_dependencies = true;
	if (!plugin_config.local_dependencies.empty()) {
		for (int i = 0; i < plugin_config.local_dependencies.size(); i++) {
			if (!FileAccess::exists(plugin_config.local_dependencies[i])) {
				valid_local_dependencies = false;
				break;
			}
		}
	}
	return valid_name && valid_binary && valid_binary_type && valid_local_dependencies;
}

static inline uint64_t get_plugin_modification_time(const PluginConfig &plugin_config, const String &config_path) {
	uint64_t last_updated = FileAccess::get_modified_time(config_path);
	last_updated = MAX(last_updated, FileAccess::get_modified_time(plugin_config.binary));

	for (int i = 0; i < plugin_config.local_dependencies.size(); i++) {
		String binary = plugin_config.local_dependencies.get(i);
		last_updated = MAX(last_updated, FileAccess::get_modified_time(binary));
	}

	return last_updated;
}

static inline PluginConfig load_plugin_config(Ref<ConfigFile> config_file, const String &path) {
	PluginConfig plugin_config = {};

	if (config_file.is_valid()) {
		Error err = config_file->load(path);
		if (err == OK) {
			String config_base_dir = path.get_base_dir();

			plugin_config.name = config_file->get_value(CONFIG_SECTION, CONFIG_NAME_KEY, String());
			plugin_config.binary_type = config_file->get_value(CONFIG_SECTION, CONFIG_BINARY_TYPE_KEY, String());

			String binary_path = config_file->get_value(CONFIG_SECTION, CONFIG_BINARY_KEY, String());
			plugin_config.binary = plugin_config.binary_type == BINARY_TYPE_LOCAL ? resolve_local_dependency_path(config_base_dir, binary_path) : binary_path;

			if (config_file->has_section(DEPENDENCIES_SECTION)) {
				Vector<String> local_dependencies_paths = config_file->get_value(DEPENDENCIES_SECTION, DEPENDENCIES_LOCAL_KEY, Vector<String>());
				if (!local_dependencies_paths.empty()) {
					for (int i = 0; i < local_dependencies_paths.size(); i++) {
						plugin_config.local_dependencies.push_back(resolve_local_dependency_path(config_base_dir, local_dependencies_paths[i]));
					}
				}

				plugin_config.remote_dependencies = config_file->get_value(DEPENDENCIES_SECTION, DEPENDENCIES_REMOTE_KEY, Vector<String>());
				plugin_config.custom_maven_repos = config_file->get_value(DEPENDENCIES_SECTION, DEPENDENCIES_CUSTOM_MAVEN_REPOS_KEY, Vector<String>());
			}

			plugin_config.valid_config = is_plugin_config_valid(plugin_config);
			plugin_config.last_updated = get_plugin_modification_time(plugin_config, path);
		}
	}

	return plugin_config;
}

static inline String get_plugins_binaries(String binary_type, Vector<PluginConfig> plugins_configs) {
	String plugins_binaries;
	if (!plugins_configs.empty()) {
		Vector<String> binaries;
		for (int i = 0; i < plugins_configs.size(); i++) {
			PluginConfig config = plugins_configs[i];
			if (!config.valid_config) {
				continue;
			}

			if (config.binary_type == binary_type) {
				binaries.push_back(config.binary);
			}

			if (binary_type == BINARY_TYPE_LOCAL) {
				binaries.append_array(config.local_dependencies);
			}

			if (binary_type == BINARY_TYPE_REMOTE) {
				binaries.append_array(config.remote_dependencies);
			}
		}

		plugins_binaries = String(PLUGIN_VALUE_SEPARATOR).join(binaries);
	}

	return plugins_binaries;
}

static inline String get_plugins_custom_maven_repos(Vector<PluginConfig> plugins_configs) {
	String custom_maven_repos;
	if (!plugins_configs.empty()) {
		Vector<String> repos_urls;
		for (int i = 0; i < plugins_configs.size(); i++) {
			PluginConfig config = plugins_configs[i];
			if (!config.valid_config) {
				continue;
			}

			repos_urls.append_array(config.custom_maven_repos);
		}

		custom_maven_repos = String(PLUGIN_VALUE_SEPARATOR).join(repos_urls);
	}
	return custom_maven_repos;
}

static inline String get_plugins_names(Vector<PluginConfig> plugins_configs) {
	String plugins_names;
	if (!plugins_configs.empty()) {
		Vector<String> names;
		for (int i = 0; i < plugins_configs.size(); i++) {
			PluginConfig config = plugins_configs[i];
			if (!config.valid_config) {
				continue;
			}

			names.push_back(config.name);
		}
		plugins_names = String(PLUGIN_VALUE_SEPARATOR).join(names);
	}

	return plugins_names;
}

#endif // GODOT_PLUGIN_CONFIG_H
