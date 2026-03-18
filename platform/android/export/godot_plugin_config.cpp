/**************************************************************************/
/*  godot_plugin_config.cpp                                               */
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

#include "godot_plugin_config.h"

#ifndef DISABLE_DEPRECATED

/*
 * Set of prebuilt plugins.
 * Currently unused, this is just for future reference:
 */
// static const PluginConfigAndroid MY_PREBUILT_PLUGIN = {
//    /*.valid_config =*/true,
//    /*.last_updated =*/0,
//    /*.name =*/"GodotPayment",
//    /*.binary_type =*/"local",
//    /*.binary =*/"res://android/build/libs/plugins/GodotPayment.release.aar",
//    /*.local_dependencies =*/{},
//     /*.remote_dependencies =*/String("com.android.billingclient:billing:2.2.1").split("|"),
//     /*.custom_maven_repos =*/{}
// };

String PluginConfigAndroid::resolve_local_dependency_path(String plugin_config_dir, String dependency_path) {
	String absolute_path;
	if (!dependency_path.is_empty()) {
		if (dependency_path.is_absolute_path()) {
			absolute_path = ProjectSettings::get_singleton()->globalize_path(dependency_path);
		} else {
			absolute_path = plugin_config_dir.path_join(dependency_path);
		}
	}

	return absolute_path;
}

PluginConfigAndroid PluginConfigAndroid::resolve_prebuilt_plugin(PluginConfigAndroid prebuilt_plugin, String plugin_config_dir) {
	PluginConfigAndroid resolved = prebuilt_plugin;
	resolved.binary = resolved.binary_type == PluginConfigAndroid::BINARY_TYPE_LOCAL ? resolve_local_dependency_path(plugin_config_dir, prebuilt_plugin.binary) : prebuilt_plugin.binary;
	if (!prebuilt_plugin.local_dependencies.is_empty()) {
		resolved.local_dependencies.clear();
		for (int i = 0; i < prebuilt_plugin.local_dependencies.size(); i++) {
			resolved.local_dependencies.push_back(resolve_local_dependency_path(plugin_config_dir, prebuilt_plugin.local_dependencies[i]));
		}
	}
	return resolved;
}

Vector<PluginConfigAndroid> PluginConfigAndroid::get_prebuilt_plugins(String plugins_base_dir) {
	Vector<PluginConfigAndroid> prebuilt_plugins;
	return prebuilt_plugins;
}

bool PluginConfigAndroid::is_plugin_config_valid(PluginConfigAndroid plugin_config) {
	bool valid_name = !plugin_config.name.is_empty();
	bool valid_binary_type = plugin_config.binary_type == PluginConfigAndroid::BINARY_TYPE_LOCAL ||
			plugin_config.binary_type == PluginConfigAndroid::BINARY_TYPE_REMOTE;

	bool valid_binary = false;
	if (valid_binary_type) {
		valid_binary = !plugin_config.binary.is_empty() &&
				(plugin_config.binary_type == PluginConfigAndroid::BINARY_TYPE_REMOTE ||
						FileAccess::exists(plugin_config.binary));
	}

	bool valid_local_dependencies = true;
	if (!plugin_config.local_dependencies.is_empty()) {
		for (int i = 0; i < plugin_config.local_dependencies.size(); i++) {
			if (!FileAccess::exists(plugin_config.local_dependencies[i])) {
				valid_local_dependencies = false;
				break;
			}
		}
	}
	return valid_name && valid_binary && valid_binary_type && valid_local_dependencies;
}

uint64_t PluginConfigAndroid::get_plugin_modification_time(const PluginConfigAndroid &plugin_config, const String &config_path) {
	uint64_t last_updated = FileAccess::get_modified_time(config_path);
	last_updated = MAX(last_updated, FileAccess::get_modified_time(plugin_config.binary));

	for (int i = 0; i < plugin_config.local_dependencies.size(); i++) {
		String binary = plugin_config.local_dependencies.get(i);
		last_updated = MAX(last_updated, FileAccess::get_modified_time(binary));
	}

	return last_updated;
}

PluginConfigAndroid PluginConfigAndroid::load_plugin_config(Ref<ConfigFile> config_file, const String &path) {
	PluginConfigAndroid plugin_config = {};

	if (config_file.is_valid()) {
		Error err = config_file->load(path);
		if (err == OK) {
			String config_base_dir = path.get_base_dir();

			plugin_config.name = config_file->get_value(PluginConfigAndroid::CONFIG_SECTION, PluginConfigAndroid::CONFIG_NAME_KEY, String());
			plugin_config.binary_type = config_file->get_value(PluginConfigAndroid::CONFIG_SECTION, PluginConfigAndroid::CONFIG_BINARY_TYPE_KEY, String());

			String binary_path = config_file->get_value(PluginConfigAndroid::CONFIG_SECTION, PluginConfigAndroid::CONFIG_BINARY_KEY, String());
			plugin_config.binary = plugin_config.binary_type == PluginConfigAndroid::BINARY_TYPE_LOCAL ? resolve_local_dependency_path(config_base_dir, binary_path) : binary_path;

			if (config_file->has_section(PluginConfigAndroid::DEPENDENCIES_SECTION)) {
				Vector<String> local_dependencies_paths = config_file->get_value(PluginConfigAndroid::DEPENDENCIES_SECTION, PluginConfigAndroid::DEPENDENCIES_LOCAL_KEY, Vector<String>());
				if (!local_dependencies_paths.is_empty()) {
					for (int i = 0; i < local_dependencies_paths.size(); i++) {
						plugin_config.local_dependencies.push_back(resolve_local_dependency_path(config_base_dir, local_dependencies_paths[i]));
					}
				}

				plugin_config.remote_dependencies = config_file->get_value(PluginConfigAndroid::DEPENDENCIES_SECTION, PluginConfigAndroid::DEPENDENCIES_REMOTE_KEY, Vector<String>());
				plugin_config.custom_maven_repos = config_file->get_value(PluginConfigAndroid::DEPENDENCIES_SECTION, PluginConfigAndroid::DEPENDENCIES_CUSTOM_MAVEN_REPOS_KEY, Vector<String>());
			}

			plugin_config.valid_config = is_plugin_config_valid(plugin_config);
			plugin_config.last_updated = get_plugin_modification_time(plugin_config, path);
		}
	}

	return plugin_config;
}

void PluginConfigAndroid::get_plugins_binaries(String binary_type, Vector<PluginConfigAndroid> plugins_configs, Vector<String> &r_result) {
	if (!plugins_configs.is_empty()) {
		for (int i = 0; i < plugins_configs.size(); i++) {
			PluginConfigAndroid config = plugins_configs[i];
			if (!config.valid_config) {
				continue;
			}

			if (config.binary_type == binary_type) {
				r_result.push_back(config.binary);
			}

			if (binary_type == PluginConfigAndroid::BINARY_TYPE_LOCAL) {
				r_result.append_array(config.local_dependencies);
			}

			if (binary_type == PluginConfigAndroid::BINARY_TYPE_REMOTE) {
				r_result.append_array(config.remote_dependencies);
			}
		}
	}
}

void PluginConfigAndroid::get_plugins_custom_maven_repos(Vector<PluginConfigAndroid> plugins_configs, Vector<String> &r_result) {
	if (!plugins_configs.is_empty()) {
		for (int i = 0; i < plugins_configs.size(); i++) {
			PluginConfigAndroid config = plugins_configs[i];
			if (!config.valid_config) {
				continue;
			}

			r_result.append_array(config.custom_maven_repos);
		}
	}
}

void PluginConfigAndroid::get_plugins_names(Vector<PluginConfigAndroid> plugins_configs, Vector<String> &r_result) {
	if (!plugins_configs.is_empty()) {
		for (int i = 0; i < plugins_configs.size(); i++) {
			PluginConfigAndroid config = plugins_configs[i];
			if (!config.valid_config) {
				continue;
			}

			r_result.push_back(config.name);
		}
	}
}

#endif // DISABLE_DEPRECATED
