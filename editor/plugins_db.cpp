/*************************************************************************/
/*  plugins_db.cpp                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#ifdef TOOLS_ENABLED

#include "plugins_db.h"

#include "core/io/config_file.h"
#include "core/io/file_access_pack.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/os.h"
#include "editor/addons_fs_manager.h"
#include "editor/editor_file_system.h"
#include "editor/editor_settings.h"

PluginsDB *PluginsDB::singleton;

PluginsDB *PluginsDB::get_singleton() {

	return singleton;
}

bool PluginsDB::_parse_plugin(const String &p_location, bool p_is_user_level, String p_expected_name_for_pack, PluginInfo *r_entry) {

	bool is_pack = p_expected_name_for_pack != "";

	ConfigFile cf;
	if (is_pack) {
		PackedData pd;
		PackedSourcePCK pck;
		ERR_FAIL_COND_V(!pck.try_open_pack(p_location, true, &pd), false);

		String cfg_path = "addons://" + p_expected_name_for_pack + "/plugin.cfg";

		FileAccess *fa = pd.try_open_path(cfg_path);
		ERR_FAIL_COND_V(!fa, false);
		String cfg_data = fa->get_as_utf8_string();
		memdelete(fa);

		ERR_FAIL_COND_V(cf.parse(cfg_data) != OK, false);
	} else {
		String cfg_path = p_location.plus_file("plugin.cfg");
		if (!FileAccess::exists(cfg_path)) {
			return false;
		}
		ERR_FAIL_COND_V(cf.load(cfg_path) != OK, false);
	}

	int model = cf.get_value("plugin", "model", 1);
	Vector<String> editor_only_paths;
	if (model >= 2) {
		editor_only_paths = cf.get_value("plugin", "editor_only_paths", Vector<String>());
	} else {
		if (p_is_user_level) {
			WARN_PRINTS("Legacy plugin '" + String(cf.get_value("plugin", "name", "")) + "' is located at user level and that's not supported. Ignoring it.");
			return false;
		}
	}

	r_entry->location = p_location;
	r_entry->model = model;
	r_entry->is_user_level = p_is_user_level;
	r_entry->is_pack = is_pack;
	r_entry->editor_only_paths = editor_only_paths;

	return true;
}

void PluginsDB::scan() {

	// Scan for plugins

	entries.clear();

	bool first_time = addons_dirs.size() == 0;
	if (first_time) {
		// Order is relevant (the first one wins)

		// - Project

		project_addons_dir = OS::get_singleton()->get_resource_dir().plus_file("addons").replace("\\", "/");
		addons_dirs.push_back(project_addons_dir);

		// - User

		if (EditorSettings::get_singleton()) {
			// Editor
			user_addons_dir = EditorSettings::get_singleton()->get_plugins_dir();
		} else {
			// Editor runtime
			EditorSettings::MainPaths paths;
			EditorSettings::get_main_paths(&paths);
			user_addons_dir = paths.config_dir.plus_file("addons");
		}
		user_addons_dir = user_addons_dir.replace("\\", "/");
		addons_dirs.push_back(user_addons_dir);

		// Discover PCKs

		if (DirAccess::exists(user_addons_dir)) {
			DirAccess *da = DirAccess::open(user_addons_dir);
			ERR_FAIL_COND(!da);

			da->list_dir_begin();
			while (true) {
				String f = da->get_next();
				if (f == "") {
					break;
				}
				if (!da->current_is_dir() && f.to_lower().ends_with(".pck")) {
					String plugin_name = f.to_lower().trim_suffix(".pck");
					PluginInfo info;
					if (_parse_plugin(user_addons_dir.plus_file(f), true, plugin_name, &info)) {
						pack_entries_cache.insert(plugin_name, info);
					} else {
						print_line("No plugin found in " + user_addons_dir.plus_file(f) + "; ignoring");
					}
				}
			}
			da->list_dir_end();

			memdelete(da);
		}
	}

	for (int i = 0; i < addons_dirs.size(); ++i) {
		if (!DirAccess::exists(addons_dirs[i])) {
			continue;
		}

		DirAccess *da = DirAccess::open(addons_dirs[i]);
		ERR_FAIL_COND(!da);

		da->list_dir_begin();
		while (true) {
			String f = da->get_next();
			if (f == "") {
				break;
			}
			if (da->current_is_dir()) {
				if (f.begins_with(".")) {
					continue;
				}

				if (!entries.has(f)) {
					PluginInfo info;
					bool is_user_level = addons_dirs[i] == user_addons_dir;
					if (_parse_plugin(addons_dirs[i].plus_file(f), is_user_level, "", &info)) {
						entries.insert(f, info);
					}
				}
			}
		}
		da->list_dir_end();

		memdelete(da);
	}

	// User-level plugins as PCK are the least prioritary; let's see if any is still relevant

	for (Map<String, PluginInfo>::Element *E = pack_entries_cache.front(); E; E = E->next()) {
		if (!entries.has(E->key())) {
			const String &pack_path = E->value().location;
			if (mounted_packs_paths.find(pack_path) == -1) {
				print_line("Mounting " + pack_path);
				ERR_CONTINUE(PackedData::get_singleton()->add_pack(pack_path, true) != OK);
				mounted_packs_paths.push_back(pack_path);
			}

			entries.insert(E->key(), E->value());
		}
	}
	for (int i = 0; i < mounted_packs_paths.size(); i++) {
		bool keep_mounted = false;
		for (Map<String, PluginInfo>::Element *F = entries.front(); F; F = F->next()) {
			const PluginInfo &pi = F->get();
			if (pi.is_pack && pi.location == mounted_packs_paths[i]) {
				keep_mounted = true;
				break;
			}
		}
		if (!keep_mounted) {
			print_line("Unmounting " + mounted_packs_paths[i]);
			PackedData::get_singleton()->remove_pack(mounted_packs_paths[i]);
			mounted_packs_paths.erase(mounted_packs_paths[i]);
		}
	}

	PackedData::get_singleton()->set_disabled(mounted_packs_paths.size() == 0);

	// Update mappings for addons://

	AddonsFileSystemManager *afsm = AddonsFileSystemManager::get_singleton();
	afsm->start_building();
	Vector<String> enabled_plugins;
	if (ProjectSettings::get_singleton()->has_setting("editor_plugins/enabled")) {
		enabled_plugins = ProjectSettings::get_singleton()->get("editor_plugins/enabled");
	}
	for (Map<String, PluginInfo>::Element *E = entries.front(); E; E = E->next()) {
		const String &name = E->key();
		const PluginInfo &info = E->get();
		if (info.model >= 2) {
			if (!info.is_pack) {
				afsm->add_subdirectory(E->key(), info.location);
			} else {
				afsm->add_pack_subdirectory(E->key(), info.location);
			}
			if (enabled_plugins.find(name) == -1) {
				afsm->set_subdirectory_hidden(E->key(), true);
			}
		}
	}
	afsm->end_building();

	// Trigger FS rescan

	if (!first_time) {
		if (EditorFileSystem::get_singleton()) {
			EditorFileSystem::get_singleton()->scan_changes();
		}
	}
}

Vector<String> PluginsDB::get_plugin_names() {

	Vector<String> names;
	for (Map<String, PluginInfo>::Element *E = entries.front(); E; E = E->next()) {
		names.push_back(E->key());
	}
	names.sort_custom<NaturalNoCaseComparator>();
	return names;
}

bool PluginsDB::get_plugin_info(const String &p_plugin_name, PluginInfo *r_plugin_info) {

	Map<String, PluginInfo>::Element *entry = entries.find(p_plugin_name);
	if (!entry) {
		return false;
	}

	*r_plugin_info = entry->get();
	return true;
}

String PluginsDB::get_plugin_abstract_path(const String &p_plugin_name) {

	Map<String, PluginInfo>::Element *entry = entries.find(p_plugin_name);
	if (!entry) {
		return "";
	}

	const PluginInfo &pi = entry->get();
	if (pi.model == 1 || !pi.is_user_level) {
		return "<project>/addons/" + p_plugin_name + "/";
	} else {
		if (pi.is_pack) {
			return "<godot_settings>/addons/" + pi.location.get_file();
		} else {
			return "<godot_settings>/addons/" + p_plugin_name + "/";
		}
	}
}

bool PluginsDB::has_plugin(const String &p_plugin_name) {

	return entries.find(p_plugin_name) != NULL;
}

bool PluginsDB::has_universal_plugin(const String &p_plugin_name) {

	Map<String, PluginInfo>::Element *entry = entries.find(p_plugin_name);
	if (!entry) {
		return false;
	}

	return entry->get().model >= 2;
}

bool PluginsDB::is_editor_only_path(const String &p_path, const Vector<String> *p_editor_only_paths) {

	if (p_path.begins_with("addons://")) {
		Vector<String> parts = p_path.strip_filesystem_prefix().split("/", false, 1);
		if (parts.size() >= 2) {
			const String &plugin_name = parts[0];
			const String &subpath = parts[1].ends_with("/") ? parts[1].trim_suffix("/") : parts[1];
			if (!p_editor_only_paths) {
				Map<String, PluginInfo>::Element *entry = entries.find(plugin_name);
				if (!entry) {
					return false;
				}
				p_editor_only_paths = &entry->value().editor_only_paths;
			}
			for (int i = 0; i < p_editor_only_paths->size(); ++i) {
				const String &eop = p_editor_only_paths->get(i);
				if (eop == ".") {
					return true;
				}
				if (subpath.begins_with(eop)) {
					// Ensure it's actually a full match of the last segment
					if (subpath.length() == eop.length() || subpath[eop.length()] == '/') {
						return true;
					}
				}
			}
		}
	}

	return false;
}

PluginsDB::PluginsDB() {
	singleton = this;
}

#endif
