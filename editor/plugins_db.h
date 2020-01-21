/*************************************************************************/
/*  plugins_db.h                                                         */
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

#ifndef PLUGINS_DB_H
#define PLUGINS_DB_H

#include "core/map.h"
#include "core/ustring.h"
#include "core/vector.h"

class PluginsDB {
	static PluginsDB *singleton;

public:
	struct PluginInfo {
		String location;
		int model;
		// model >= 2
		bool is_user_level;
		bool is_pack;
		Vector<String> editor_only_paths;
	};

private:
	String project_addons_dir;
	String user_addons_dir;

	Vector<String> addons_dirs;
	Map<String, PluginInfo> pack_entries_cache;
	Map<String, PluginInfo> entries;
	Vector<String> mounted_packs_paths;

	bool _parse_plugin(const String &p_location, bool p_is_user_level, String p_expected_name_for_pack, PluginInfo *r_entry);

public:
	static PluginsDB *get_singleton();

	void scan();

	Vector<String> get_plugin_names();
	bool get_plugin_info(const String &p_plugin_name, PluginInfo *r_plugin_info);
	String get_plugin_abstract_path(const String &p_plugin_name);
	bool has_plugin(const String &p_plugin_name);
	bool has_universal_plugin(const String &p_plugin_name);

	bool is_editor_only_path(const String &p_path, const Vector<String> *p_editor_only_paths = NULL);

	PluginsDB();
};

#endif
