/*************************************************************************/
/*  editor_settings.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http:/www.godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "editor_settings.h"

#include "core/io/compression.h"
#include "core/io/config_file.h"
#include "core/io/file_access_memory.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/io/translation_loader_po.h"
#include "core/os/dir_access.h"
#include "core/os/file_access.h"
#include "core/os/keyboard.h"
#include "core/os/os.h"
#include "core/project_settings.h"
#include "core/version.h"
#include "editor/editor_node.h"
#include "editor/translations.gen.h"
#include "scene/main/node.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"

Ref<EditorSettings> EditorSettings::singleton = NULL;

EditorSettings *EditorSettings::get_singleton() {

	return singleton.ptr();
}

bool EditorSettings::_set(const StringName &p_name, const Variant &p_value) {

	_THREAD_SAFE_METHOD_

	if (p_name.operator String() == "shortcuts") {

		Array arr = p_value;
		ERR_FAIL_COND_V(arr.size() && arr.size() & 1, true);
		for (int i = 0; i < arr.size(); i += 2) {

			String name = arr[i];
			Ref<InputEvent> shortcut = arr[i + 1];

			Ref<ShortCut> sc;
			sc.instance();
			sc->set_shortcut(shortcut);
			add_shortcut(name, sc);
		}

		return true;
	}

	if (p_value.get_type() == Variant::NIL)
		props.erase(p_name);
	else {

		if (props.has(p_name))
			props[p_name].variant = p_value;
		else
			props[p_name] = VariantContainer(p_value, last_order++);

		if (save_changed_setting) {
			props[p_name].save = true;
		}
	}

	emit_signal("settings_changed");
	return true;
}

bool EditorSettings::_get(const StringName &p_name, Variant &r_ret) const {

	_THREAD_SAFE_METHOD_

	if (p_name.operator String() == "shortcuts") {

		Array arr;
		for (const Map<String, Ref<ShortCut> >::Element *E = shortcuts.front(); E; E = E->next()) {

			Ref<ShortCut> sc = E->get();

			if (optimize_save) {
				if (!sc->has_meta("original")) {
					continue; //this came from settings but is not any longer used
				}

				Ref<InputEvent> original = sc->get_meta("original");
				if (sc->is_shortcut(original) || (original.is_null() && sc->get_shortcut().is_null()))
					continue; //not changed from default, don't save
			}

			arr.push_back(E->key());
			arr.push_back(sc->get_shortcut());
		}
		r_ret = arr;
		return true;
	}

	const VariantContainer *v = props.getptr(p_name);
	if (!v) {
		print_line("EditorSettings::_get - Warning, not found: " + String(p_name));
		return false;
	}
	r_ret = v->variant;
	return true;
}

struct _EVCSort {

	String name;
	Variant::Type type;
	int order;
	bool save;

	bool operator<(const _EVCSort &p_vcs) const { return order < p_vcs.order; }
};

void EditorSettings::_get_property_list(List<PropertyInfo> *p_list) const {

	_THREAD_SAFE_METHOD_

	const String *k = NULL;
	Set<_EVCSort> vclist;

	while ((k = props.next(k))) {

		const VariantContainer *v = props.getptr(*k);

		if (v->hide_from_editor)
			continue;

		_EVCSort vc;
		vc.name = *k;
		vc.order = v->order;
		vc.type = v->variant.get_type();
		vc.save = v->save;

		vclist.insert(vc);
	}

	for (Set<_EVCSort>::Element *E = vclist.front(); E; E = E->next()) {

		int pinfo = 0;
		if (E->get().save || !optimize_save) {
			pinfo |= PROPERTY_USAGE_STORAGE;
		}

		if (!E->get().name.begins_with("_") && !E->get().name.begins_with("projects/")) {
			pinfo |= PROPERTY_USAGE_EDITOR;
		} else {
			pinfo |= PROPERTY_USAGE_STORAGE; //hiddens must always be saved
		}

		PropertyInfo pi(E->get().type, E->get().name);
		pi.usage = pinfo;
		if (hints.has(E->get().name))
			pi = hints[E->get().name];

		p_list->push_back(pi);
	}

	p_list->push_back(PropertyInfo(Variant::ARRAY, "shortcuts", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR)); //do not edit
}

bool EditorSettings::has(String p_var) const {

	_THREAD_SAFE_METHOD_

	return props.has(p_var);
}

void EditorSettings::erase(String p_var) {

	_THREAD_SAFE_METHOD_

	props.erase(p_var);
}

void EditorSettings::raise_order(const String &p_name) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_COND(!props.has(p_name));
	props[p_name].order = ++last_order;
}

Variant _EDITOR_DEF(const String &p_var, const Variant &p_default) {

	if (EditorSettings::get_singleton()->has(p_var))
		return EditorSettings::get_singleton()->get(p_var);
	EditorSettings::get_singleton()->set(p_var, p_default);
	return p_default;
}

Variant _EDITOR_GET(const String &p_var) {

	ERR_FAIL_COND_V(!EditorSettings::get_singleton()->has(p_var), Variant())
	return EditorSettings::get_singleton()->get(p_var);
}

static Dictionary _get_builtin_script_templates() {
	Dictionary templates;

	//No Comments
	templates["no_comments.gd"] =
			"extends %BASE%\n"
			"\n"
			"func _ready():\n"
			"%TS%pass\n";

	//Empty
	templates["empty.gd"] =
			"extends %BASE%"
			"\n"
			"\n";

	return templates;
}

static void _create_script_templates(const String &p_path) {

	Dictionary templates = _get_builtin_script_templates();
	List<Variant> keys;
	templates.get_key_list(&keys);
	FileAccess *file = FileAccess::create(FileAccess::ACCESS_FILESYSTEM);

	DirAccess *dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	dir->change_dir(p_path);
	for (int i = 0; i < keys.size(); i++) {
		if (!dir->file_exists(keys[i])) {
			Error err = file->reopen(p_path.plus_file((String)keys[i]), FileAccess::WRITE);
			ERR_FAIL_COND(err != OK);
			file->store_string(templates[keys[i]]);
			file->close();
		}
	}

	memdelete(dir);
	memdelete(file);
}

void EditorSettings::create() {

	if (singleton.ptr())
		return; //pointless

	DirAccess *dir = NULL;
	Variant meta;

	String config_path;
	String config_dir;
	Ref<ConfigFile> extra_config = memnew(ConfigFile);

	String exe_path = OS::get_singleton()->get_executable_path().get_base_dir();
	DirAccess *d = DirAccess::create_for_path(exe_path);
	bool self_contained = false;

	if (d->file_exists(exe_path + "/._sc_")) {
		self_contained = true;
		extra_config->load(exe_path + "/._sc_");
	} else if (d->file_exists(exe_path + "/_sc_")) {
		self_contained = true;
		extra_config->load(exe_path + "/_sc_");
	}
	memdelete(d);

	if (self_contained) {
		// editor is self contained
		config_path = exe_path;
		config_dir = "editor_data";
	} else {

		if (OS::get_singleton()->has_environment("APPDATA")) {
			// Most likely under windows, save here
			config_path = OS::get_singleton()->get_environment("APPDATA");
			config_dir = String(_MKSTR(VERSION_SHORT_NAME)).capitalize();
		} else if (OS::get_singleton()->has_environment("HOME")) {

			config_path = OS::get_singleton()->get_environment("HOME");
			config_dir = "." + String(_MKSTR(VERSION_SHORT_NAME)).to_lower();
		}
	};

	ClassDB::register_class<EditorSettings>(); //otherwise it can't be unserialized
	String config_file_path;

	if (config_path != "") {

		dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		if (dir->change_dir(config_path) != OK) {
			ERR_PRINT("Cannot find path for config directory!");
			memdelete(dir);
			goto fail;
		}

		if (dir->change_dir(config_dir) != OK) {
			dir->make_dir(config_dir);
			if (dir->change_dir(config_dir) != OK) {
				ERR_PRINT("Cannot create config directory!");
				memdelete(dir);
				goto fail;
			}
		}

		if (dir->change_dir("templates") != OK) {
			dir->make_dir("templates");
		} else {

			dir->change_dir("..");
		}

		if (dir->change_dir("text_editor_themes") != OK) {
			dir->make_dir("text_editor_themes");
		} else {
			dir->change_dir("..");
		}

		if (dir->change_dir("script_templates") != OK) {
			dir->make_dir("script_templates");
		} else {
			dir->change_dir("..");
		}
		_create_script_templates(dir->get_current_dir() + "/script_templates");

		if (dir->change_dir("tmp") != OK) {
			dir->make_dir("tmp");
		} else {

			dir->change_dir("..");
		}

		if (dir->change_dir("config") != OK) {
			dir->make_dir("config");
		} else {

			dir->change_dir("..");
		}

		dir->change_dir("config");

		String pcp = ProjectSettings::get_singleton()->get_resource_path();
		if (pcp.ends_with("/"))
			pcp = config_path.substr(0, pcp.size() - 1);
		pcp = pcp.get_file() + "-" + pcp.md5_text();

		if (dir->change_dir(pcp)) {
			dir->make_dir(pcp);
		} else {
			dir->change_dir("..");
		}

		dir->change_dir("..");

		// path at least is validated, so validate config file

		String config_file_name = "editor_settings-" + String(_MKSTR(VERSION_MAJOR)) + ".tres";
		config_file_path = config_path + "/" + config_dir + "/" + config_file_name;

		String open_path = config_file_path;

		if (!dir->file_exists(config_file_name)) {

			goto fail;
		}

		memdelete(dir);

		singleton = ResourceLoader::load(open_path, "EditorSettings");

		if (singleton.is_null()) {
			WARN_PRINT("Could not open config file.");
			goto fail;
		}

		singleton->save_changed_setting = true;
		singleton->config_file_path = config_file_path;
		singleton->project_config_path = pcp;
		singleton->settings_path = config_path + "/" + config_dir;

		if (OS::get_singleton()->is_stdout_verbose()) {

			print_line("EditorSettings: Load OK!");
		}

		singleton->setup_language();
		singleton->setup_network();
		singleton->load_favorites();
		singleton->list_text_editor_themes();

		return;
	}

fail:

	// patch init projects
	if (extra_config->has_section("init_projects")) {
		Vector<String> list = extra_config->get_value("init_projects", "list");
		for (int i = 0; i < list.size(); i++) {

			list[i] = exe_path + "/" + list[i];
		};
		extra_config->set_value("init_projects", "list", list);
	};

	singleton = Ref<EditorSettings>(memnew(EditorSettings));
	singleton->save_changed_setting = true;
	singleton->config_file_path = config_file_path;
	singleton->settings_path = config_path + "/" + config_dir;
	singleton->_load_defaults(extra_config);
	singleton->setup_language();
	singleton->setup_network();
	singleton->list_text_editor_themes();
}

String EditorSettings::get_settings_path() const {

	return settings_path;
}

void EditorSettings::setup_language() {

	String lang = get("interface/editor_language");
	if (lang == "en")
		return; //none to do

	for (int i = 0; i < translations.size(); i++) {
		if (translations[i]->get_locale() == lang) {
			TranslationServer::get_singleton()->set_tool_translation(translations[i]);
			break;
		}
	}
}

void EditorSettings::setup_network() {

	List<IP_Address> local_ip;
	IP::get_singleton()->get_local_addresses(&local_ip);
	String lip;
	String hint;
	String current = has("network/debug/remote_host") ? get("network/debug/remote_host") : "";
	int port = has("network/debug/remote_port") ? (int)get("network/debug/remote_port") : 6007;

	for (List<IP_Address>::Element *E = local_ip.front(); E; E = E->next()) {

		String ip = E->get();

		if (lip == "")
			lip = ip;
		if (ip == current)
			lip = current; //so it saves
		if (hint != "")
			hint += ",";
		hint += ip;
	}

	set("network/debug/remote_host", lip);
	add_property_hint(PropertyInfo(Variant::STRING, "network/debug/remote_host", PROPERTY_HINT_ENUM, hint));

	set("network/debug/remote_port", port);
	add_property_hint(PropertyInfo(Variant::INT, "network/debug/remote_port", PROPERTY_HINT_RANGE, "1,65535,1"));
}

void EditorSettings::save() {

	//_THREAD_SAFE_METHOD_

	if (!singleton.ptr())
		return;

	if (singleton->config_file_path == "") {
		ERR_PRINT("Cannot save EditorSettings config, no valid path");
		return;
	}

	Error err = ResourceSaver::save(singleton->config_file_path, singleton);

	if (err != OK) {
		ERR_PRINT("Can't Save!");
		return;
	}

	if (OS::get_singleton()->is_stdout_verbose()) {
		print_line("EditorSettings Save OK!");
	}
}

void EditorSettings::destroy() {

	if (!singleton.ptr())
		return;
	save();
	singleton = Ref<EditorSettings>();
}

void EditorSettings::_load_defaults(Ref<ConfigFile> p_extra_config) {

	_THREAD_SAFE_METHOD_

	{
		String lang_hint = "en";
		String host_lang = OS::get_singleton()->get_locale();

		String best;

		for (int i = 0; i < translations.size(); i++) {
			String locale = translations[i]->get_locale();
			lang_hint += ",";
			lang_hint += locale;

			if (host_lang == locale) {
				best = locale;
			}

			if (best == String() && host_lang.begins_with(locale)) {
				best = locale;
			}
		}

		if (best == String()) {
			best = "en";
		}

		set("interface/editor_language", best);
		hints["interface/editor_language"] = PropertyInfo(Variant::STRING, "interface/editor_language", PROPERTY_HINT_ENUM, lang_hint, PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	}

	set("interface/hidpi_mode", 0);
	hints["interface/hidpi_mode"] = PropertyInfo(Variant::INT, "interface/hidpi_mode", PROPERTY_HINT_ENUM, "Auto,VeryLoDPI,LoDPI,MidDPI,HiDPI", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/show_script_in_scene_tabs", false);
	set("interface/font_size", 14);
	hints["interface/font_size"] = PropertyInfo(Variant::INT, "interface/font_size", PROPERTY_HINT_RANGE, "10,40,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/source_font_size", 14);
	hints["interface/source_font_size"] = PropertyInfo(Variant::INT, "interface/source_font_size", PROPERTY_HINT_RANGE, "8,96,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/custom_font", "");
	hints["interface/custom_font"] = PropertyInfo(Variant::STRING, "interface/custom_font", PROPERTY_HINT_GLOBAL_FILE, "*.font", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/dim_editor_on_dialog_popup", true);
	set("interface/dim_amount", 0.6f);
	hints["interface/dim_amount"] = PropertyInfo(Variant::REAL, "interface/dim_amount", PROPERTY_HINT_RANGE, "0,1,0.01", PROPERTY_USAGE_DEFAULT);
	set("interface/dim_transition_time", 0.08f);
	hints["interface/dim_transition_time"] = PropertyInfo(Variant::REAL, "interface/dim_transition_time", PROPERTY_HINT_RANGE, "0,1,0.001", PROPERTY_USAGE_DEFAULT);

	set("interface/separate_distraction_mode", false);

	set("interface/save_each_scene_on_quit", true); // Regression
	set("interface/quit_confirmation", true);

	set("interface/theme/preset", 0);
	hints["interface/theme/preset"] = PropertyInfo(Variant::INT, "interface/theme/preset", PROPERTY_HINT_ENUM, "Default,Grey,Godot 2,Arc,Light,Custom", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/theme/icon_and_font_color", 0);
	hints["interface/theme/icon_and_font_color"] = PropertyInfo(Variant::INT, "interface/theme/icon_and_font_color", PROPERTY_HINT_ENUM, "Auto,Dark,Light", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/theme/base_color", Color::html("#323b4f"));
	hints["interface/theme/accent_color"] = PropertyInfo(Variant::COLOR, "interface/theme/accent_color", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/theme/accent_color", Color::html("#699ce8"));
	hints["interface/theme/base_color"] = PropertyInfo(Variant::COLOR, "interface/theme/base_color", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/theme/contrast", 0.25);
	hints["interface/theme/contrast"] = PropertyInfo(Variant::REAL, "interface/theme/contrast", PROPERTY_HINT_RANGE, "0.01, 1, 0.01");
	set("interface/theme/highlight_tabs", false);
	set("interface/theme/border_size", 1);
	hints["interface/theme/border_size"] = PropertyInfo(Variant::INT, "interface/theme/border_size", PROPERTY_HINT_RANGE, "0,2,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);
	set("interface/theme/custom_theme", "");
	hints["interface/theme/custom_theme"] = PropertyInfo(Variant::STRING, "interface/theme/custom_theme", PROPERTY_HINT_GLOBAL_FILE, "*.res,*.tres,*.theme", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);

	set("interface/scene_tabs/show_extension", false);
	set("interface/scene_tabs/show_thumbnail_on_hover", true);
	set("interface/scene_tabs/resize_if_many_tabs", true);
	set("interface/scene_tabs/minimum_width", 50);
	hints["interface/scene_tabs/minimum_width"] = PropertyInfo(Variant::INT, "interface/scene_tabs/minimum_width", PROPERTY_HINT_RANGE, "50,500,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);

	set("filesystem/directories/autoscan_project_path", "");
	hints["filesystem/directories/autoscan_project_path"] = PropertyInfo(Variant::STRING, "filesystem/directories/autoscan_project_path", PROPERTY_HINT_GLOBAL_DIR);
	set("filesystem/directories/default_project_path", OS::get_singleton()->has_environment("HOME") ? OS::get_singleton()->get_environment("HOME") : OS::get_singleton()->get_system_dir(OS::SYSTEM_DIR_DOCUMENTS));
	hints["filesystem/directories/default_project_path"] = PropertyInfo(Variant::STRING, "filesystem/directories/default_project_path", PROPERTY_HINT_GLOBAL_DIR);
	set("filesystem/directories/default_project_export_path", "");
	hints["global/default_project_export_path"] = PropertyInfo(Variant::STRING, "global/default_project_export_path", PROPERTY_HINT_GLOBAL_DIR);
	set("interface/show_script_in_scene_tabs", false);

	set("text_editor/theme/color_theme", "Adaptive");
	hints["text_editor/theme/color_theme"] = PropertyInfo(Variant::STRING, "text_editor/theme/color_theme", PROPERTY_HINT_ENUM, "Adaptive,Default");

	set("text_editor/theme/line_spacing", 4);

	_load_default_text_editor_theme();

	set("text_editor/highlighting/syntax_highlighting", true);

	set("text_editor/highlighting/highlight_all_occurrences", true);
	set("text_editor/cursor/scroll_past_end_of_file", false);

	set("text_editor/indent/type", 0);
	hints["text_editor/indent/type"] = PropertyInfo(Variant::INT, "text_editor/indent/type", PROPERTY_HINT_ENUM, "Tabs,Spaces");
	set("text_editor/indent/size", 4);
	hints["text_editor/indent/size"] = PropertyInfo(Variant::INT, "text_editor/indent/size", PROPERTY_HINT_RANGE, "1, 64, 1"); // size of 0 crashes.
	set("text_editor/indent/auto_indent", true);
	set("text_editor/indent/convert_indent_on_save", false);
	set("text_editor/indent/draw_tabs", true);

	set("text_editor/line_numbers/show_line_numbers", true);
	set("text_editor/line_numbers/line_numbers_zero_padded", false);
	set("text_editor/line_numbers/show_breakpoint_gutter", true);
	set("text_editor/line_numbers/show_line_length_guideline", false);
	set("text_editor/line_numbers/line_length_guideline_column", 80);
	hints["text_editor/line_numbers/line_length_guideline_column"] = PropertyInfo(Variant::INT, "text_editor/line_numbers/line_length_guideline_column", PROPERTY_HINT_RANGE, "20, 160, 10");

	set("text_editor/open_scripts/smooth_scrolling", true);
	set("text_editor/open_scripts/v_scroll_speed", 80);
	set("text_editor/open_scripts/show_members_overview", true);

	set("text_editor/files/trim_trailing_whitespace_on_save", false);
	set("text_editor/completion/idle_parse_delay", 2);
	set("text_editor/tools/create_signal_callbacks", true);
	set("text_editor/files/autosave_interval_secs", 0);

	set("text_editor/cursor/block_caret", false);
	set("text_editor/cursor/caret_blink", false);
	set("text_editor/cursor/caret_blink_speed", 0.65);
	hints["text_editor/cursor/caret_blink_speed"] = PropertyInfo(Variant::REAL, "text_editor/cursor/caret_blink_speed", PROPERTY_HINT_RANGE, "0.1, 10, 0.1");

	set("text_editor/theme/font", "");
	hints["text_editor/theme/font"] = PropertyInfo(Variant::STRING, "text_editor/theme/font", PROPERTY_HINT_GLOBAL_FILE, "*.font");
	set("text_editor/completion/auto_brace_complete", false);
	set("text_editor/files/restore_scripts_on_load", true);
	set("text_editor/completion/complete_file_paths", true);
	set("text_editor/files/maximum_recent_files", 20);
	hints["text_editor/files/maximum_recent_files"] = PropertyInfo(Variant::INT, "text_editor/files/maximum_recent_files", PROPERTY_HINT_RANGE, "1, 200, 0");

	set("docks/scene_tree/start_create_dialog_fully_expanded", false);
	set("docks/scene_tree/draw_relationship_lines", false);
	set("docks/scene_tree/relationship_line_color", Color::html("464646"));

	set("editors/grid_map/pick_distance", 5000.0);

	set("editors/3d/grid_color", Color(1, 1, 1, 0.2));
	hints["editors/3d/grid_color"] = PropertyInfo(Variant::COLOR, "editors/3d/grid_color", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESTART_IF_CHANGED);

	set("editors/3d/default_fov", 55.0);
	set("editors/3d/default_z_near", 0.1);
	set("editors/3d/default_z_far", 500.0);

	set("editors/3d/navigation_scheme", 0);
	hints["editors/3d/navigation_scheme"] = PropertyInfo(Variant::INT, "editors/3d/navigation_scheme", PROPERTY_HINT_ENUM, "Godot,Maya,Modo");
	set("editors/3d/zoom_style", 0);
	hints["editors/3d/zoom_style"] = PropertyInfo(Variant::INT, "editors/3d/zoom_style", PROPERTY_HINT_ENUM, "Vertical, Horizontal");
	set("editors/3d/orbit_modifier", 0);
	hints["editors/3d/orbit_modifier"] = PropertyInfo(Variant::INT, "editors/3d/orbit_modifier", PROPERTY_HINT_ENUM, "None,Shift,Alt,Meta,Ctrl");
	set("editors/3d/pan_modifier", 1);
	hints["editors/3d/pan_modifier"] = PropertyInfo(Variant::INT, "editors/3d/pan_modifier", PROPERTY_HINT_ENUM, "None,Shift,Alt,Meta,Ctrl");
	set("editors/3d/zoom_modifier", 4);
	hints["editors/3d/zoom_modifier"] = PropertyInfo(Variant::INT, "editors/3d/zoom_modifier", PROPERTY_HINT_ENUM, "None,Shift,Alt,Meta,Ctrl");
	set("editors/3d/emulate_numpad", false);
	set("editors/3d/emulate_3_button_mouse", false);
	set("editors/3d/warped_mouse_panning", true);

	set("editors/3d/orbit_sensitivity", 0.4);

	set("editors/3d/orbit_inertia", 0.2);
	hints["editors/3d/orbit_inertia"] = PropertyInfo(Variant::REAL, "editors/3d/orbit_inertia", PROPERTY_HINT_RANGE, "0.0, 1, 0.01");

	set("editors/3d/freelook_inertia", 0.2);
	hints["editors/3d/freelook_inertia"] = PropertyInfo(Variant::REAL, "editors/3d/freelook_inertia", PROPERTY_HINT_RANGE, "0.0, 1, 0.01");

	set("editors/3d/freelook_base_speed", 0.5);
	hints["editors/3d/freelook_base_speed"] = PropertyInfo(Variant::REAL, "editors/3d/freelook_base_speed", PROPERTY_HINT_RANGE, "0.0, 10, 0.1");

	set("editors/3d/freelook_activation_modifier", 0);
	hints["editors/3d/freelook_activation_modifier"] = PropertyInfo(Variant::INT, "editors/3d/freelook_activation_modifier", PROPERTY_HINT_ENUM, "None,Shift,Alt,Meta,Ctrl");

	set("editors/3d/freelook_modifier_speed_factor", 5.0);

	set("editors/2d/bone_width", 5);
	set("editors/2d/bone_color1", Color(1.0, 1.0, 1.0, 0.9));
	set("editors/2d/bone_color2", Color(0.75, 0.75, 0.75, 0.9));
	set("editors/2d/bone_selected_color", Color(0.9, 0.45, 0.45, 0.9));
	set("editors/2d/bone_ik_color", Color(0.9, 0.9, 0.45, 0.9));
	set("editors/2d/keep_margins_when_changing_anchors", false);
	set("editors/2d/warped_mouse_panning", true);
	set("editors/2d/scroll_to_pan", false);
	set("editors/2d/pan_speed", 20);

	set("editors/poly_editor/point_grab_radius", 8);
	set("editors/poly_editor/show_previous_outline", true);

	set("run/window_placement/rect", 1);
	hints["run/window_placement/rect"] = PropertyInfo(Variant::INT, "run/window_placement/rect", PROPERTY_HINT_ENUM, "Top Left,Centered,Custom Position,Force Maximized,Force Fullscreen");
	String screen_hints = TTR("Default (Same as Editor)");
	for (int i = 0; i < OS::get_singleton()->get_screen_count(); i++) {
		screen_hints += ",Monitor " + itos(i + 1);
	}
	set("run/window_placement/rect_custom_position", Vector2());
	set("run/window_placement/screen", 0);
	hints["run/window_placement/screen"] = PropertyInfo(Variant::INT, "run/window_placement/screen", PROPERTY_HINT_ENUM, screen_hints);

	set("filesystem/on_save/compress_binary_resources", true);
	set("filesystem/on_save/save_modified_external_resources", true);

	set("text_editor/tools/create_signal_callbacks", true);

	set("filesystem/file_dialog/show_hidden_files", false);
	set("filesystem/file_dialog/display_mode", 0);
	hints["filesystem/file_dialog/display_mode"] = PropertyInfo(Variant::INT, "filesystem/file_dialog/display_mode", PROPERTY_HINT_ENUM, "Thumbnails,List");
	set("filesystem/file_dialog/thumbnail_size", 64);
	hints["filesystem/file_dialog/thumbnail_size"] = PropertyInfo(Variant::INT, "filesystem/file_dialog/thumbnail_size", PROPERTY_HINT_RANGE, "32,128,16");

	set("docks/filesystem/display_mode", 0);
	hints["docks/filesystem/display_mode"] = PropertyInfo(Variant::INT, "docks/filesystem/display_mode", PROPERTY_HINT_ENUM, "Thumbnails,List");
	set("docks/filesystem/thumbnail_size", 64);
	hints["docks/filesystem/thumbnail_size"] = PropertyInfo(Variant::INT, "docks/filesystem/thumbnail_size", PROPERTY_HINT_RANGE, "32,128,16");
	set("docks/filesystem/display_mode", 0);
	hints["docks/filesystem/display_mode"] = PropertyInfo(Variant::INT, "docks/filesystem/display_mode", PROPERTY_HINT_ENUM, "Thumbnails,List");
	set("docks/filesystem/always_show_folders", true);

	set("editors/animation/autorename_animation_tracks", true);
	set("editors/animation/confirm_insert_track", true);

	set("docks/property_editor/texture_preview_width", 48);
	set("docks/property_editor/auto_refresh_interval", 0.3);
	set("text_editor/help/doc_path", "");
	set("text_editor/help/show_help_index", true);

	set("filesystem/import/ask_save_before_reimport", false);

	set("filesystem/import/pvrtc_texture_tool", "");
#ifdef WINDOWS_ENABLED
	hints["filesystem/import/pvrtc_texture_tool"] = PropertyInfo(Variant::STRING, "filesystem/import/pvrtc_texture_tool", PROPERTY_HINT_GLOBAL_FILE, "*.exe");
#else
	hints["filesystem/import/pvrtc_texture_tool"] = PropertyInfo(Variant::STRING, "filesystem/import/pvrtc_texture_tool", PROPERTY_HINT_GLOBAL_FILE, "");
#endif
	set("filesystem/import/pvrtc_fast_conversion", false);

	set("run/auto_save/save_before_running", true);
	set("run/output/always_clear_output_on_play", true);
	set("run/output/always_open_output_on_play", true);
	set("run/output/always_close_output_on_stop", false);
	set("filesystem/resources/save_compressed_resources", true);
	set("filesystem/resources/auto_reload_modified_images", true);

	set("filesystem/import/automatic_reimport_on_sources_changed", true);

	if (p_extra_config.is_valid()) {

		if (p_extra_config->has_section("init_projects") && p_extra_config->has_section_key("init_projects", "list")) {

			Vector<String> list = p_extra_config->get_value("init_projects", "list");
			for (int i = 0; i < list.size(); i++) {

				String name = list[i].replace("/", "::");
				set("projects/" + name, list[i]);
			};
		};

		if (p_extra_config->has_section("presets")) {

			List<String> keys;
			p_extra_config->get_section_keys("presets", &keys);

			for (List<String>::Element *E = keys.front(); E; E = E->next()) {

				String key = E->get();
				Variant val = p_extra_config->get_value("presets", key);
				set(key, val);
			};
		};
	};
}

void EditorSettings::_load_default_text_editor_theme() {
	set("text_editor/highlighting/background_color", Color::html("3b000000"));
	set("text_editor/highlighting/completion_background_color", Color::html("2C2A32"));
	set("text_editor/highlighting/completion_selected_color", Color::html("434244"));
	set("text_editor/highlighting/completion_existing_color", Color::html("21dfdfdf"));
	set("text_editor/highlighting/completion_scroll_color", Color::html("ffffff"));
	set("text_editor/highlighting/completion_font_color", Color::html("aaaaaa"));
	set("text_editor/highlighting/caret_color", Color::html("aaaaaa"));
	set("text_editor/highlighting/caret_background_color", Color::html("000000"));
	set("text_editor/highlighting/line_number_color", Color::html("66aaaaaa"));
	set("text_editor/highlighting/text_color", Color::html("aaaaaa"));
	set("text_editor/highlighting/text_selected_color", Color::html("000000"));
	set("text_editor/highlighting/keyword_color", Color::html("ffffb3"));
	set("text_editor/highlighting/base_type_color", Color::html("a4ffd4"));
	set("text_editor/highlighting/engine_type_color", Color::html("83d3ff"));
	set("text_editor/highlighting/function_color", Color::html("66a2ce"));
	set("text_editor/highlighting/member_variable_color", Color::html("e64e59"));
	set("text_editor/highlighting/comment_color", Color::html("676767"));
	set("text_editor/highlighting/string_color", Color::html("ef6ebe"));
	set("text_editor/highlighting/number_color", Color::html("EB9532"));
	set("text_editor/highlighting/symbol_color", Color::html("badfff"));
	set("text_editor/highlighting/selection_color", Color::html("6ca9c2"));
	set("text_editor/highlighting/brace_mismatch_color", Color(1, 0.2, 0.2));
	set("text_editor/highlighting/current_line_color", Color(0.3, 0.5, 0.8, 0.15));
	set("text_editor/highlighting/line_length_guideline_color", Color(0.3, 0.5, 0.8, 0.1));
	set("text_editor/highlighting/mark_color", Color(1.0, 0.4, 0.4, 0.4));
	set("text_editor/highlighting/breakpoint_color", Color(0.8, 0.8, 0.4, 0.2));
	set("text_editor/highlighting/word_highlighted_color", Color(0.8, 0.9, 0.9, 0.15));
	set("text_editor/highlighting/search_result_color", Color(0.05, 0.25, 0.05, 1));
	set("text_editor/highlighting/search_result_border_color", Color(0.1, 0.45, 0.1, 1));
}

void EditorSettings::notify_changes() {

	_THREAD_SAFE_METHOD_

	SceneTree *sml = Object::cast_to<SceneTree>(OS::get_singleton()->get_main_loop());

	if (!sml) {
		return;
	}

	Node *root = sml->get_root()->get_child(0);

	if (!root) {
		return;
	}
	root->propagate_notification(NOTIFICATION_EDITOR_SETTINGS_CHANGED);
}

void EditorSettings::_add_property_info_bind(const Dictionary &p_info) {

	ERR_FAIL_COND(!p_info.has("name"));
	ERR_FAIL_COND(!p_info.has("type"));

	PropertyInfo pinfo;
	pinfo.name = p_info["name"];
	ERR_FAIL_COND(!props.has(pinfo.name));
	pinfo.type = Variant::Type(p_info["type"].operator int());
	ERR_FAIL_INDEX(pinfo.type, Variant::VARIANT_MAX);

	if (p_info.has("hint"))
		pinfo.hint = PropertyHint(p_info["hint"].operator int());
	if (p_info.has("hint_string"))
		pinfo.hint_string = p_info["hint_string"];

	add_property_hint(pinfo);
}

void EditorSettings::add_property_hint(const PropertyInfo &p_hint) {

	_THREAD_SAFE_METHOD_

	hints[p_hint.name] = p_hint;
}

void EditorSettings::set_favorite_dirs(const Vector<String> &p_favorites_dirs) {

	favorite_dirs = p_favorites_dirs;
	FileAccess *f = FileAccess::open(get_project_settings_path().plus_file("favorite_dirs"), FileAccess::WRITE);
	if (f) {
		for (int i = 0; i < favorite_dirs.size(); i++)
			f->store_line(favorite_dirs[i]);
		memdelete(f);
	}
}

Vector<String> EditorSettings::get_favorite_dirs() const {

	return favorite_dirs;
}

void EditorSettings::set_recent_dirs(const Vector<String> &p_recent_dirs) {

	recent_dirs = p_recent_dirs;
	FileAccess *f = FileAccess::open(get_project_settings_path().plus_file("recent_dirs"), FileAccess::WRITE);
	if (f) {
		for (int i = 0; i < recent_dirs.size(); i++)
			f->store_line(recent_dirs[i]);
		memdelete(f);
	}
}

Vector<String> EditorSettings::get_recent_dirs() const {

	return recent_dirs;
}

String EditorSettings::get_project_settings_path() const {

	return get_settings_path().plus_file("config").plus_file(project_config_path);
}

void EditorSettings::load_favorites() {

	FileAccess *f = FileAccess::open(get_project_settings_path().plus_file("favorite_dirs"), FileAccess::READ);
	if (f) {
		String line = f->get_line().strip_edges();
		while (line != "") {
			favorite_dirs.push_back(line);
			line = f->get_line().strip_edges();
		}
		memdelete(f);
	}

	f = FileAccess::open(get_project_settings_path().plus_file("recent_dirs"), FileAccess::READ);
	if (f) {
		String line = f->get_line().strip_edges();
		while (line != "") {
			recent_dirs.push_back(line);
			line = f->get_line().strip_edges();
		}
		memdelete(f);
	}
}

void EditorSettings::list_text_editor_themes() {
	String themes = "Adaptive,Default";
	DirAccess *d = DirAccess::open(settings_path + "/text_editor_themes");
	if (d) {
		d->list_dir_begin();
		String file = d->get_next();
		while (file != String()) {
			if (file.get_extension() == "tet" && file.get_basename().to_lower() != "default" && file.get_basename().to_lower() != "adaptive") {
				themes += "," + file.get_basename();
			}
			file = d->get_next();
		}
		d->list_dir_end();
		memdelete(d);
	}
	add_property_hint(PropertyInfo(Variant::STRING, "text_editor/theme/color_theme", PROPERTY_HINT_ENUM, themes));
}

void EditorSettings::load_text_editor_theme() {
	if (get("text_editor/theme/color_theme") == "Default" || get("text_editor/theme/color_theme") == "Adaptive") {
		_load_default_text_editor_theme(); // sorry for "Settings changed" console spam
		return;
	}

	String theme_path = get_settings_path() + "/text_editor_themes/" + get("text_editor/theme/color_theme") + ".tet";

	Ref<ConfigFile> cf = memnew(ConfigFile);
	Error err = cf->load(theme_path);

	if (err != OK) {
		return;
	}

	List<String> keys;
	cf->get_section_keys("color_theme", &keys);

	for (List<String>::Element *E = keys.front(); E; E = E->next()) {
		String key = E->get();
		String val = cf->get_value("color_theme", key);

		// don't load if it's not already there!
		if (has("text_editor/highlighting/" + key)) {

			// make sure it is actually a color
			if (val.is_valid_html_color() && key.find("color") >= 0) {
				props["text_editor/highlighting/" + key].variant = Color::html(val); // change manually to prevent "Settings changed" console spam
			}
		}
	}
	emit_signal("settings_changed");
	// if it doesn't load just use what is currently loaded
}

bool EditorSettings::import_text_editor_theme(String p_file) {

	if (!p_file.ends_with(".tet")) {
		return false;
	} else {
		if (p_file.get_file().to_lower() == "default.tet") {
			return false;
		}

		DirAccess *d = DirAccess::open(settings_path + "/text_editor_themes");
		if (d) {
			d->copy(p_file, settings_path + "/text_editor_themes/" + p_file.get_file());
			memdelete(d);
			return true;
		}
	}
	return false;
}

bool EditorSettings::save_text_editor_theme() {

	String p_file = get("text_editor/theme/color_theme");

	if (p_file.get_file().to_lower() == "default" || p_file.get_file().to_lower() == "adaptive") {
		return false;
	}
	String theme_path = get_settings_path() + "/text_editor_themes/" + p_file + ".tet";
	return _save_text_editor_theme(theme_path);
}

bool EditorSettings::save_text_editor_theme_as(String p_file) {
	if (!p_file.ends_with(".tet")) {
		p_file += ".tet";
	}

	if (p_file.get_file().to_lower() == "default.tet" || p_file.get_file().to_lower() == "adaptive.tet") {
		return false;
	}
	if (_save_text_editor_theme(p_file)) {

		// switch to theme is saved in the theme directory
		list_text_editor_themes();
		String theme_name = p_file.substr(0, p_file.length() - 4).get_file();

		if (p_file.get_base_dir() == get_settings_path() + "/text_editor_themes") {
			set("text_editor/theme/color_theme", theme_name);
			load_text_editor_theme();
		}
		return true;
	}
	return false;
}

Vector<String> EditorSettings::get_script_templates(const String &p_extension) {

	Vector<String> templates;
	DirAccess *d = DirAccess::open(settings_path + "/script_templates");
	if (d) {
		d->list_dir_begin();
		String file = d->get_next();
		while (file != String()) {
			if (file.get_extension() == p_extension) {
				templates.push_back(file.get_basename());
			}
			file = d->get_next();
		}
		d->list_dir_end();
		memdelete(d);
	}
	return templates;
}

bool EditorSettings::_save_text_editor_theme(String p_file) {
	String theme_section = "color_theme";
	Ref<ConfigFile> cf = memnew(ConfigFile); // hex is better?
	cf->set_value(theme_section, "background_color", ((Color)get("text_editor/highlighting/background_color")).to_html());
	cf->set_value(theme_section, "completion_background_color", ((Color)get("text_editor/highlighting/completion_background_color")).to_html());
	cf->set_value(theme_section, "completion_selected_color", ((Color)get("text_editor/highlighting/completion_selected_color")).to_html());
	cf->set_value(theme_section, "completion_existing_color", ((Color)get("text_editor/highlighting/completion_existing_color")).to_html());
	cf->set_value(theme_section, "completion_scroll_color", ((Color)get("text_editor/highlighting/completion_scroll_color")).to_html());
	cf->set_value(theme_section, "completion_font_color", ((Color)get("text_editor/highlighting/completion_font_color")).to_html());
	cf->set_value(theme_section, "caret_color", ((Color)get("text_editor/highlighting/caret_color")).to_html());
	cf->set_value(theme_section, "caret_background_color", ((Color)get("text_editor/highlighting/caret_background_color")).to_html());
	cf->set_value(theme_section, "line_number_color", ((Color)get("text_editor/highlighting/line_number_color")).to_html());
	cf->set_value(theme_section, "text_color", ((Color)get("text_editor/highlighting/text_color")).to_html());
	cf->set_value(theme_section, "text_selected_color", ((Color)get("text_editor/highlighting/text_selected_color")).to_html());
	cf->set_value(theme_section, "keyword_color", ((Color)get("text_editor/highlighting/keyword_color")).to_html());
	cf->set_value(theme_section, "base_type_color", ((Color)get("text_editor/highlighting/base_type_color")).to_html());
	cf->set_value(theme_section, "engine_type_color", ((Color)get("text_editor/highlighting/engine_type_color")).to_html());
	cf->set_value(theme_section, "function_color", ((Color)get("text_editor/highlighting/function_color")).to_html());
	cf->set_value(theme_section, "member_variable_color", ((Color)get("text_editor/highlighting/member_variable_color")).to_html());
	cf->set_value(theme_section, "comment_color", ((Color)get("text_editor/highlighting/comment_color")).to_html());
	cf->set_value(theme_section, "string_color", ((Color)get("text_editor/highlighting/string_color")).to_html());
	cf->set_value(theme_section, "number_color", ((Color)get("text_editor/highlighting/number_color")).to_html());
	cf->set_value(theme_section, "symbol_color", ((Color)get("text_editor/highlighting/symbol_color")).to_html());
	cf->set_value(theme_section, "selection_color", ((Color)get("text_editor/highlighting/selection_color")).to_html());
	cf->set_value(theme_section, "brace_mismatch_color", ((Color)get("text_editor/highlighting/brace_mismatch_color")).to_html());
	cf->set_value(theme_section, "current_line_color", ((Color)get("text_editor/highlighting/current_line_color")).to_html());
	cf->set_value(theme_section, "line_length_guideline_color", ((Color)get("text_editor/highlighting/line_length_guideline_color")).to_html());
	cf->set_value(theme_section, "mark_color", ((Color)get("text_editor/highlighting/mark_color")).to_html());
	cf->set_value(theme_section, "breakpoint_color", ((Color)get("text_editor/highlighting/breakpoint_color")).to_html());
	cf->set_value(theme_section, "word_highlighted_color", ((Color)get("text_editor/highlighting/word_highlighted_color")).to_html());
	cf->set_value(theme_section, "search_result_color", ((Color)get("text_editor/highlighting/search_result_color")).to_html());
	cf->set_value(theme_section, "search_result_border_color", ((Color)get("text_editor/highlighting/search_result_border_color")).to_html());

	Error err = cf->save(p_file);

	if (err == OK) {
		return true;
	}
	return false;
}

void EditorSettings::add_shortcut(const String &p_name, Ref<ShortCut> &p_shortcut) {

	shortcuts[p_name] = p_shortcut;
}

bool EditorSettings::is_shortcut(const String &p_name, const Ref<InputEvent> &p_event) const {

	const Map<String, Ref<ShortCut> >::Element *E = shortcuts.find(p_name);
	if (!E) {
		ERR_EXPLAIN("Unknown Shortcut: " + p_name);
		ERR_FAIL_V(false);
	}

	return E->get()->is_shortcut(p_event);
}

Ref<ShortCut> EditorSettings::get_shortcut(const String &p_name) const {

	const Map<String, Ref<ShortCut> >::Element *E = shortcuts.find(p_name);
	if (!E)
		return Ref<ShortCut>();

	return E->get();
}

void EditorSettings::get_shortcut_list(List<String> *r_shortcuts) {

	for (const Map<String, Ref<ShortCut> >::Element *E = shortcuts.front(); E; E = E->next()) {

		r_shortcuts->push_back(E->key());
	}
}

void EditorSettings::set_optimize_save(bool p_optimize) {

	optimize_save = p_optimize;
}

Variant EditorSettings::get_project_metadata(const String &p_section, const String &p_key, Variant p_default) {
	Ref<ConfigFile> cf = memnew(ConfigFile);
	String path = get_project_settings_path().plus_file("project_metadata.cfg");
	Error err = cf->load(path);
	if (err != OK) {
		return p_default;
	}
	return cf->get_value(p_section, p_key, p_default);
}

void EditorSettings::set_project_metadata(const String &p_section, const String &p_key, Variant p_data) {
	Ref<ConfigFile> cf = memnew(ConfigFile);
	String path = get_project_settings_path().plus_file("project_metadata.cfg");
	cf->load(path);
	cf->set_value(p_section, p_key, p_data);
	cf->save(path);
}

void EditorSettings::_bind_methods() {

	ClassDB::bind_method(D_METHOD("erase", "property"), &EditorSettings::erase);
	ClassDB::bind_method(D_METHOD("get_settings_path"), &EditorSettings::get_settings_path);
	ClassDB::bind_method(D_METHOD("get_project_settings_path"), &EditorSettings::get_project_settings_path);

	ClassDB::bind_method(D_METHOD("add_property_info", "info"), &EditorSettings::_add_property_info_bind);

	ClassDB::bind_method(D_METHOD("set_favorite_dirs", "dirs"), &EditorSettings::set_favorite_dirs);
	ClassDB::bind_method(D_METHOD("get_favorite_dirs"), &EditorSettings::get_favorite_dirs);

	ClassDB::bind_method(D_METHOD("set_recent_dirs", "dirs"), &EditorSettings::set_recent_dirs);
	ClassDB::bind_method(D_METHOD("get_recent_dirs"), &EditorSettings::get_recent_dirs);

	ADD_SIGNAL(MethodInfo("settings_changed"));
}

EditorSettings::EditorSettings() {

	last_order = 0;
	optimize_save = true;
	save_changed_setting = true;

	EditorTranslationList *etl = _editor_translations;

	while (etl->data) {

		Vector<uint8_t> data;
		data.resize(etl->uncomp_size);
		Compression::decompress(data.ptr(), etl->uncomp_size, etl->data, etl->comp_size, Compression::MODE_DEFLATE);

		FileAccessMemory *fa = memnew(FileAccessMemory);
		fa->open_custom(data.ptr(), data.size());

		Ref<Translation> tr = TranslationLoaderPO::load_translation(fa, NULL, "translation_" + String(etl->lang));

		if (tr.is_valid()) {
			tr->set_locale(etl->lang);
			translations.push_back(tr);
		}

		etl++;
	}

	_load_defaults();
}

EditorSettings::~EditorSettings() {
}

Ref<ShortCut> ED_GET_SHORTCUT(const String &p_path) {

	Ref<ShortCut> sc = EditorSettings::get_singleton()->get_shortcut(p_path);
	if (!sc.is_valid()) {
		ERR_EXPLAIN("Used ED_GET_SHORTCUT with invalid shortcut: " + p_path);
		ERR_FAIL_COND_V(!sc.is_valid(), sc);
	}

	return sc;
}

Ref<ShortCut> ED_SHORTCUT(const String &p_path, const String &p_name, uint32_t p_keycode) {

	Ref<InputEventKey> ie;
	if (p_keycode) {
		ie.instance();

		ie->set_unicode(p_keycode & KEY_CODE_MASK);
		ie->set_scancode(p_keycode & KEY_CODE_MASK);
		ie->set_shift(bool(p_keycode & KEY_MASK_SHIFT));
		ie->set_alt(bool(p_keycode & KEY_MASK_ALT));
		ie->set_control(bool(p_keycode & KEY_MASK_CTRL));
		ie->set_metakey(bool(p_keycode & KEY_MASK_META));
	}

	Ref<ShortCut> sc = EditorSettings::get_singleton()->get_shortcut(p_path);
	if (sc.is_valid()) {

		sc->set_name(p_name); //keep name (the ones that come from disk have no name)
		sc->set_meta("original", ie); //to compare against changes
		return sc;
	}

	sc.instance();
	sc->set_name(p_name);
	sc->set_shortcut(ie);
	sc->set_meta("original", ie); //to compare against changes
	EditorSettings::get_singleton()->add_shortcut(p_path, sc);

	return sc;
}
