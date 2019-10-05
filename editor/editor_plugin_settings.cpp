/*************************************************************************/
/*  editor_plugin_settings.cpp                                           */
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

#include "editor_plugin_settings.h"

#include "core/io/config_file.h"
#include "core/os/file_access.h"
#include "core/os/main_loop.h"
#include "core/project_settings.h"
#include "editor_node.h"
#include "scene/gui/margin_container.h"

void EditorPluginSettings::_notification(int p_what) {

	if (p_what == MainLoop::NOTIFICATION_WM_FOCUS_IN) {
		update_plugins();
	} else if (p_what == Node::NOTIFICATION_READY) {
		plugin_config_dialog->connect("plugin_ready", EditorNode::get_singleton(), "_on_plugin_ready");
		plugin_list->connect("button_pressed", this, "_cell_button_pressed");
	}
}

void EditorPluginSettings::update_plugins() {

	plugin_list->clear();

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	Error err = da->change_dir("res://addons");
	if (err != OK) {
		memdelete(da);
		return;
	}

	updating = true;

	TreeItem *root = plugin_list->create_item();

	da->list_dir_begin();

	String d = da->get_next();

	Vector<String> plugins;

	while (d != String()) {

		bool dir = da->current_is_dir();
		String path = "res://addons/" + d + "/plugin.cfg";

		if (dir && FileAccess::exists(path)) {

			plugins.push_back(d);
		}

		d = da->get_next();
	}

	da->list_dir_end();
	memdelete(da);

	plugins.sort();

	for (int i = 0; i < plugins.size(); i++) {

		Ref<ConfigFile> cf;
		cf.instance();
		String path = "res://addons/" + plugins[i] + "/plugin.cfg";

		Error err2 = cf->load(path);

		if (err2 != OK) {
			WARN_PRINTS("Can't load plugin config: " + path);
		} else {
			bool key_missing = false;

			if (!cf->has_section_key("plugin", "name")) {
				WARN_PRINTS("Plugin config misses \"plugin/name\" key: " + path);
				key_missing = true;
			}
			if (!cf->has_section_key("plugin", "author")) {
				WARN_PRINTS("Plugin config misses \"plugin/author\" key: " + path);
				key_missing = true;
			}
			if (!cf->has_section_key("plugin", "version")) {
				WARN_PRINTS("Plugin config misses \"plugin/version\" key: " + path);
				key_missing = true;
			}
			if (!cf->has_section_key("plugin", "description")) {
				WARN_PRINTS("Plugin config misses \"plugin/description\" key: " + path);
				key_missing = true;
			}
			if (!cf->has_section_key("plugin", "script")) {
				WARN_PRINTS("Plugin config misses \"plugin/script\" key: " + path);
				key_missing = true;
			}

			if (!key_missing) {
				String d2 = plugins[i];
				String name = cf->get_value("plugin", "name");
				String author = cf->get_value("plugin", "author");
				String version = cf->get_value("plugin", "version");
				String description = cf->get_value("plugin", "description");
				String script = cf->get_value("plugin", "script");

				TreeItem *item = plugin_list->create_item(root);
				item->set_text(0, name);
				item->set_tooltip(0, TTR("Name:") + " " + name + "\n" + TTR("Path:") + " " + path + "\n" + TTR("Main Script:") + " " + script + "\n" + TTR("Description:") + " " + description);
				item->set_metadata(0, d2);
				item->set_text(1, version);
				item->set_metadata(1, script);
				item->set_text(2, author);
				item->set_metadata(2, description);
				item->set_cell_mode(3, TreeItem::CELL_MODE_RANGE);
				item->set_range_config(3, 0, 1, 1);
				item->set_text(3, "Inactive,Active");
				item->set_editable(3, true);
				item->add_button(4, get_icon("Edit", "EditorIcons"), BUTTON_PLUGIN_EDIT, false, TTR("Edit Plugin"));

				if (EditorNode::get_singleton()->is_addon_plugin_enabled(d2)) {
					item->set_custom_color(3, get_color("success_color", "Editor"));
					item->set_range(3, 1);
				} else {
					item->set_custom_color(3, get_color("disabled_font_color", "Editor"));
					item->set_range(3, 0);
				}
			}
		}
	}

	updating = false;
}

void EditorPluginSettings::_plugin_activity_changed() {

	if (updating)
		return;

	TreeItem *ti = plugin_list->get_edited();
	ERR_FAIL_COND(!ti);
	bool active = ti->get_range(3);
	String name = ti->get_metadata(0);

	EditorNode::get_singleton()->set_addon_plugin_enabled(name, active, true);

	bool is_active = EditorNode::get_singleton()->is_addon_plugin_enabled(name);

	if (is_active != active) {
		updating = true;
		ti->set_range(3, is_active ? 1 : 0);
		updating = false;
	}

	if (is_active)
		ti->set_custom_color(3, get_color("success_color", "Editor"));
	else
		ti->set_custom_color(3, get_color("disabled_font_color", "Editor"));
}

void EditorPluginSettings::_create_clicked() {
	plugin_config_dialog->config("");
	plugin_config_dialog->popup_centered();
}

void EditorPluginSettings::_cell_button_pressed(Object *p_item, int p_column, int p_id) {
	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	if (!item)
		return;
	if (p_id == BUTTON_PLUGIN_EDIT) {
		if (p_column == 4) {
			String dir = item->get_metadata(0);
			plugin_config_dialog->config("res://addons/" + dir + "/plugin.cfg");
			plugin_config_dialog->popup_centered();
		}
	}
}

void EditorPluginSettings::_bind_methods() {

	ClassDB::bind_method("update_plugins", &EditorPluginSettings::update_plugins);
	ClassDB::bind_method("_create_clicked", &EditorPluginSettings::_create_clicked);
	ClassDB::bind_method("_plugin_activity_changed", &EditorPluginSettings::_plugin_activity_changed);
	ClassDB::bind_method("_cell_button_pressed", &EditorPluginSettings::_cell_button_pressed);
}

EditorPluginSettings::EditorPluginSettings() {

	plugin_config_dialog = memnew(PluginConfigDialog);
	plugin_config_dialog->config("");
	add_child(plugin_config_dialog);

	HBoxContainer *title_hb = memnew(HBoxContainer);
	title_hb->add_child(memnew(Label(TTR("Installed Plugins:"))));
	title_hb->add_spacer();
	create_plugin = memnew(Button(TTR("Create")));
	create_plugin->connect("pressed", this, "_create_clicked");
	title_hb->add_child(create_plugin);
	update_list = memnew(Button(TTR("Update")));
	update_list->connect("pressed", this, "update_plugins");
	title_hb->add_child(update_list);
	add_child(title_hb);

	plugin_list = memnew(Tree);
	plugin_list->set_v_size_flags(SIZE_EXPAND_FILL);
	plugin_list->set_columns(5);
	plugin_list->set_column_titles_visible(true);
	plugin_list->set_column_title(0, TTR("Name:"));
	plugin_list->set_column_title(1, TTR("Version:"));
	plugin_list->set_column_title(2, TTR("Author:"));
	plugin_list->set_column_title(3, TTR("Status:"));
	plugin_list->set_column_title(4, TTR("Edit:"));
	plugin_list->set_column_expand(0, true);
	plugin_list->set_column_expand(1, false);
	plugin_list->set_column_expand(2, false);
	plugin_list->set_column_expand(3, false);
	plugin_list->set_column_expand(4, false);
	plugin_list->set_column_min_width(1, 100 * EDSCALE);
	plugin_list->set_column_min_width(2, 250 * EDSCALE);
	plugin_list->set_column_min_width(3, 80 * EDSCALE);
	plugin_list->set_column_min_width(4, 40 * EDSCALE);
	plugin_list->set_hide_root(true);
	plugin_list->connect("item_edited", this, "_plugin_activity_changed");

	VBoxContainer *mc = memnew(VBoxContainer);
	mc->add_child(plugin_list);
	mc->set_v_size_flags(SIZE_EXPAND_FILL);
	mc->set_h_size_flags(SIZE_EXPAND_FILL);

	add_child(mc);

	updating = false;
}
