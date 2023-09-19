/**************************************************************************/
/*  editor_plugin_settings.cpp                                            */
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

#include "editor_plugin_settings.h"

#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/os/main_loop.h"
#include "editor/editor_node.h"
#include "editor/editor_scale.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/tree.h"

void EditorPluginSettings::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			update_plugins();
		} break;

		case Node::NOTIFICATION_READY: {
			plugin_config_dialog->connect("plugin_ready", callable_mp(EditorNode::get_singleton(), &EditorNode::_on_plugin_ready));
			plugin_list->connect("button_clicked", callable_mp(this, &EditorPluginSettings::_cell_button_pressed));
		} break;
	}
}

void EditorPluginSettings::update_plugins() {
	plugin_list->clear();
	updating = true;
	TreeItem *root = plugin_list->create_item();

	Vector<String> plugins = _get_plugins("res://addons");
	plugins.sort();

	for (int i = 0; i < plugins.size(); i++) {
		Ref<ConfigFile> cf;
		cf.instantiate();
		const String path = plugins[i];

		Error err2 = cf->load(path);

		if (err2 != OK) {
			WARN_PRINT("Can't load plugin config: " + path);
		} else {
			bool key_missing = false;

			if (!cf->has_section_key("plugin", "name")) {
				WARN_PRINT("Plugin config misses \"plugin/name\" key: " + path);
				key_missing = true;
			}
			if (!cf->has_section_key("plugin", "author")) {
				WARN_PRINT("Plugin config misses \"plugin/author\" key: " + path);
				key_missing = true;
			}
			if (!cf->has_section_key("plugin", "version")) {
				WARN_PRINT("Plugin config misses \"plugin/version\" key: " + path);
				key_missing = true;
			}
			if (!cf->has_section_key("plugin", "description")) {
				WARN_PRINT("Plugin config misses \"plugin/description\" key: " + path);
				key_missing = true;
			}
			if (!cf->has_section_key("plugin", "script")) {
				WARN_PRINT("Plugin config misses \"plugin/script\" key: " + path);
				key_missing = true;
			}

			if (!key_missing) {
				String name = cf->get_value("plugin", "name");
				String author = cf->get_value("plugin", "author");
				String version = cf->get_value("plugin", "version");
				String description = cf->get_value("plugin", "description");
				String scr = cf->get_value("plugin", "script");

				const PackedInt32Array boundaries = TS->string_get_word_breaks(description, "", 80);
				String wrapped_description;

				for (int j = 0; j < boundaries.size(); j += 2) {
					const int start = boundaries[j];
					const int end = boundaries[j + 1];
					wrapped_description += "\n" + description.substr(start, end - start + 1).rstrip("\n");
				}

				TreeItem *item = plugin_list->create_item(root);
				item->set_text(0, name);
				item->set_tooltip_text(0, TTR("Name:") + " " + name + "\n" + TTR("Path:") + " " + path + "\n" + TTR("Main Script:") + " " + scr + "\n" + TTR("Description:") + " " + wrapped_description);
				item->set_metadata(0, path);
				item->set_text(1, version);
				item->set_metadata(1, scr);
				item->set_text(2, author);
				item->set_metadata(2, description);
				item->set_cell_mode(3, TreeItem::CELL_MODE_CHECK);
				item->set_text(3, TTR("Enable"));
				bool is_active = EditorNode::get_singleton()->is_addon_plugin_enabled(path);
				item->set_checked(3, is_active);
				item->set_editable(3, true);
				item->add_button(4, get_editor_theme_icon(SNAME("Edit")), BUTTON_PLUGIN_EDIT, false, TTR("Edit Plugin"));
			}
		}
	}

	updating = false;
}

void EditorPluginSettings::_plugin_activity_changed() {
	if (updating) {
		return;
	}

	TreeItem *ti = plugin_list->get_edited();
	ERR_FAIL_NULL(ti);
	bool active = ti->is_checked(3);
	String name = ti->get_metadata(0);

	EditorNode::get_singleton()->set_addon_plugin_enabled(name, active, true);

	bool is_active = EditorNode::get_singleton()->is_addon_plugin_enabled(name);

	if (is_active != active) {
		updating = true;
		ti->set_checked(3, is_active);
		updating = false;
	}
}

void EditorPluginSettings::_create_clicked() {
	plugin_config_dialog->config("");
	plugin_config_dialog->popup_centered();
}

void EditorPluginSettings::_cell_button_pressed(Object *p_item, int p_column, int p_id, MouseButton p_button) {
	if (p_button != MouseButton::LEFT) {
		return;
	}
	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	if (!item) {
		return;
	}
	if (p_id == BUTTON_PLUGIN_EDIT) {
		if (p_column == 4) {
			String dir = item->get_metadata(0);
			plugin_config_dialog->config(dir);
			plugin_config_dialog->popup_centered();
		}
	}
}

Vector<String> EditorPluginSettings::_get_plugins(const String &p_dir) {
	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	Error err = da->change_dir(p_dir);
	if (err != OK) {
		return Vector<String>();
	}

	Vector<String> plugins;
	da->list_dir_begin();
	for (String path = da->get_next(); !path.is_empty(); path = da->get_next()) {
		if (path[0] == '.' || !da->current_is_dir()) {
			continue;
		}

		const String full_path = p_dir.path_join(path);
		const String plugin_config = full_path.path_join("plugin.cfg");
		if (FileAccess::exists(plugin_config)) {
			plugins.push_back(plugin_config);
		} else {
			plugins.append_array(_get_plugins(full_path));
		}
	}

	da->list_dir_end();
	return plugins;
}

void EditorPluginSettings::_bind_methods() {
}

EditorPluginSettings::EditorPluginSettings() {
	ProjectSettings::get_singleton()->add_hidden_prefix("editor_plugins/");

	plugin_config_dialog = memnew(PluginConfigDialog);
	plugin_config_dialog->config("");
	add_child(plugin_config_dialog);

	HBoxContainer *title_hb = memnew(HBoxContainer);
	Label *l = memnew(Label(TTR("Installed Plugins:")));
	l->set_theme_type_variation("HeaderSmall");
	title_hb->add_child(l);
	title_hb->add_spacer();
	Button *create_plugin = memnew(Button(TTR("Create New Plugin")));
	create_plugin->connect("pressed", callable_mp(this, &EditorPluginSettings::_create_clicked));
	title_hb->add_child(create_plugin);
	add_child(title_hb);

	plugin_list = memnew(Tree);
	plugin_list->set_v_size_flags(SIZE_EXPAND_FILL);
	plugin_list->set_columns(5);
	plugin_list->set_column_titles_visible(true);
	plugin_list->set_column_title(0, TTR("Name"));
	plugin_list->set_column_title(1, TTR("Version"));
	plugin_list->set_column_title(2, TTR("Author"));
	plugin_list->set_column_title(3, TTR("Status"));
	plugin_list->set_column_title(4, TTR("Edit"));
	plugin_list->set_column_expand(0, true);
	plugin_list->set_column_clip_content(0, true);
	plugin_list->set_column_expand(1, false);
	plugin_list->set_column_clip_content(1, true);
	plugin_list->set_column_expand(2, false);
	plugin_list->set_column_clip_content(2, true);
	plugin_list->set_column_expand(3, false);
	plugin_list->set_column_clip_content(3, true);
	plugin_list->set_column_expand(4, false);
	plugin_list->set_column_clip_content(4, true);
	plugin_list->set_column_custom_minimum_width(1, 100 * EDSCALE);
	plugin_list->set_column_custom_minimum_width(2, 250 * EDSCALE);
	plugin_list->set_column_custom_minimum_width(3, 80 * EDSCALE);
	plugin_list->set_column_custom_minimum_width(4, 40 * EDSCALE);
	plugin_list->set_hide_root(true);
	plugin_list->connect("item_edited", callable_mp(this, &EditorPluginSettings::_plugin_activity_changed), CONNECT_DEFERRED);

	VBoxContainer *mc = memnew(VBoxContainer);
	mc->add_child(plugin_list);
	mc->set_v_size_flags(SIZE_EXPAND_FILL);
	mc->set_h_size_flags(SIZE_EXPAND_FILL);

	add_child(mc);
}
