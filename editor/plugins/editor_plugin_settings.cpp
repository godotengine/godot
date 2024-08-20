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
#include "editor/editor_settings.h"
#include "editor/editor_string_names.h"
#include "editor/themes/editor_scale.h"
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
		Ref<ConfigFile> cfg;
		cfg.instantiate();
		const String &path = plugins[i];

		Error err = cfg->load(path);

		if (err != OK) {
			WARN_PRINT("Can't load plugin config at: " + path);
		} else {
			Vector<String> missing_keys;
			for (const String required_key : { "name", "author", "version", "description", "script" }) {
				if (!cfg->has_section_key("plugin", required_key)) {
					missing_keys.append("\"plugin/" + required_key + "\"");
				}
			}

			if (!missing_keys.is_empty()) {
				WARN_PRINT(vformat("Plugin config at \"%s\" is missing the following keys: %s", path, String(",").join(missing_keys)));
			} else {
				String name = cfg->get_value("plugin", "name");
				String author = cfg->get_value("plugin", "author");
				String version = cfg->get_value("plugin", "version");
				String description = cfg->get_value("plugin", "description");
				String scr = cfg->get_value("plugin", "script");

				List<String> plugin_settings;
				cfg->get_sections(&plugin_settings);

				bool is_enabled = EditorNode::get_singleton()->is_addon_plugin_enabled(path);
				Color disabled_color = get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor));

				const PackedInt32Array boundaries = TS->string_get_word_breaks(description, "", 80);
				String wrapped_description;

				for (int j = 0; j < boundaries.size(); j += 2) {
					const int start = boundaries[j];
					const int end = boundaries[j + 1];
					wrapped_description += "\n" + description.substr(start, end - start + 1).rstrip("\n");
				}

				TreeItem *item = plugin_list->create_item(root);
				if (is_enabled) {
					TreeItem *info = item->create_child(0);
					TreeItem *info_author = info->create_child(0);
					TreeItem *info_version = info->create_child(1);
					TreeItem *info_script = info->create_child(2);
					info->set_text(COLUMN_NAME, TTR("Info"));
					info->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Info")));
					info->set_collapsed(true);
					info_author->set_text(COLUMN_NAME, TTR("Author"));
					info_author->set_text(COLUMN_STATUS, author);
					info_author->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Rename")));
					info_version->set_text(COLUMN_NAME, TTR("Version"));
					info_version->set_text(COLUMN_STATUS, version);
					info_version->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Tools")));
					info_script->set_text(COLUMN_NAME, TTR("Script"));
					info_script->set_text(COLUMN_STATUS, scr);
					info_script->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Script")));
					TreeItem *settings = item->create_child(1);
					settings->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("GDScript")));
					settings->set_text(COLUMN_NAME, TTR("Settings"));
					settings->set_collapsed(true);

					for (const String &section : plugin_settings) {
						if (section == String("plugin")) {
							continue;
						}
						TreeItem *setting = settings->create_child(0);
						List<String> keys;
						cfg->get_section_keys(section, &keys);
						setting->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Folder")));
						setting->set_text(COLUMN_NAME, section);
						setting->set_metadata(COLUMN_VERSION, name);
						for (const String &key : keys) {
							Variant value = cfg->get_value(section, key);
							if (value.get_type() == Variant::BOOL) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_CHECK);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_checked(COLUMN_STATUS, value);
								keyi->set_text(COLUMN_STATUS, value ? "true" : "false");
								keyi->set_metadata(COLUMN_STATUS, "key_bool");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": bool");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("bool")));
							} else if (value.get_type() == Variant::STRING) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_string");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": String");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("String")));
							} else if (value.get_type() == Variant::INT) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_int");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": int");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("int")));
							} else if (value.get_type() == Variant::FLOAT) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_float");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": float");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("float")));
							} else if (value.get_type() == Variant::VECTOR2) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_vector2");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Vector2");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Vector2")));
							} else if (value.get_type() == Variant::VECTOR3) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_vector3");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Vector3");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Vector3")));
							} else if (value.get_type() == Variant::DICTIONARY) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_dictionary");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Dictionary");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Dictionary")));
							} else if (value.get_type() == Variant::ARRAY) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_array");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Array");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Array")));
							} else if (value.get_type() == Variant::AABB) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_aabb");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": AABB");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("AABB")));
							} else if (value.get_type() == Variant::BASIS) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_basis");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Basis");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Basis")));
							} else if (value.get_type() == Variant::COLOR) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_color");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Color");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Color")));
							} else if (value.get_type() == Variant::NODE_PATH) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_nodepath");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": NodePath");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("NodePath")));
							} else if (value.get_type() == Variant::STRING_NAME) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_stringname");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": StringName");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("StringName")));
							} else if (value.get_type() == Variant::CALLABLE) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_callable");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Callable");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Callable")));
							}  else if (value.get_type() == Variant::RID) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_rid");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": RID");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("RID")));
							}  else if (value.get_type() == Variant::TRANSFORM2D) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_transform2d");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Transform2D");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Transform2D")));
							}  else if (value.get_type() == Variant::TRANSFORM3D) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_transform3d");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Transform3D");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Transform3D")));
							} else if (value.get_type() == Variant::RECT2) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_rect2");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Rect2");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Rect2")));
							} else if (value.get_type() == Variant::RECT2I) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_rect2i");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Rect2i");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Rect2i")));
							} else if (value.get_type() == Variant::SIGNAL) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_signal");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Signal");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Signal")));
							} else if (value.get_type() == Variant::VECTOR2I) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_vector2i");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Vector2i");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Vector2i")));
							} else if (value.get_type() == Variant::VECTOR3I) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_vector3i");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Vector3i");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Vector3i")));
							} else if (value.get_type() == Variant::VECTOR4) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_vector4");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Vector4");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Vector4")));
							} else if (value.get_type() == Variant::VECTOR4I) {
								TreeItem *keyi = setting->create_child(0);
								keyi->set_text(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_NAME, key);
								keyi->set_metadata(COLUMN_AUTHOR, section);
								keyi->set_metadata(COLUMN_VERSION, name);
								keyi->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_STRING);
								keyi->set_editable(COLUMN_STATUS, true);
								keyi->set_text(COLUMN_STATUS, value);
								keyi->set_metadata(COLUMN_STATUS, "key_vector4i");
								keyi->set_tooltip_text(COLUMN_NAME, key + ": Vector4i");
								keyi->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("Vector4i")));
							}
						}
					}
				} else {
					item->set_custom_color(COLUMN_NAME, disabled_color);
				}
				item->set_text(COLUMN_NAME, name);
				item->set_icon(COLUMN_NAME, plugin_list->get_editor_theme_icon(SNAME("EditorPlugin")));
				item->set_tooltip_text(COLUMN_NAME, vformat(TTR("Name: %s\nPath: %s\nMain Script: %s\n\n%s"), name, path, scr, wrapped_description));
				item->set_metadata(COLUMN_NAME, path);
				item->set_metadata(COLUMN_STATUS, path);
				item->set_custom_font(COLUMN_VERSION, get_theme_font("source", EditorStringName(EditorFonts)));
				item->set_metadata(COLUMN_VERSION, scr);
				item->set_metadata(COLUMN_AUTHOR, description);
				item->set_cell_mode(COLUMN_STATUS, TreeItem::CELL_MODE_CHECK);
				item->set_text(COLUMN_STATUS, (is_enabled) ? TTR("Enabled") : TTR("Disabled"));
				item->set_checked(COLUMN_STATUS, is_enabled);
				item->set_editable(COLUMN_STATUS, true);
			}
		}
	}

	updating = false;
}

void EditorPluginSettings::_plugin_activity_changed() {
	if (updating) {
		return;
	}
	Vector<String> plugins = _get_plugins("res://addons");
	plugins.sort();
	TreeItem *ti = plugin_list->get_edited();
	String plugin_name = ti->get_metadata(COLUMN_VERSION);
	String name = ti->get_metadata(COLUMN_NAME);
	String setting = ti->get_metadata(COLUMN_STATUS);
	String section = ti->get_metadata(COLUMN_AUTHOR);

	ERR_FAIL_NULL(ti);
	if (ti->get_cell_mode(COLUMN_STATUS) == TreeItem::CELL_MODE_CHECK) {
		bool checked = ti->is_checked(COLUMN_STATUS);
		if (setting == "key_bool") {
			Ref<ConfigFile> cfg;
			cfg.instantiate();
			String path = "";
			for (int i = 0; i < plugins.size(); i++) {
				Error err = cfg->load(plugins[i]);
				if (err != OK) {
					WARN_PRINT("Can't load plugin config at: " + path);
				} else {
					if (cfg->get_value("plugin", "name") == plugin_name) {
						path = plugins[i];
						break;
					} else {
						continue;
					}
				}
			}
			Error err = cfg->load(path);
			if (err != OK) {
				WARN_PRINT("Can't load plugin config at: " + path);
			} else {
				cfg->set_value(section, name, checked);
				ti->set_text(COLUMN_STATUS, cfg->get_value(section, name) ? "true" : "false");
				cfg->save(path);
			}
		} else {
			EditorNode::get_singleton()->set_addon_plugin_enabled(name, checked, true);
			bool is_enabled = EditorNode::get_singleton()->is_addon_plugin_enabled(name);
			if (is_enabled != checked) {
				updating = true;
				ti->set_checked(COLUMN_STATUS, is_enabled);
				updating = false;
			}
			if (is_enabled) {
				update_plugins();
			} else {
				ti->set_custom_color(COLUMN_NAME, get_theme_color(SNAME("font_disabled_color"), EditorStringName(Editor)));
				ti->remove_child(ti->get_child(0));
				ti->remove_child(ti->get_child(0));
				ti->set_text(COLUMN_STATUS, TTR("Disabled"));
			}
		}
	} else {
		Ref<ConfigFile> cfg;
		cfg.instantiate();
		String path = "";
		for (int i = 0; i < plugins.size(); i++) {
			Error err = cfg->load(plugins[i]);
			if (err != OK) {
				WARN_PRINT("Can't load plugin config at: " + path);
			} else {
				if (cfg->get_value("plugin", "name") == plugin_name) {
					path = plugins[i];
					break;
				} else {
					continue;
				}
			}
		}
		Error err = cfg->load(path);
		if (err != OK) {
			WARN_PRINT("Can't load plugin config at: " + path);
		} else {
			Variant value = ti->get_text(COLUMN_STATUS);
			cfg->set_value(section, name, value);
			ti->set_text(COLUMN_STATUS, cfg->get_value(section, name, value));
			cfg->save(path);
		}
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
		String dir = item->get_metadata(COLUMN_NAME);
		plugin_config_dialog->config(dir);
		plugin_config_dialog->popup_centered();
	} else if (p_id == BUTTON_PLUGIN_UNINSTALL) {
		String dir = item->get_metadata(COLUMN_NAME);
		plugin_config_dialog->config(dir);
		plugin_config_dialog->popup_centered();
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

EditorPluginSettings::EditorPluginSettings() {
	ProjectSettings::get_singleton()->add_hidden_prefix("editor_plugins/");

	plugin_config_dialog = memnew(PluginConfigDialog);
	plugin_config_dialog->config("");
	add_child(plugin_config_dialog);

	HBoxContainer *title_hb = memnew(HBoxContainer);
	Label *label = memnew(Label(TTR("Installed Plugins:")));
	label->set_theme_type_variation("HeaderSmall");
	title_hb->add_child(label);
	title_hb->add_spacer();
	Button *create_plugin_button = memnew(Button(TTR("Create New Plugin")));
	create_plugin_button->connect(SceneStringName(pressed), callable_mp(this, &EditorPluginSettings::_create_clicked));
	title_hb->add_child(create_plugin_button);
	add_child(title_hb);

	plugin_list = memnew(Tree);
	plugin_list->set_v_size_flags(SIZE_EXPAND_FILL);
	plugin_list->set_columns(COLUMN_MAX);
	plugin_list->set_column_titles_visible(false);
	/*plugin_list->set_column_title(COLUMN_STATUS, TTR("Value"));
	plugin_list->set_column_title(COLUMN_NAME, TTR("Setting"));
	plugin_list->set_column_title(COLUMN_VERSION, TTR("Version"));
	plugin_list->set_column_title(COLUMN_AUTHOR, TTR("Author"));
	plugin_list->set_column_title(COLUMN_EDIT, TTR("Edit"));
	plugin_list->set_column_title_alignment(COLUMN_STATUS, HORIZONTAL_ALIGNMENT_LEFT);
	plugin_list->set_column_title_alignment(COLUMN_NAME, HORIZONTAL_ALIGNMENT_LEFT);
	plugin_list->set_column_title_alignment(COLUMN_VERSION, HORIZONTAL_ALIGNMENT_LEFT);
	plugin_list->set_column_title_alignment(COLUMN_AUTHOR, HORIZONTAL_ALIGNMENT_LEFT);
	plugin_list->set_column_title_alignment(COLUMN_EDIT, HORIZONTAL_ALIGNMENT_LEFT);*/
	plugin_list->set_column_expand(COLUMN_STATUS, true);
	plugin_list->set_column_expand(COLUMN_NAME, true);
	plugin_list->set_column_expand(COLUMN_VERSION, false);
	plugin_list->set_column_expand(COLUMN_AUTHOR, true);
	plugin_list->set_column_expand(COLUMN_EDIT, false);
	plugin_list->set_column_expand(COLUMN_PADDING_RIGHT, true);
	plugin_list->set_column_clip_content(COLUMN_STATUS, true);
	plugin_list->set_column_clip_content(COLUMN_NAME, true);
	plugin_list->set_column_clip_content(COLUMN_VERSION, true);
	plugin_list->set_column_clip_content(COLUMN_AUTHOR, true);
	plugin_list->set_column_clip_content(COLUMN_EDIT, true);
	plugin_list->set_column_custom_minimum_width(COLUMN_STATUS, 80 * EDSCALE);
	plugin_list->set_column_custom_minimum_width(COLUMN_VERSION, 100 * EDSCALE);
	plugin_list->set_column_custom_minimum_width(COLUMN_AUTHOR, 250 * EDSCALE);
	plugin_list->set_column_custom_minimum_width(COLUMN_EDIT, 40 * EDSCALE);
	plugin_list->set_column_custom_minimum_width(COLUMN_PADDING_RIGHT, 10 * EDSCALE);
	plugin_list->set_hide_root(true);
	plugin_list->connect("item_edited", callable_mp(this, &EditorPluginSettings::_plugin_activity_changed), CONNECT_DEFERRED);

	VBoxContainer *mc = memnew(VBoxContainer);
	mc->add_child(plugin_list);
	mc->set_v_size_flags(SIZE_EXPAND_FILL);
	mc->set_h_size_flags(SIZE_EXPAND_FILL);

	add_child(mc);
}
