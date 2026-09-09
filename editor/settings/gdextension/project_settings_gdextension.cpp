/**************************************************************************/
/*  project_settings_gdextension.cpp                                      */
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

#include "project_settings_gdextension.h"

#include "core/config/project_settings.h"
#include "core/extension/gdextension_manager.h"
#include "core/io/config_file.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/label.h"
#include "scene/gui/tree.h"

class GDExtensionPluginCreatorBase;

void ProjectSettingsGDExtension::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED:
		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			if (is_visible_in_tree()) {
				_update_extension_tree();
			}
		} break;
		case NOTIFICATION_READY: {
			extension_list->connect("button_clicked", callable_mp(this, &ProjectSettingsGDExtension::_cell_button_pressed));
			extension_list->connect("item_activated", callable_mp(this, &ProjectSettingsGDExtension::_on_item_activated));
		} break;
	}
}

void ProjectSettingsGDExtension::_cell_button_pressed(Object *p_item, int p_column, int p_id, MouseButton p_button) {
	if (p_button != MouseButton::LEFT) {
		return;
	}
	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	if (item == nullptr) {
		return;
	}
	if (p_id == COLUMN_RELOAD) {
		const String path = item->get_metadata(COLUMN_PATH);
		GDExtensionManager::get_singleton()->reload_extension(path);
		_update_extension_tree();
	}
}

void ProjectSettingsGDExtension::_on_item_activated() {
	TreeItem *item = extension_list->get_selected();
	if (item == nullptr) {
		return;
	}
	const String global_path = ProjectSettings::get_singleton()->globalize_path(item->get_metadata(COLUMN_PATH));
	OS::get_singleton()->shell_open(global_path);
}

void ProjectSettingsGDExtension::_update_extension_tree() {
	extension_list->clear();
	TreeItem *root = extension_list->create_item();
	GDExtensionManager *gdext_man = GDExtensionManager::get_singleton();
	const Vector<String> extensions = gdext_man->get_loaded_extensions();
	for (const String &extension_path : extensions) {
		const Ref<GDExtension> gdextension = gdext_man->get_extension(extension_path);
		ERR_CONTINUE(gdextension.is_null());
		TreeItem *item = extension_list->create_item(root);
		item->set_text(COLUMN_PATH, extension_path);
		Ref<ConfigFile> gdext_config;
		gdext_config.instantiate();
		Error err = gdext_config->load(extension_path);
		ERR_CONTINUE(err != OK);
		item->set_text(COLUMN_MIN_VERSION, gdext_config->get_value("configuration", "compatibility_minimum", ""));
		item->set_text(COLUMN_MAX_VERSION, gdext_config->get_value("configuration", "compatibility_maximum", ""));
		if (gdextension->is_reloadable()) {
			item->add_button(COLUMN_RELOAD, get_editor_theme_icon(SNAME("Reload")), COLUMN_RELOAD, false, TTRC("Reload Extension"));
		}
		item->set_metadata(COLUMN_PATH, extension_path);
	}
}

ProjectSettingsGDExtension::ProjectSettingsGDExtension() {
	// Create the title label.
	HBoxContainer *title_hb = memnew(HBoxContainer);
	Label *label = memnew(Label(TTRC("Installed GDExtensions:")));
	label->set_theme_type_variation("HeaderSmall");
	title_hb->add_child(label);
	title_hb->add_spacer();
	add_child(title_hb);
	// Create the tree.
	extension_list = memnew(Tree);
	extension_list->set_v_size_flags(SIZE_EXPAND_FILL);
	extension_list->set_hide_root(true);
	extension_list->set_theme_type_variation("TreeTable");
	extension_list->set_hide_folding(true);
	// Configure tree columns.
	extension_list->set_columns(COLUMN_MAX);
	extension_list->set_column_titles_visible(true);
	extension_list->set_column_title(COLUMN_PATH, TTRC("Path"));
	extension_list->set_column_title(COLUMN_MIN_VERSION, TTRC("Min Version"));
	extension_list->set_column_title(COLUMN_MAX_VERSION, TTRC("Max Version"));
	extension_list->set_column_title(COLUMN_RELOAD, TTRC("Reload"));
	extension_list->set_column_title_alignment(COLUMN_PATH, HORIZONTAL_ALIGNMENT_LEFT);
	extension_list->set_column_title_alignment(COLUMN_MIN_VERSION, HORIZONTAL_ALIGNMENT_LEFT);
	extension_list->set_column_title_alignment(COLUMN_MAX_VERSION, HORIZONTAL_ALIGNMENT_LEFT);
	extension_list->set_column_title_alignment(COLUMN_RELOAD, HORIZONTAL_ALIGNMENT_LEFT);
	extension_list->set_column_expand(COLUMN_PATH, true);
	extension_list->set_column_expand(COLUMN_MIN_VERSION, false);
	extension_list->set_column_expand(COLUMN_MAX_VERSION, false);
	extension_list->set_column_expand(COLUMN_RELOAD, false);
	extension_list->set_column_clip_content(COLUMN_PATH, true);
	extension_list->set_column_clip_content(COLUMN_MIN_VERSION, true);
	extension_list->set_column_clip_content(COLUMN_MAX_VERSION, true);
	extension_list->set_column_clip_content(COLUMN_RELOAD, true);
	extension_list->set_column_custom_minimum_width(COLUMN_PATH, 300 * EDSCALE);
	extension_list->set_column_custom_minimum_width(COLUMN_MIN_VERSION, 100 * EDSCALE);
	extension_list->set_column_custom_minimum_width(COLUMN_MAX_VERSION, 100 * EDSCALE);
	extension_list->set_column_custom_minimum_width(COLUMN_RELOAD, 40 * EDSCALE);
	add_child(extension_list);
}
