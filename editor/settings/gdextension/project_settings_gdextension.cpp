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
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/settings/gdextension/gdextension_edit_dialog.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/tree.h"

class GDExtensionPluginCreatorBase;

constexpr int BUTTON_EDIT = 0;
constexpr int BUTTON_RELOAD = 1;

void ProjectSettingsGDExtension::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED:
		case NOTIFICATION_WM_WINDOW_FOCUS_IN: {
			if (is_visible_in_tree()) {
				_update_extension_tree();
			}
		} break;
		case NOTIFICATION_READY: {
			edit_dialog->connect("gdextension_editor_closed", callable_mp(this, &ProjectSettingsGDExtension::_update_extension_tree));
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
	if (p_id == BUTTON_EDIT) {
		const String path = item->get_metadata(COLUMN_PATH);
		edit_dialog->load_gdextension_config(path);
		edit_dialog->popup_centered(Size2(400, 300) * EDSCALE);
	} else if (p_id == BUTTON_RELOAD) {
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
	Vector<String> extensions = GDExtensionManager::get_singleton()->get_loaded_extensions();
	GDExtensionManager *gdext_man = GDExtensionManager::get_singleton();
	for (const String &extension_path : extensions) {
		const Ref<GDExtension> gdextension = gdext_man->get_extension(extension_path);
		ERR_CONTINUE(gdextension.is_null());
		TreeItem *item = extension_list->create_item(root);
		item->set_text(COLUMN_PATH, extension_path);
		item->add_button(COLUMN_EDIT, get_editor_theme_icon(SNAME("Edit")), BUTTON_EDIT, false, TTRC("Edit Extension"));
		if (gdextension->is_reloadable()) {
			item->add_button(COLUMN_RELOAD, get_editor_theme_icon(SNAME("Reload")), BUTTON_RELOAD, false, TTRC("Reload Extension"));
		}
		item->set_metadata(COLUMN_PATH, extension_path);
	}
}

ProjectSettingsGDExtension::ProjectSettingsGDExtension() {
	edit_dialog = memnew(GDExtensionEditDialog);
	add_child(edit_dialog);
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
	extension_list->set_column_title(COLUMN_EDIT, TTRC("Edit"));
	extension_list->set_column_title(COLUMN_RELOAD, TTRC("Reload"));
	extension_list->set_column_title_alignment(COLUMN_PATH, HORIZONTAL_ALIGNMENT_LEFT);
	extension_list->set_column_title_alignment(COLUMN_EDIT, HORIZONTAL_ALIGNMENT_LEFT);
	extension_list->set_column_title_alignment(COLUMN_RELOAD, HORIZONTAL_ALIGNMENT_LEFT);
	extension_list->set_column_expand(COLUMN_PATH, true);
	extension_list->set_column_expand(COLUMN_EDIT, false);
	extension_list->set_column_expand(COLUMN_RELOAD, false);
	extension_list->set_column_clip_content(COLUMN_PATH, true);
	extension_list->set_column_clip_content(COLUMN_EDIT, true);
	extension_list->set_column_clip_content(COLUMN_RELOAD, true);
	extension_list->set_column_custom_minimum_width(COLUMN_PATH, 300 * EDSCALE);
	extension_list->set_column_custom_minimum_width(COLUMN_EDIT, 40 * EDSCALE);
	extension_list->set_column_custom_minimum_width(COLUMN_RELOAD, 40 * EDSCALE);
	add_child(extension_list);
}
