/*************************************************************************/
/*  project_export.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "project_export.h"

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/image_loader.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/os/os.h"
#include "core/string/optimized_translation.h"
#include "core/version_generated.gen.h"
#include "editor_data.h"
#include "editor_node.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "scene/gui/box_container.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/tab_container.h"
#include "servers/display_server.h"

void ProjectExportDialog::_theme_changed() {
	duplicate_preset->set_icon(presets->get_theme_icon(SNAME("Duplicate"), SNAME("EditorIcons")));
	delete_preset->set_icon(presets->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
}

void ProjectExportDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (!is_visible()) {
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "export", Rect2(get_position(), get_size()));
			}
		} break;
		case NOTIFICATION_READY: {
			duplicate_preset->set_icon(presets->get_theme_icon(SNAME("Duplicate"), SNAME("EditorIcons")));
			delete_preset->set_icon(presets->get_theme_icon(SNAME("Remove"), SNAME("EditorIcons")));
			connect("confirmed", callable_mp(this, &ProjectExportDialog::_export_pck_zip));
			_update_export_all();
		} break;
	}
}

void ProjectExportDialog::popup_export() {
	add_preset->get_popup()->clear();
	for (int i = 0; i < EditorExport::get_singleton()->get_export_platform_count(); i++) {
		Ref<EditorExportPlatform> plat = EditorExport::get_singleton()->get_export_platform(i);

		add_preset->get_popup()->add_icon_item(plat->get_logo(), plat->get_name());
	}

	_update_presets();
	if (presets->get_current() >= 0) {
		_update_current_preset(); // triggers rescan for templates if newly installed
	}

	// Restore valid window bounds or pop up at default size.
	Rect2 saved_size = EditorSettings::get_singleton()->get_project_metadata("dialog_bounds", "export", Rect2());
	if (saved_size != Rect2()) {
		popup(saved_size);
	} else {
		popup_centered_clamped(Size2(900, 700) * EDSCALE, 0.8);
	}
}

void ProjectExportDialog::_add_preset(int p_platform) {
	Ref<EditorExportPreset> preset = EditorExport::get_singleton()->get_export_platform(p_platform)->create_preset();
	ERR_FAIL_COND(!preset.is_valid());

	String name = EditorExport::get_singleton()->get_export_platform(p_platform)->get_name();
	bool make_runnable = true;
	int attempt = 1;
	while (true) {
		bool valid = true;

		for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
			Ref<EditorExportPreset> p = EditorExport::get_singleton()->get_export_preset(i);
			if (p->get_platform() == preset->get_platform() && p->is_runnable()) {
				make_runnable = false;
			}
			if (p->get_name() == name) {
				valid = false;
				break;
			}
		}

		if (valid) {
			break;
		}

		attempt++;
		name = EditorExport::get_singleton()->get_export_platform(p_platform)->get_name() + " " + itos(attempt);
	}

	preset->set_name(name);
	if (make_runnable) {
		preset->set_runnable(make_runnable);
	}
	EditorExport::get_singleton()->add_export_preset(preset);
	_update_presets();
	_edit_preset(EditorExport::get_singleton()->get_export_preset_count() - 1);
}

void ProjectExportDialog::_force_update_current_preset_parameters() {
	// Force the parameters section to refresh its UI.
	parameters->edit(nullptr);
	_update_current_preset();
}

void ProjectExportDialog::_update_current_preset() {
	_edit_preset(presets->get_current());
}

void ProjectExportDialog::_update_presets() {
	updating = true;

	Ref<EditorExportPreset> current;
	if (presets->get_current() >= 0 && presets->get_current() < presets->get_item_count()) {
		current = get_current_preset();
	}

	int current_idx = -1;
	presets->clear();
	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
		Ref<EditorExportPreset> preset = EditorExport::get_singleton()->get_export_preset(i);
		if (preset == current) {
			current_idx = i;
		}

		String name = preset->get_name();
		if (preset->is_runnable()) {
			name += " (" + TTR("Runnable") + ")";
		}
		preset->update_files_to_export();
		presets->add_item(name, preset->get_platform()->get_logo());
	}

	if (current_idx != -1) {
		presets->select(current_idx);
	}

	updating = false;
}

void ProjectExportDialog::_update_export_all() {
	bool can_export = EditorExport::get_singleton()->get_export_preset_count() > 0;

	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
		Ref<EditorExportPreset> preset = EditorExport::get_singleton()->get_export_preset(i);
		bool needs_templates;
		String error;
		if (preset->get_export_path().is_empty() || !preset->get_platform()->can_export(preset, error, needs_templates)) {
			can_export = false;
			break;
		}
	}

	export_all_button->set_disabled(!can_export);

	if (can_export) {
		export_all_button->set_tooltip(TTR("Export the project for all the presets defined."));
	} else {
		export_all_button->set_tooltip(TTR("All presets must have an export path defined for Export All to work."));
	}
}

void ProjectExportDialog::_edit_preset(int p_index) {
	if (p_index < 0 || p_index >= presets->get_item_count()) {
		name->set_text("");
		name->set_editable(false);
		export_path->hide();
		runnable->set_disabled(true);
		parameters->edit(nullptr);
		presets->deselect_all();
		duplicate_preset->set_disabled(true);
		delete_preset->set_disabled(true);
		sections->hide();
		export_error->hide();
		export_templates_error->hide();
		return;
	}

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(p_index);
	ERR_FAIL_COND(current.is_null());

	updating = true;

	presets->select(p_index);
	sections->show();

	name->set_editable(true);
	export_path->show();
	duplicate_preset->set_disabled(false);
	delete_preset->set_disabled(false);
	name->set_text(current->get_name());

	List<String> extension_list = current->get_platform()->get_binary_extensions(current);
	Vector<String> extension_vector;
	for (int i = 0; i < extension_list.size(); i++) {
		extension_vector.push_back("*." + extension_list[i]);
	}

	export_path->setup(extension_vector, false, true);
	export_path->update_property();
	runnable->set_disabled(false);
	runnable->set_pressed(current->is_runnable());
	parameters->edit(current.ptr());

	export_filter->select(current->get_export_filter());
	include_filters->set_text(current->get_include_filter());
	exclude_filters->set_text(current->get_exclude_filter());

	_fill_resource_tree();

	bool needs_templates;
	String error;
	if (!current->get_platform()->can_export(current, error, needs_templates)) {
		if (!error.is_empty()) {
			Vector<String> items = error.split("\n", false);
			error = "";
			for (int i = 0; i < items.size(); i++) {
				if (i > 0) {
					error += "\n";
				}
				error += " - " + items[i];
			}

			export_error->set_text(error);
			export_error->show();
		} else {
			export_error->hide();
		}
		if (needs_templates) {
			export_templates_error->show();
		} else {
			export_templates_error->hide();
		}

		export_button->set_disabled(true);
		get_ok_button()->set_disabled(true);

	} else {
		export_error->hide();
		export_templates_error->hide();
		export_button->set_disabled(false);
		get_ok_button()->set_disabled(false);
	}

	custom_features->set_text(current->get_custom_features());
	_update_feature_list();
	_update_export_all();
	child_controls_changed();

	String enc_in_filters_str = current->get_enc_in_filter();
	String enc_ex_filters_str = current->get_enc_ex_filter();
	if (!updating_enc_filters) {
		enc_in_filters->set_text(enc_in_filters_str);
		enc_ex_filters->set_text(enc_ex_filters_str);
	}

	bool enc_pck_mode = current->get_enc_pck();
	enc_pck->set_pressed(enc_pck_mode);

	enc_directory->set_disabled(!enc_pck_mode);
	enc_in_filters->set_editable(enc_pck_mode);
	enc_ex_filters->set_editable(enc_pck_mode);
	script_key->set_editable(enc_pck_mode);

	bool enc_directory_mode = current->get_enc_directory();
	enc_directory->set_pressed(enc_directory_mode);

	int script_export_mode = current->get_script_export_mode();
	script_mode->select(script_export_mode);

	String key = current->get_script_encryption_key();
	if (!updating_script_key) {
		script_key->set_text(key);
	}
	if (enc_pck_mode) {
		script_key->set_editable(true);

		bool key_valid = _validate_script_encryption_key(key);
		if (key_valid) {
			script_key_error->hide();
		} else {
			script_key_error->show();
		}
	} else {
		script_key->set_editable(false);
		script_key_error->hide();
	}

	updating = false;
}

void ProjectExportDialog::_update_feature_list() {
	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	Set<String> fset;
	List<String> features;

	current->get_platform()->get_platform_features(&features);
	current->get_platform()->get_preset_features(current, &features);

	String custom = current->get_custom_features();
	Vector<String> custom_list = custom.split(",");
	for (int i = 0; i < custom_list.size(); i++) {
		String f = custom_list[i].strip_edges();
		if (!f.is_empty()) {
			features.push_back(f);
		}
	}

	for (const String &E : features) {
		fset.insert(E);
	}

	custom_feature_display->clear();
	for (Set<String>::Element *E = fset.front(); E; E = E->next()) {
		String f = E->get();
		if (E->next()) {
			f += ", ";
		}
		custom_feature_display->add_text(f);
	}
}

void ProjectExportDialog::_custom_features_changed(const String &p_text) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_custom_features(p_text);
	_update_feature_list();
}

void ProjectExportDialog::_tab_changed(int) {
	_update_feature_list();
}

void ProjectExportDialog::_update_parameters(const String &p_edited_property) {
	_update_current_preset();
}

void ProjectExportDialog::_runnable_pressed() {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	if (runnable->is_pressed()) {
		for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
			Ref<EditorExportPreset> p = EditorExport::get_singleton()->get_export_preset(i);
			if (p->get_platform() == current->get_platform()) {
				p->set_runnable(current == p);
			}
		}
	} else {
		current->set_runnable(false);
	}

	_update_presets();
}

void ProjectExportDialog::_name_changed(const String &p_string) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_name(p_string);
	_update_presets();
}

void ProjectExportDialog::set_export_path(const String &p_value) {
	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_export_path(p_value);
}

String ProjectExportDialog::get_export_path() {
	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND_V(current.is_null(), String(""));

	return current->get_export_path();
}

Ref<EditorExportPreset> ProjectExportDialog::get_current_preset() const {
	return EditorExport::get_singleton()->get_export_preset(presets->get_current());
}

void ProjectExportDialog::_export_path_changed(const StringName &p_property, const Variant &p_value, const String &p_field, bool p_changing) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_export_path(p_value);
	_update_presets();
	_update_export_all();
}

void ProjectExportDialog::_enc_filters_changed(const String &p_filters) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_enc_in_filter(enc_in_filters->get_text());
	current->set_enc_ex_filter(enc_ex_filters->get_text());

	updating_enc_filters = true;
	_update_current_preset();
	updating_enc_filters = false;
}

void ProjectExportDialog::_open_key_help_link() {
	OS::get_singleton()->shell_open(vformat("%s/development/compiling/compiling_with_script_encryption_key.html", VERSION_DOCS_URL));
}

void ProjectExportDialog::_enc_pck_changed(bool p_pressed) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_enc_pck(p_pressed);
	enc_directory->set_disabled(!p_pressed);
	enc_in_filters->set_editable(p_pressed);
	enc_ex_filters->set_editable(p_pressed);
	script_key->set_editable(p_pressed);

	_update_current_preset();
}

void ProjectExportDialog::_enc_directory_changed(bool p_pressed) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_enc_directory(p_pressed);

	_update_current_preset();
}

void ProjectExportDialog::_script_export_mode_changed(int p_mode) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_script_export_mode(p_mode);

	_update_current_preset();
}

void ProjectExportDialog::_script_encryption_key_changed(const String &p_key) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	current->set_script_encryption_key(p_key);

	updating_script_key = true;
	_update_current_preset();
	updating_script_key = false;
}

bool ProjectExportDialog::_validate_script_encryption_key(const String &p_key) {
	bool is_valid = false;

	if (!p_key.is_empty() && p_key.is_valid_hex_number(false) && p_key.length() == 64) {
		is_valid = true;
	}
	return is_valid;
}

void ProjectExportDialog::_duplicate_preset() {
	Ref<EditorExportPreset> current = get_current_preset();
	if (current.is_null()) {
		return;
	}

	Ref<EditorExportPreset> preset = current->get_platform()->create_preset();
	ERR_FAIL_COND(!preset.is_valid());

	String name = current->get_name() + " (copy)";
	bool make_runnable = true;
	while (true) {
		bool valid = true;

		for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
			Ref<EditorExportPreset> p = EditorExport::get_singleton()->get_export_preset(i);
			if (p->get_platform() == preset->get_platform() && p->is_runnable()) {
				make_runnable = false;
			}
			if (p->get_name() == name) {
				valid = false;
				break;
			}
		}

		if (valid) {
			break;
		}

		name += " (copy)";
	}

	preset->set_name(name);
	if (make_runnable) {
		preset->set_runnable(make_runnable);
	}
	preset->set_export_filter(current->get_export_filter());
	preset->set_include_filter(current->get_include_filter());
	preset->set_exclude_filter(current->get_exclude_filter());
	preset->set_custom_features(current->get_custom_features());

	for (const PropertyInfo &E : current->get_properties()) {
		preset->set(E.name, current->get(E.name));
	}

	EditorExport::get_singleton()->add_export_preset(preset);
	_update_presets();
	_edit_preset(EditorExport::get_singleton()->get_export_preset_count() - 1);
}

void ProjectExportDialog::_delete_preset() {
	Ref<EditorExportPreset> current = get_current_preset();
	if (current.is_null()) {
		return;
	}

	delete_confirm->set_text(vformat(TTR("Delete preset '%s'?"), current->get_name()));
	delete_confirm->popup_centered();
}

void ProjectExportDialog::_delete_preset_confirm() {
	int idx = presets->get_current();
	_edit_preset(-1);
	export_button->set_disabled(true);
	get_ok_button()->set_disabled(true);
	EditorExport::get_singleton()->remove_export_preset(idx);
	_update_presets();

	// The Export All button might become enabled (if all other presets have an export path defined),
	// or it could be disabled (if there are no presets anymore).
	_update_export_all();
}

Variant ProjectExportDialog::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	if (p_from == presets) {
		int pos = presets->get_item_at_position(p_point, true);

		if (pos >= 0) {
			Dictionary d;
			d["type"] = "export_preset";
			d["preset"] = pos;

			HBoxContainer *drag = memnew(HBoxContainer);
			TextureRect *tr = memnew(TextureRect);
			tr->set_texture(presets->get_item_icon(pos));
			drag->add_child(tr);
			Label *label = memnew(Label);
			label->set_text(presets->get_item_text(pos));
			drag->add_child(label);

			presets->set_drag_preview(drag);

			return d;
		}
	}

	return Variant();
}

bool ProjectExportDialog::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	if (p_from == presets) {
		Dictionary d = p_data;
		if (!d.has("type") || String(d["type"]) != "export_preset") {
			return false;
		}

		if (presets->get_item_at_position(p_point, true) < 0 && !presets->is_pos_at_end_of_items(p_point)) {
			return false;
		}
	}

	return true;
}

void ProjectExportDialog::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (p_from == presets) {
		Dictionary d = p_data;
		int from_pos = d["preset"];

		int to_pos = -1;

		if (presets->get_item_at_position(p_point, true) >= 0) {
			to_pos = presets->get_item_at_position(p_point, true);
		}

		if (to_pos == -1 && !presets->is_pos_at_end_of_items(p_point)) {
			return;
		}

		if (to_pos == from_pos) {
			return;
		} else if (to_pos > from_pos) {
			to_pos--;
		}

		Ref<EditorExportPreset> preset = EditorExport::get_singleton()->get_export_preset(from_pos);
		EditorExport::get_singleton()->remove_export_preset(from_pos);
		EditorExport::get_singleton()->add_export_preset(preset, to_pos);

		_update_presets();
		if (to_pos >= 0) {
			_edit_preset(to_pos);
		} else {
			_edit_preset(presets->get_item_count() - 1);
		}
	}
}

void ProjectExportDialog::_export_type_changed(int p_which) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	if (current.is_null()) {
		return;
	}

	current->set_export_filter(EditorExportPreset::ExportFilter(p_which));
	updating = true;
	_fill_resource_tree();
	updating = false;
}

void ProjectExportDialog::_filter_changed(const String &p_filter) {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	if (current.is_null()) {
		return;
	}

	current->set_include_filter(include_filters->get_text());
	current->set_exclude_filter(exclude_filters->get_text());
}

void ProjectExportDialog::_fill_resource_tree() {
	include_files->clear();
	include_label->hide();
	include_margin->hide();

	Ref<EditorExportPreset> current = get_current_preset();
	if (current.is_null()) {
		return;
	}

	EditorExportPreset::ExportFilter f = current->get_export_filter();

	if (f == EditorExportPreset::EXPORT_ALL_RESOURCES) {
		return;
	}

	include_label->show();
	include_margin->show();

	TreeItem *root = include_files->create_item();

	_fill_tree(EditorFileSystem::get_singleton()->get_filesystem(), root, current, f == EditorExportPreset::EXPORT_SELECTED_SCENES);
}

bool ProjectExportDialog::_fill_tree(EditorFileSystemDirectory *p_dir, TreeItem *p_item, Ref<EditorExportPreset> &current, bool p_only_scenes) {
	p_item->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	p_item->set_icon(0, presets->get_theme_icon(SNAME("folder"), SNAME("FileDialog")));
	p_item->set_text(0, p_dir->get_name() + "/");
	p_item->set_editable(0, true);
	p_item->set_metadata(0, p_dir->get_path());

	bool used = false;
	bool checked = true;
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {
		TreeItem *subdir = include_files->create_item(p_item);
		if (_fill_tree(p_dir->get_subdir(i), subdir, current, p_only_scenes)) {
			used = true;
			checked = checked && subdir->is_checked(0);
		} else {
			memdelete(subdir);
		}
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {
		String type = p_dir->get_file_type(i);
		if (p_only_scenes && type != "PackedScene") {
			continue;
		}
		if (type == "TextFile") {
			continue;
		}

		TreeItem *file = include_files->create_item(p_item);
		file->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		file->set_text(0, p_dir->get_file(i));

		String path = p_dir->get_file_path(i);

		file->set_icon(0, EditorNode::get_singleton()->get_class_icon(type));
		file->set_editable(0, true);
		file->set_checked(0, current->has_export_file(path));
		file->set_metadata(0, path);
		checked = checked && file->is_checked(0);

		used = true;
	}

	p_item->set_checked(0, checked);
	return used;
}

void ProjectExportDialog::_tree_changed() {
	if (updating) {
		return;
	}

	Ref<EditorExportPreset> current = get_current_preset();
	if (current.is_null()) {
		return;
	}

	TreeItem *item = include_files->get_edited();
	if (!item) {
		return;
	}

	String path = item->get_metadata(0);
	bool added = item->is_checked(0);

	if (path.ends_with("/")) {
		_check_dir_recursive(item, added);
	} else {
		if (added) {
			current->add_export_file(path);
		} else {
			current->remove_export_file(path);
		}
	}
	_refresh_parent_checks(item); // Makes parent folder checked if all files/folders are checked.
}

void ProjectExportDialog::_check_dir_recursive(TreeItem *p_dir, bool p_checked) {
	for (TreeItem *child = p_dir->get_first_child(); child; child = child->get_next()) {
		String path = child->get_metadata(0);

		child->set_checked(0, p_checked);
		if (path.ends_with("/")) {
			_check_dir_recursive(child, p_checked);
		} else {
			if (p_checked) {
				get_current_preset()->add_export_file(path);
			} else {
				get_current_preset()->remove_export_file(path);
			}
		}
	}
}

void ProjectExportDialog::_refresh_parent_checks(TreeItem *p_item) {
	TreeItem *parent = p_item->get_parent();
	if (!parent) {
		return;
	}

	bool checked = true;
	for (TreeItem *child = parent->get_first_child(); child; child = child->get_next()) {
		checked = checked && child->is_checked(0);
		if (!checked) {
			break;
		}
	}
	parent->set_checked(0, checked);

	_refresh_parent_checks(parent);
}

void ProjectExportDialog::_export_pck_zip() {
	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());

	String dir = current->get_export_path().get_base_dir();
	export_pck_zip->set_current_dir(dir);

	export_pck_zip->popup_file_dialog();
}

void ProjectExportDialog::_export_pck_zip_selected(const String &p_path) {
	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());
	Ref<EditorExportPlatform> platform = current->get_platform();
	ERR_FAIL_COND(platform.is_null());

	if (p_path.ends_with(".zip")) {
		platform->export_zip(current, export_pck_zip_debug->is_pressed(), p_path);
	} else if (p_path.ends_with(".pck")) {
		platform->export_pack(current, export_pck_zip_debug->is_pressed(), p_path);
	}
}

void ProjectExportDialog::_open_export_template_manager() {
	EditorNode::get_singleton()->open_export_template_manager();
	hide();
}

void ProjectExportDialog::_validate_export_path(const String &p_path) {
	// Disable export via OK button or Enter key if LineEdit has an empty filename
	bool invalid_path = (p_path.get_file().get_basename().is_empty());

	// Check if state change before needlessly messing with signals
	if (invalid_path && export_project->get_ok_button()->is_disabled()) {
		return;
	}
	if (!invalid_path && !export_project->get_ok_button()->is_disabled()) {
		return;
	}

	if (invalid_path) {
		export_project->get_ok_button()->set_disabled(true);
		export_project->get_line_edit()->disconnect("text_submitted", Callable(export_project, "_file_submitted"));
	} else {
		export_project->get_ok_button()->set_disabled(false);
		export_project->get_line_edit()->connect("text_submitted", Callable(export_project, "_file_submitted"));
	}
}

void ProjectExportDialog::_export_project() {
	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());
	Ref<EditorExportPlatform> platform = current->get_platform();
	ERR_FAIL_COND(platform.is_null());

	export_project->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	export_project->clear_filters();

	List<String> extension_list = platform->get_binary_extensions(current);
	for (int i = 0; i < extension_list.size(); i++) {
		export_project->add_filter("*." + extension_list[i] + " ; " + platform->get_name() + " Export");
	}

	if (!current->get_export_path().is_empty()) {
		export_project->set_current_path(current->get_export_path());
	} else {
		if (extension_list.size() >= 1) {
			export_project->set_current_file(default_filename + "." + extension_list[0]);
		} else {
			export_project->set_current_file(default_filename);
		}
	}

	// Ensure that signal is connected if previous attempt left it disconnected
	// with _validate_export_path.
	// FIXME: This is a hack, we should instead change EditorFileDialog to allow
	// disabling validation by the "text_submitted" signal.
	if (!export_project->get_line_edit()->is_connected("text_submitted", Callable(export_project, "_file_submitted"))) {
		export_project->get_ok_button()->set_disabled(false);
		export_project->get_line_edit()->connect("text_submitted", Callable(export_project, "_file_submitted"));
	}

	export_project->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	export_project->popup_file_dialog();
}

void ProjectExportDialog::_export_project_to_path(const String &p_path) {
	// Save this name for use in future exports (but drop the file extension)
	default_filename = p_path.get_file().get_basename();
	EditorSettings::get_singleton()->set_project_metadata("export_options", "default_filename", default_filename);

	Ref<EditorExportPreset> current = get_current_preset();
	ERR_FAIL_COND(current.is_null());
	Ref<EditorExportPlatform> platform = current->get_platform();
	ERR_FAIL_COND(platform.is_null());
	current->set_export_path(p_path);

	Error err = platform->export_project(current, export_debug->is_pressed(), p_path, 0);
	if (err != OK && err != ERR_SKIP) {
		if (err == ERR_FILE_NOT_FOUND) {
			error_dialog->set_text(vformat(TTR("Failed to export the project for platform '%s'.\nExport templates seem to be missing or invalid."), platform->get_name()));
		} else { // Assume misconfiguration. FIXME: Improve error handling and preset config validation.
			error_dialog->set_text(vformat(TTR("Failed to export the project for platform '%s'.\nThis might be due to a configuration issue in the export preset or your export settings."), platform->get_name()));
		}

		ERR_PRINT(vformat("Failed to export the project for platform '%s'.", platform->get_name()));
		error_dialog->show();
		error_dialog->popup_centered(Size2(300, 80));
	}
}

void ProjectExportDialog::_export_all_dialog() {
	export_all_dialog->show();
	export_all_dialog->popup_centered(Size2(300, 80));
}

void ProjectExportDialog::_export_all_dialog_action(const String &p_str) {
	export_all_dialog->hide();

	_export_all(p_str != "release");
}

void ProjectExportDialog::_export_all(bool p_debug) {
	String mode = p_debug ? TTR("Debug") : TTR("Release");
	EditorProgress ep("exportall", TTR("Exporting All") + " " + mode, EditorExport::get_singleton()->get_export_preset_count(), true);

	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
		Ref<EditorExportPreset> preset = EditorExport::get_singleton()->get_export_preset(i);
		ERR_FAIL_COND(preset.is_null());
		Ref<EditorExportPlatform> platform = preset->get_platform();
		ERR_FAIL_COND(platform.is_null());

		ep.step(preset->get_name(), i);

		Error err = platform->export_project(preset, p_debug, preset->get_export_path(), 0);
		if (err != OK && err != ERR_SKIP) {
			if (err == ERR_FILE_BAD_PATH) {
				error_dialog->set_text(TTR("The given export path doesn't exist:") + "\n" + preset->get_export_path().get_base_dir());
			} else {
				error_dialog->set_text(TTR("Export templates for this platform are missing/corrupted:") + " " + platform->get_name());
			}
			error_dialog->show();
			error_dialog->popup_centered(Size2(300, 80));
			ERR_PRINT("Failed to export project");
		}
	}
}

void ProjectExportDialog::_bind_methods() {
	ClassDB::bind_method("_get_drag_data_fw", &ProjectExportDialog::get_drag_data_fw);
	ClassDB::bind_method("_can_drop_data_fw", &ProjectExportDialog::can_drop_data_fw);
	ClassDB::bind_method("_drop_data_fw", &ProjectExportDialog::drop_data_fw);
	ClassDB::bind_method("_export_all", &ProjectExportDialog::_export_all);
	ClassDB::bind_method("set_export_path", &ProjectExportDialog::set_export_path);
	ClassDB::bind_method("get_export_path", &ProjectExportDialog::get_export_path);
	ClassDB::bind_method("get_current_preset", &ProjectExportDialog::get_current_preset);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "export_path"), "set_export_path", "get_export_path");
}

ProjectExportDialog::ProjectExportDialog() {
	set_title(TTR("Export"));

	VBoxContainer *main_vb = memnew(VBoxContainer);
	main_vb->connect("theme_changed", callable_mp(this, &ProjectExportDialog::_theme_changed));
	add_child(main_vb);
	HSplitContainer *hbox = memnew(HSplitContainer);
	main_vb->add_child(hbox);
	hbox->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	// Presets list.

	VBoxContainer *preset_vb = memnew(VBoxContainer);
	preset_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbox->add_child(preset_vb);

	Label *l = memnew(Label(TTR("Presets")));
	l->set_theme_type_variation("HeaderSmall");

	HBoxContainer *preset_hb = memnew(HBoxContainer);
	preset_hb->add_child(l);
	preset_hb->add_spacer();
	preset_vb->add_child(preset_hb);

	add_preset = memnew(MenuButton);
	add_preset->set_text(TTR("Add..."));
	add_preset->get_popup()->connect("index_pressed", callable_mp(this, &ProjectExportDialog::_add_preset));
	preset_hb->add_child(add_preset);
	MarginContainer *mc = memnew(MarginContainer);
	preset_vb->add_child(mc);
	mc->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	presets = memnew(ItemList);
#ifndef _MSC_VER
#warning must reimplement drag forward
#endif
	//presets->set_drag_forwarding(this);
	mc->add_child(presets);
	presets->connect("item_selected", callable_mp(this, &ProjectExportDialog::_edit_preset));
	duplicate_preset = memnew(Button);
	duplicate_preset->set_flat(true);
	preset_hb->add_child(duplicate_preset);
	duplicate_preset->connect("pressed", callable_mp(this, &ProjectExportDialog::_duplicate_preset));
	delete_preset = memnew(Button);
	delete_preset->set_flat(true);
	preset_hb->add_child(delete_preset);
	delete_preset->connect("pressed", callable_mp(this, &ProjectExportDialog::_delete_preset));

	// Preset settings.

	VBoxContainer *settings_vb = memnew(VBoxContainer);
	settings_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbox->add_child(settings_vb);

	name = memnew(LineEdit);
	settings_vb->add_margin_child(TTR("Name:"), name);
	name->connect("text_changed", callable_mp(this, &ProjectExportDialog::_name_changed));
	runnable = memnew(CheckButton);
	runnable->set_text(TTR("Runnable"));
	runnable->set_tooltip(TTR("If checked, the preset will be available for use in one-click deploy.\nOnly one preset per platform may be marked as runnable."));
	runnable->connect("pressed", callable_mp(this, &ProjectExportDialog::_runnable_pressed));
	settings_vb->add_child(runnable);

	export_path = memnew(EditorPropertyPath);
	settings_vb->add_child(export_path);
	export_path->set_label(TTR("Export Path"));
	export_path->set_object_and_property(this, "export_path");
	export_path->set_save_mode();
	export_path->connect("property_changed", callable_mp(this, &ProjectExportDialog::_export_path_changed));

	// Subsections.

	sections = memnew(TabContainer);
	sections->set_tab_alignment(TabContainer::ALIGNMENT_LEFT);
	sections->set_use_hidden_tabs_for_min_size(true);
	settings_vb->add_child(sections);
	sections->set_v_size_flags(Control::SIZE_EXPAND_FILL);

	// Main preset parameters.

	parameters = memnew(EditorInspector);
	sections->add_child(parameters);
	parameters->set_name(TTR("Options"));
	parameters->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	parameters->connect("property_edited", callable_mp(this, &ProjectExportDialog::_update_parameters));
	EditorExport::get_singleton()->connect("export_presets_updated", callable_mp(this, &ProjectExportDialog::_force_update_current_preset_parameters));

	// Resources export parameters.

	VBoxContainer *resources_vb = memnew(VBoxContainer);
	sections->add_child(resources_vb);
	resources_vb->set_name(TTR("Resources"));

	export_filter = memnew(OptionButton);
	export_filter->add_item(TTR("Export all resources in the project"));
	export_filter->add_item(TTR("Export selected scenes (and dependencies)"));
	export_filter->add_item(TTR("Export selected resources (and dependencies)"));
	export_filter->add_item(TTR("Export all resources in the project except resources checked below"));
	resources_vb->add_margin_child(TTR("Export Mode:"), export_filter);
	export_filter->connect("item_selected", callable_mp(this, &ProjectExportDialog::_export_type_changed));

	include_label = memnew(Label);
	include_label->set_text(TTR("Resources to export:"));
	resources_vb->add_child(include_label);
	include_margin = memnew(MarginContainer);
	include_margin->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	resources_vb->add_child(include_margin);

	include_files = memnew(Tree);
	include_margin->add_child(include_files);
	include_files->connect("item_edited", callable_mp(this, &ProjectExportDialog::_tree_changed));

	include_filters = memnew(LineEdit);
	resources_vb->add_margin_child(
			TTR("Filters to export non-resource files/folders\n(comma-separated, e.g: *.json, *.txt, docs/*)"),
			include_filters);
	include_filters->connect("text_changed", callable_mp(this, &ProjectExportDialog::_filter_changed));

	exclude_filters = memnew(LineEdit);
	resources_vb->add_margin_child(
			TTR("Filters to exclude files/folders from project\n(comma-separated, e.g: *.json, *.txt, docs/*)"),
			exclude_filters);
	exclude_filters->connect("text_changed", callable_mp(this, &ProjectExportDialog::_filter_changed));

	script_mode = memnew(OptionButton);
	resources_vb->add_margin_child(TTR("GDScript Export Mode:"), script_mode);
	script_mode->add_item(TTR("Text"), (int)EditorExportPreset::MODE_SCRIPT_TEXT);
	script_mode->add_item(TTR("Compiled Bytecode (Faster Loading)"), (int)EditorExportPreset::MODE_SCRIPT_COMPILED);
	script_mode->connect("item_selected", callable_mp(this, &ProjectExportDialog::_script_export_mode_changed));

	// Feature tags.

	VBoxContainer *feature_vb = memnew(VBoxContainer);
	feature_vb->set_name(TTR("Features"));
	custom_features = memnew(LineEdit);
	custom_features->connect("text_changed", callable_mp(this, &ProjectExportDialog::_custom_features_changed));
	feature_vb->add_margin_child(TTR("Custom (comma-separated):"), custom_features);
	custom_feature_display = memnew(RichTextLabel);
	custom_feature_display->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	feature_vb->add_margin_child(TTR("Feature List:"), custom_feature_display, true);
	sections->add_child(feature_vb);

	// Script export parameters.

	updating_script_key = false;
	updating_enc_filters = false;

	VBoxContainer *sec_vb = memnew(VBoxContainer);
	sec_vb->set_name(TTR("Encryption"));

	enc_pck = memnew(CheckButton);
	enc_pck->connect("toggled", callable_mp(this, &ProjectExportDialog::_enc_pck_changed));
	enc_pck->set_text(TTR("Encrypt Exported PCK"));
	sec_vb->add_child(enc_pck);

	enc_directory = memnew(CheckButton);
	enc_directory->connect("toggled", callable_mp(this, &ProjectExportDialog::_enc_directory_changed));
	enc_directory->set_text(TTR("Encrypt Index (File Names and Info)"));
	sec_vb->add_child(enc_directory);

	enc_in_filters = memnew(LineEdit);
	enc_in_filters->connect("text_changed", callable_mp(this, &ProjectExportDialog::_enc_filters_changed));
	sec_vb->add_margin_child(
			TTR("Filters to include files/folders\n(comma-separated, e.g: *.tscn, *.tres, scenes/*)"),
			enc_in_filters);

	enc_ex_filters = memnew(LineEdit);
	enc_ex_filters->connect("text_changed", callable_mp(this, &ProjectExportDialog::_enc_filters_changed));
	sec_vb->add_margin_child(
			TTR("Filters to exclude files/folders\n(comma-separated, e.g: *.stex, *.import, music/*)"),
			enc_ex_filters);

	script_key = memnew(LineEdit);
	script_key->connect("text_changed", callable_mp(this, &ProjectExportDialog::_script_encryption_key_changed));
	script_key_error = memnew(Label);
	script_key_error->set_text(String::utf8("•  ") + TTR("Invalid Encryption Key (must be 64 hexadecimal characters long)"));
	script_key_error->add_theme_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_theme_color(SNAME("error_color"), SNAME("Editor")));
	sec_vb->add_margin_child(TTR("Encryption Key (256-bits as hexadecimal):"), script_key);
	sec_vb->add_child(script_key_error);
	sections->add_child(sec_vb);

	Label *sec_info = memnew(Label);
	sec_info->set_text(TTR("Note: Encryption key needs to be stored in the binary,\nyou need to build the export templates from source."));
	sec_vb->add_child(sec_info);

	LinkButton *sec_more_info = memnew(LinkButton);
	sec_more_info->set_text(TTR("More Info..."));
	sec_more_info->connect("pressed", callable_mp(this, &ProjectExportDialog::_open_key_help_link));
	sec_vb->add_child(sec_more_info);

	sections->connect("tab_changed", callable_mp(this, &ProjectExportDialog::_tab_changed));

	// Disable by default.
	name->set_editable(false);
	export_path->hide();
	runnable->set_disabled(true);
	duplicate_preset->set_disabled(true);
	delete_preset->set_disabled(true);
	script_key_error->hide();
	sections->hide();
	parameters->edit(nullptr);

	// Deletion dialog.

	delete_confirm = memnew(ConfirmationDialog);
	add_child(delete_confirm);
	delete_confirm->get_ok_button()->set_text(TTR("Delete"));
	delete_confirm->connect("confirmed", callable_mp(this, &ProjectExportDialog::_delete_preset_confirm));

	// Export buttons, dialogs and errors.

	updating = false;

	get_cancel_button()->set_text(TTR("Close"));
	get_ok_button()->set_text(TTR("Export PCK/ZIP..."));
	export_button = add_button(TTR("Export Project..."), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "export");
	export_button->connect("pressed", callable_mp(this, &ProjectExportDialog::_export_project));
	// Disable initially before we select a valid preset
	export_button->set_disabled(true);
	get_ok_button()->set_disabled(true);

	export_all_dialog = memnew(ConfirmationDialog);
	add_child(export_all_dialog);
	export_all_dialog->set_title(TTR("Export All"));
	export_all_dialog->set_text(TTR("Choose an export mode:"));
	export_all_dialog->get_ok_button()->hide();
	export_all_dialog->add_button(TTR("Debug"), true, "debug");
	export_all_dialog->add_button(TTR("Release"), true, "release");
	export_all_dialog->connect("custom_action", callable_mp(this, &ProjectExportDialog::_export_all_dialog_action));

	export_all_button = add_button(TTR("Export All..."), !DisplayServer::get_singleton()->get_swap_cancel_ok(), "export");
	export_all_button->connect("pressed", callable_mp(this, &ProjectExportDialog::_export_all_dialog));
	export_all_button->set_disabled(true);

	export_pck_zip = memnew(EditorFileDialog);
	export_pck_zip->add_filter("*.zip ; " + TTR("ZIP File"));
	export_pck_zip->add_filter("*.pck ; " + TTR("Godot Project Pack"));
	export_pck_zip->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	export_pck_zip->set_file_mode(EditorFileDialog::FILE_MODE_SAVE_FILE);
	add_child(export_pck_zip);
	export_pck_zip->connect("file_selected", callable_mp(this, &ProjectExportDialog::_export_pck_zip_selected));

	export_error = memnew(Label);
	main_vb->add_child(export_error);
	export_error->hide();
	export_error->add_theme_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_theme_color(SNAME("error_color"), SNAME("Editor")));

	export_templates_error = memnew(HBoxContainer);
	main_vb->add_child(export_templates_error);
	export_templates_error->hide();

	Label *export_error2 = memnew(Label);
	export_templates_error->add_child(export_error2);
	export_error2->add_theme_color_override("font_color", EditorNode::get_singleton()->get_gui_base()->get_theme_color(SNAME("error_color"), SNAME("Editor")));
	export_error2->set_text(String::utf8("•  ") + TTR("Export templates for this platform are missing:") + " ");

	error_dialog = memnew(AcceptDialog);
	error_dialog->set_title(TTR("Error"));
	error_dialog->set_text(TTR("Export templates for this platform are missing/corrupted:") + " ");
	main_vb->add_child(error_dialog);
	error_dialog->hide();

	LinkButton *download_templates = memnew(LinkButton);
	download_templates->set_text(TTR("Manage Export Templates"));
	download_templates->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	export_templates_error->add_child(download_templates);
	download_templates->connect("pressed", callable_mp(this, &ProjectExportDialog::_open_export_template_manager));

	export_project = memnew(EditorFileDialog);
	export_project->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	add_child(export_project);
	export_project->connect("file_selected", callable_mp(this, &ProjectExportDialog::_export_project_to_path));
	export_project->get_line_edit()->connect("text_changed", callable_mp(this, &ProjectExportDialog::_validate_export_path));

	export_debug = memnew(CheckBox);
	export_debug->set_text(TTR("Export With Debug"));
	export_debug->set_pressed(true);
	export_debug->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	export_project->get_vbox()->add_child(export_debug);

	export_pck_zip_debug = memnew(CheckBox);
	export_pck_zip_debug->set_text(TTR("Export With Debug"));
	export_pck_zip_debug->set_pressed(true);
	export_pck_zip_debug->set_h_size_flags(Control::SIZE_SHRINK_CENTER);
	export_pck_zip->get_vbox()->add_child(export_pck_zip_debug);

	set_hide_on_ok(false);

	editor_icons = "EditorIcons";

	default_filename = EditorSettings::get_singleton()->get_project_metadata("export_options", "default_filename", "");
	// If no default set, use project name
	if (default_filename.is_empty()) {
		// If no project name defined, use a sane default
		default_filename = ProjectSettings::get_singleton()->get("application/config/name");
		if (default_filename.is_empty()) {
			default_filename = "UnnamedProject";
		}
	}
}

ProjectExportDialog::~ProjectExportDialog() {
}
