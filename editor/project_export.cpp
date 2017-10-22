/*************************************************************************/
/*  project_export.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "project_export.h"
#include "compressed_translation.h"
#include "editor_data.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "io/image_loader.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/dir_access.h"
#include "os/file_access.h"
#include "os/os.h"
#include "project_settings.h"
#include "scene/gui/box_container.h"
#include "scene/gui/margin_container.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/tab_container.h"

void ProjectExportDialog::_notification(int p_what) {

	switch (p_what) {
		case NOTIFICATION_READY: {
			delete_preset->set_icon(get_icon("Remove", "EditorIcons"));
			connect("confirmed", this, "_export_pck_zip");
			custom_feature_display->get_parent_control()->add_style_override("panel", get_stylebox("bg", "Tree"));
		} break;
		case NOTIFICATION_POPUP_HIDE: {
			EditorSettings::get_singleton()->set("interface/dialogs/export_bounds", get_rect());
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

	// Restore valid window bounds or pop up at default size.
	if (EditorSettings::get_singleton()->has_setting("interface/dialogs/export_bounds")) {
		popup(EditorSettings::get_singleton()->get("interface/dialogs/export_bounds"));
	} else {
		popup_centered_ratio();
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

		if (valid)
			break;

		attempt++;
		name = EditorExport::get_singleton()->get_export_platform(p_platform)->get_name() + " " + itos(attempt);
	}

	preset->set_name(name);
	if (make_runnable)
		preset->set_runnable(make_runnable);
	EditorExport::get_singleton()->add_export_preset(preset);
	_update_presets();
	_edit_preset(EditorExport::get_singleton()->get_export_preset_count() - 1);
}

void ProjectExportDialog::_update_presets() {

	updating = true;

	Ref<EditorExportPreset> current;
	if (presets->get_current() >= 0 && presets->get_current() < presets->get_item_count())
		current = EditorExport::get_singleton()->get_export_preset(presets->get_current());

	int current_idx = -1;
	presets->clear();
	for (int i = 0; i < EditorExport::get_singleton()->get_export_preset_count(); i++) {
		Ref<EditorExportPreset> preset = EditorExport::get_singleton()->get_export_preset(i);
		if (preset == current) {
			current_idx = i;
		}

		String name = preset->get_name();
		if (preset->is_runnable())
			name += " (" + TTR("Runnable") + ")";
		presets->add_item(name, preset->get_platform()->get_logo());
	}

	if (current_idx != -1) {
		presets->select(current_idx);
		//_edit_preset(current_idx);
	}

	updating = false;
}

void ProjectExportDialog::_edit_preset(int p_index) {

	if (p_index < 0 || p_index >= presets->get_item_count()) {
		name->set_text("");
		name->set_editable(false);
		runnable->set_disabled(true);
		parameters->edit(NULL);
		delete_preset->set_disabled(true);
		sections->hide();
		patches->clear();
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
	delete_preset->set_disabled(false);
	name->set_text(current->get_name());
	runnable->set_disabled(false);
	runnable->set_pressed(current->is_runnable());
	parameters->edit(current.ptr());

	export_filter->select(current->get_export_filter());
	include_filters->set_text(current->get_include_filter());
	exclude_filters->set_text(current->get_exclude_filter());

	patches->clear();
	TreeItem *patch_root = patches->create_item();
	Vector<String> patchlist = current->get_patches();
	for (int i = 0; i < patchlist.size(); i++) {
		TreeItem *patch = patches->create_item(patch_root);
		patch->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		String file = patchlist[i].get_file();
		patch->set_editable(0, true);
		patch->set_text(0, file.get_file().replace("*", ""));
		if (file.ends_with("*"))
			patch->set_checked(0, true);
		patch->set_tooltip(0, patchlist[i]);
		patch->set_metadata(0, i);
		patch->add_button(0, get_icon("Remove", "EditorIcons"), 0);
		patch->add_button(0, get_icon("folder", "FileDialog"), 1);
	}

	TreeItem *patch_add = patches->create_item(patch_root);
	patch_add->set_metadata(0, patchlist.size());
	if (patchlist.size() == 0)
		patch_add->set_text(0, "Add initial export..");
	else
		patch_add->set_text(0, "Add previous patches..");

	patch_add->add_button(0, get_icon("folder", "FileDialog"), 1);

	_fill_resource_tree();

	bool needs_templates;
	String error;
	if (!current->get_platform()->can_export(current, error, needs_templates)) {

		if (error != String()) {

			Vector<String> items = error.split("\n");
			error = "";
			for (int i = 0; i < items.size(); i++) {
				if (i > 0)
					error += "\n";
				error += " - " + items[i];
			}

			export_error->set_text(error);
			export_error->show();
		} else {
			export_error->hide();
		}
		if (needs_templates)
			export_templates_error->show();
		else
			export_templates_error->hide();

		export_button->set_disabled(true);

	} else {
		export_error->hide();
		export_templates_error->hide();
		export_button->set_disabled(false);
	}

	custom_features->set_text(current->get_custom_features());
	_update_feature_list();

	updating = false;
}

void ProjectExportDialog::_update_feature_list() {

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());

	Set<String> fset;
	List<String> features;

	current->get_platform()->get_platform_features(&features);
	current->get_platform()->get_preset_features(current, &features);

	String custom = current->get_custom_features();
	Vector<String> custom_list = custom.split(",");
	for (int i = 0; i < custom_list.size(); i++) {
		String f = custom_list[i].strip_edges();
		if (f != String()) {
			features.push_back(f);
		}
	}

	for (List<String>::Element *E = features.front(); E; E = E->next()) {
		fset.insert(E->get());
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

	if (updating)
		return;

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());

	current->set_custom_features(p_text);
	_update_feature_list();
}

void ProjectExportDialog::_tab_changed(int) {
	_update_feature_list();
}

void ProjectExportDialog::_patch_button_pressed(Object *p_item, int p_column, int p_id) {

	TreeItem *ti = (TreeItem *)p_item;

	patch_index = ti->get_metadata(0);

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());

	if (p_id == 0) {
		Vector<String> patches = current->get_patches();
		ERR_FAIL_INDEX(patch_index, patches.size());
		patch_erase->set_text(vformat(TTR("Delete patch '" + patches[patch_index].get_file() + "' from list?")));
		patch_erase->popup_centered_minsize();
	} else {
		patch_dialog->popup_centered_ratio();
	}
}

void ProjectExportDialog::_patch_edited() {

	TreeItem *item = patches->get_edited();
	if (!item)
		return;
	int index = item->get_metadata(0);

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());

	Vector<String> patches = current->get_patches();

	ERR_FAIL_INDEX(index, patches.size());

	String patch = patches[index].replace("*", "");

	if (item->is_checked(0)) {
		patch += "*";
	}

	current->set_patch(index, patch);
}

void ProjectExportDialog::_patch_selected(const String &p_path) {

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());

	Vector<String> patches = current->get_patches();

	if (patch_index >= patches.size()) {

		current->add_patch(ProjectSettings::get_singleton()->get_resource_path().path_to(p_path) + "*");
	} else {
		String enabled = patches[patch_index].ends_with("*") ? String("*") : String();
		current->set_patch(patch_index, ProjectSettings::get_singleton()->get_resource_path().path_to(p_path) + enabled);
	}

	_edit_preset(presets->get_current());
}

void ProjectExportDialog::_patch_deleted() {

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());

	Vector<String> patches = current->get_patches();
	if (patch_index < patches.size()) {

		current->remove_patch(patch_index);
		_edit_preset(presets->get_current());
	}
}

void ProjectExportDialog::_update_parameters(const String &p_edited_property) {

	_edit_preset(presets->get_current());
	parameters->update_tree();
}

void ProjectExportDialog::_runnable_pressed() {

	if (updating)
		return;

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
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

	if (updating)
		return;

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());

	current->set_name(p_string);
	_update_presets();
}

void ProjectExportDialog::_delete_preset() {

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	if (current.is_null())
		return;

	delete_confirm->set_text(vformat(TTR("Delete preset '%s'?"), current->get_name()));
	delete_confirm->popup_centered_minsize();
}

void ProjectExportDialog::_delete_preset_confirm() {

	int idx = presets->get_current();
	parameters->edit(NULL); //to avoid crash
	_edit_preset(-1);
	EditorExport::get_singleton()->remove_export_preset(idx);
	_update_presets();
	_edit_preset(presets->get_current());
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

			set_drag_preview(drag);

			return d;
		}
	} else if (p_from == patches) {

		TreeItem *item = patches->get_item_at_position(p_point);

		if (item && item->get_cell_mode(0) == TreeItem::CELL_MODE_CHECK) {

			int metadata = item->get_metadata(0);
			Dictionary d;
			d["type"] = "export_patch";
			d["patch"] = metadata;

			Label *label = memnew(Label);
			label->set_text(item->get_text(0));
			set_drag_preview(label);

			return d;
		}
	}

	return Variant();
}

bool ProjectExportDialog::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {

	if (p_from == presets) {
		Dictionary d = p_data;
		if (!d.has("type") || String(d["type"]) != "export_preset")
			return false;

		if (presets->get_item_at_position(p_point, true) < 0 && !presets->is_pos_at_end_of_items(p_point))
			return false;
	} else if (p_from == patches) {

		Dictionary d = p_data;
		if (!d.has("type") || String(d["type"]) != "export_patch")
			return false;

		patches->set_drop_mode_flags(Tree::DROP_MODE_ON_ITEM);

		TreeItem *item = patches->get_item_at_position(p_point);

		if (!item) {

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

		if (to_pos == -1 && !presets->is_pos_at_end_of_items(p_point))
			return;

		if (to_pos == from_pos)
			return;
		else if (to_pos > from_pos) {
			to_pos--;
		}

		Ref<EditorExportPreset> preset = EditorExport::get_singleton()->get_export_preset(from_pos);
		EditorExport::get_singleton()->remove_export_preset(from_pos);
		EditorExport::get_singleton()->add_export_preset(preset, to_pos);

		_update_presets();
		if (to_pos >= 0)
			_edit_preset(to_pos);
		else
			_edit_preset(presets->get_item_count() - 1);
	} else if (p_from == patches) {

		Dictionary d = p_data;
		if (!d.has("type") || String(d["type"]) != "export_patch")
			return;

		int from_pos = d["patch"];

		TreeItem *item = patches->get_item_at_position(p_point);
		if (!item)
			return;

		int to_pos = item->get_cell_mode(0) == TreeItem::CELL_MODE_CHECK ? int(item->get_metadata(0)) : -1;

		if (to_pos == from_pos)
			return;
		else if (to_pos > from_pos) {
			to_pos--;
		}

		Ref<EditorExportPreset> preset = EditorExport::get_singleton()->get_export_preset(presets->get_current());
		String patch = preset->get_patch(from_pos);
		preset->remove_patch(from_pos);
		preset->add_patch(patch, to_pos);

		_edit_preset(presets->get_current());
	}
}

void ProjectExportDialog::_export_type_changed(int p_which) {

	if (updating)
		return;

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	if (current.is_null())
		return;

	current->set_export_filter(EditorExportPreset::ExportFilter(p_which));
	updating = true;
	_fill_resource_tree();
	updating = false;
}

void ProjectExportDialog::_filter_changed(const String &p_filter) {

	if (updating)
		return;

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	if (current.is_null())
		return;

	current->set_include_filter(include_filters->get_text());
	current->set_exclude_filter(exclude_filters->get_text());
}

void ProjectExportDialog::_fill_resource_tree() {

	include_files->clear();
	include_label->hide();
	include_margin->hide();

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	if (current.is_null())
		return;

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

	p_item->set_icon(0, get_icon("folder", "FileDialog"));
	p_item->set_text(0, p_dir->get_name() + "/");

	bool used = false;
	for (int i = 0; i < p_dir->get_subdir_count(); i++) {

		TreeItem *subdir = include_files->create_item(p_item);
		if (_fill_tree(p_dir->get_subdir(i), subdir, current, p_only_scenes) == false) {
			memdelete(subdir);
		} else {
			used = true;
		}
	}

	for (int i = 0; i < p_dir->get_file_count(); i++) {

		String type = p_dir->get_file_type(i);
		if (p_only_scenes && type != "PackedScene")
			continue;

		TreeItem *file = include_files->create_item(p_item);
		file->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
		file->set_text(0, p_dir->get_file(i));

		Ref<Texture> tex;
		if (has_icon(type, editor_icons)) {
			tex = get_icon(type, editor_icons);
		} else {
			tex = get_icon("Object", editor_icons);
		}

		String path = p_dir->get_file_path(i);

		file->set_icon(0, tex);
		file->set_editable(0, true);
		file->set_checked(0, current->has_export_file(path));
		file->set_metadata(0, path);

		used = true;
	}

	return used;
}

void ProjectExportDialog::_tree_changed() {

	if (updating)
		return;

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	if (current.is_null())
		return;

	TreeItem *item = include_files->get_edited();
	if (!item)
		return;

	String path = item->get_metadata(0);
	bool added = item->is_checked(0);

	if (added) {
		current->add_export_file(path);
	} else {
		current->remove_export_file(path);
	}
}

void ProjectExportDialog::_export_pck_zip() {

	export_pck_zip->popup_centered_ratio();
}

void ProjectExportDialog::_export_pck_zip_selected(const String &p_path) {

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());
	Ref<EditorExportPlatform> platform = current->get_platform();
	ERR_FAIL_COND(platform.is_null());

	if (p_path.ends_with(".zip")) {
		platform->save_zip(current, p_path);
	} else if (p_path.ends_with(".pck")) {
		platform->save_pack(current, p_path);
	}
}

void ProjectExportDialog::_open_export_template_manager() {

	EditorNode::get_singleton()->open_export_template_manager();
	hide();
}

void ProjectExportDialog::_export_project() {

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());
	Ref<EditorExportPlatform> platform = current->get_platform();
	ERR_FAIL_COND(platform.is_null());

	export_project->set_access(FileDialog::ACCESS_FILESYSTEM);
	export_project->clear_filters();
	String extension = platform->get_binary_extension();
	if (extension != String()) {
		export_project->add_filter("*." + extension + " ; " + platform->get_name() + " Export");
	}

	export_project->popup_centered_ratio();
}

void ProjectExportDialog::_export_project_to_path(const String &p_path) {

	Ref<EditorExportPreset> current = EditorExport::get_singleton()->get_export_preset(presets->get_current());
	ERR_FAIL_COND(current.is_null());
	Ref<EditorExportPlatform> platform = current->get_platform();
	ERR_FAIL_COND(platform.is_null());

	Error err = platform->export_project(current, export_debug->is_pressed(), p_path, 0);
	if (err != OK) {
		error_dialog->set_text(TTR("Export templates for this platform are missing/corrupted: ") + platform->get_name());
		error_dialog->show();
		error_dialog->popup_centered_minsize(Size2(300, 80));
		ERR_PRINT("Failed to export project");
	}
}

void ProjectExportDialog::_bind_methods() {

	ClassDB::bind_method("_add_preset", &ProjectExportDialog::_add_preset);
	ClassDB::bind_method("_edit_preset", &ProjectExportDialog::_edit_preset);
	ClassDB::bind_method("_update_parameters", &ProjectExportDialog::_update_parameters);
	ClassDB::bind_method("_runnable_pressed", &ProjectExportDialog::_runnable_pressed);
	ClassDB::bind_method("_name_changed", &ProjectExportDialog::_name_changed);
	ClassDB::bind_method("_delete_preset", &ProjectExportDialog::_delete_preset);
	ClassDB::bind_method("_delete_preset_confirm", &ProjectExportDialog::_delete_preset_confirm);
	ClassDB::bind_method("get_drag_data_fw", &ProjectExportDialog::get_drag_data_fw);
	ClassDB::bind_method("can_drop_data_fw", &ProjectExportDialog::can_drop_data_fw);
	ClassDB::bind_method("drop_data_fw", &ProjectExportDialog::drop_data_fw);
	ClassDB::bind_method("_export_type_changed", &ProjectExportDialog::_export_type_changed);
	ClassDB::bind_method("_filter_changed", &ProjectExportDialog::_filter_changed);
	ClassDB::bind_method("_tree_changed", &ProjectExportDialog::_tree_changed);
	ClassDB::bind_method("_patch_button_pressed", &ProjectExportDialog::_patch_button_pressed);
	ClassDB::bind_method("_patch_selected", &ProjectExportDialog::_patch_selected);
	ClassDB::bind_method("_patch_deleted", &ProjectExportDialog::_patch_deleted);
	ClassDB::bind_method("_patch_edited", &ProjectExportDialog::_patch_edited);
	ClassDB::bind_method("_export_pck_zip", &ProjectExportDialog::_export_pck_zip);
	ClassDB::bind_method("_export_pck_zip_selected", &ProjectExportDialog::_export_pck_zip_selected);
	ClassDB::bind_method("_open_export_template_manager", &ProjectExportDialog::_open_export_template_manager);
	ClassDB::bind_method("_export_project", &ProjectExportDialog::_export_project);
	ClassDB::bind_method("_export_project_to_path", &ProjectExportDialog::_export_project_to_path);
	ClassDB::bind_method("_custom_features_changed", &ProjectExportDialog::_custom_features_changed);
	ClassDB::bind_method("_tab_changed", &ProjectExportDialog::_tab_changed);
}
ProjectExportDialog::ProjectExportDialog() {

	set_title(TTR("Export"));
	set_resizable(true);

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);
	HBoxContainer *hbox = memnew(HBoxContainer);
	main_vb->add_child(hbox);
	hbox->set_v_size_flags(SIZE_EXPAND_FILL);

	VBoxContainer *preset_vb = memnew(VBoxContainer);
	preset_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbox->add_child(preset_vb);

	HBoxContainer *preset_hb = memnew(HBoxContainer);
	preset_hb->add_child(memnew(Label(TTR("Presets"))));
	preset_hb->add_spacer();
	preset_vb->add_child(preset_hb);

	add_preset = memnew(MenuButton);
	add_preset->set_text(TTR("Add.."));
	add_preset->get_popup()->connect("index_pressed", this, "_add_preset");
	preset_hb->add_child(add_preset);
	MarginContainer *mc = memnew(MarginContainer);
	preset_vb->add_child(mc);
	mc->set_v_size_flags(SIZE_EXPAND_FILL);
	presets = memnew(ItemList);
	presets->set_drag_forwarding(this);
	mc->add_child(presets);
	presets->connect("item_selected", this, "_edit_preset");
	delete_preset = memnew(ToolButton);
	preset_hb->add_child(delete_preset);
	delete_preset->connect("pressed", this, "_delete_preset");

	VBoxContainer *settings_vb = memnew(VBoxContainer);
	settings_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	hbox->add_child(settings_vb);

	name = memnew(LineEdit);
	settings_vb->add_margin_child(TTR("Name:"), name);
	name->connect("text_changed", this, "_name_changed");
	runnable = memnew(CheckButton);
	runnable->set_text(TTR("Runnable"));
	runnable->connect("pressed", this, "_runnable_pressed");
	settings_vb->add_child(runnable);

	sections = memnew(TabContainer);
	sections->set_tab_align(TabContainer::ALIGN_LEFT);
	settings_vb->add_child(sections);
	sections->set_v_size_flags(SIZE_EXPAND_FILL);

	parameters = memnew(PropertyEditor);
	sections->add_child(parameters);
	parameters->set_name(TTR("Options"));
	parameters->hide_top_label();
	parameters->set_v_size_flags(SIZE_EXPAND_FILL);

	parameters->connect("property_edited", this, "_update_parameters");

	VBoxContainer *resources_vb = memnew(VBoxContainer);
	sections->add_child(resources_vb);
	resources_vb->set_name(TTR("Resources"));

	export_filter = memnew(OptionButton);
	export_filter->add_item(TTR("Export all resources in the project"));
	export_filter->add_item(TTR("Export selected scenes (and dependencies)"));
	export_filter->add_item(TTR("Export selected resources (and dependencies)"));
	resources_vb->add_margin_child(TTR("Export Mode:"), export_filter);
	export_filter->connect("item_selected", this, "_export_type_changed");

	include_label = memnew(Label);
	include_label->set_text(TTR("Resources to export:"));
	resources_vb->add_child(include_label);
	include_margin = memnew(MarginContainer);
	include_margin->set_v_size_flags(SIZE_EXPAND_FILL);
	resources_vb->add_child(include_margin);

	include_files = memnew(Tree);
	include_margin->add_child(include_files);
	include_files->connect("item_edited", this, "_tree_changed");

	include_filters = memnew(LineEdit);
	resources_vb->add_margin_child(TTR("Filters to export non-resource files (comma separated, e.g: *.json, *.txt)"), include_filters);
	include_filters->connect("text_changed", this, "_filter_changed");

	exclude_filters = memnew(LineEdit);
	resources_vb->add_margin_child(TTR("Filters to exclude files from project (comma separated, e.g: *.json, *.txt)"), exclude_filters);
	exclude_filters->connect("text_changed", this, "_filter_changed");

	VBoxContainer *patch_vb = memnew(VBoxContainer);
	sections->add_child(patch_vb);
	patch_vb->set_name(TTR("Patches"));

	patches = memnew(Tree);
	patch_vb->add_child(patches);
	patches->set_v_size_flags(SIZE_EXPAND_FILL);
	patches->set_hide_root(true);
	patches->connect("button_pressed", this, "_patch_button_pressed");
	patches->connect("item_edited", this, "_patch_edited");
	patches->set_drag_forwarding(this);
	patches->set_edit_checkbox_cell_only_when_checkbox_is_pressed(true);

	HBoxContainer *patches_hb = memnew(HBoxContainer);
	patch_vb->add_child(patches_hb);
	patches_hb->add_spacer();
	patch_export = memnew(Button);
	patch_export->set_text(TTR("Make Patch"));
	patches_hb->add_child(patch_export);
	patches_hb->add_spacer();

	patch_dialog = memnew(FileDialog);
	patch_dialog->add_filter("*.pck ; Pack File");
	patch_dialog->set_mode(FileDialog::MODE_OPEN_FILE);
	patch_dialog->connect("file_selected", this, "_patch_selected");
	add_child(patch_dialog);

	patch_erase = memnew(ConfirmationDialog);
	patch_erase->get_ok()->set_text(TTR("Delete"));
	patch_erase->connect("confirmed", this, "_patch_deleted");
	add_child(patch_erase);

	VBoxContainer *feature_vb = memnew(VBoxContainer);
	feature_vb->set_name(TTR("Features"));
	custom_features = memnew(LineEdit);
	custom_features->connect("text_changed", this, "_custom_features_changed");
	feature_vb->add_margin_child(TTR("Custom (comma-separated):"), custom_features);
	Panel *features_panel = memnew(Panel);
	custom_feature_display = memnew(RichTextLabel);
	features_panel->add_child(custom_feature_display);
	custom_feature_display->set_anchors_and_margins_preset(Control::PRESET_WIDE, Control::PRESET_MODE_MINSIZE, 10 * EDSCALE);
	custom_feature_display->set_v_size_flags(SIZE_EXPAND_FILL);
	feature_vb->add_margin_child(TTR("Feature List:"), features_panel, true);
	sections->add_child(feature_vb);

	sections->connect("tab_changed", this, "_tab_changed");

	//disable by default
	name->set_editable(false);
	runnable->set_disabled(true);
	delete_preset->set_disabled(true);
	sections->hide();
	parameters->edit(NULL);

	delete_confirm = memnew(ConfirmationDialog);
	add_child(delete_confirm);
	delete_confirm->get_ok()->set_text(TTR("Delete"));
	delete_confirm->connect("confirmed", this, "_delete_preset_confirm");

	updating = false;

	get_cancel()->set_text(TTR("Close"));
	get_ok()->set_text(TTR("Export PCK/Zip"));
	export_button = add_button(TTR("Export Project"), !OS::get_singleton()->get_swap_ok_cancel(), "export");

	export_pck_zip = memnew(FileDialog);
	export_pck_zip->add_filter("*.zip ; ZIP File");
	export_pck_zip->add_filter("*.pck ; Godot Game Pack");
	export_pck_zip->set_access(FileDialog::ACCESS_FILESYSTEM);
	export_pck_zip->set_mode(FileDialog::MODE_SAVE_FILE);
	add_child(export_pck_zip);
	export_pck_zip->connect("file_selected", this, "_export_pck_zip_selected");

	export_error = memnew(Label);
	main_vb->add_child(export_error);
	export_error->hide();
	export_error->add_color_override("font_color", get_color("error_color", "Editor"));

	export_templates_error = memnew(HBoxContainer);
	main_vb->add_child(export_templates_error);
	export_templates_error->hide();

	Label *export_error2 = memnew(Label);
	export_templates_error->add_child(export_error2);
	export_error2->add_color_override("font_color", get_color("error_color", "Editor"));
	export_error2->set_text(" - " + TTR("Export templates for this platform are missing:") + " ");

	error_dialog = memnew(AcceptDialog);
	error_dialog->set_title("Error");
	error_dialog->set_text(TTR("Export templates for this platform are missing/corrupted:") + " ");
	main_vb->add_child(error_dialog);
	error_dialog->hide();

	LinkButton *download_templates = memnew(LinkButton);
	download_templates->set_text(TTR("Manage Export Templates"));
	export_templates_error->add_child(download_templates);
	download_templates->connect("pressed", this, "_open_export_template_manager");

	export_project = memnew(FileDialog);
	export_project->set_access(FileDialog::ACCESS_FILESYSTEM);
	add_child(export_project);
	export_project->connect("file_selected", this, "_export_project_to_path");
	export_button->connect("pressed", this, "_export_project");

	export_debug = memnew(CheckButton);
	export_debug->set_text(TTR("Export With Debug"));
	export_debug->set_pressed(true);
	export_project->get_vbox()->add_child(export_debug);

	set_hide_on_ok(false);

	editor_icons = "EditorIcons";
}

ProjectExportDialog::~ProjectExportDialog() {
}
