/*************************************************************************/
/*  editor_export_preset.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "editor_export.h"

bool EditorExportPreset::_set(const StringName &p_name, const Variant &p_value) {
	if (values.has(p_name)) {
		values[p_name] = p_value;
		EditorExport::singleton->save_presets();
		return true;
	}

	return false;
}

bool EditorExportPreset::_get(const StringName &p_name, Variant &r_ret) const {
	if (values.has(p_name)) {
		r_ret = values[p_name];
		return true;
	}

	return false;
}

void EditorExportPreset::_get_property_list(List<PropertyInfo> *p_list) const {
	for (const PropertyInfo &E : properties) {
		if (platform->get_export_option_visibility(E.name, values)) {
			p_list->push_back(E);
		}
	}
}

Ref<EditorExportPlatform> EditorExportPreset::get_platform() const {
	return platform;
}

void EditorExportPreset::update_files_to_export() {
	Vector<String> to_remove;
	for (const String &E : selected_files) {
		if (!FileAccess::exists(E)) {
			to_remove.push_back(E);
		}
	}
	for (int i = 0; i < to_remove.size(); ++i) {
		selected_files.erase(to_remove[i]);
	}
}

Vector<String> EditorExportPreset::get_files_to_export() const {
	Vector<String> files;
	for (const String &E : selected_files) {
		files.push_back(E);
	}
	return files;
}

void EditorExportPreset::set_name(const String &p_name) {
	name = p_name;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_name() const {
	return name;
}

void EditorExportPreset::set_runnable(bool p_enable) {
	runnable = p_enable;
	EditorExport::singleton->save_presets();
}

bool EditorExportPreset::is_runnable() const {
	return runnable;
}

void EditorExportPreset::set_export_filter(ExportFilter p_filter) {
	export_filter = p_filter;
	EditorExport::singleton->save_presets();
}

EditorExportPreset::ExportFilter EditorExportPreset::get_export_filter() const {
	return export_filter;
}

void EditorExportPreset::set_include_filter(const String &p_include) {
	include_filter = p_include;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_include_filter() const {
	return include_filter;
}

void EditorExportPreset::set_export_path(const String &p_path) {
	export_path = p_path;
	/* NOTE(SonerSound): if there is a need to implement a PropertyHint that specifically indicates a relative path,
	 * this should be removed. */
	if (export_path.is_absolute_path()) {
		String res_path = OS::get_singleton()->get_resource_dir();
		export_path = res_path.path_to_file(export_path);
	}
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_export_path() const {
	return export_path;
}

void EditorExportPreset::set_exclude_filter(const String &p_exclude) {
	exclude_filter = p_exclude;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_exclude_filter() const {
	return exclude_filter;
}

void EditorExportPreset::add_export_file(const String &p_path) {
	selected_files.insert(p_path);
	EditorExport::singleton->save_presets();
}

void EditorExportPreset::remove_export_file(const String &p_path) {
	selected_files.erase(p_path);
	EditorExport::singleton->save_presets();
}

bool EditorExportPreset::has_export_file(const String &p_path) {
	return selected_files.has(p_path);
}

void EditorExportPreset::set_custom_features(const String &p_custom_features) {
	custom_features = p_custom_features;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_custom_features() const {
	return custom_features;
}

void EditorExportPreset::set_enc_in_filter(const String &p_filter) {
	enc_in_filters = p_filter;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_enc_in_filter() const {
	return enc_in_filters;
}

void EditorExportPreset::set_enc_ex_filter(const String &p_filter) {
	enc_ex_filters = p_filter;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_enc_ex_filter() const {
	return enc_ex_filters;
}

void EditorExportPreset::set_enc_pck(bool p_enabled) {
	enc_pck = p_enabled;
	EditorExport::singleton->save_presets();
}

bool EditorExportPreset::get_enc_pck() const {
	return enc_pck;
}

void EditorExportPreset::set_enc_directory(bool p_enabled) {
	enc_directory = p_enabled;
	EditorExport::singleton->save_presets();
}

bool EditorExportPreset::get_enc_directory() const {
	return enc_directory;
}

void EditorExportPreset::set_script_export_mode(int p_mode) {
	script_mode = p_mode;
	EditorExport::singleton->save_presets();
}

int EditorExportPreset::get_script_export_mode() const {
	return script_mode;
}

void EditorExportPreset::set_script_encryption_key(const String &p_key) {
	script_key = p_key;
	EditorExport::singleton->save_presets();
}

String EditorExportPreset::get_script_encryption_key() const {
	return script_key;
}

EditorExportPreset::EditorExportPreset() {}
