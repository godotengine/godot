/**************************************************************************/
/*  editor_extension_source_code_plugin.cpp                               */
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

#include "editor_extension_source_code_plugin.h"

#include "editor/editor_node.h"

bool EditorExtensionSourceCodePlugin::can_handle_object(const Object *p_object) const {
	bool can_handle = false;
	GDVIRTUAL_CALL(_can_handle_object, p_object, can_handle);
	return can_handle;
}

bool EditorExtensionSourceCodePlugin::overrides_external_editor() const {
	bool overrides = false;
	GDVIRTUAL_CALL(_overrides_external_editor, overrides);
	return overrides;
}

String EditorExtensionSourceCodePlugin::get_source_path(const StringName &p_class_name) const {
	String path;
	GDVIRTUAL_CALL(_get_source_path, p_class_name, path);
	return path;
}

StringName EditorExtensionSourceCodePlugin::get_class_name_from_source_path(const String &p_source_path) const {
	StringName class_name;
	GDVIRTUAL_CALL(_get_class_name_from_source_path, p_source_path, class_name);
	return class_name;
}

Error EditorExtensionSourceCodePlugin::open_in_external_editor(const String &p_source_path, int p_line, int p_col) const {
	Error err = ERR_BUG;
	GDVIRTUAL_CALL(_open_in_external_editor, p_source_path, p_line, p_col, err);
	return err;
}

String EditorExtensionSourceCodePlugin::get_language_name() const {
	String language_name;
	GDVIRTUAL_CALL(_get_language_name, language_name);
	return language_name;
}

Ref<Texture2D> EditorExtensionSourceCodePlugin::get_language_icon() const {
	Ref<Texture2D> icon;
	GDVIRTUAL_CALL(_get_language_icon, icon);
	return icon;
}

int EditorExtensionSourceCodePlugin::get_path_count() const {
	int path_count = 1;
	GDVIRTUAL_CALL(_get_path_count, path_count);
	ERR_FAIL_COND_V_MSG(path_count < 1, 1, "Must have at least one path.");
	return path_count;
}

String EditorExtensionSourceCodePlugin::get_path_label(int p_path_index) const {
	String label;
	GDVIRTUAL_CALL(_get_path_label, p_path_index, label);
	return label;
}

void EditorExtensionSourceCodePlugin::configure_select_path_dialog(int p_path_index, EditorFileDialog *p_dialog) const {
	GDVIRTUAL_CALL(_configure_select_path_dialog, p_path_index, p_dialog);
}

String EditorExtensionSourceCodePlugin::adjust_path(int p_path_index, const String &p_class_name, const String &p_base_path, const String &p_old_path) const {
	String new_path;
	if (GDVIRTUAL_CALL(_adjust_path, p_path_index, p_class_name, p_base_path, p_old_path, new_path)) {
		return new_path;
	}
	return p_old_path;
}

String EditorExtensionSourceCodePlugin::adjust_script_name_casing(const String &p_script_name, ScriptLanguage::ScriptNameCasing p_auto_casing) {
	return EditorNode::adjust_script_name_casing(p_script_name, p_auto_casing);
}

bool EditorExtensionSourceCodePlugin::is_using_templates() const {
	bool using_templates = false;
	GDVIRTUAL_CALL(_is_using_templates, using_templates);
	return using_templates;
}

PackedStringArray EditorExtensionSourceCodePlugin::get_available_templates(const String &p_base_class_name) const {
	PackedStringArray templates;
	GDVIRTUAL_CALL(_get_available_templates, p_base_class_name, templates);
	return templates;
}

String EditorExtensionSourceCodePlugin::get_template_display_name(const String &p_template_id) const {
	String display_name;
	GDVIRTUAL_CALL(_get_template_display_name, p_template_id, display_name);
	return display_name;
}

String EditorExtensionSourceCodePlugin::get_template_description(const String &p_template_id) const {
	String description;
	GDVIRTUAL_CALL(_get_template_description, p_template_id, description);
	return description;
}

TypedArray<Dictionary> EditorExtensionSourceCodePlugin::get_template_options(const String &p_template_id) const {
	TypedArray<Dictionary> template_options;
	GDVIRTUAL_CALL(_get_template_options, p_template_id, template_options);
	return template_options;
}

bool EditorExtensionSourceCodePlugin::can_use_template_files() const {
	bool can_use = false;
	GDVIRTUAL_CALL(_can_use_template_files, can_use);
	return can_use;
}

bool EditorExtensionSourceCodePlugin::can_handle_template_file(const String &p_template_path) {
	bool can_handle = false;
	GDVIRTUAL_CALL(_can_handle_template_file, p_template_path, can_handle);
	return can_handle;
}

String EditorExtensionSourceCodePlugin::get_template_file_display_name(const String &p_template_id) {
	String display_name;
	GDVIRTUAL_CALL(_get_template_file_display_name, p_template_id, display_name);
	return display_name;
}

String EditorExtensionSourceCodePlugin::get_template_file_description(const String &p_template_id) {
	String description;
	GDVIRTUAL_CALL(_get_template_file_description, p_template_id, description);
	return description;
}

bool EditorExtensionSourceCodePlugin::can_create_class_source() const {
	bool can_create = false;
	GDVIRTUAL_CALL(_can_create_class_source, can_create);
	return can_create;
}

Error EditorExtensionSourceCodePlugin::create_class_source(const String &p_class_name, const String &p_base_class_name, const PackedStringArray &p_paths) const {
	Error err = ERR_UNAVAILABLE;
	GDVIRTUAL_CALL(_create_class_source, p_class_name, p_base_class_name, p_paths, err);
	return err;
}

Error EditorExtensionSourceCodePlugin::create_class_source_from_template_id(const String &p_class_name, const String &p_base_class_name, const PackedStringArray &p_paths, const String &p_template_id, const Dictionary &p_template_options) const {
	Error err = ERR_UNAVAILABLE;
	GDVIRTUAL_CALL(_create_class_source_from_template_id, p_class_name, p_base_class_name, p_paths, p_template_id, p_template_options, err);
	return err;
}

Error EditorExtensionSourceCodePlugin::create_class_source_from_template_file(const String &p_class_name, const String &p_base_class_name, const PackedStringArray &p_paths, const String &p_template_path) {
	Error err = ERR_UNAVAILABLE;
	GDVIRTUAL_CALL(_create_class_source_from_template_file, p_class_name, p_base_class_name, p_paths, p_template_path, err);
	return err;
}

void EditorExtensionSourceCodePlugin::validate_class_name(ValidationContext *p_validation_context, const String &p_class_name) const {
	GDVIRTUAL_CALL(_validate_class_name, p_validation_context, p_class_name);
}

void EditorExtensionSourceCodePlugin::validate_path(ValidationContext *p_validation_context, int p_path_index, const String &p_path) const {
	GDVIRTUAL_CALL(_validate_path, p_validation_context, p_path_index, p_path);
}

void EditorExtensionSourceCodePlugin::validate_template_option(ValidationContext *p_validation_context, const String &p_template_id, const String &p_option_name, const Variant &p_value) const {
	GDVIRTUAL_CALL(_validate_template_option, p_validation_context, p_template_id, p_option_name, p_value);
}

bool EditorExtensionSourceCodePlugin::can_edit_class_source() const {
	bool can_edit = false;
	GDVIRTUAL_CALL(_can_edit_class_source, can_edit);
	return can_edit;
}

Error EditorExtensionSourceCodePlugin::add_method_func(const StringName &p_class_name, const String &p_method_name, const PackedStringArray &p_args) const {
	Error err = ERR_UNAVAILABLE;
	GDVIRTUAL_CALL(_add_method_func, p_class_name, p_method_name, p_args, err);
	return err;
}

void EditorExtensionSourceCodePlugin::_bind_methods() {
	GDVIRTUAL_BIND(_can_handle_object, "object");

	GDVIRTUAL_BIND(_get_source_path, "class_name");
	GDVIRTUAL_BIND(_get_class_name_from_source_path, "source_path");

	GDVIRTUAL_BIND(_overrides_external_editor);
	GDVIRTUAL_BIND(_open_in_external_editor, "source_path", "line", "col");

	GDVIRTUAL_BIND(_get_language_name);
	GDVIRTUAL_BIND(_get_language_icon);

	GDVIRTUAL_BIND(_get_path_count);
	GDVIRTUAL_BIND(_get_path_label, "path_index");
	GDVIRTUAL_BIND(_configure_select_path_dialog, "path_index", "dialog");
	GDVIRTUAL_BIND(_adjust_path, "path_index", "class_name", "base_path", "old_path");
	ClassDB::bind_static_method("EditorExtensionSourceCodePlugin", D_METHOD("adjust_script_name_casing", "script_name", "auto_casing"), &EditorExtensionSourceCodePlugin::adjust_script_name_casing);

	GDVIRTUAL_BIND(_is_using_templates);
	GDVIRTUAL_BIND(_get_available_templates, "base_class_name");
	GDVIRTUAL_BIND(_get_template_display_name, "template_id");
	GDVIRTUAL_BIND(_get_template_description, "template_id");
	GDVIRTUAL_BIND(_get_template_options, "template_id");

	GDVIRTUAL_BIND(_can_use_template_files);
	GDVIRTUAL_BIND(_can_handle_template_file, "template_path");
	GDVIRTUAL_BIND(_get_template_file_display_name, "template_path");
	GDVIRTUAL_BIND(_get_template_file_description, "template_path");

	GDVIRTUAL_BIND(_can_create_class_source);
	GDVIRTUAL_BIND(_create_class_source, "class_name", "base_class_name", "paths");
	GDVIRTUAL_BIND(_create_class_source_from_template_id, "class_name", "base_class_name", "paths", "template_id", "template_options");
	GDVIRTUAL_BIND(_create_class_source_from_template_file, "class_name", "base_class_name", "paths", "template_path");

	GDVIRTUAL_BIND(_validate_class_name, "validation_context", "class_name");
	GDVIRTUAL_BIND(_validate_path, "validation_context", "path_index", "path");
	GDVIRTUAL_BIND(_validate_template_option, "validation_context", "template_id", "option_name", "value");

	GDVIRTUAL_BIND(_can_edit_class_source);
	GDVIRTUAL_BIND(_add_method_func, "class_name", "method_name", "args");
}
