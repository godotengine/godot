/**************************************************************************/
/*  editor_extension_source_code_plugin.h                                 */
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

#pragma once

#include "core/error/error_list.h"
#include "core/object/gdvirtual.gen.inc"
#include "core/object/ref_counted.h"
#include "core/object/script_language.h"
#include "core/string/ustring.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/validation_context.h"
#include "scene/resources/texture.h"

class EditorExtensionSourceCodePlugin : public RefCounted {
	GDCLASS(EditorExtensionSourceCodePlugin, RefCounted);

protected:
	static void _bind_methods();

	GDVIRTUAL1RC_REQUIRED(bool, _can_handle_object, const Object *);

	GDVIRTUAL1RC(String, _get_source_path, StringName);
	GDVIRTUAL1RC(StringName, _get_class_name_from_source_path, String);

	GDVIRTUAL0RC(bool, _overrides_external_editor);
	GDVIRTUAL3RC(Error, _open_in_external_editor, String, int, int);

	GDVIRTUAL0RC_REQUIRED(String, _get_language_name);
	GDVIRTUAL0RC(Ref<Texture2D>, _get_language_icon);

	GDVIRTUAL0RC(int, _get_path_count);
	GDVIRTUAL1RC(String, _get_path_label, int);
	GDVIRTUAL2C(_configure_select_path_dialog, int, EditorFileDialog *);
	GDVIRTUAL4RC(String, _adjust_path, int, String, String, String);

	GDVIRTUAL0RC(bool, _is_using_templates);
	GDVIRTUAL1RC(PackedStringArray, _get_available_templates, String);
	GDVIRTUAL1RC(String, _get_template_display_name, String);
	GDVIRTUAL1RC(String, _get_template_description, String);
	GDVIRTUAL1RC(TypedArray<Dictionary>, _get_template_options, String);

	GDVIRTUAL0RC(bool, _can_use_template_files);
	GDVIRTUAL1RC(bool, _can_handle_template_file, String);
	GDVIRTUAL1RC(String, _get_template_file_display_name, String);
	GDVIRTUAL1RC(String, _get_template_file_description, String);

	GDVIRTUAL0RC(bool, _can_create_class_source);
	GDVIRTUAL3RC(Error, _create_class_source, String, String, PackedStringArray);
	GDVIRTUAL5RC(Error, _create_class_source_from_template_id, String, String, PackedStringArray, String, Dictionary);
	GDVIRTUAL4RC(Error, _create_class_source_from_template_file, String, String, PackedStringArray, String);

	GDVIRTUAL2C(_validate_class_name, ValidationContext *, String);
	GDVIRTUAL3C(_validate_path, ValidationContext *, int, String);
	GDVIRTUAL4C(_validate_template_option, ValidationContext *, String, String, Variant);

	GDVIRTUAL0RC(bool, _can_edit_class_source);
	GDVIRTUAL3RC(Error, _add_method_func, StringName, String, PackedStringArray);

public:
	virtual bool can_handle_object(const Object *p_object) const;

	virtual String get_source_path(const StringName &p_class_name) const;
	virtual StringName get_class_name_from_source_path(const String &p_source_path) const;

	virtual bool overrides_external_editor() const;
	virtual Error open_in_external_editor(const String &p_source_path, int p_line, int p_col) const;

	virtual String get_language_name() const;
	virtual Ref<Texture2D> get_language_icon() const;

	virtual int get_path_count() const;
	virtual String get_path_label(int p_path_index) const;
	virtual void configure_select_path_dialog(int p_path_index, EditorFileDialog *p_dialog) const;
	virtual String adjust_path(int p_path_index, const String &p_class_name, const String &p_base_path, const String &p_old_path) const;
	static String adjust_script_name_casing(const String &p_script_name, ScriptLanguage::ScriptNameCasing p_auto_casing);

	virtual bool is_using_templates() const;
	virtual PackedStringArray get_available_templates(const String &p_base_class_name) const;
	virtual String get_template_display_name(const String &p_template_id) const;
	virtual String get_template_description(const String &p_template_id) const;
	virtual TypedArray<Dictionary> get_template_options(const String &p_template_id) const;

	virtual bool can_use_template_files() const;
	virtual bool can_handle_template_file(const String &p_template_path);
	virtual String get_template_file_display_name(const String &p_template_path);
	virtual String get_template_file_description(const String &p_template_path);

	virtual bool can_create_class_source() const;
	virtual Error create_class_source(const String &p_class_name, const String &p_base_class_name, const PackedStringArray &p_paths) const;
	virtual Error create_class_source_from_template_id(const String &p_class_name, const String &p_base_class_name, const PackedStringArray &p_paths, const String &p_template_id, const Dictionary &p_template_options) const;
	virtual Error create_class_source_from_template_file(const String &p_class_name, const String &p_base_class_name, const PackedStringArray &p_paths, const String &p_template_path);

	virtual void validate_class_name(ValidationContext *p_validation_context, const String &p_class_name) const;
	virtual void validate_path(ValidationContext *p_validation_context, int p_path_index, const String &p_path) const;
	virtual void validate_template_option(ValidationContext *p_validation_context, const String &p_template_id, const String &p_option_name, const Variant &p_value) const;

	virtual bool can_edit_class_source() const;
	virtual Error add_method_func(const StringName &p_class_name, const String &p_method_name, const PackedStringArray &p_args) const;
};
