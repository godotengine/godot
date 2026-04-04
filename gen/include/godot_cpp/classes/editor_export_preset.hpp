/**************************************************************************/
/*  editor_export_preset.hpp                                              */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class StringName;

class EditorExportPreset : public RefCounted {
	GDEXTENSION_CLASS(EditorExportPreset, RefCounted)

public:
	enum ExportFilter {
		EXPORT_ALL_RESOURCES = 0,
		EXPORT_SELECTED_SCENES = 1,
		EXPORT_SELECTED_RESOURCES = 2,
		EXCLUDE_SELECTED_RESOURCES = 3,
		EXPORT_CUSTOMIZED = 4,
	};

	enum FileExportMode {
		MODE_FILE_NOT_CUSTOMIZED = 0,
		MODE_FILE_STRIP = 1,
		MODE_FILE_KEEP = 2,
		MODE_FILE_REMOVE = 3,
	};

	enum ScriptExportMode {
		MODE_SCRIPT_TEXT = 0,
		MODE_SCRIPT_BINARY_TOKENS = 1,
		MODE_SCRIPT_BINARY_TOKENS_COMPRESSED = 2,
	};

	bool has(const StringName &p_property) const;
	PackedStringArray get_files_to_export() const;
	Dictionary get_customized_files() const;
	int32_t get_customized_files_count() const;
	bool has_export_file(const String &p_path);
	EditorExportPreset::FileExportMode get_file_export_mode(const String &p_path, EditorExportPreset::FileExportMode p_default = (EditorExportPreset::FileExportMode)0) const;
	Variant get_project_setting(const StringName &p_name);
	String get_preset_name() const;
	bool is_runnable() const;
	bool are_advanced_options_enabled() const;
	bool is_dedicated_server() const;
	EditorExportPreset::ExportFilter get_export_filter() const;
	String get_include_filter() const;
	String get_exclude_filter() const;
	String get_custom_features() const;
	PackedStringArray get_patches() const;
	String get_export_path() const;
	String get_encryption_in_filter() const;
	String get_encryption_ex_filter() const;
	bool get_encrypt_pck() const;
	bool get_encrypt_directory() const;
	String get_encryption_key() const;
	EditorExportPreset::ScriptExportMode get_script_export_mode() const;
	Variant get_or_env(const StringName &p_name, const String &p_env_var) const;
	String get_version(const StringName &p_name, bool p_windows_version) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorExportPreset::ExportFilter);
VARIANT_ENUM_CAST(EditorExportPreset::FileExportMode);
VARIANT_ENUM_CAST(EditorExportPreset::ScriptExportMode);

