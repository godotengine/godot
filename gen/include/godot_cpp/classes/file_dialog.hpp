/**************************************************************************/
/*  file_dialog.hpp                                                       */
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

#include <godot_cpp/classes/confirmation_dialog.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class LineEdit;
class VBoxContainer;

class FileDialog : public ConfirmationDialog {
	GDEXTENSION_CLASS(FileDialog, ConfirmationDialog)

public:
	enum FileMode {
		FILE_MODE_OPEN_FILE = 0,
		FILE_MODE_OPEN_FILES = 1,
		FILE_MODE_OPEN_DIR = 2,
		FILE_MODE_OPEN_ANY = 3,
		FILE_MODE_SAVE_FILE = 4,
	};

	enum Access {
		ACCESS_RESOURCES = 0,
		ACCESS_USERDATA = 1,
		ACCESS_FILESYSTEM = 2,
	};

	enum DisplayMode {
		DISPLAY_THUMBNAILS = 0,
		DISPLAY_LIST = 1,
	};

	enum Customization {
		CUSTOMIZATION_HIDDEN_FILES = 0,
		CUSTOMIZATION_CREATE_FOLDER = 1,
		CUSTOMIZATION_FILE_FILTER = 2,
		CUSTOMIZATION_FILE_SORT = 3,
		CUSTOMIZATION_FAVORITES = 4,
		CUSTOMIZATION_RECENT = 5,
		CUSTOMIZATION_LAYOUT = 6,
		CUSTOMIZATION_OVERWRITE_WARNING = 7,
		CUSTOMIZATION_DELETE = 8,
	};

	void clear_filters();
	void add_filter(const String &p_filter, const String &p_description = String(), const String &p_mime_type = String());
	void set_filters(const PackedStringArray &p_filters);
	PackedStringArray get_filters() const;
	void clear_filename_filter();
	void set_filename_filter(const String &p_filter);
	String get_filename_filter() const;
	String get_option_name(int32_t p_option) const;
	PackedStringArray get_option_values(int32_t p_option) const;
	int32_t get_option_default(int32_t p_option) const;
	void set_option_name(int32_t p_option, const String &p_name);
	void set_option_values(int32_t p_option, const PackedStringArray &p_values);
	void set_option_default(int32_t p_option, int32_t p_default_value_index);
	void set_option_count(int32_t p_count);
	int32_t get_option_count() const;
	void add_option(const String &p_name, const PackedStringArray &p_values, int32_t p_default_value_index);
	Dictionary get_selected_options() const;
	String get_current_dir() const;
	String get_current_file() const;
	String get_current_path() const;
	void set_current_dir(const String &p_dir);
	void set_current_file(const String &p_file);
	void set_current_path(const String &p_path);
	void set_mode_overrides_title(bool p_override);
	bool is_mode_overriding_title() const;
	void set_file_mode(FileDialog::FileMode p_mode);
	FileDialog::FileMode get_file_mode() const;
	void set_display_mode(FileDialog::DisplayMode p_mode);
	FileDialog::DisplayMode get_display_mode() const;
	VBoxContainer *get_vbox();
	LineEdit *get_line_edit();
	void set_access(FileDialog::Access p_access);
	FileDialog::Access get_access() const;
	void set_root_subfolder(const String &p_dir);
	String get_root_subfolder() const;
	void set_show_hidden_files(bool p_show);
	bool is_showing_hidden_files() const;
	void set_use_native_dialog(bool p_native);
	bool get_use_native_dialog() const;
	void set_customization_flag_enabled(FileDialog::Customization p_flag, bool p_enabled);
	bool is_customization_flag_enabled(FileDialog::Customization p_flag) const;
	void deselect_all();
	static void set_favorite_list(const PackedStringArray &p_favorites);
	static PackedStringArray get_favorite_list();
	static void set_recent_list(const PackedStringArray &p_recents);
	static PackedStringArray get_recent_list();
	static void set_get_icon_callback(const Callable &p_callback);
	static void set_get_thumbnail_callback(const Callable &p_callback);
	void popup_file_dialog();
	void invalidate();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		ConfirmationDialog::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(FileDialog::FileMode);
VARIANT_ENUM_CAST(FileDialog::Access);
VARIANT_ENUM_CAST(FileDialog::DisplayMode);
VARIANT_ENUM_CAST(FileDialog::Customization);

