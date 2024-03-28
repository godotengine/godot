/*************************************************************************/
/*  native_file_dialog.h                                                 */
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

#ifndef NATIVE_FILE_DIALOG_H
#define NATIVE_FILE_DIALOG_H

#include "scene/main/node.h"
#include "thirdparty/portable-file-dialogs/portable-file-dialogs.h"

class NativeFileDialog : public Node {
	GDCLASS(NativeFileDialog, Node);

public:
	enum NativeFileMode {
		NATIVE_FILE_MODE_OPEN_FILE,
		NATIVE_FILE_MODE_OPEN_FILES,
		NATIVE_FILE_MODE_OPEN_DIR,
		NATIVE_FILE_MODE_SAVE_FILE,
	};

private:
	NativeFileMode assigned_mode = NativeFileMode::NATIVE_FILE_MODE_OPEN_FILE;
	NativeFileMode active_mode = NativeFileMode::NATIVE_FILE_MODE_OPEN_FILE;

	bool waiting = false;
	bool supported = true;
	bool sent_signal = false;

	String title = "";
	String start_directory = "";

	Vector<String> stored_results = Vector<String>();
	Vector<String> filters;

	// PFD uses classes instead of normal function calls so need separate members
	union {
		pfd::open_file *open_file_dialog = nullptr;
		pfd::select_folder *open_dir_dialog;
		pfd::save_file *save_file_dialog;
	};

	inline std::string _std_string(const String &gd_string);
	inline std::string _std_string_with_fallback(const String &gd_string, const String &fallback);

	void _build_std_filters();
	void _update_std_filters_if_necessary();

	bool _has_results();
	void _fetch_results();
	void _emit_signals_if_necessary();

	inline String _get_first_result();

	bool std_filters_built = false;
	std::vector<std::string> std_filters;

	static void _bind_methods();

protected:
	void _notification(int p_what);

public:
	static bool is_supported() { return pfd::settings::available(); }

	void clear_filters();
	void add_filter(const String &p_filter);
	void set_filters(const Vector<String> &p_filters);
	Vector<String> get_filters() const { return filters; };

	void set_file_mode(NativeFileMode p_mode) { assigned_mode = p_mode; }
	NativeFileMode get_file_mode() const { return assigned_mode; }

	void set_start_directory(String p_start_directory) { start_directory = p_start_directory; }
	String get_start_directory() { return start_directory; }

	void set_title(String p_title) { title = p_title; }
	String get_title() { return title; }

	bool has_results();
	String get_result();
	Vector<String> get_results();

	void show();
	void hide();

	NativeFileDialog();
	~NativeFileDialog();
};

VARIANT_ENUM_CAST(NativeFileDialog::NativeFileMode);

#endif // NATIVE_FILE_DIALOG_H
