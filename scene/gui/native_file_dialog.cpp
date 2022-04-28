/*************************************************************************/
/*  native_file_dialog.cpp                                               */
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

#include "native_file_dialog.h"
#include "thirdparty/portable-file-dialogs/portable-file-dialogs.h"

inline std::string NativeFileDialog::_std_string(const String &gd_string) {
	return std::string(gd_string.utf8().get_data());
}

inline std::string NativeFileDialog::_std_string_with_fallback(const String &gd_string, const String &fallback) {
	if (gd_string.is_empty()) {
		return std::string(fallback.utf8().get_data());
	} else {
		return std::string(gd_string.utf8().get_data());
	}
}

void NativeFileDialog::_build_std_filters() {
	// pfd takes filters in the form of a paired std::vector, so we need to
	// convert prior to calling up the dialog
	std_filters.clear();

	if (filters.size() > 1) {
		std::string all_filters_desc;
		std::string all_filters_flt;
		const int max_desc_filters = 5;
		for (int i = 0; i < filters.size(); i++) {
			String flt = filters[i].get_slice(";", 0).strip_edges();
			std::string std_flt = std::string(flt.utf8().get_data());
			if (i > 0) {
				all_filters_flt += " ";
			}
			if (i < max_desc_filters) {
				if (i > 0) {
					all_filters_desc += ", ";
				}
				all_filters_desc += std_flt;
			}
			all_filters_flt += std_flt;
		}
		if (max_desc_filters < filters.size()) {
			all_filters_desc += ", ...";
		}
		all_filters_desc = _std_string(RTR("All Recognized")) + " (" + all_filters_desc + ")";
		std_filters.push_back(all_filters_desc);
		std_filters.push_back(all_filters_flt);
	}

	for (int i = 0; i < filters.size(); i++) {
		String flt = filters[i].get_slice(";", 0).strip_edges();
		String desc = filters[i].get_slice(";", 1).strip_edges();
		if (desc.length()) {
			std_filters.push_back(_std_string(tr(desc) + " (" + flt + ")"));
			std_filters.push_back(_std_string(flt.replace(",", " ")));
		} else {
			std_filters.push_back(_std_string(flt));
			std_filters.push_back(_std_string(flt.replace(",", " ")));
		}
	}
}

void NativeFileDialog::_update_std_filters_if_necessary() {
	if (!std_filters_built) {
		std_filters_built = true;
		_build_std_filters();
	}
}

bool NativeFileDialog::_has_results() {
	if (!supported || !waiting) {
		return false;
	}
	switch (active_mode) {
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILE:
			[[fallthrough]];
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILES: {
			return open_file_dialog->ready();
		}
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_DIR: {
			return open_dir_dialog->ready();
		}
		case NativeFileMode::NATIVE_FILE_MODE_SAVE_FILE: {
			return save_file_dialog->ready();
		}
	}
	return false;
}

void NativeFileDialog::_fetch_results() {
	if (!supported || !waiting) {
		return;
	}
	stored_results.clear();
	switch (active_mode) {
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILE:
			[[fallthrough]];
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILES: {
			std::vector<std::string> results = open_file_dialog->result();
			if (!results.empty()) {
				for (auto r : results) {
					stored_results.push_back(r.c_str());
				}
			}
		} break;
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_DIR: {
			stored_results.push_back(open_dir_dialog->result().c_str());
		} break;
		case NativeFileMode::NATIVE_FILE_MODE_SAVE_FILE: {
			stored_results.push_back(open_dir_dialog->result().c_str());
		} break;
	}
}

inline String NativeFileDialog::_get_first_result() {
	if (stored_results.is_empty()) {
		return String();
	} else {
		return stored_results[0];
	}
}

void NativeFileDialog::_emit_signals_if_necessary() {
	if (waiting && !sent_signal) {
		switch (active_mode) {
			case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILE:
				emit_signal("file_selected", _get_first_result());
				[[fallthrough]];
			case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILES: {
				emit_signal("files_selected", stored_results);
			} break;
			case NativeFileMode::NATIVE_FILE_MODE_OPEN_DIR: {
				emit_signal("dir_selected", _get_first_result());
			} break;
			case NativeFileMode::NATIVE_FILE_MODE_SAVE_FILE: {
				emit_signal("file_selected", _get_first_result());
			} break;
		}
		sent_signal = true;
	}
}

void NativeFileDialog::add_filter(const String &p_filter) {
	ERR_FAIL_COND_MSG(p_filter.begins_with("."), "Filter must be \"filename.extension\", can't start with dot.");
	filters.push_back(p_filter);
	std_filters_built = false;
}

void NativeFileDialog::clear_filters() {
	filters.clear();
	std_filters_built = false;
}

void NativeFileDialog::set_filters(const Vector<String> &p_filters) {
	filters = p_filters;
	std_filters_built = false;
}

bool NativeFileDialog::has_results() {
	if (!supported) {
		return false;
	}
	if (waiting && _has_results()) {
		_fetch_results();
		_emit_signals_if_necessary();
	}
	return _has_results();
}

String NativeFileDialog::get_result() {
	if (!supported) {
		return String();
	}
	if (waiting) {
		_fetch_results();
		_emit_signals_if_necessary();
	}
	return _get_first_result();
}

Vector<String> NativeFileDialog::get_results() {
	if (!supported) {
		return Vector<String>();
	}
	if (waiting) {
		_fetch_results();
		_emit_signals_if_necessary();
	}
	return stored_results;
}

void NativeFileDialog::show() {
	if (!supported) {
		return;
	}
	if (waiting) {
		hide();
	}

	sent_signal = false;
	active_mode = assigned_mode;
	waiting = true;

	switch (active_mode) {
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILE:
			[[fallthrough]];
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILES: {
			_update_std_filters_if_necessary();
			open_file_dialog = new pfd::open_file(
					_std_string_with_fallback(title, RTR(active_mode == NativeFileMode::NATIVE_FILE_MODE_OPEN_FILE ? "Open file" : "Open files")),
					_std_string(start_directory),
					std_filters,
					active_mode == NativeFileMode::NATIVE_FILE_MODE_OPEN_FILES ? pfd::opt::multiselect : pfd::opt::none);
		} break;
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_DIR: {
			open_dir_dialog = new pfd::select_folder(
					_std_string_with_fallback(title, RTR("Open folder")),
					_std_string(start_directory));
		} break;
		case NativeFileMode::NATIVE_FILE_MODE_SAVE_FILE: {
			_update_std_filters_if_necessary();
			save_file_dialog = new pfd::save_file(
					_std_string_with_fallback(title, RTR("Save file")),
					_std_string(start_directory),
					std_filters);
		} break;
	}
}

void NativeFileDialog::hide() {
	if (!supported || !waiting) {
		return;
	}
	waiting = false;
	switch (active_mode) {
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILE:
			[[fallthrough]];
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_FILES: {
			if (!open_file_dialog->ready()) {
				open_file_dialog->kill();
			}
		} break;
		case NativeFileMode::NATIVE_FILE_MODE_OPEN_DIR: {
			if (!open_dir_dialog->ready()) {
				open_dir_dialog->kill();
			}
		} break;
		case NativeFileMode::NATIVE_FILE_MODE_SAVE_FILE: {
			if (!save_file_dialog->ready()) {
				save_file_dialog->kill();
			}
		} break;
	}
}

void NativeFileDialog::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(true);
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (waiting && _has_results()) {
				_fetch_results();
				_emit_signals_if_necessary();
			}
		} break;
	}
}

void NativeFileDialog::_bind_methods() {
	ClassDB::bind_method(D_METHOD("show"), &NativeFileDialog::show);
	ClassDB::bind_method(D_METHOD("hide"), &NativeFileDialog::hide);

	ClassDB::bind_method(D_METHOD("has_results"), &NativeFileDialog::has_results);
	ClassDB::bind_method(D_METHOD("get_result"), &NativeFileDialog::get_result);
	ClassDB::bind_method(D_METHOD("get_results"), &NativeFileDialog::get_results);

	ClassDB::bind_method(D_METHOD("set_start_directory"), &NativeFileDialog::set_start_directory);
	ClassDB::bind_method(D_METHOD("get_start_directory"), &NativeFileDialog::get_start_directory);
	ClassDB::bind_method(D_METHOD("set_file_mode"), &NativeFileDialog::set_file_mode);
	ClassDB::bind_method(D_METHOD("get_file_mode"), &NativeFileDialog::get_file_mode);
	ClassDB::bind_method(D_METHOD("set_title", "title"), &NativeFileDialog::set_title);
	ClassDB::bind_method(D_METHOD("get_title"), &NativeFileDialog::get_title);

	ClassDB::bind_method(D_METHOD("clear_filters"), &NativeFileDialog::clear_filters);
	ClassDB::bind_method(D_METHOD("add_filter", "filter"), &NativeFileDialog::add_filter);
	ClassDB::bind_method(D_METHOD("set_filters", "filters"), &NativeFileDialog::set_filters);
	ClassDB::bind_method(D_METHOD("get_filters"), &NativeFileDialog::get_filters);

	ClassDB::bind_static_method(SNAME("NativeFileDialog"), D_METHOD("is_supported"), &NativeFileDialog::is_supported);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "start_directory"), "set_start_directory", "get_start_directory");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "file_mode", PROPERTY_HINT_ENUM, "Open File,Open Files,Open Folder,Save File"), "set_file_mode", "get_file_mode");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "title"), "set_title", "get_title");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "filters"), "set_filters", "get_filters");

	ADD_SIGNAL(MethodInfo("file_selected", PropertyInfo(Variant::STRING, "path")));
	ADD_SIGNAL(MethodInfo("files_selected", PropertyInfo(Variant::PACKED_STRING_ARRAY, "paths")));
	ADD_SIGNAL(MethodInfo("dir_selected", PropertyInfo(Variant::STRING, "dir")));

	BIND_ENUM_CONSTANT(NATIVE_FILE_MODE_OPEN_FILE);
	BIND_ENUM_CONSTANT(NATIVE_FILE_MODE_OPEN_FILES);
	BIND_ENUM_CONSTANT(NATIVE_FILE_MODE_OPEN_DIR);
	BIND_ENUM_CONSTANT(NATIVE_FILE_MODE_SAVE_FILE);
}

NativeFileDialog::NativeFileDialog() {
	supported = pfd::settings::available();
}

NativeFileDialog::~NativeFileDialog() {
	if (open_file_dialog != nullptr) {
		delete open_file_dialog;
	}
}
